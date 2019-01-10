import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Reshape, Concatenate, Dense, Input, Lambda, Flatten, Conv1D)
from tensorflow.keras.initializers import RandomNormal
from tensorflow_probability.python.distributions import RelaxedOneHotCategorical
from numpy.random import shuffle
from numpy.linalg import matrix_rank
from collections import Counter

from .. import util
from ..world.linear import Linear as World

class Linear:
    """A model for describing simple mutations of a 1d world."""

    default_cfg = {
        # Actual batch_size == batch_size * num_concepts
        'batch_size': 20,
        'epochs': 20000,
         # How often to anneal temperature
         # More like a traditional epoch due to small dataset size
        'superepoch': 2000,
        'e_dense_size': 30,
        'd_dense_size': 20,
        'd_hidden_size': 30,
        'world_size': 20,
        'world_depth': 2,
        'world_init_objs': 2,
        'conv_filters': 4,
        'conv_kernel_size': 2,
        'num_worlds': 20000,
        'sentence_len': 2,
        'vocab_size': 20,

        'learning_rate': 1e-2,
        'temp_init': 3,
        'temp_decay': 0.85,
        'train_st': False,
        'test_prop': 0.1,
        'dropout_rate': 0.0,
    }

    def __init__(self, cfg=None, logdir='log'):
        if cfg is None:
            self.cfg = Linear.default_cfg
        else:
            self.cfg = {**Linear.default_cfg, **cfg} 

        self.world_shape = (self.cfg['world_size'], self.cfg['world_depth'])

        self.stored_layers = {}

        self.sess = tf.Session()
        self.initialize_graph()
        self.generate_train_and_test()
        self.train_writer = tf.summary.FileWriter(
                logdir + '/train',
                self.sess.graph
                )
        self.test_writer = tf.summary.FileWriter(
                logdir + '/test',
                self.sess.graph
                )

    def get_layer(self, name, layer_type, *args, **kwargs):
        scope = tf.get_variable_scope().name + '/'
        reuse = tf.get_variable_scope().reuse
        if scope + name in self.stored_layers and reuse:
            return self.stored_layers[scope + name]
        layer = layer_type(*args, name=name, **kwargs)
        if reuse:
            self.stored_layers[scope + name] = layer
        return layer

    def gs_sampler(self, logits):
        """Sampling function for Gumbel-Softmax"""
        dist = RelaxedOneHotCategorical(temperature=self.temperature, logits=logits)
        sample = dist.sample()
        y_hard = tf.one_hot(tf.argmax(sample, -1), self.cfg['vocab_size'])
        # y_hard is the value that gets used but the gradient flows through logits
        y = tf.stop_gradient(y_hard - logits) + logits

        return tf.cond(self.straight_through, lambda: y, lambda: sample)

    def initialize_encoder(self, world_0, world_goal):
        e_conv = self.get_layer(
                'e_conv',
                Conv1D,
                filters=self.cfg['conv_filters'],
                kernel_size=self.cfg['conv_kernel_size'],
                activation='relu',
                use_bias=False,
                )

        e_conv_flatten = self.get_layer(
                'e_conv_flatten',
                Flatten,
                )

        e_world_0 = e_conv_flatten(e_conv(self.world_0))
        e_world_goal = e_conv_flatten(e_conv(self.world_goal))

        e_x = self.get_layer('e_concat', Concatenate)([e_world_0, e_world_goal])
        e_x = self.get_layer(
                'e_conv_dense',
                Dense,
                self.cfg['e_dense_size'],
                activation='relu',
                )(e_x)
        e_x = self.get_layer(
                "e_word_dense",
                Dense,
                self.cfg['vocab_size']*self.cfg['sentence_len'],
                )(e_x)

        e_raw_output = self.get_layer(
                'e_word_reshape',
                Reshape,
                (self.cfg['sentence_len'], self.cfg['vocab_size']),
                )(e_x)
        return e_raw_output

    def initialize_communication(self, raw_outputs):
        categorical_output = self.gs_sampler

        argmax_selector = lambda: tf.one_hot(
                tf.argmax(self.e_raw_output, -1),
                self.e_raw_output.shape[-1]
                )
        gumbel_softmax_selector = lambda: Lambda(categorical_output)(raw_outputs)
        utterance = tf.cond(self.use_argmax,
                argmax_selector,
                gumbel_softmax_selector,
                name="argmax_cond",
                )

        dropout_lambda = Lambda(
                lambda x: tf.layers.dropout(
                    x,
                    noise_shape=(tf.shape(utterance)[0], self.cfg['sentence_len'], 1),
                    rate=self.dropout_rate,
                    training=tf.logical_not(self.use_argmax),
                    ),
                name="dropout_lambda",
                )
        utt_dropout = dropout_lambda(utterance)
        return utterance, utt_dropout

    def initialize_decoder(self, utt_dropout, world_0):
        weight_shape = (
                1,
                self.cfg['vocab_size'],
                self.cfg['d_dense_size'],
                )
        bias_shape = (
                self.cfg['sentence_len'],
                1, # Extra dim used below
                self.cfg['d_dense_size'],
                )
        d_fc_w = tf.get_variable(
                'd_fc_w',
                initializer=tf.initializers.truncated_normal(
                    0., 1e-2)(tf.constant(weight_shape)),
                dtype=tf.float32,
                )
        d_fc_b = tf.get_variable(
                'd_fc_b',
                initializer=tf.constant(1e-1, shape=bias_shape),
                dtype=tf.float32,
                )

        batch_size = tf.shape(utt_dropout)[0]

        tiled = tf.reshape(
                tf.tile(d_fc_w, (batch_size, self.cfg['sentence_len'], 1)),
                (batch_size,) + (self.cfg['sentence_len'],) + weight_shape[1:],
                )
        utt_dropout_reshaped = tf.reshape(
                utt_dropout,
                (-1, self.cfg['sentence_len'], 1, self.cfg['vocab_size']))

        #d_x = tf.nn.relu(tf.matmul(utt_dropout_reshaped, tiled) + d_fc_b)
        d_x = tf.matmul(utt_dropout_reshaped, tiled) + d_fc_b
        #d_x = tf.nn.relu(tf.matmul(utt_dropout_reshaped, tiled))
        d_x = self.get_layer(
                'decoder_flatten',
                Flatten,
                )(d_x)
        d_conv = self.get_layer(
                'd_conv',
                Conv1D,
                filters=self.cfg['conv_filters'],
                kernel_size=self.cfg['conv_kernel_size'],
                activation='relu',
                use_bias=False,
                )

        d_x = self.get_layer(
                'd_concat',
                Concatenate,
                )([d_x, Flatten(name='e_conv_flatten')(d_conv(world_0))])

        # I do not know if I need to get_layer() this
        d_x = tf.layers.batch_normalization(d_x, renorm=True)

        NUM_FILTERS = 8
        d_x = self.get_layer(
                'd_hidden',
                Dense,
                #self.cfg['d_hidden_size'],
                self.cfg['world_size']*NUM_FILTERS,
                activation='relu',
                )(d_x)

        # Currently unused
        d_x_conv = Reshape((self.cfg['world_size'], NUM_FILTERS))(d_x)

        d_filter_w = tf.get_variable(
                'd_filter_w',
                initializer=tf.initializers.truncated_normal(mean=0.,
                    # Fix this stupidity
                    stddev=1e-2)((1, 2, NUM_FILTERS)),
                dtype=tf.float32,
                )

        # Currently unused
        d_x_conv = tf.contrib.nn.conv1d_transpose(
                d_x_prime,
                d_filter_w,
                #(batch_size, self.cfg['world_size'], 2),
                (batch_size, self.cfg['world_size'], 2),
                1,
                name='d_conv_transpose',
                )

        d_x = self.get_layer(
                "decoder_class",
                Dense,
                np.prod(self.world_shape),
                activation=None,
                use_bias=False,
                )(d_x)
        #d_output = d_x_prime
        d_output = tf.reshape(d_x, (-1,) + self.world_shape)
        d_sigmoid = tf.nn.sigmoid(d_output)
        return d_output, d_sigmoid

    def initialize_graph(self):
        #with tf.name_scope("hyperparameters"):
        with tf.variable_scope("hyperparameters", reuse=tf.AUTO_REUSE):
            self.dropout_rate = tf.placeholder(tf.float32, shape=(),
                    name='dropout_rate')
            self.use_argmax = tf.placeholder(tf.bool, shape=(),
                    name='use_argmax')
            self.temperature = tf.placeholder(tf.float32, shape=(),
                    name='temperature')
            self.straight_through = tf.placeholder(tf.bool, shape=(),
                    name='straight_through')

        with tf.name_scope("environment"):
            with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
                self.world_0 = Input(shape=self.world_shape, dtype=tf.float32)
                self.world_goal = Input(shape=self.world_shape, dtype=tf.float32)
                self.e_raw_output = self.initialize_encoder(
                        self.world_0,
                        self.world_goal
                        )
            with tf.variable_scope("communication", reuse=tf.AUTO_REUSE):
                self.utterance, self.utt_dropout = self.initialize_communication(self.e_raw_output)
            with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
                self.d_output, self.d_sigmoid = self.initialize_decoder(self.utt_dropout, self.world_0)

                # Experimental
                self.input_ph = tf.placeholder(name='input_ph',
                        shape=self.utt_dropout.shape, dtype=tf.float32)
                _, self.test_sigmoid = self.initialize_decoder(self.input_ph, self.world_0)

        with tf.variable_scope("training", reuse=tf.AUTO_REUSE):
            optmizier = tf.train.AdamOptimizer(self.cfg['learning_rate'])
            self.loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.d_output, labels=self.world_goal)

            tf.summary.scalar('loss', tf.reduce_mean(self.loss))

            self.train_step = optmizier.minimize(self.loss)

        with tf.variable_scope("testing", reuse=tf.AUTO_REUSE):
            word_correct = tf.cast(
                    tf.math.equal(
                            tf.argmax(self.d_output, -1),
                            tf.argmax(self.world_goal, -1),
                            ),
                    tf.int32
                    )
            self.correct = tf.reduce_prod(word_correct, axis=-1)
            self.accuracy = (
                    #tf.reduce_sum(tf.reduce_prod(self.correct, -1))
                    tf.reduce_sum(self.correct, -1)
                    / tf.shape(self.correct)[0]
                    )

        self.init_op = tf.initializers.global_variables()
        self.sess.run(self.init_op)
        self.summary = tf.summary.merge_all()

    def generate_train_and_test(self, verbose=False):
        data_set = set()
        i = 0
        while len(data_set) < self.cfg['num_worlds']:
            w = World(*self.world_shape, self.cfg['world_init_objs'],
                    unique_objs=False)
            data_set.add((w, w.apply(World.random_swap1())))
            data_set.add((w, w.apply(World.random_create())))
            data_set.add((w, w.apply(World.random_destroy())))
            i += 1
            if i > self.cfg['world_size'] ** 3:
                if verbose:
                    print(f"Warning: only generated "
                    f"{len(data_set)}/{self.cfg['num_worlds']} worlds")
                break

        data_list = list(data_set)
        shuffle(data_list)
        test_data = np.array([
                (w[0].world, w[1].world)
                    for w in data_list[:len(data_list) // 8]
                ]).transpose((1, 0, 2, 3))
        train_data = np.array([
                (w[0].world, w[1].world)
                    for w in data_list[len(data_list) // 8:]
                ]).transpose((1, 0, 2, 3))
        

        self.train_fd = {
            self.world_0.name: train_data[0],
            self.world_goal.name: train_data[1],
            self.temperature.name: self.cfg['temp_init'],
            self.straight_through.name: self.cfg['train_st'],
            self.dropout_rate.name: self.cfg['dropout_rate'],
            self.use_argmax.name: False,
        }
        self.test_fd = {
            self.world_0.name: test_data[0],
            self.world_goal.name: test_data[1],
            self.temperature.name: self.cfg['temp_init'], # Unused
            self.straight_through.name: True, # Unused
            self.dropout_rate.name: 0., # Unused
            self.use_argmax.name: True,
        }
        #return train_data, test_data

    def run(self, verbose=False):
        # The labels are unused because they are the same as the input
        for i in range(self.cfg['epochs']):
            indexes = np.random.choice(np.arange(len(self.train_fd[self.world_0.name]), dtype=np.int64), size=self.cfg['batch_size'])
            epoch_fd = {
                    **self.train_fd,
                    self.world_0.name:
                        self.train_fd[self.world_0.name][indexes],
                    self.world_goal.name:
                        self.train_fd[self.world_goal.name][indexes],
                    }
            self.sess.run(self.train_step, feed_dict=epoch_fd)

            if i % self.cfg['superepoch'] == 0:
                self.train_fd[self.temperature.name] *= self.cfg['temp_decay']

                train_fd_use_argmax = {
                        **self.train_fd,
                        self.use_argmax.name: True
                        }
                superepoch = i // self.cfg['superepoch']
                summary = self.sess.run(
                        self.summary,
                        feed_dict=train_fd_use_argmax
                        )
                self.train_writer.add_summary(summary, superepoch)
                summary = self.sess.run(self.summary, feed_dict=self.test_fd)
                self.test_writer.add_summary(summary, superepoch)

                if verbose:
                    # TODO Fix this redundancy
                    train_loss = self.sess.run(
                            self.loss,
                            feed_dict=train_fd_use_argmax
                            )
                    test_loss = self.sess.run(
                            self.loss,
                            feed_dict=self.test_fd
                            )
                    print(f"superepoch {i // self.cfg['superepoch']}\t"
                          f"train loss: {train_loss.mean():.3f}\t"
                          f"test: {test_loss.mean():.3f}")

    def train(self, inputs, labels=None, verbose=False):
        # The labels are unused because they are the same as the input
        train_fd = {
            self.e_inputs.name: inputs,
            self.temperature.name: self.cfg['temp_init'],
            self.straight_through.name: self.cfg['train_st'],
            self.dropout_rate.name: self.cfg['dropout_rate'],
            self.use_argmax.name: False,
        }

        for i in range(self.cfg['epochs']):
            self.sess.run(self.train_step, feed_dict=train_fd)
            if i % self.cfg['superepoch'] == 0:
                train_fd[self.temperature.name] *= self.cfg['temp_decay']
                if verbose:
                    summary = self.sess.run(self.summary, feed_dict=train_fd)
                    self.file_writer.add_summary(
                            summary,
                            i // self.cfg['superepoch']
                            )
                    # TODO Fix this redundancy
                    loss = self.sess.run(self.loss, feed_dict=train_fd)
                    print(f"superepoch {i // self.cfg['superepoch']}\t"
                          f"training loss: {loss.mean():.3f}")

    def test(self, verbose=False):
        # The labels are unused because they are the same as the input
        all_losses, accuracy = self.sess.run(
                (self.loss, self.accuracy), feed_dict=self.test_fd
                )
        losses = np.apply_along_axis(np.average, -1, all_losses)
        # TODO Add accuracy
        if verbose:
            print(f"test loss: {np.average(losses):.3f}\t"
                  f"acc: {accuracy:.3f}")

    def examples(self, n):
        outputs, raw_utts = self.sess.run((self.d_output, self.utterance), feed_dict=self.test_fd)
        argmaxes = lambda x: np.array([np.argmax(y) for y in x])
        utts = np.array([argmaxes(x) for x in raw_utts])

        indexes = [i for i, x in enumerate(utts) if x[0] == 0]
        for i in range(min(n, outputs.shape[0])):
            print(argmaxes(self.test_fd[self.world_0.name][indexes][i]))
            print(argmaxes(self.test_fd[self.world_goal.name][indexes][i]))

            print(utts[indexes][i])

            print(argmaxes(outputs[indexes][i]))
            print()

    def interactive_test_world(self):
        while True:
            try:
                oh_0 = [int(x) for x in input("w_0\t").split()]
                oh_goal  = [int(x) for x in input("w_goal\t").split()]
            except ValueError:
                pass
            if 99 in oh_0 or 99 in oh_goal:
                break
            oh_0 += [0]*(self.world_shape[0] - len(oh_0))
            oh_goal += [0]*(self.world_shape[0] - len(oh_goal))
            w_0 = np.zeros(self.world_shape)
            w_goal = np.zeros(self.world_shape)
            w_0[np.arange(self.world_shape[0]), oh_0] = 1
            w_goal[np.arange(self.world_shape[0]), oh_goal] = 1

            fd = {
                    **self.test_fd,
                    self.world_0.name: [w_0],
                    self.world_goal.name: [w_goal], 
                    }

            outputs, raw_utts = self.sess.run((self.d_output, self.utterance),
                    feed_dict=fd)
            argmaxes = lambda x: np.array([np.argmax(y) for y in x])
            utts = np.array([argmaxes(x) for x in raw_utts])
            print(utts[0])
            print(argmaxes(outputs[0]))

    def interactive_test_utterance(self):
        w_0 = World(*self.world_shape, self.cfg['world_init_objs'],
                unique_objs=False)
        while True:
            try:
                raw_utt = [int(x) for x in input("utt\t").split()]
            except ValueError:
                pass
            if 99 in raw_utt:
                break
            if 98 in raw_utt:
                w_0 = World(*self.world_shape, self.cfg['world_init_objs'],
                        unique_objs=False)
                continue

            utt = np.zeros(
                    (self.cfg['sentence_len'], self.cfg['vocab_size'])
                    )
            utt[np.arange(self.cfg['sentence_len']), raw_utt] = 1.

            fd = {
                    **self.test_fd,
                    self.world_0.name: [w_0.world],
                    #self.world_goal.name: [w_goal], 
                    self.input_ph.name: [utt],
                    }

            outputs = self.sess.run(self.test_sigmoid,
                    feed_dict=fd)
            argmaxes = lambda x: np.array([np.argmax(y) for y in x])
            print(' '.join(w_0))
            print(' '.join(argmaxes(outputs[0])))
            print()

    def get_word_counts(self):
        fd = {
                **self.test_fd,
                self.world_0.name: self.train_fd[self.world_0.name],
                self.world_goal.name: self.train_fd[self.world_goal.name],
                }
        raw_utts = self.sess.run( self.utterance, feed_dict=fd)
        argmaxes = lambda x: [str(i)+':'+str(np.argmax(y)) for i, y in enumerate(x)]
        #utts = [z for y in [argmaxes(x) for x in raw_utts] for z in y]
        utts = [str(argmaxes(x)) for x in raw_utts]
        counts = Counter(utts)
        import code; code.interact(local=locals())

    def test_mutation_locality(self, n=100):
        print("Generating examples...")
        #mutation = World.create(5, 1)
        #mutation = World.swap(3,-1)
        mutations = []
        for i in range(3):
            mutations += [World.create(i, 1), World.destroy(i)]
        examples = [[] for _ in mutations]
        while min(len(e) for e in examples) < n:
            w_0 = World(*self.world_shape, self.cfg['world_init_objs'],
                    unique_objs=False)
            for e, m in zip(examples, mutations):
                w_1 = w_0.apply(m)
                if w_0 != w_1 and len(e) < n:
                    e.append((w_0, w_1))
        #examples = np.reshape(examples, (len(examples)*n, 2)).transpose()
        print("Done.")

        counts = []
        for e in examples:
            fd = {
                    **self.test_fd,
                    self.world_0.name: [x.world for x in np.transpose(e)[0]],
                    self.world_goal.name: [x.world for x in np.transpose(e)[1]],
                    }
            # TODO Keep track of which ones are correct
            raw_utts, correct = self.sess.run((self.utterance, self.correct), feed_dict=fd)
            argmaxes = lambda x: [str(i)+':'+str(np.argmax(y)) for i, y in enumerate(x)]
            #utts = [z for y in [argmaxes(utt) for utt, c in zip(raw_utts,
                #correct) if c] for z in y]
            utts = []
            for c, utt in zip(correct, raw_utts):
                if c:
                    utts += argmaxes(utt)
            counts.append(Counter(utts))
            #print(counts)
        for i, x in enumerate(counts):
            print(' ' * 5 * i, end='')
            for y in counts[i+1:]:
                print(f'{util.get_word_alignment(x, y):.1f}', end='  ')
            print()
        import code; code.interact(local=locals())
