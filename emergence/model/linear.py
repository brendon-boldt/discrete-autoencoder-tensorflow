import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Concatenate, Dense, Input, Lambda, Flatten, Conv1D)
from tensorflow.keras.initializers import RandomNormal
from tensorflow_probability.python.distributions import RelaxedOneHotCategorical
from numpy.random import shuffle
from numpy.linalg import matrix_rank

from .. import util
from ..world.linear import Linear as World

class Linear:
    """A model for describing simple mutations of a 1d world."""

    default_cfg = {
        # Actual batch_size == batch_size * num_concepts
        'batch_size': 20,
        'epochs': 10000,
         # How often to anneal temperature
         # More like a traditional epoch due to small dataset size
        'superepoch': 2000,
        'e_dense_size': 50,
        'd_dense_size': 10,
        'world_size': 10,
        'world_depth': 2,
        'world_init_objs': 2,
        'conv_filters': 4,
        'conv_kernel_size': 2,
        'num_worlds': 800,
        'sentence_len': 2,
        'vocab_size': 10,

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

    def gs_sampler(self, logits):
        """Sampling function for Gumbel-Softmax"""
        dist = RelaxedOneHotCategorical(temperature=self.temperature, logits=logits)
        sample = dist.sample()
        y_hard = tf.one_hot(tf.argmax(sample, -1), self.cfg['vocab_size'])
        # y_hard is the value that gets used but the gradient flows through logits
        y = tf.stop_gradient(y_hard - logits) + logits

        return tf.cond(self.straight_through, lambda: y, lambda: sample)

    def initialize_encoder(self):
        self.world_0 = Input(shape=self.world_shape, dtype=tf.float32)
        self.world_goal = Input(shape=self.world_shape, dtype=tf.float32)

        e_conv = Conv1D(
                filters=self.cfg['conv_filters'],
                kernel_size=self.cfg['conv_kernel_size'],
                activation='relu',
                use_bias=False,
                name='e_conv',
                )

        e_conv_flatten = Flatten(name='e_conv_flatten')

        e_world_0 = e_conv_flatten(e_conv(self.world_0))
        e_world_goal = e_conv_flatten(e_conv(self.world_goal))
        e_x = Concatenate()([e_world_0, e_world_goal])
        e_x = Dense(self.cfg['e_dense_size'],
                activation='relu',
                name='e_conv_dense'
                )(e_x)
        e_x = Dense(self.cfg['vocab_size']*self.cfg['sentence_len'],
                name="encoder_word_dense")(e_x)

        self.e_raw_output = tf.keras.layers.Reshape((self.cfg['sentence_len'],
                self.cfg['vocab_size']))(e_x)

    def initialize_communication(self):
        categorical_output = self.gs_sampler

        argmax_selector = lambda: tf.one_hot(
                tf.argmax(self.e_raw_output, -1),
                self.e_raw_output.shape[-1]
                )
        gumbel_softmax_selector = lambda: Lambda(categorical_output)(self.e_raw_output)
        self.utterance = tf.cond(self.use_argmax,
                argmax_selector,
                gumbel_softmax_selector,
                name="argmax_cond",
                )

        dropout_lambda = Lambda(
                lambda x: tf.layers.dropout(
                    x,
                    noise_shape=(tf.shape(self.utterance)[0], self.cfg['sentence_len'], 1),
                    rate=self.dropout_rate,
                    training=tf.logical_not(self.use_argmax),
                    ),
                name="dropout_lambda",
                )
        self.utt_dropout = dropout_lambda(self.utterance)

    def initialize_decoder(self):
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
        d_fc_w = tf.Variable(
                tf.initializers.truncated_normal(
                    0., 1e-2)(tf.constant(weight_shape)),
                dtype=tf.float32,
                expected_shape=weight_shape,
                )
        d_fc_b = tf.Variable(
                tf.constant(1e-1, shape=bias_shape),
                dtype=tf.float32,
                expected_shape=bias_shape,
                )

        batch_size = tf.shape(self.utt_dropout)[0]

        tiled = tf.reshape(
                tf.tile(d_fc_w, (batch_size, self.cfg['sentence_len'], 1)),
                (batch_size,) + (self.cfg['sentence_len'],) + weight_shape[1:],
                )
        utt_dropout_reshaped = tf.reshape(
                self.utt_dropout,
                (-1, self.cfg['sentence_len'], 1, self.cfg['vocab_size']))

        d_x = tf.nn.relu(tf.matmul(utt_dropout_reshaped, tiled) + d_fc_b)
        d_x = Flatten(name='decoder_flatten')(d_x)

        self.world_0 # -> conv -> flatten -> concatenate
        d_conv = Conv1D(
                filters=self.cfg['conv_filters'],
                kernel_size=self.cfg['conv_kernel_size'],
                activation='relu',
                use_bias=False,
                name='d_conv',
                )

        d_x = Concatenate()(
                [d_x, Flatten(name='e_conv_flatten')(d_conv(self.world_0))]
                )


        d_x = tf.layers.batch_normalization(d_x, renorm=True)

        d_x = Dense(
                #np.prod(self.world_shape),
                self.cfg['d_dense_size'],
                activation=None,
                name='decoder_output'
                )(d_x)


        #self.d_output = Dense(self.cfg['num_concepts'],
        d_x = Dense(np.prod(self.world_shape),
                name="decoder_class",
                activation=None,
                use_bias=False,
                )(d_x)
        self.d_output = tf.reshape(d_x, (-1,) + self.world_shape)
        self.d_sigmoid = tf.nn.sigmoid(self.d_output)

    def initialize_graph(self):
        with tf.name_scope("hyperparameters"):
            self.dropout_rate = tf.placeholder(tf.float32, shape=(),
                    name='dropout_rate')
            self.use_argmax = tf.placeholder(tf.bool, shape=(),
                    name='use_argmax')
            self.temperature = tf.placeholder(tf.float32, shape=(),
                    name='temperature')
            self.straight_through = tf.placeholder(tf.bool, shape=(),
                    name='straight_through')

        with tf.name_scope("environment"):
            with tf.name_scope("encoder"):
                self.initialize_encoder()
            with tf.name_scope("communication"):
                self.initialize_communication()
            with tf.name_scope("decoder"):
                self.initialize_decoder()

        with tf.name_scope("training"):
            optmizier = tf.train.AdamOptimizer(self.cfg['learning_rate'])
            self.loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.d_output, labels=self.world_goal)

            tf.summary.scalar('loss', tf.reduce_mean(self.loss))

            self.train_step = optmizier.minimize(self.loss)

        self.init_op = tf.initializers.global_variables()
        self.sess.run(self.init_op)
        self.summary = tf.summary.merge_all()

    def generate_train_and_test(self):
        train_data = []
        test_data = []
        # TODO Do this counting better
        for _ in range(self.cfg['num_worlds']):
            w = World(*self.world_shape, self.cfg['world_init_objs'],
                    unique_objs=False)
            train_data.append((w, w.apply(World.random_swap1())))
            train_data.append((w, w.apply(World.random_create())))
            train_data.append((w, w.apply(World.random_destroy())))
        train_data = list(zip(*train_data))

        for _ in range(self.cfg['num_worlds'] // 8):
            w = World(*self.world_shape, self.cfg['world_init_objs'],
                    unique_objs=False)
            test_data.append((w, w.apply(World.random_swap1())))
            test_data.append((w, w.apply(World.random_create())))
            test_data.append((w, w.apply(World.random_destroy())))
        test_data = list(zip(*test_data))
        

        self.train_fd = {
            self.world_0.name: np.array([w.world for w in train_data[0]]),
            self.world_goal.name: np.array([w.world for w in train_data[1]]),
            self.temperature.name: self.cfg['temp_init'],
            self.straight_through.name: self.cfg['train_st'],
            self.dropout_rate.name: self.cfg['dropout_rate'],
            self.use_argmax.name: False,
        }
        self.test_fd = {
            self.world_0.name: np.array([w.world for w in test_data[0]]),
            self.world_goal.name: np.array([w.world for w in test_data[1]]),
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
                    loss = self.sess.run(
                            self.loss,
                            feed_dict=train_fd_use_argmax
                            )
                    print(f"superepoch {i // self.cfg['superepoch']}\t"
                          f"training loss: {loss.mean():.3f}")

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
        all_losses = self.sess.run(self.loss, feed_dict=self.test_fd)
        losses = np.apply_along_axis(np.average, -1, all_losses)
        # TODO Add accuracy
        if verbose:
            print(f"test loss\t"
                  f"avg: {np.average(losses):.3f}\t"
                  f"max: {np.max(losses):.3f}")

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
        raise NotImplementedError
        while True:
            try:
                raw_utt = [int(x) for x in input("utt\t").split()]
            except ValueError:
                pass
            if 99 in raw_utt:
                break
            if 98 in raw_utt:
                pass # New world

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



