import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Reshape, Concatenate, Dense, Input, Lambda, Flatten, Conv1D)
from tensorflow.keras.initializers import RandomNormal
from tensorflow_probability.python.distributions import RelaxedOneHotCategorical
from numpy.random import shuffle, randint
from numpy.linalg import matrix_rank
from collections import Counter

from .. import util
from ..world.linear import Linear as World

class Linear:
    """A model for describing simple mutations of a 1d world."""

    default_cfg = {
        # Actual batch_size == batch_size * num_concepts
        'batch_size': 20,
        'iters': 20000,
        'iters_per_epoch': 2000,
         # How often to anneal temperature
         # More like a traditional epoch due to small dataset size
        'e_dense_size': 40,
        'd_dense_size': 40,
        'd_hidden_size': 40,
        'world_size': 10,
        'world_depth': 2,
        #'world_init_objs': 2,
        'world_init_objs': (2,2+1),
        'conv_filters': 4,
        'conv_kernel_size': 3,
        'num_worlds': 20000,

        'sentence_len': 2,
        'vocab_size': 10,

        'learning_rate': 1e-2,
        'temp_init': 10,#3,
        'temp_decay': 1-2e-1,#1 - 15e-2,
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
        self.train_writer = tf.summary.FileWriter(
                logdir + '/train',
                self.sess.graph
                )
        self.valid_writer = tf.summary.FileWriter(
                logdir + '/valid',
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

    def init_eval_iter(self, worlds):
        fd = {
                self.world_placeholders[k]: v
                for k, v in worlds.items()
                }
        self.sess.run(self.ds_eval_iter.initializer, feed_dict=fd)

    def initialize_datasets(self):
        data_set = set()
        attempts = 0
        while len(data_set) < self.cfg['num_worlds']:
            init_objs = randint(*self.cfg['world_init_objs'])
            w = World(*self.world_shape, init_objs,
                    unique_objs=False)
            worlds = [w]
            #w_create = w.apply(World.random_create())
            #w_destroy = w.apply(World.random_destroy())
            #w_swap = w.apply(World.random_swap1())

            # TODO include random world

            if attempts % 4 == 0:
                worlds.append(w.apply(World.random_create()))
            if attempts % 4 == 1:
                worlds.append(w.apply(World.random_destroy()))
            if attempts % 4 == 2:
                worlds.append(w.apply(World.random_swap1()))
            if attempts % 4 == 3:
                while True:
                    w_swap = w.apply(World.random_swap1())
                    if w_swap != worlds[-1]:
                        worlds.append(w_swap)
                        break
            # Add with cartesian product?
            for i in range(len(worlds)):
                for j in range(i, len(worlds)):
                    data_set.add((
                        w,
                        worlds[i],
                        worlds[j],
                        i == j,
                        ))
            attempts += 1
            # TODO use better heruristic
            if attempts >  2 * self.cfg['world_size'] ** 3:
                if False:
                    print(f"Warning: only generated "
                    f"{len(data_set)}/{self.cfg['num_worlds']} worlds")
                break
        print(len(data_set))

        data_list = list(data_set)
        shuffle(data_list)
        data_list = [
                    (w[0].world, w[1].world, w[2].world, w[3])
                    for w in data_list
                    ]
        part_size = int(self.cfg['test_prop'] * len(data_list))
        def to_dict(arr):
            d = {
                    'world_0': [],
                    'world_goal': [],
                    'world_target': [],
                    'label': [],
                    }
            for tup in arr:
                d['world_0'].append(tup[0])
                d['world_goal'].append(tup[1])
                d['world_target'].append(tup[2])
                d['label'].append(tup[3])
            return {
                    'world_0': np.array(
                        d['world_0'],
                        dtype=np.float32),
                    'world_goal': np.array(
                        d['world_goal'],
                        dtype=np.float32),
                    'world_target': np.array(
                        d['world_target'],
                        dtype=np.float32),
                    'label': np.array(
                        d['label'],
                        dtype=np.bool)
                    }

        self.world_pairs_train = to_dict(data_list[2 * part_size:])
        self.world_pairs_test = to_dict(data_list[:part_size])
        self.world_pairs_valid = to_dict(data_list[part_size:2 * part_size])

        # TODO make this an intializable iterator if memory is an issue
        ds_train = tf.data.Dataset.from_tensor_slices(self.world_pairs_train)
        ds_train = ds_train.repeat()
        ds_train = ds_train.shuffle(self.cfg['iters'] * self.cfg['batch_size'])
        ds_train = ds_train.batch(self.cfg['batch_size'])
        ds_train_iter = ds_train.make_one_shot_iterator()
        ds_train_handle = self.sess.run(ds_train_iter.string_handle())

        # Here I left off; I am pretty sure the next thing I wanted to do was
        # to implement a classification game instead of reconstructing the
        # world as the task. I also wanted to determine if there was
        # a difference between learning against distractor images or just doing
        # straight classfication.

        self.world_placeholders = {
                'world_0': tf.placeholder(
                        tf.float32, 
                        name="world_0",
                        shape=(None, *self.world_shape),
                        ),
                'world_goal': tf.placeholder(
                        tf.float32, 
                        name="world_goal",
                        shape=(None, *self.world_shape),
                        ),
                'world_target': tf.placeholder(
                        tf.float32, 
                        name="world_target",
                        shape=(None, *self.world_shape),
                        ),
                'label': tf.placeholder(
                        tf.bool, 
                        name="world_label",
                        shape=(None,),
                        ),
                }
        ds_eval = tf.data.Dataset.from_tensors(self.world_placeholders)
        self.ds_eval_iter = ds_eval.make_initializable_iterator()
        ds_eval_handle = self.sess.run(self.ds_eval_iter.string_handle())

        ds_handle = tf.placeholder(tf.string, shape=())
        ds_iterator = tf.data.Iterator.from_string_handle(
                ds_handle,
                ds_train.output_types,
                ds_train.output_shapes,
                )

        self.train_fd = {
            ds_handle: ds_train_handle,
            self.temperature.name: self.cfg['temp_init'],
            self.straight_through: self.cfg['train_st'],
            self.dropout_rate: self.cfg['dropout_rate'],
            self.use_argmax: False,
        }
        self.eval_fd = {
            ds_handle: ds_eval_handle,
            self.temperature.name: self.cfg['temp_init'], # Unused
            self.straight_through: True, # Unused
            self.dropout_rate: 0., # Unused
            self.use_argmax: True,
        }

        return ds_iterator.get_next()

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

        e_world_0 = e_conv_flatten(e_conv(world_0))
        e_world_goal = e_conv_flatten(e_conv(world_goal))

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

    def initialize_decoder(self, utt_dropout, world_0, world_target):
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
        d_utt_hidden = self.get_layer(
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
        d_conv_flatten = self.get_layer(
                'd_conv_flatten',
                Flatten,
                )

        w_0_flat = d_conv_flatten(d_conv(world_0))
        w_target_flat = d_conv_flatten(d_conv(world_target))

        d_world_combined = self.get_layer(
                'd_concat',
                Concatenate,
                )([w_0_flat, w_target_flat])

        # I do not know if I need to get_layer() this
        # Where to add this back in?
        #d_x = tf.layers.batch_normalization(d_x, renorm=True)

        d_utt_embed = self.get_layer(
                'd_utt_to_embed',
                Dense,
                self.cfg['d_hidden_size'],
                activation='relu',
                )(d_utt_hidden)
        d_world_embed = self.get_layer(
                'd_world_to_embed',
                Dense,
                self.cfg['d_hidden_size'],
                activation='relu',
                )(d_world_combined)

        d_everything = self.get_layer(
                'd_concat_final',
                Concatenate,
                )([d_world_embed, d_utt_embed])

        # One article said to check out CCA

        # Currently unused
        #d_x_conv = Reshape((self.cfg['world_size'], NUM_FILTERS))(d_x)

        d_x = self.get_layer(
                "decoder_class",
                Dense,
                1,
                activation=None,
                use_bias=False,
                )(d_everything)

        return tf.squeeze(d_x)

    def initialize_graph(self):
        with tf.variable_scope("hyperparameters", reuse=tf.AUTO_REUSE):
            self.dropout_rate = tf.placeholder(tf.float32, shape=(),
                    name='dropout_rate')
            self.use_argmax = tf.placeholder(tf.bool, shape=(),
                    name='use_argmax')
            self.temperature = tf.placeholder(tf.float32, shape=(),
                    name='temperature')
            self.straight_through = tf.placeholder(tf.bool, shape=(),
                    name='straight_through')

        with tf.variable_scope("dataset", reuse=tf.AUTO_REUSE):
            inputs = self.initialize_datasets()
            # For debugging
            self.inputs = inputs

        with tf.name_scope("environment"):
            with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
                self.e_raw_output = self.initialize_encoder(
                        inputs['world_0'],
                        inputs['world_goal'],
                        )
            with tf.variable_scope("communication", reuse=tf.AUTO_REUSE):
                self.utterance, self.utt_dropout = self.initialize_communication(self.e_raw_output)
            with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
                #self.d_output, self.d_sigmoid = self.initialize_decoder(self.utt_dropout, inputs['world_0'])
                self.d_output = self.initialize_decoder(
                        self.utt_dropout,
                        inputs['world_0'],
                        inputs['world_target']
                        )

                #self.d_only_utt = tf.placeholder(name='d_only_utt',
                #        shape=self.utt_dropout.shape, dtype=tf.float32)
                #self.d_only_output, self.d_only_sigmoid = self.initialize_decoder(
                #        self.d_only_utt,
                #        inputs['world_0']
                #        )

        with tf.variable_scope("training", reuse=tf.AUTO_REUSE):
            optmizier = tf.train.AdamOptimizer(self.cfg['learning_rate'])
            self.loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.cast(inputs['label'], tf.float32),
                    logits=self.d_output,
                    )
            tf.summary.scalar('loss', tf.reduce_mean(self.loss))
            self.train_step = optmizier.minimize(self.loss)

            #d_only_optimizer = tf.train.AdamOptimizer(self.cfg['learning_rate'])
            #d_only_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            #        logits=self.d_only_output,
            #        labels=inputs['world_goal']
            #        )
            #self.d_only_train_step = d_only_optimizer.minimize(d_only_loss)

        with tf.variable_scope("evaluation", reuse=tf.AUTO_REUSE):
            word_correct = tf.cast(
                    tf.math.equal(
                            tf.argmax(self.d_output, -1),
                            tf.argmax(inputs['world_goal'], -1),
                            ),
                    tf.int32
                    )
            self.correct = tf.reduce_prod(word_correct, axis=-1)
            self.accuracy = (
                    tf.reduce_sum(self.correct, -1)
                    / tf.shape(self.correct)[0]
                    )

        self.init_op = tf.initializers.global_variables()
        self.sess.run(self.init_op)
        self.summary = tf.summary.merge_all()

    def run(self, verbose=False):
        loss_avg = 0.
        loss_count = 0
        for i in range(self.cfg['iters']):
            _, loss = self.sess.run(
                    (self.train_step, self.loss),
                    feed_dict=self.train_fd
                    )
            loss_count += 1
            loss_avg = loss.mean()/loss_count + loss_avg * (1 - 1/loss_count)


            if i % self.cfg['iters_per_epoch'] == 0:
                if i != 0:
                    self.train_fd[self.temperature.name] *= self.cfg['temp_decay']

                train_loss = loss_avg
                loss_avg, loss_count = 0, 0


                # TODO find a better way to record this
                summary = tf.Summary(value=[tf.Summary.Value(tag="training/loss", simple_value=train_loss)])
                self.train_writer.add_summary(summary, i)
                self.init_eval_iter(self.world_pairs_valid)
                summary, valid_loss = self.sess.run((self.summary, self.loss), feed_dict=self.eval_fd)
                self.valid_writer.add_summary(summary, i)

                if verbose:
                    print(f"epoch {i//self.cfg['iters_per_epoch']} \t"
                          f"train loss: {train_loss.mean():.3f}\t"
                          f"valid: {valid_loss.mean():.3f}")
        #self.test(verbose=verbose)
        self.test(verbose=True)

    def test(self, verbose=False):
        self.init_eval_iter(self.world_pairs_test)
        all_losses = self.sess.run(
                self.loss, feed_dict=self.eval_fd
                )
        #all_losses, accuracy = self.sess.run(
        #        (self.loss, self.accuracy), feed_dict=self.eval_fd
        #        )
        losses = np.apply_along_axis(np.average, -1, all_losses)
        if verbose:
            print(f"loss: {np.average(losses):.3f}")
            #print(f"test acc: {accuracy:.3f}\t"
            #      f"loss: {np.average(losses):.3f}")
