import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Input, Concatenate,
        BatchNormalization, RepeatVector, Lambda, Flatten)
from tensorflow.keras.initializers import RandomNormal
import tensorflow_probability as tfp
from numpy.random import shuffle
from numpy.linalg import matrix_rank

ROHC = tfp.distributions.RelaxedOneHotCategorical
np.set_printoptions(precision=2, sign=' ')
#np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def permutations(n):
    ma = [2**i for i in range(n)]
    arr = []
    for i in range(2**n):
        arr.append([i//b % 2 for b in ma])
    return np.array(arr)

def tt_split(arr, test_split=1.0):
    indexes = list(range(len(arr)))
    shuffle(indexes)
    train, test = [], []
    n = arr.shape[1]
    covered = []
    for i in indexes:
        mr_covered = matrix_rank(covered)
        if sum(arr[i]) == 0.:
            test.append(i)
            train.append(i)
        if mr_covered == n:
            if len(test)/len(arr) < test_split:
                test.append(i)
            else:
                train.append(i)
        else:
            covered.append(arr[i])
            train.append(i)
            #if matrix_rank(covered + [arr[i]]) > mr_covered:
            #    covered.append(arr[i])
            #    train.append(i)
            #else:
            #    test.append(i)

    return train, test 

def ohvs_to_words(ohvs):
    sentence = ""
    for v in ohvs:
        sentence += chr(ord('a')+np.argmax(v))
    return sentence


def sampler(logits, temp, size, straight_through):
    """Sampling function for Gumbel-Softmax"""
    dist = ROHC(temperature=temp, logits=logits)
    sample = dist.sample()
    y_hard = tf.one_hot(tf.argmax(sample, -1), size)
    # y_hard is the value that gets used but the gradient flows through logits
    y = tf.stop_gradient(y_hard - logits) + logits

    return tf.cond(straight_through, lambda: y, lambda: sample)

def identityInitializer(shape, **kwargs):
    return np.identity(shape[0])

class AgentPair:


    default_cfg = {
        # Actual batch_size == batch_size * num_concepts
        'batch_size': 7,
        'epochs': 8000,
         # How often to anneal temperature
         # More like a traditional epoch due to small dataset size
        'superepoch': 200,
        'e_dense_size': 14,
        'd_dense_size': 22,
        'input_dim': 8,
        'num_concepts': 7,
        'sentence_len': 7,
        'vocab_size': 2,

        'temp_init': 3,
        'temp_decay': 0.85,
        'train_st': 0,
        'test_prop': 0.1,
        'dropout_rate': 0.15,
        
        'verbose': False,
    }
    

    def __init__(self, cfg=None):
        if cfg is None:
            self.cfg = AgentPair.default_cfg
        else:
            self.cfg = {**AgentPair.default_cfg, **cfg} 
        self.sess = tf.Session()

        self.generate_data()
        real_batch_size = self.train_fd['e_input:0'].shape[0]

        dropout_rate = tf.placeholder(tf.float32, shape=(),
                name='dropout_rate')
        use_argmax = tf.placeholder(tf.bool, shape=(),
                name='use_argmax')

        # Encoder inputs
        e_inputs = Input(shape=(self.cfg['num_concepts'],), name='e_input')
        e_temp = tf.placeholder(tf.float32, shape=(), name='e_temp')
        e_st = tf.placeholder(tf.bool, shape=(), name='e_st')

        # Generate a static vector space of "concepts"
        #e_x = Dense(self.cfg['input_dim'],
        e_x = Dense(self.cfg['num_concepts'],
                trainable=False,
                kernel_initializer=RandomNormal(0., 2.),
                #kernel_initializer=identityInitializer,
                use_bias=False,
                name='concept_space',)(e_inputs)

        # Dense layer for encocder
        e_x = Dense(self.cfg['e_dense_size'],
                activation='relu',
                name='encoder_h0')(e_x)
        #e_x = tf.layers.batch_normalization(e_x, renorm=True)
        # The generic keras BN was NaN'ing, but tf.keras might be okay
        #e_x = BatchNormalization()(e_x)
        e_x = Dense(self.cfg['vocab_size']*self.cfg['sentence_len'],
                name="encoder_word_dense")(e_x)


        e_x = tf.keras.layers.Reshape((self.cfg['sentence_len'],
                self.cfg['vocab_size']))(e_x)

        # Generate GS sampling layer
        categorical = lambda x: (
            sampler(x, e_temp, self.cfg['vocab_size'], e_st))
        self.e_output = tf.cond(use_argmax,
                lambda: tf.one_hot(tf.argmax(e_x, -1), e_x.shape[-1]),
                lambda: Lambda(categorical)(e_x))

        if True:
            #self.e_output = Lambda(lambda x: tf.layers.dropout(self.e_output,
            self.e_output = Lambda(lambda x: tf.layers.dropout(x,
                    #noise_shape=(real_batch_size, self.cfg['sentence_len'], 1),
                    noise_shape=(tf.shape(self.e_output)[0], self.cfg['sentence_len'], 1),
                    rate=dropout_rate,
                    training=tf.logical_not(use_argmax),))(self.e_output)
        else:
            self.e_output = tf.layers.Dropout(
                    noise_shape=(real_batch_size, self.cfg['sentence_len'], 1),
                    #noise_shape=(real_batch_size, 6, 1),
                    rate=dropout_rate,
                    training=tf.logical_not(use_argmax),)(self.e_output)
        
        # Decoder input
        d_x = Flatten(name='decoder_flatten')(self.e_output)
        d_x = Dense(self.cfg['d_dense_size'],
                activation='relu',
                name='decoder_input')(d_x)
        #d_x1 = BatchNormalization()(d_x0)
        d_x = tf.layers.batch_normalization(d_x, renorm=True)

        d_x = Dense(self.cfg['input_dim'], activation=None,
                name='decoder_output')(d_x)
        d_output = Dense(self.cfg['num_concepts'],
                name="decoder_class",
                activation=None,)(d_x)
        #self.d_softmax = tf.nn.softmax(d_output)
        self.d_sigmoid = tf.nn.sigmoid(d_output)

        e_inputs = tf.stop_gradient(e_inputs)
        optmizier = tf.train.AdamOptimizer()
        #self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                #logits=d_output, labels=tf.argmax(e_inputs, axis=-1))
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_output, labels=e_inputs)

        self.train = optmizier.minimize(self.loss)

    def generate_data(self):
        all_input = permutations(self.cfg['num_concepts'])
        train_i, test_i = tt_split(all_input, self.cfg['test_prop'])
        train_i = np.repeat(train_i, self.cfg['batch_size'], axis=0)
        shuffle(train_i)

        train_input = all_input[train_i]
        test_input = all_input[test_i]

        self.train_fd = {
            'e_input:0': train_input,
            'e_temp:0': self.cfg['temp_init'],
            'e_st:0': self.cfg['train_st'],
            'dropout_rate:0': self.cfg['dropout_rate'],
            'use_argmax:0': False,
        }

        self.test_fd = {
            'e_input:0': test_input,
            #'e_input:0': [[0,0,0,0]]*10,
            'e_temp:0': 1e-8, # Not used
            'e_st:0': 1,
            'dropout_rate:0': 0.,
            'use_argmax:0': True,
        }

    def interactive_run(self):
        self.sess.run(tf.initializers.global_variables())
        for i in range(self.cfg['epochs']):
            self.sess.run(self.train, feed_dict=self.train_fd)
            if i % self.cfg['superepoch'] == 0:
                self.train_fd['e_temp:0'] *= self.cfg['temp_decay']
                if self.cfg['verbose']:
                    loss = self.sess.run(self.loss, feed_dict=self.train_fd)
                    print(loss.mean())

        results = self.sess.run(self.d_sigmoid, feed_dict=self.test_fd)
        utt = self.sess.run(self.e_output, feed_dict=self.test_fd)
        test_input = self.test_fd['e_input:0'] 
        #results = self.sess.run(self.d_sigmoid, feed_dict=self.train_fd)
        #test_input = self.train_fd['e_input:0'] 
        #print(self.sess.run(self.e_output, feed_dict=self.test_fd))
        all_losses = self.sess.run(self.loss, feed_dict=self.test_fd)
        losses = np.apply_along_axis(np.average, -1, all_losses)
        print('\ntest_loss')
        print(np.average(losses), np.max(losses))
        for i in range(len(test_input)):
            sent = ohvs_to_words(utt[i])
            #print(f'{test_input[i]} -> {sent} -> {results[i]}')
        if np.average(losses) < 0.1:
            inputs = permutations(self.cfg['num_concepts'])
            fd = {
                'e_input:0': inputs,
                #'e_input:0': [[0,0,0,0]]*10,
                'e_temp:0': 1e-8, # Not used
                'e_st:0': 1,
                'dropout_rate:0': 0.,
                'use_argmax:0': True,
            }
            for i in range(2**self.cfg['num_concepts']):
                utt = self.sess.run(self.e_output, feed_dict=fd)
                sent = ohvs_to_words(utt[i])
                print(f'{inputs[i]} -> {sent} -> {results[i]}')
                results = self.sess.run(self.d_sigmoid, feed_dict=fd)
        return np.average(losses) 
        #score = sum([1 for i,r in enumerate(results) if i == np.argmax(r)])
        #print(f"{score}/{self.cfg['num_concepts']}")

    def get_performance(self):
        self.sess.run(tf.initializers.global_variables())
        for i in range(self.cfg['epochs']):
            self.sess.run(self.train, feed_dict=self.train_fd)
            if i % self.cfg['superepoch'] == 0:
                self.train_fd['e_temp:0'] *= self.cfg['temp_decay']

        all_losses = self.sess.run(self.loss, feed_dict=self.test_fd)
        losses = np.apply_along_axis(np.average, -1, all_losses)
        #print(np.average(losses), np.max(losses))
        return {
            'average': np.average(losses),
            'max': np.max(losses),
        }

if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    cfg = {
        'verbose': True,
        'dropout_rate': 0.1,
    }
    results = []
    for i in range(2):
        print(f'{i}: ', end='')
        ap = AgentPair(cfg)
        results.append(ap.interactive_run())
        tf.reset_default_graph()

