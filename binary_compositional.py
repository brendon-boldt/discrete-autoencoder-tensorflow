import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Input, Concatenate,
        BatchNormalization, RepeatVector, Lambda, Flatten)
from tensorflow.keras.initializers import RandomNormal
import tensorflow_probability as tfp

ROHC = tfp.distributions.RelaxedOneHotCategorical
np.set_printoptions(precision=2, sign=' ')
#np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def permutations(n):
    ma = [2**i for i in range(n)]
    arr = []
    for i in range(n**2 - 1):
        arr.append([i//b % 2 for b in ma])
    return np.array(arr)

from numpy.random import shuffle
from numpy.linalg import matrix_rank

def tt_split(arr, test_split=1.0):
    indexes = list(range(len(arr)))
    shuffle(indexes)
    train, test = [], []
    n = arr.shape[1]
    covered = []
    for i in indexes:
        mr_covered = matrix_rank(covered)
        if mr_covered == n:
            if len(test)/len(arr) < test_split:
                test.append(i)
            else:
                train.append(i)
        else:
            if matrix_rank(covered + [arr[i]]) > mr_covered:
                covered.append(arr[i])
                train.append(i)
            else:
                test.append(i)

    return train, test 



def sampler(logits, temp, size, straight_through):
    """Sampling function for Gumbel-Softmax"""
    dist = ROHC(temperature=temp, logits=logits)
    sample = dist.sample()
    y_hard = tf.one_hot(tf.argmax(sample, -1), size)
    # y_hard is the value that gets used but the gradient flows through logits
    y = tf.stop_gradient(y_hard - logits) + logits

    # TODO Make this make more sense
    pred = tf.reshape(tf.slice(straight_through, [0,0], [1,1]), ())
    return tf.where(pred, y, sample)

class AgentPair:

    def __init__(self, cfg):
        self.cfg = cfg
        self.sess = tf.Session()

        # Encoder inputs
        e_inputs = Input(shape=(cfg['num_concepts'],), name='e_oh')
        e_temp = Input(shape=(1,), dtype='float32', name='e_temp')
        e_st = Input(shape=(1,), dtype='bool', name='e_st')

        # Generate a static vector space of "concepts"
        e_x = Dense(cfg['input_dim'],
                trainable=False,
                kernel_initializer=RandomNormal(),
                use_bias=False,
                name='concept_space',)(e_inputs)

        # Dense layer for encocder
        e_x = Dense(cfg['e_dense_size'],
                activation='relu',
                name='encoder_h0')(e_x)
        e_x = tf.layers.batch_normalization(e_x, renorm=True)
        # The generic keras BN was NaN'ing, but tf.keras might be okay
        #e_x = BatchNormalization()(e_x)
        e_x = Dense(cfg['vocab_size']*cfg['sentence_len'],
                name="encoder_word_dense")(e_x)
        e_x = tf.keras.layers.Reshape((cfg['sentence_len'],
                cfg['vocab_size']))(e_x)

        # Generate GS sampling layer
        categorical = lambda x: (
            sampler(x, e_temp, cfg['vocab_size'], e_st))
        self.e_output = Lambda(categorical)(e_x)

        
        # Decoder input
        d_x = Flatten(name='decoder_flatten')(self.e_output)
        d_x = Dense(cfg['d_dense_size'],
                activation='relu',
                name='decoder_input')(d_x)
        #d_x1 = BatchNormalization()(d_x0)
        d_x = tf.layers.batch_normalization(d_x, renorm=True)

        d_x = Dense(cfg['input_dim'], activation=None,
                name='decoder_output')(d_x)
        d_output = Dense(cfg['num_concepts'],
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

        #train_input = np.random.permutation(np.repeat(
            #np.identity(cfg['num_concepts']), cfg['batch_size'], axis=0))
        all_input = permutations(cfg['num_concepts'])
        train_i, test_i = tt_split(all_input, cfg['test_prop'])
        train_i = np.repeat(train_i, cfg['batch_size'], axis=0)
        shuffle(train_i)

        train_input = all_input[train_i]
        test_input = all_input[test_i]

        self.train_fd = {
            'e_oh:0': train_input,
            'e_temp:0': [[cfg['temp_init']]],
            'e_st:0': [[cfg['train_st']]],
        }

        self.test_fd = {
            # TODO change this tensor name since it is no longer accurate
            'e_oh:0': test_input,
            'e_temp:0': [[1e-8]],
            'e_st:0': [[1]],
        }

    def run(self):
        self.sess.run(tf.initializers.global_variables())
        for i in range(self.cfg['epochs']):
            self.sess.run(self.train, feed_dict=self.train_fd)
            if i % self.cfg['superepoch'] == 0:
                self.train_fd['e_temp:0'][0][0] *= self.cfg['temp_decay']
                if self.cfg['verbose']:
                    pass

        results = self.sess.run(self.d_sigmoid, feed_dict=self.test_fd)
        test_input = self.test_fd['e_oh:0'] 
        #results = self.sess.run(self.d_sigmoid, feed_dict=self.train_fd)
        #test_input = self.train_fd['e_oh:0'] 
        for i in range(len(test_input)):
            print(f'{test_input[i]} -> {results[i]}')
        print()
        #score = sum([1 for i,r in enumerate(results) if i == np.argmax(r)])
        #print(f"{score}/{self.cfg['num_concepts']}")

default_config = {
    # Actual batch_size == batch_size * num_concepts
    'batch_size': 4,
    'epochs': 2000,
     # How often to anneal temperature
     # More like a traditional epoch due to small dataset size
    'superepoch': 100,
    'e_dense_size': 4,
    'd_dense_size': 4,
    'sentence_len': 4,
    'vocab_size': 2,
    'input_dim': 5,
    'num_concepts': 4,
    'temp_init': 5,
    'temp_decay': 0.9,
    'train_st': 1,
    'test_prop': 0.4,
    
    'verbose': False,
}

if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    cfg = default_config
    for _ in range(3):
        ap = AgentPair(cfg)
        ap.run()
        tf.reset_default_graph()
