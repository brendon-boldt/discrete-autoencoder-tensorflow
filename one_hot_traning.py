import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Input, Concatenate,
        BatchNormalization, RepeatVector, Lambda, Flatten)
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import backend as K
import tensorflow_probability as tfp

ROHC = tfp.distributions.RelaxedOneHotCategorical
np.set_printoptions(precision=2, sign=' ')
#np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


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

        # Encoder inputs
        e_inputs = Input(shape=(self.cfg['input_vocab_size'],), name='e_oh')
        e_temp = Input(shape=(1,), dtype='float32', name='e_temp')
        e_st = Input(shape=(1,), dtype='bool', name='e_st')

        # Generate a static vector space of "concepts"
        e_x = Dense(self.cfg['input_dim'],
                trainable=False,
                kernel_initializer=RandomNormal(),
                use_bias=False,
                name='concept_space',)(e_inputs)

        # Dense layer for encocder
        e_x = Dense(self.cfg['e_d1_size'],
                activation='relu',
                name='encoder_h0')(e_x)
        #e_x2 = BatchNormalization()(e_x1)
        #e_x2 = tf.layers.batch_normalization(e_x1)
        e_x = tf.layers.batch_normalization(e_x, renorm=True)
        e_x = Dense(self.cfg['vocab_size']*self.cfg['sentence_len'],
                name="encoder_word_dense")(e_x)
        #e_x3 = RepeatVector(self.cfg['sentence_len'])(e_x2)
        #e_x4 = Dense(self.cfg['vocab_size'], name="encoder_word_dense")(e_x3)
        e_x = tf.keras.layers.Reshape((self.cfg['sentence_len'],
                self.cfg['vocab_size']))(e_x)

        categorical = lambda x: (
            sampler(x, e_temp, self.cfg['vocab_size'], e_st))
        self.e_output = Lambda(categorical)(e_x)


        '''
        # (?,n) => (?,l,v)
        e_output = []
        for i in range(config['sentence_len']):
            logits = Dense(config['vocab_size'],
                    activation=None,
                    name='encoder_logits'+str(i))(e_x)
            #alt_outputs.append(logits)
            categorical = lambda x:
                sampler(x, e_temp, self.cfg['vocab_size'], e_st)
            e_output.append(keras.layers.Lambda(categorical)(logits))
        e_output = tf.stack(e_output)
        '''

        
        # Decoder input
        d_x0 = Dense(self.cfg['d_d0_size'],
                activation='relu',
                name='decoder_input')(self.e_output)
        #d_x1 = BatchNormalization()(d_x0)
        d_x1 = Flatten(name='decoder_flatten')(d_x0)
        d_x2 = tf.layers.batch_normalization(d_x1, renorm=True)

        d_x3 = Dense(self.cfg['input_dim'], activation=None,
                name='decoder_output')(d_x2)
        d_output = Dense(self.cfg['input_vocab_size'],
                name="decoder_class",
                activation=None,)(d_x3)
        self.d_softmax = tf.nn.softmax(d_output)

        e_inputs = tf.stop_gradient(e_inputs)
        optmizier = tf.train.AdamOptimizer()
        #self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                #logits=d_output, labels=e_inputs)
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=d_output, labels=tf.argmax(e_inputs, axis=-1))

        #self.loss = e_inputs * -tf.log(self.d_softmax+1e-8)

        self.train = optmizier.minimize(self.loss)


        train_input = np.random.permutation(np.repeat(
            np.identity(self.cfg['input_vocab_size']), self.cfg['batch_size'], axis=0))

        self.train_fd = {
            'e_oh:0': train_input,
            'e_temp:0': [[self.cfg['temp_init']]],
            'e_st:0': [[self.cfg['train_st']]],
        }

        self.test_fd = {
            'e_oh:0': np.identity(self.cfg['input_vocab_size']),
            'e_temp:0': [[1e-8]],
            'e_st:0': [[1]],
        }
        #for i in range(self.cfg['epochs']*self.cfg['batch_size']):

    def run(self):
        sess = K.get_session()
        sess.run(tf.initializers.global_variables())
        for i in range(self.cfg['epochs']):
            sess.run(self.train, feed_dict=self.train_fd)
            if i % 200 == 0:
                loss_val =sess.run(self.loss, feed_dict=self.test_fd) 
                #print(np.average(loss_val))
                #print('>'+'#'* int(10*np.average(loss_val)))
                e_out = sess.run(self.e_output, feed_dict=self.test_fd)
                unique = np.unique(e_out, axis=0).shape[0]
                #print(f'{unique} ', end='')
                #if unique >= self.cfg['input_vocab_size']:
                self.train_fd['e_temp:0'][0][0] *= self.cfg['temp_decay']
                #self.train_fd['e_temp:0'][0][0] /= self.cfg['temp_decay']

        results = sess.run(self.d_softmax, feed_dict=self.test_fd)
        score = sum([1 for i,r in enumerate(results) if i == np.argmax(r)])
        #my_loss = np.zeros((4,))
        loss_val =sess.run(self.loss, feed_dict=self.test_fd) 
        #print(results)
        #print(loss_val)
        print(f"{score}/{self.cfg['input_vocab_size']}")

default_config = {
    'batch_size': 32 // 4,
    'epochs': 8000,
    #'e_d0_size': 30,
    'e_d1_size': 30,
    'd_d0_size': 30,
    #'d_d1_size': 30,
    'sentence_len': 4,
    'vocab_size': 4,
    'input_dim': 5,
    'input_vocab_size': 10,
    'temp_init': 5,
    'temp_decay': 0.9,
    'train_st': 0,
}

if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    cfg = default_config
    for _ in range(10):
        ag = AgentPair(cfg)
        ag.run()
        K.clear_session()
        tf.reset_default_graph()
