import numpy as np
'''
import keras
from keras.models import Model
from keras.layers import (Dense, Dropout, Input, Concatenate,
        BatchNormalization, RepeatVector, Lambda, Flatten)
from keras.initializers import RandomNormal
from keras import backend as K
'''
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

default_config = {
    'batch_size': 600,
    'epochs': 15,
    'e_d0_size': 30,
    'e_d1_size': 30,
    'd_d0_size': 30,
    'd_d1_size': 30,
    'sentence_len': 8,
    'vocab_size': 2,
    'input_dim': 5,
    'input_vocab_size': 4,
}

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
        e_x0 = Dense(self.cfg['input_dim'],
                trainable=False,
                kernel_initializer=RandomNormal(),
                use_bias=False,
                name='concept_space',)(e_inputs)

        # Dense layer for encocder
        e_x1 = Dense(self.cfg['e_d1_size'],
                activation='relu',
                name='encoder_h0')(e_x0)
        #e_x2 = BatchNormalization()(e_x1)
        e_x2 = tf.layers.batch_normalization(e_x1, renorm=True)
        e_x3 = RepeatVector(self.cfg['sentence_len'])(e_x2)
        e_x4 = Dense(self.cfg['vocab_size'], name="encoder_word_dense")(e_x3)

        categorical = lambda x: (
            sampler(x, e_temp, self.cfg['vocab_size'], e_st))
        e_output = Lambda(categorical)(e_x4)


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
                name='decoder_input')(e_output)
        d_x1 = BatchNormalization()(d_x0)
        d_x2 = Flatten(name='decoder_flatten')(d_x1)

        d_x3 = Dense(self.cfg['input_dim'], activation=None,
                name='decoder_output')(d_x2)
        d_output = Dense(self.cfg['input_vocab_size'],
                name="decoder_class",
                activation=None,)(d_x3)
        d_softmax = tf.nn.softmax(d_output)

        e_inputs = tf.stop_gradient(e_inputs)
        optmizier = tf.train.AdamOptimizer()
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=d_output, labels=e_inputs)
        #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                #logits=d_output, labels=tf.argmax(e_inputs))
        train = optmizier.minimize(loss)

        sess = K.get_session()
        sess.run(tf.initializers.global_variables())

        train_fd = {
            'e_oh:0': np.random.permutation(np.repeat(np.identity(4), 10, axis=0)),
            'e_temp:0': [[5]],
            'e_st:0': [[0]],
        }

        test_fd = {
            'e_oh:0': np.identity(4),
            'e_temp:0': [[1]],
            'e_st:0': [[1]],
        }
        temp_decay = 0.9
        #for i in range(self.cfg['epochs']*self.cfg['batch_size']):
        for i in range(5000):
            sess.run(train, feed_dict=train_fd)
            if i % 500 == 0:
                loss_val =sess.run(loss, feed_dict=test_fd) 
                train_fd['e_temp:0'][0][0] *= temp_decay
                print(loss_val)

        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        results = sess.run(d_softmax, feed_dict=test_fd)
        print(results)


if __name__ == '__main__':
    cfg = default_config
    ap = AgentPair(cfg)
