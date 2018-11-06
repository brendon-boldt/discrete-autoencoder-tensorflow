import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Input, Concatenate, BatchNormalization
from keras.initializers import RandomNormal
from keras import backend as K
import tensorflow as tf
import tensorflow_probability as tfp

ROHC = tfp.distributions.RelaxedOneHotCategorical
#np.set_printoptions(precision=2, sign=' ')
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def astr(arr):
    """Convert an array to nice string"""
    strs = ["%.2f " % (x,) for x in arr]
    return " ".join(strs)

default_config = {
    'batch_size': 600,
    'epochs': 15,
    'e_d0_size': 30,
    'e_d1_size': 30,
    'd_d0_size': 30,
    'd_d1_size': 30,
    'sentence_len': 8,
    'vocab_size': 2,
    'input_dim': 100,
    'input_vocab_size': 100,
}

config = default_config

class DistConfig:
    def __init__(self):
        self.m_mean = 0.0
        self.m_variance = 1.0
        self.v_mean = 0.0
        self.v_variance = 0.25

def generate_embeddings(size, dims, config=None):
    config = config or DistConfig()
    means = np.random.normal(config.m_mean, config.m_variance, (dims,))
    variances = np.random.lognormal(config.v_mean, config.v_variance, (dims,))
    words = []
    for _ in range(size):
        words.append(np.random.multivariate_normal(means, np.diag(variances)))
    return np.array(words)
#import pdb; pdb.set_trace()


config['input_vals'] = generate_embeddings(
        config['input_vocab_size'],
        config['input_dim'])

def sampler(logits, temp, straight_through):
    dist = ROHC(temperature=temp, logits=logits)
    sample = dist.sample()
    y_hard = tf.one_hot(tf.argmax(sample, -1), config['vocab_size'])
    y = tf.stop_gradient(y_hard - logits) + logits

    pred = tf.reshape(tf.slice(straight_through, [0,0], [1,1]), ())
    return tf.where(pred, y, sample)

def difference_matrix(m, f):
    sz = m.shape[0]
    dm = np.zeros((sz,sz))
    for i in range(sz):
        for j in range(sz):
            dm[i][j] = f(m[i], m[j])
    return dm

def ohvs_to_words(ohvs):
    sentence = ""
    for v in ohvs:
        sentence += chr(ord('a')+np.argmax(v))
    return sentence

def main():
    #e_inputs = Input(shape=(config['input_dim'],))
    e_inputs = Input(shape=(config['input_vocab_size'],), name='oh_input')
    e_temp = Input(shape=(1,), dtype='float32')
    e_st = Input(shape=(1,), dtype='bool')

    # Translate OH to conept/embedding
    e_x = Dense(config['input_dim'],
            trainable=False,
            kernel_initializer=RandomNormal(),
            use_bias=False,
            name='concept_space',)(e_inputs)

    ''' Layer not needed
    e_x = Dense(config['e_d0_size'],
            kernel_initializer=RandomNormal(),
            name='encoder_inputs')(e_inputs)
    '''
    e_x = Dense(config['e_d1_size'],
            activation='relu',
            name='encoder_h0')(e_x)
    e_x = BatchNormalization()(e_x)
    e_outputs = []
    alt_outputs = []

    for i in range(config['sentence_len']):
        logits = Dense(config['vocab_size'],
                activation=None,
                name='encoder_logits'+str(i))(e_x)
        alt_outputs.append(logits)
        categorical = lambda x: sampler(x, e_temp, e_st)
        e_outputs.append(keras.layers.Lambda(categorical)(logits))

    e_model = Model(inputs=e_inputs, outputs=e_outputs)
    e_model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    d_input = Dense(config['d_d0_size'],
            activation='relu',
            name='decoder_input')
    d_inputs = []
    for word in e_outputs:
        d_inputs.append(d_input(word))

    # Keras doesn't like an array size of 1
    if config['sentence_len'] == 1:
        d_x = d_inputs[0]
    else:
        d_x = Concatenate()(d_inputs)
    ''' Don't need
    d_x = Dense(config['d_d1_size'],
            activation='relu',
            name='decoder_h0')(d_x)
    '''
    d_x = BatchNormalization()(d_x)
    d_output = Dense(config['input_dim'], activation=None,
            name='decoder_output')(d_x)
    d_class = Dense(config['input_vocab_size'],
            name="decoder_class",
            activation='softmax',)(d_output)

    optimizer = keras.optimizers.Adam()
    #optimizer = keras.optimizers.RMSprop(lr=0.01)
    #model = Model(inputs=[e_inputs, e_temp, e_st], outputs=d_output)
    model = Model(inputs=[e_inputs, e_temp, e_st], outputs=d_class)
    model.compile(optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['accuracy'])
    #sentence_model = Model(inputs=e_inputs, outputs=alt_outputs)
    sentence_model = Model(inputs=[e_inputs, e_temp, e_st], outputs=e_outputs)
    #sentence_model.compile(optimizer='rmsprop', loss='mean_squared_error')
    sentence_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    test_data = [np.identity(config['input_vocab_size']),
            np.repeat([1e-3], len(config['input_vals'])),
            np.repeat([True], len(config['input_vals']))]
    #output_data = np.repeat(config['input_vals'], config['batch_size'], axis=0)
    output_data = np.repeat(np.identity(config['input_vocab_size']), config['batch_size'], axis=0)

    cosine_dist = lambda x, y: np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

    temp = 25
    temp_decay = 0.8
    for i in range(config['epochs']):
        #input_data = [np.repeat(config['input_vals'], config['batch_size'], axis=0),
        input_data = [np.repeat(np.identity(config['input_vocab_size']),config['batch_size'], axis=0),
                np.repeat([[temp]], config['batch_size']*config['input_vocab_size'], axis=0),
                np.repeat([[False]], config['batch_size']*config['input_vocab_size'], axis=0)]
        #import pdb; pdb.set_trace()
        model.fit(input_data,
                output_data,
                epochs=1,
                verbose=0,
                shuffle=True,)
        sentences = np.array([sentence_model.predict(test_data)])
        unique = np.unique(sentences, axis=2).shape[2]
        print(unique, end=' ')
        if unique >= config['input_vals'].shape[0]:
            temp *= temp_decay
        else:
            temp /= temp_decay
    print()
            

    predictions = model.predict(test_data)
    sentences = np.array([sentence_model.predict(test_data)])
    if config['sentence_len'] == 1:
        pass
        #sentences = np.array([[sentences[0]], [sentences[1]]])
    unique = np.unique(sentences, axis=2).shape[2]
    '''
    if unique != config['input_vocab_size']:
        for i in range(len(predictions)):
            #print(test_data[0][i], ohvs_to_words(sentences[0,:,i]), predictions[i])
            print(ohvs_to_words(sentences[0,:,i]), predictions[i])
            #print(test_data[0][i],sentences[0,:,i], predictions[i])
        #print(model.layers[1].weights[0].eval(K.get_session()))
        weights = model.layers[1].weights[0].eval(K.get_session())
        #print(difference_matrix(weights, lambda x,y: np.linalg.norm(x-y)))
        #print(difference_matrix(weights, cosine_dist))
    else:
        print("Converged.")
    '''

    sess = K.get_session()
    del sess



if __name__ == "__main__":
    for _ in range(10):
        main()
        print("=======================")
