import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Input, Concatenate
from keras.initializers import RandomNormal
from keras import backend as K
import tensorflow as tf
import tensorflow_probability as tfp

ROHC = tfp.distributions.RelaxedOneHotCategorical

default_config = {
    'batch_size': 1000,
    'epochs': 30,
    'e_d0_size': 12,
    'e_d1_size': 12,
    'd_d0_size': 12,
    'd_d1_size': 12,
    'sentence_len': 4,
    'vocab_size': 2,
    'input_len': 1,
    'input_dim': 1, # Not fully implemented
    'input_vals': [-1., 0., 5.]
}

config = default_config

def sampler(logits, temp, straight_through):
    dist = ROHC(temperature=temp, logits=logits)
    sample = dist.sample()
    y_hard = tf.one_hot(tf.argmax(sample, -1), config['vocab_size'])
    y = tf.stop_gradient(y_hard - logits) + logits

    pred = tf.reshape(tf.slice(straight_through, [0,0], [1,1]), ())
    return tf.where(pred, y, sample)

def ohvs_to_words(ohvs):
    sentence = ""
    for v in ohvs:
        sentence += chr(ord('a')+np.argmax(v))
    return sentence

def main():
    e_inputs = Input(shape=(config['input_len'],))
    e_temp = Input(shape=(1,), dtype='float32')
    e_st = Input(shape=(1,), dtype='bool')

    e_x = Dense(config['e_d0_size'],
            kernel_initializer=RandomNormal(),
            name='encoder_inputs')(e_inputs)
    e_x = Dense(config['e_d1_size'],
            activation='relu',
            name='encoder_h0')(e_x)
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
    d_x = Dense(config['d_d1_size'],
            activation='relu',
            name='decoder_h0')(d_x)
    d_output = Dense(config['input_dim'], activation=None,
            name='decoder_output')(d_x)

    optimizer = keras.optimizers.Adam()
    #optimizer = keras.optimizers.RMSprop(lr=0.01)
    model = Model(inputs=[e_inputs, e_temp, e_st], outputs=d_output)
    model.compile(optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['accuracy'])
    #sentence_model = Model(inputs=e_inputs, outputs=alt_outputs)
    sentence_model = Model(inputs=[e_inputs, e_temp, e_st], outputs=e_outputs)
    sentence_model.compile(optimizer='rmsprop', loss='mean_squared_error')

    #test_data = np.transpose([[0., 1., True], [1., 1., True]]).tolist()
    test_data = [config['input_vals'],
            [1e-3]*len(config['input_vals']),
            [True]*len(config['input_vals'])]
    output_data = config['input_vals']*config['batch_size']
    for i in range(config['epochs']):
        temp = 2/(i+1)
        input_data = [config['input_vals']*config['batch_size'],
                [temp]*config['batch_size']*len(config['input_vals']),
                [True]*config['batch_size']*len(config['input_vals'])]
        model.fit(input_data,
                output_data,
                epochs=1,
                verbose=0)

    predictions = model.predict(test_data)
    sentences = np.array([sentence_model.predict(test_data)])
    if config['sentence_len'] == 1:
        sentences = np.array([[sentences[0]], [sentences[1]]])
    #import pdb; pdb.set_trace()
    for i in range(len(predictions)):
        print(test_data[0][i], ohvs_to_words(sentences[0,:,i]), predictions[i])
        #print(test_data[0][i],sentences[0,:,i], predictions[i])

    sess = keras.backend.get_session()
    del sess



if __name__ == "__main__":
    for _ in range(2):
        main()
        print("=======================")
