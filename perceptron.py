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
    'vocab_size': 2,
    'batch_size': 1000,
    'sentence_len': 1,
}

config = default_config

def sampler(logits, temp, straight_through):
    dist = ROHC(temperature=temp, logits=logits)
    y_hard = tf.one_hot(tf.argmax(dist.sample(),-1), config['vocab_size'])
    y = tf.stop_gradient(y_hard - logits) + logits
    softmax = tf.nn.softmax(dist.sample())
    pred = tf.reshape(tf.slice(straight_through, [0,0], [1,1]), ())
    return tf.where(pred, y, softmax)

def ohvs_to_words(ohvs):
    sentence = ""
    for v in ohvs:
        sentence += chr(ord('a')+np.argmax(v))
    return sentence

def main():
    e_inputs = Input(shape=(1,))
    e_temp = Input(shape=(1,), dtype='float32')
    e_st = Input(shape=(1,), dtype='bool')

    e_x = Dense(64,
            kernel_initializer=RandomNormal())(e_inputs)
    e_x = Dense(64, activation='relu')(e_x)
    e_outputs = []
    alt_outputs = []

    for _ in range(config['sentence_len']):
        logits = Dense(config['vocab_size'], activation=None)(e_x)
        alt_outputs.append(logits)
        categorical = lambda x: sampler(x, e_temp, e_st)
        e_outputs.append(keras.layers.Lambda(categorical)(logits))

    e_model = Model(inputs=e_inputs, outputs=e_outputs)
    e_model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    d_input = Dense(8, activation='relu')
    d_inputs = []
    for word in e_outputs:
        d_inputs.append(d_input(word))

    d_x = d_inputs[0]
    d_x = Dense(64, activation='relu')(d_x)
    d_output = Dense(1, activation=None)(d_x)

    model = Model(inputs=[e_inputs, e_temp, e_st], outputs=d_output)
    model.compile(optimizer='rmsprop',
            loss='mean_squared_error',
            metrics=['accuracy'])
    sentence_model = Model(inputs=e_inputs, outputs=alt_outputs)
    sentence_model.compile(optimizer='rmsprop', loss='mean_squared_error')

    test_data = np.transpose([[0., 1., True], [1., 1., True]]).tolist()
    output_data = [0., 1.] * (config['batch_size']//2)
    for i in range(30):
        temp = 2/(i+1)
        input_data = [[0.,1.]*(config['batch_size']//2),
                [temp]*config['batch_size'],
                [True]*config['batch_size']]
        model.fit(input_data,
                output_data, epochs=1)

    predictions = model.predict(test_data)
    sentences = np.array([sentence_model.predict(np.array(test_data)[:,0])])
    for i in range(len(predictions)):
        print(test_data[i][0], ohvs_to_words(sentences[:,i]), predictions[i])
        print(test_data[i][0],sentences[:,i], predictions[i])

    import gc; gc.collect()


if __name__ == "__main__":
    main()
