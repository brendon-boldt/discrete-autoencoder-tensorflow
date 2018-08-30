import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Input, Concatenate
from keras.initializers import RandomNormal
from keras import backend as K
#from tensorflow.contrib.distributions import RelaxedOneHotCategorical
import tensorflow as tf
ROHC = tf.contrib.distributions.RelaxedOneHotCategorical

vocab_size = 2
batch_size = 1000
sentence_len = 1

def e_generator():
    inputs = np.random.randint(0, 6, size=batch_size)
    outputs = np.zeros((batch_size, vocab_size))
    outputs[np.arange(batch_size), inputs] = 1
    return inputs, [outputs]*sentence_len

def sampler(logits, temp, straight_through):
    dist = ROHC(temperature=temp, logits=logits)
    '''
    if straight_through:
        y_hard = tf.one_hot(tf.argmax(dist.sample(),-1), vocab_size)
        y = tf.stop_gradient(y_hard - logits) + logits
        return y
    else:
        softmax = tf.nn.softmax(dist.sample())
        return softmax
    '''
    y_hard = tf.one_hot(tf.argmax(dist.sample(),-1), vocab_size)
    y = tf.stop_gradient(y_hard - logits) + logits
    softmax = tf.nn.softmax(dist.sample())
    pred = tf.reshape(tf.slice(straight_through, [0,0], [1,1]), ())
    return tf.where(pred, y, softmax)
    #return tf.cond(pred,
            #lambda: y,
            #lambda: softmax)

e_inputs = Input(shape=(1,))
e_temp = Input(shape=(1,), dtype='float32')
e_st = Input(shape=(1,), dtype='bool')

e_x = Dense(64,
        #activation='relu',
        kernel_initializer=RandomNormal())(e_inputs)
e_x = Dense(64, activation='relu')(e_x)
e_outputs = []
alt_outputs = []
#for _ in range(num_words):

for _ in range(sentence_len):
    logits = Dense(vocab_size, activation=None)(e_x)
    alt_outputs.append(logits)
    categorical = lambda x: sampler(x, e_temp, e_st)
    e_outputs.append(keras.layers.Lambda(categorical)(logits))

e_model = Model(inputs=e_inputs, outputs=e_outputs)
e_model.compile(optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])


if False:
    e_data = e_generator()
    e_model.fit(e_data[0], e_data[1], epochs=30) 

    test_inputs = np.random.randint(0, 2, size=2)
    test_outputs = np.array(e_model.predict(test_inputs))
    for i in range(len(test_inputs)):
        print(test_inputs[i], '=>', test_outputs[:,i])
    exit()

d_input = Dense(8, activation='relu')
d_inputs = []
for word in e_outputs:
    d_inputs.append(d_input(word))

d_x = d_inputs[0]
#d_x = Concatenate()(d_inputs)
d_x = Dense(64, activation='relu')(d_x)
d_output = Dense(1, activation=None)(d_x)

model = Model(inputs=[e_inputs, e_temp, e_st], outputs=d_output)
model.compile(optimizer='rmsprop',
        loss='mean_squared_error',
        metrics=['accuracy'])
#sentence_model = Model(inputs=e_inputs, outputs=e_outputs)
sentence_model = Model(inputs=e_inputs, outputs=alt_outputs)
sentence_model.compile(optimizer='rmsprop', loss='mean_squared_error')

#data = [0., 1., 2., 3., 4., 5.]
epoch_size = 1000
#input_data = [[0.,1.]*(epoch_size//2), [1.]*epoch_size, [False]]
test_data = np.transpose([[0., 1., True], [1., 1., True]]).tolist()
output_data = [0., 1.] * (epoch_size//2)
for i in range(30):
    temp = 2/(i+1)
    input_data = [[0.,1.]*(epoch_size//2), [temp]*epoch_size, [True]*epoch_size]
    model.fit(input_data,
            output_data, epochs=1)

def ohvs_to_words(ohvs):
    sentence = ""
    for v in ohvs:
        sentence += chr(ord('a')+np.argmax(v))
    return sentence

predictions = model.predict(test_data)
#sentences = np.array(sentence_model.predict(data))
sentences = np.array([sentence_model.predict(np.array(test_data)[:,0])])
for i in range(len(predictions)):
    print(test_data[i][0], ohvs_to_words(sentences[:,i]), predictions[i])
    print(test_data[i][0],sentences[:,i], predictions[i])
