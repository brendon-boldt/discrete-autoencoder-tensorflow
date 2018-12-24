import os
import shutil

import tensorflow as tf
import random

import emergence as em

def get_data(size=3):
    data = []
    for i in range(size):
        for j in range(size):
            arr = [0]*size*2
            arr[i], arr[size+j] = 1, 1
            data.append(arr)
    random.shuffle(data)
    #return data[:-size], data[-size:]
    return data[1:], data[:1]

def run_binary_model():
    model_cfg = {
        'epochs': 3000,
        'batch_size': 6,
        'num_concepts': 6,
        'test_prop': 0.2,
        'e_dense_size': 4,
        'd_dense_size': 2,
        'sentence_len': 2,
        'vocab_size': 6,
        #'learning_rate': 1e-3
    }
    logdir = 'log'
    if os.path.isdir(logdir):
        shutil.rmtree(logdir)
    model = em.BinaryModel(cfg=model_cfg, logdir=logdir)

    train, test = get_data(size=3)
    model.set_train_data(train)
    model.set_test_data(test)

    model.run(verbose=False)
    test_result = model.test(verbose=False)
    if test_result < 0.05:
        print(test)
        print(test_result)
        model.output_test_space(verbose=True)
    tf.reset_default_graph()
    del model

if __name__ == '__main__':
    while True:
        run_binary_model()
