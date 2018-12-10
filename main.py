import shutil

import tensorflow as tf

import emergence as em

if __name__ == '__main__':
    model_cfg = {
        'epochs': 200,
    }
    model = em.BinaryModel(model_cfg)
    logdir = 'log'
    shutil.rmtree(logdir)
    # TODO Improve logging
    tf.summary.FileWriter(logdir, model.sess.graph)
    train, test = model.generate_train_and_test()
    model.train(*train, verbose=True)
    model.test(*test, verbose=True)
