import os
import shutil

import tensorflow as tf

import emergence as em


def run_binary_model():
    model_cfg = {
        'epochs': 5000,
        'batch_size': 6,
        'num_concepts': 8,
        'test_prop': 0.1,
    }
    logdir = 'log'
    if os.path.isdir(logdir):
        shutil.rmtree(logdir)
    model = em.BinaryModel(cfg=model_cfg, logdir=logdir)
    train, test = model.generate_train_and_test()
    model.train(*train, verbose=True)
    model.test(*test, verbose=True)
    model.file_writer.close()

if __name__ == '__main__':
    run_binary_model()
