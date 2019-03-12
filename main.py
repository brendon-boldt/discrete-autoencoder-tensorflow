import os
import shutil

import tensorflow as tf

import emergence as em


def run_binary_model():
    model_cfg = {
        'epochs': 5000,
        'batch_size': 4,
        'num_concepts': 6,
        'test_prop': 0.2,
        'e_dense_size': 20,
        'sentence_len': 6,
    }
    logdir = 'log'
    if os.path.isdir(logdir):
        shutil.rmtree(logdir)
    model = em.BinaryModel(cfg=model_cfg, logdir=logdir)
    model.run(verbose=True)
    model.test(verbose=True)
    #model.output_test_space(verbose=True)

def run_linear_model():
    model_cfg = {
            #'train_st': True,
            #'test_prop': 0.4,
            }
    logdir = 'log'
    if os.path.isdir(logdir):
        shutil.rmtree(logdir)
    model = em.LinearModel(cfg=model_cfg, logdir=logdir)
    model.run(verbose=True)

    #em.explore.linear.examples(model, 10)
    #em.explore.linear.interactive_test_world(model, )
    #em.explore.linear.interactive_test_utterance(model, )
    #em.explore.linear.get_word_counts(model,)
    #em.explore.linear.test_mutation_locality(model, )

if __name__ == '__main__':
    try:
        #while True:
        for _ in range(10):
            tf.reset_default_graph()
            run_linear_model()
    except KeyboardInterrupt:
        pass
