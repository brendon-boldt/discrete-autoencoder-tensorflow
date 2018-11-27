import numpy as np
from binary_compositional import AgentPair 
from hyperopt import hp, tpe, fmin, Trials
import tensorflow as tf

ITERS = 1

count = 0
def do_run(cfg):
    scores = []
    #for k in ['batch_size', 'epochs', 'superepoch', 'e_dense_size',
    #        'd_dense_size']:
    for k in ['e_dense_size', 'd_dense_size']:
        cfg[k] = int(cfg[k])
    #cfg['train_st'] = cfg['train_st'][1]
    for _ in range(ITERS):
        ap = AgentPair(cfg)
        scores.append(ap.get_performance())
        tf.reset_default_graph()
    loss = np.average([s['average'] for s in scores])
    global count
    count += 1
    print(count)
    print(cfg)
    print(loss)
    print()
    return loss

if __name__ == '__main__':
    space = {
        # Actual batch_size == batch_size * num_concepts
        #'batch_size': hp.qlognormal('batch_size', 2.0, 0.3, 1),#4,
        #'epochs': hp.qlognormal('epochs', 8.3, 0.3, 100),#4000,
         # How often to anneal temperature
         # More like a traditional epoch due to small dataset size
        #'superepoch': hp.qlognormal('superepoch', 5.3, 0.2, 10),#200,
        'e_dense_size': hp.qlognormal('e_dense_size', 2.5, 0.4, 1),#20,
        'd_dense_size': hp.qlognormal('d_dense_size', 3., 0.5, 1),#20,
        #'input_dim': 8,
        #'num_concepts': 7,
        #'sentence_len': 7,
        #'vocab_size': 2,

        'temp_init': hp.lognormal('temp_init', 1.2, 0.4),#4,
        'temp_decay': hp.uniform('temp_decay', 0.8, 1),#0.9,
        #'train_st': hp.choice('train_st', [
        #    ('st_false', 0),
        #    ('st_true', 1),
        #]),
        #'test_prop': 0.1,
        'dropout_rate': hp.uniform('dropout_rate', 0, 0.4),#0.3,
        
        #'verbose': True,
    }
    trials = Trials()
    best = fmin(do_run, space=space, max_evals=50, algo=tpe.suggest,
            trials=trials)
    import pdb; pdb.set_trace()
