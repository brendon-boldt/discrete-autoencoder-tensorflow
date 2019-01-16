import numpy as np
from collections import Counter

from .. import util
from ..world.linear import Linear as World

def examples(model, n):
    raise NotImplementedError
    outputs, raw_utts = model.sess.run((model.d_output, model.utterance), feed_dict=model.test_fd)
    argmaxes = lambda x: np.array([np.argmax(y) for y in x])
    utts = np.array([argmaxes(x) for x in raw_utts])

    indexes = [i for i, x in enumerate(utts) if x[0] == 0]
    for i in range(min(n, outputs.shape[0])):
        print(argmaxes(model.test_fd[model.next_batch][0][indexes][i]))
        print(argmaxes(model.test_fd[model.next_batch][1][indexes][i]))
        #print(argmaxes(model.test_fd[model.world_0.name][indexes][i]))
        #print(argmaxes(model.test_fd[model.world_goal.name][indexes][i]))

        print(utts[indexes][i])

        print(argmaxes(outputs[indexes][i]))
        print()

def interactive_test_world(model):
    raise NotImplementedError
    while True:
        try:
            oh_0 = [int(x) for x in input("w_0\t").split()]
            oh_goal  = [int(x) for x in input("w_goal\t").split()]
        except ValueError:
            pass
        if 99 in oh_0 or 99 in oh_goal:
            break
        oh_0 += [0]*(model.world_shape[0] - len(oh_0))
        oh_goal += [0]*(model.world_shape[0] - len(oh_goal))
        w_0 = np.zeros(model.world_shape)
        w_goal = np.zeros(model.world_shape)
        w_0[np.arange(model.world_shape[0]), oh_0] = 1
        w_goal[np.arange(model.world_shape[0]), oh_goal] = 1

        fd = {
                **model.test_fd,
                model.world_0.name: [w_0],
                model.world_goal.name: [w_goal], 
                }

        outputs, raw_utts = model.sess.run((model.d_output, model.utterance),
                feed_dict=fd)
        argmaxes = lambda x: np.array([np.argmax(y) for y in x])
        utts = np.array([argmaxes(x) for x in raw_utts])
        print(utts[0])
        print(argmaxes(outputs[0]))

def interactive_test_utterance(model):
    raise NotImplementedError
    w_0 = World(*model.world_shape, model.cfg['world_init_objs'],
            unique_objs=False)
    while True:
        try:
            raw_utt = [int(x) for x in input("utt\t").split()]
        except ValueError:
            pass
        if 99 in raw_utt:
            break
        if 98 in raw_utt:
            w_0 = World(*model.world_shape, model.cfg['world_init_objs'],
                    unique_objs=False)
            continue

        utt = np.zeros(
                (model.cfg['sentence_len'], model.cfg['vocab_size'])
                )
        utt[np.arange(model.cfg['sentence_len']), raw_utt] = 1.

        fd = {
                **model.test_fd,
                model.world_0.name: [w_0.world],
                model.d_only_utt.name: [utt],
                }

        outputs = model.sess.run(model.d_only_sigmoid,
                feed_dict=fd)
        argmaxes = lambda x: np.array([np.argmax(y) for y in x])
        print(' '.join(str(x) for x in  w_0.as_argmax()))
        print(' '.join(str(x) for x in  argmaxes(outputs[0])))
        print()

def get_word_counts(model):
    data = np.concatenate((
            model.world_pairs_train,
            model.world_pairs_valid,
            model.world_pairs_test,
            ))
            
    fd = {model.world_pair_ph: data}
    model.sess.run(model.ds_eval_iter.initializer, feed_dict=fd)
    raw_utts, correct = model.sess.run((model.utterance, model.correct), feed_dict=model.eval_fd)

    raw_utts = [x for x, y in zip(raw_utts, correct) if y] 
    argmaxes = lambda x: [str(i)+':'+str(np.argmax(y)) for i, y in enumerate(x)]
    #utts = [z for y in [argmaxes(x) for x in raw_utts] for z in y]
    utts = [str(argmaxes(x)) for x in raw_utts]
    words = [str(x) for y in raw_utts for x in argmaxes(y)]
    utt_counts = Counter(utts)
    word_counts = Counter(words)
    for k in utt_counts:
        utt_counts[k] /= np.shape(raw_utts)[0]
    for k in word_counts:
        word_counts[k] /= (np.shape(raw_utts)[0] * np.shape(raw_utts)[1])
        
    print(utt_counts)
    print(word_counts)
    import code; code.interact(local=locals())

def test_mutation_locality(model, n=100):
    raise NotImplementedError
    print("Generating examples...")
    #mutation = World.create(5, 1)
    #mutation = World.swap(3,-1)
    mutations = []
    for i in range(3):
        mutations += [World.create(i, 1), World.destroy(i)]
    examples = [[] for _ in mutations]
    while min(len(e) for e in examples) < n:
        w_0 = World(*model.world_shape, model.cfg['world_init_objs'],
                unique_objs=False)
        for e, m in zip(examples, mutations):
            w_1 = w_0.apply(m)
            if w_0 != w_1 and len(e) < n:
                e.append((w_0, w_1))
    #examples = np.reshape(examples, (len(examples)*n, 2)).transpose()
    print("Done.")

    counts = []
    for e in examples:
        fd = {
                **model.test_fd,
                model.world_0.name: [x.world for x in np.transpose(e)[0]],
                model.world_goal.name: [x.world for x in np.transpose(e)[1]],
                }
        # TODO Keep track of which ones are correct
        raw_utts, correct = model.sess.run((model.utterance, model.correct), feed_dict=fd)
        argmaxes = lambda x: [str(i)+':'+str(np.argmax(y)) for i, y in enumerate(x)]
        #utts = [z for y in [argmaxes(utt) for utt, c in zip(raw_utts,
            #correct) if c] for z in y]
        utts = []
        for c, utt in zip(correct, raw_utts):
            if c:
                utts += argmaxes(utt)
        counts.append(Counter(utts))
        #print(counts)
    for i, x in enumerate(counts):
        print(' ' * 5 * i, end='')
        for y in counts[i+1:]:
            print(f'{util.get_word_alignment(x, y):.1f}', end='  ')
        print()
    import code; code.interact(local=locals())

def mock_decode(world_0, utt):
    w = world_0.copy()
    if utt[0] == 0:
        # Sakujo
        w[(utt[1], utt[1]), (0, 1)] = 1, 0
    elif utt[0] == 1:
        # Create
        w[(utt[1], utt[1]), (0, 1)] = 0, 1
    elif utt[0] == 2:
        # Swap right
        temp = w[utt[1]]
        w[utt[1]] = w[utt[1] + 1]
        w[utt[1] + 1] = temp
    elif utt[0] == 3:
        # Swap left
        temp = w[utt[1]]
        w[utt[1]] = w[utt[1] - 1]
        w[utt[1] - 1] = temp
    return w

def test_encoder_only(model):
    pass
