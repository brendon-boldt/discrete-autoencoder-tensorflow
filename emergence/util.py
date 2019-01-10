import numpy as np

def ohvs_to_words(ohvs):
    sentence = ""
    for v in ohvs:
        sentence += chr(ord('a')+np.argmax(v))
    return sentence

def get_word_alignment(x, y):
    total = 0
    matched = 0
    for k in x:
        total += x[k]
        matched += min(x[k], y.get(k, 0))
    if total == 0:
        return np.nan
    return matched / total
