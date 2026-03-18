import numpy as np


def vectorize(text, vocab):
    vector = np.zeros(len(vocab))

    for word in text.lower().split():
        if word in vocab.word2idx:
            idx = vocab.word2idx[word]
            vector[idx] += 1

    return vector
