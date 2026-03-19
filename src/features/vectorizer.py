import numpy as np


def vectorize(text, vocab):
    vector = np.zeros(len(vocab))

    tokens = text.lower().split()

    # Count term frequency
    tf_counter = {}

    for word in tokens:
        if word in vocab.word2idx:
            tf_counter[word] = tf_counter.get(word, 0) + 1

    # Compute TF-IDF
    for word, tf in tf_counter.items():
        idx = vocab.word2idx[word]
        idf = vocab.idf[word]

        vector[idx] = tf * idf

    return vector
