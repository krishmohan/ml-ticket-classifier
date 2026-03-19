from collections import Counter
import math


class Vocabulary:
    def __init__(self, min_freq=1):
        self.word2idx = {}
        self.idx2word = {}
        self.min_freq = min_freq
        self.idf = {}

    def build(self, texts):
        word_counter = Counter()
        doc_freq = Counter()

        # Count frequencies
        for text in texts:
            tokens = text.lower().split()
            word_counter.update(tokens)

            unique_tokens = set(tokens)
            doc_freq.update(unique_tokens)

        # Build vocab
        idx = 0
        N = len(texts)

        for word, freq in word_counter.items():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word

                # Compute IDF
                df = doc_freq[word]
                self.idf[word] = math.log((N + 1) / (df + 1)) + 1

                idx += 1

    def __len__(self):
        return len(self.word2idx)
