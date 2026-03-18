from collections import Counter


class Vocabulary:
    def __init__(self, min_freq=1):
        self.word2idx = {}
        self.idx2word = {}
        self.min_freq = min_freq

    def build(self, texts):
        counter = Counter()

        for text in texts:
            tokens = text.lower().split()
            counter.update(tokens)

        idx = 0
        for word, freq in counter.items():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def __len__(self):
        return len(self.word2idx)
