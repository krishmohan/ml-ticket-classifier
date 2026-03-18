from src.features.vocab import Vocabulary
from src.features.vectorizer import vectorize

texts = ["apple launches product", "new product launch", "apple product"]

vocab = Vocabulary(min_freq=1)
vocab.build(texts)

print("Vocab:", vocab.word2idx)

sample = "apple product"
vec = vectorize(sample, vocab)

print("Vector:", vec)
