from src.data.dataset import TextDataset
from src.features.vocab import Vocabulary
from src.features.vectorizer import vectorize

# dummy data
data = [
    {"text": "apple product", "label": 1},
    {"text": "new launch", "label": 0},
]

# build vocab
texts = [x["text"] for x in data]
vocab = Vocabulary(min_freq=1)
vocab.build(texts)


# vectorizer wrapper
def vectorizer_fn(text):
    return vectorize(text, vocab)


# dataset
dataset = TextDataset(data, vectorizer_fn)

# test
print("Dataset size:", len(dataset))

X, y = dataset[0]

print("X shape:", X.shape)
print("y:", y)


from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=2, shuffle=True)

for X_batch, y_batch in loader:
    print("Batch X shape:", X_batch.shape)
    print("Batch y shape:", y_batch.shape)
