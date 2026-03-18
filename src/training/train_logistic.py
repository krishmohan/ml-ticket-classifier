import torch
from torch.utils.data import DataLoader

from src.data.dataset import TextDataset
from src.features.vocab import Vocabulary
from src.features.vectorizer import vectorize
from src.models.logistic import LogisticRegression
from src.training.train import train_one_epoch, evaluate

import json


# ---------- Helpers ----------
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


# ---------- Load Data ----------
train_data = load_jsonl("data/splits/train.jsonl")
val_data = load_jsonl("data/splits/val.jsonl")


# ---------- Build Vocabulary (ONLY from train) ----------
texts = [x["text"] for x in train_data]

vocab = Vocabulary(min_freq=2)
vocab.build(texts)

print("Vocab size:", len(vocab))


# ---------- Vectorizer Wrapper ----------
def vectorizer_fn(text):
    return vectorize(text, vocab)


# ---------- Dataset ----------
train_dataset = TextDataset(train_data, vectorizer_fn)
val_dataset = TextDataset(val_data, vectorizer_fn)


# ---------- DataLoader ----------
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# ---------- Model ----------
input_dim = len(vocab)
num_classes = 4

model = LogisticRegression(input_dim, num_classes)


# ---------- Loss & Optimizer ----------
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ---------- Training Loop ----------
epochs = 5

for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_acc = evaluate(model, val_loader)

    print(f"Epoch {epoch}: Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")
