import torch
from torch.utils.data import DataLoader

from src.data.dataset import TextDataset
from src.features.vocab import Vocabulary
from src.features.vectorizer import vectorize
from src.models.logistic import LogisticRegression

import json


# ---------- Device Configuration ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- Helpers ----------
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


# ---------- Load Data ----------
val_data = load_jsonl("data/splits/val.jsonl")
train_data = load_jsonl("data/splits/train.jsonl")


# ---------- Rebuild vocab (same as training) ----------
texts = [x["text"] for x in train_data]

vocab = Vocabulary(min_freq=2)  # IMPORTANT: match your experiment
vocab.build(texts)


def vectorizer_fn(text):
    return vectorize(text, vocab)


# ---------- Dataset ----------
val_dataset = TextDataset(val_data, vectorizer_fn)
val_loader = DataLoader(val_dataset, batch_size=32)


# ---------- Load Model ----------
input_dim = len(vocab)
num_classes = 4

model = LogisticRegression(input_dim, num_classes)
model.load_state_dict(torch.load("artifacts/logistic_model.pt", map_location=device))
model.to(device)
model.eval()


# ---------- Collect Errors ----------
errors = []

with torch.no_grad():
    for i, (X_batch, y_batch) in enumerate(val_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)
        preds = outputs.argmax(dim=1)

        for j in range(len(preds)):
            if preds[j] != y_batch[j]:
                idx = i * 32 + j
                sample = val_data[idx]

                errors.append(
                    {
                        "text": sample["text"],
                        "true": y_batch[j].item(),
                        "pred": preds[j].item(),
                    }
                )


# ---------- Print Some Errors ----------
print(f"Total errors: {len(errors)}")

for i in range(10):
    print("\n---")
    print("TEXT:", errors[i]["text"])
    print("TRUE:", errors[i]["true"])
    print("PRED:", errors[i]["pred"])
