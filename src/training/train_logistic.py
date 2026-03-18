import torch
from torch.utils.data import DataLoader

from src.data.dataset import TextDataset
from src.features.vocab import Vocabulary
from src.features.vectorizer import vectorize
from src.models.logistic import LogisticRegression
from src.training.train import train_one_epoch, evaluate

import json
import argparse
import os


# ---------- Helpers ----------
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


# ---------- Main ----------
def main(args):
    # ---------- Device ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- Load Data ----------
    train_data = load_jsonl(args.train_path)
    val_data = load_jsonl(args.val_path)

    # ---------- Build Vocabulary ----------
    texts = [x["text"] for x in train_data]

    vocab = Vocabulary(min_freq=args.min_freq)
    vocab.build(texts)

    print("Vocab size:", len(vocab))

    # ---------- Vectorizer ----------
    def vectorizer_fn(text):
        return vectorize(text, vocab)

    # ---------- Dataset ----------
    train_dataset = TextDataset(train_data, vectorizer_fn)
    val_dataset = TextDataset(val_data, vectorizer_fn)

    # ---------- DataLoader ----------
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # ---------- Model ----------
    input_dim = len(vocab)
    num_classes = args.num_classes

    model = LogisticRegression(input_dim, num_classes).to(device)

    # ---------- Loss & Optimizer ----------
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ---------- Training ----------
    best_acc = 0.0

    os.makedirs(args.artifact_dir, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}: Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")

        # ---------- Save Best Model ----------
        if val_acc > best_acc:
            best_acc = val_acc

            save_path = os.path.join(args.artifact_dir, "logistic_model.pt")
            torch.save(model.state_dict(), save_path)

            print(f"Saved best model → {save_path}")


# ---------- Entry ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default="data/splits/train.jsonl")
    parser.add_argument("--val_path", type=str, default="data/splits/val.jsonl")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=4)

    parser.add_argument("--artifact_dir", type=str, default="artifacts")

    args = parser.parse_args()

    main(args)