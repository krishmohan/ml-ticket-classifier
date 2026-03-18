import json
from sklearn.model_selection import train_test_split
import os


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


def main():
    print("Loading raw data...")
    data = load_jsonl("data/raw/train.jsonl")

    print("Splitting data...")

    train, temp = train_test_split(data, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    save_jsonl(train, "data/splits/train.jsonl")
    save_jsonl(val, "data/splits/val.jsonl")
    save_jsonl(test, "data/splits/test.jsonl")

    print("Done.")


if __name__ == "__main__":
    main()
