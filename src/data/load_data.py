from datasets import load_dataset
import os
import json


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


def main():
    print("Loading dataset...")
    dataset = load_dataset("ag_news")

    print("Saving raw data...")

    save_jsonl(dataset["train"], "data/raw/train.jsonl")
    save_jsonl(dataset["test"], "data/raw/test.jsonl")

    print("Done.")


if __name__ == "__main__":
    main()
