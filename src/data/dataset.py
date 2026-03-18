import torch


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, vectorizer):
        """
        data: list of dicts → [{"text": ..., "label": ...}, ...]
        vectorizer: function(text) → numpy vector
        """
        self.data = data
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text = item["text"]
        label = item["label"]

        # Step 1: vectorize text
        vector = self.vectorizer(text)

        # Step 2: convert to tensor
        X = torch.tensor(vector, dtype=torch.float32)

        # Step 3: label to tensor
        y = torch.tensor(label, dtype=torch.long)

        return X, y
