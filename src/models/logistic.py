import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        # Linear layer: input → output
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """
        x: (batch_size, input_dim)
        returns: (batch_size, num_classes)
        """
        return self.linear(x)
