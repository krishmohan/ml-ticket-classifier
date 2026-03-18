import torch


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()  # set model to training mode

    total_loss = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 1. Reset gradients
        optimizer.zero_grad()

        # 2. Forward pass
        outputs = model(X_batch)

        # 3. Compute loss
        loss = criterion(outputs, y_batch)

        # 4. Backward pass (compute gradients)
        loss.backward()

        # 5. Update weights
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()  # evaluation mode

    correct = 0
    total = 0

    with torch.no_grad():  # no gradient computation
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return correct / total
