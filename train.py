from config import DEVICE
import torch.nn.functional as F

def train_epoch(model, loader, optimizer):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = F.mse_loss(y_hat, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x)
    return total / len(loader.dataset)
