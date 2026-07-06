import torch
import torch.nn.functional as F
from config import DEVICE


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total_loss += F.mse_loss(logits, y).item() * len(x)
    return total_loss / len(loader.dataset)
