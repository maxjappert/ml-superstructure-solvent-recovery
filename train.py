from config import DEVICE
import torch.nn.functional as F

def train_epoch(model, loader, optimizer, output):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        y_hat = model(x)
        if output == 'feasibility':
            loss = F.binary_cross_entropy_with_logits(y_hat, y)
        elif output == 'fractions' or output == 'cost':
            loss = F.mse_loss(y_hat, y)
        else:
            print('wrong output type')
            exit(-1)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x)
    return total / len(loader.dataset)
