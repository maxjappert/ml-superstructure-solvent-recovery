from config import DEVICE, loss_scalar_fractions, loss_scalar_cost
import torch.nn.functional as F

from models import get_losses


def train_epoch(model, loader, optimizer,):
    model.train()
    total_loss_total, total_loss_feasibility, total_loss_recovery, total_loss_purity, total_correct, total_loss_cost_per_kg, total_loss_cost_per_year = 0,0,0,0,0,0,0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        # todo add feasibility masking
        losses = get_losses(model, x, y)

        # todo a bodge but keeps outliers at bay (wherever they might come from)
        if losses['cost_per_kg'] > 1000:
            continue

        total_loss_total += losses['total'].item() * len(x)
        total_loss_feasibility += losses['feasibility'].item() * len(x)
        total_loss_recovery += losses['recovery'].item() * len(x)
        total_loss_purity += losses['purity'].item() * len(x)
        total_loss_cost_per_kg += losses['cost_per_kg'].item() * len(x)

        total_correct += losses['num_correct'].item()

        losses['total'].backward()
        optimizer.step()


    return {
        'total loss': total_loss_total / len(loader.dataset),
        'feasibility loss': total_loss_feasibility / len(loader.dataset),
        'feasibility accuracy': total_correct / len(loader.dataset),
        'recovery loss': total_loss_recovery / len(loader.dataset),
        'purity loss': total_loss_purity / len(loader.dataset),
        'cost per kg loss': total_loss_cost_per_kg / len(loader.dataset),
        # 'cost per year loss': total_loss_cost_per_year / len(loader.dataset)
    }
