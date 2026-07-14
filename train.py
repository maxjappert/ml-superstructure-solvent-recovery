from config import DEVICE, loss_scalar_fractions, loss_scalar_cost
import torch.nn.functional as F

from models import get_losses, LossBreakdown


def train_epoch(model, loader, optimizer,):
    model.train()
    total_losses = LossBreakdown.zeros()

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        # todo add feasibility masking
        losses = get_losses(model, x, y)

        # todo a bodge but keeps outliers at bay (wherever they might come from)
        if losses.cost_per_kg_nll > 1000:
            continue

        losses.total.backward()
        optimizer.step()

        total_losses.total += losses.total
        total_losses.feasibility_bce += losses.feasibility_bce
        total_losses.feasibility_brier += losses.feasibility_brier
        total_losses.recovery_nll += losses.recovery_nll
        total_losses.recovery_rmse += losses.recovery_rmse
        total_losses.purity_nll += losses.purity_nll
        total_losses.purity_rmse += losses.purity_rmse
        total_losses.cost_per_kg_nll += losses.cost_per_kg_nll
        total_losses.cost_per_kg_rmse += losses.cost_per_kg_rmse
        total_losses.num_correct += losses.num_correct

    return total_losses