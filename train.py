from config import DEVICE
import torch.nn.functional as F

def train_epoch(model, loader, optimizer,):
    model.train()
    total_loss_total, total_loss_feasibility, total_loss_recovery, total_loss_purity, total_correct = 0,0,0,0,0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        y_hat = model(x)
        y_hat_feasibility = y_hat[:,0]
        y_hat_recovery_mu = y_hat[:,1]
        y_hat_recovery_logvar = y_hat[:,2]
        y_hat_purity_mu = y_hat[:,3]
        y_hat_purity_logvar = y_hat[:,4]
        loss_feasibility = F.binary_cross_entropy_with_logits(y_hat_feasibility, y[:,0])
        loss_recovery = F.gaussian_nll_loss(y_hat_recovery_mu, y[:,1], y_hat_recovery_logvar.exp(), reduction='mean')
        loss_purity = F.gaussian_nll_loss(y_hat_purity_mu, y[:,2], y_hat_purity_logvar.exp(), reduction='mean')

        loss_total = loss_feasibility + loss_recovery + loss_purity

        total_loss_total += loss_total.item() * len(x)
        total_loss_feasibility += loss_feasibility.item() * len(x)
        total_loss_recovery += loss_recovery.item() * len(x)
        total_loss_purity += loss_purity.item() * len(x)

        preds = (F.sigmoid(y_hat_feasibility) > 0.5)
        num_correct = (preds == y[:,0]).sum()
        total_correct += num_correct

        loss_total.backward()
        optimizer.step()
    return {
        'total loss': total_loss_total / len(loader.dataset),
        'feasibility loss': total_loss_feasibility / len(loader.dataset),
        'feasibility accuracy': total_correct / len(loader.dataset),
        'recovery loss': total_loss_recovery / len(loader.dataset),
        'purity loss': total_loss_purity / len(loader.dataset),
    }
