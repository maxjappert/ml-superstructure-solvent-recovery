import torch
from torch import nn
import torch.nn.functional as F

from config import DEVICE, loss_scalar_fractions, loss_scalar_cost


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        dropout_rate = 0.2

        self.net_shared = nn.Sequential(nn.Linear(24, 128),
                                 nn.ReLU(),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(128, 256),
                                 nn.ReLU(),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(256, 512),
                                 nn.ReLU())

        self.net_feasibility = nn.Sequential(nn.Linear(512, 256),
                                             nn.ReLU(),
                                             nn.Dropout(dropout_rate),
                                             nn.Linear(256, 128),
                                             nn.ReLU(),
                                             nn.Dropout(dropout_rate),
                                             nn.Linear(128, 1),)

        self.net_fractions = nn.Sequential(nn.Linear(512, 256),
                                             nn.ReLU(),
                                             nn.Dropout(dropout_rate),
                                             nn.Linear(256, 128),
                                             nn.ReLU(),
                                             nn.Dropout(dropout_rate),
                                             nn.Linear(128, 4),
                                             nn.ReLU())

        self.net_cost = nn.Sequential(nn.Linear(512, 256),
                                             nn.ReLU(),
                                             nn.Dropout(dropout_rate),
                                             nn.Linear(256, 128),
                                             nn.ReLU(),
                                             nn.Dropout(dropout_rate),
                                             nn.Linear(128, 2),
                                             nn.ReLU())




    def forward(self, x):
        hidden = self.net_shared(x)
        y_hat_feasibility = self.net_feasibility(hidden)
        y_hat_fractions = self.net_fractions(hidden)
        y_hat_cost = self.net_cost(hidden)

        # y_hat_combined = torch.cat([y_hat_feasibility, y_hat_fractions, y_hat_cost], dim=1)

        return {
            'feasibility': y_hat_feasibility.squeeze(),
            'recovery_mu': y_hat_fractions[:,0],
            'recovery_logvar': y_hat_fractions[:,1],
            'purity_mu': y_hat_fractions[:,2],
            'purity_logvar': y_hat_fractions[:,3],
            'cost_per_kg_mu_z': y_hat_cost[:,0],
            'cost_per_kg_logvar_z': y_hat_cost[:,1],
            # 'cost_per_year_mu_z': y_hat_cost[:,2],
            # 'cost_per_year_logvar_z': y_hat_cost[:,3]
        }

        return y_hat_combined

def load_model(name):
    model = Model()
    model.load_state_dict(torch.load(name)['model_state_dict'])
    model.to(DEVICE)
    return model


def get_losses(model, x, y):
    y_hat = model(x)

    loss_feasibility = F.binary_cross_entropy_with_logits(y_hat['feasibility'], y[:, 0])
    loss_recovery = loss_scalar_fractions * F.gaussian_nll_loss(y_hat['recovery_mu'], y[:, 1],
                                                                y_hat['recovery_logvar'].exp(), reduction='mean')
    loss_purity = loss_scalar_fractions * F.gaussian_nll_loss(y_hat['purity_mu'], y[:, 2], y_hat['purity_logvar'].exp(),
                                                              reduction='mean')
    loss_cost_per_kg = loss_scalar_cost * F.gaussian_nll_loss(y_hat['cost_per_kg_mu_z'], y[:, 3],
                                                              y_hat['cost_per_kg_logvar_z'].exp(), reduction='mean')

    loss_total = loss_feasibility + loss_recovery + loss_purity + loss_cost_per_kg  # + loss_cost_per_year

    preds = (torch.sigmoid(y_hat['feasibility']) > 0.5)
    num_correct = (preds == y[:, 0]).sum()

    return {
        'total': loss_total,
        'feasibility': loss_feasibility,
        'num_correct': num_correct,
        'recovery': loss_recovery,
        'purity': loss_purity,
        'cost_per_kg': loss_cost_per_kg,
    }