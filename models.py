import torch
from torch import nn

from config import DEVICE


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.net_shared = nn.Sequential(nn.Linear(24, 128),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(128, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(256, 256))

        self.net_feasibility = nn.Sequential(nn.Linear(256, 128),
                                             nn.ReLU(),
                                             nn.Dropout(0.5),
                                             nn.Linear(128, 1),)

        self.net_fractions = nn.Sequential(nn.Linear(256, 128),
                                             nn.ReLU(),
                                             nn.Dropout(0.5),
                                             nn.Linear(128, 4),)

        self.net_cost = nn.Sequential(nn.Linear(256, 128),
                                             nn.ReLU(),
                                             nn.Dropout(0.5),
                                             nn.Linear(128, 2),)




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
