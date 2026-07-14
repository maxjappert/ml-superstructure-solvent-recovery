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



    def forward(self, x):
        hidden = self.net_shared(x)
        y_hat_feasibility = self.net_feasibility(hidden)
        y_hat_fractions = self.net_fractions(hidden)

        y_hat_combined = torch.cat([y_hat_feasibility, y_hat_fractions], dim=1)

        return y_hat_combined

def load_model(name):
    model = Model()
    model.load_state_dict(torch.load(name)['model_state_dict'])
    model.to(DEVICE)
    return model