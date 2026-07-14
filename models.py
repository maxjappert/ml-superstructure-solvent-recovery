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



    def forward(self, x):
        y_hat = self.net_shared(x)
        y_hat = self.net_feasibility(y_hat)

        return y_hat

def load_model(name):
    model = Model()
    model.load_state_dict(torch.load(name)['model_state_dict'])
    model.to(DEVICE)
    return model