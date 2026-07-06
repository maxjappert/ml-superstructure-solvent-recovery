import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(30, 128),
                                 nn.ReLU(),
                                 # nn.Dropout(0.5),
                                 nn.Linear(128, 256),
                                 nn.ReLU(),
                                 # nn.Linear(256, 512),
                                 # nn.ReLU(),
                                 # nn.Linear(512, 256),
                                 # nn.ReLU(),
                                 # nn.Dropout(0.5),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 # nn.Dropout(0.5),
                                 nn.Linear(128, 4))

    def forward(self, x):
        output = self.net(x)
        return output
