import torch
from torch import nn

from config import DEVICE


class Model(nn.Module):
    def __init__(self, output):
        super().__init__()

        self.output = output

        if output == 'feasibility':
            num_outputs = 1
        elif output == 'fractions' or output == 'cost':
            num_outputs = 2
        else:
            print('illegal output specification')
            exit(-1)

        self.net = nn.Sequential(nn.Linear(30, 128),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(128, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(128, num_outputs))

    def forward(self, x):
        y_hat = self.net(x)

        if self.output == 'fractions':
            y_hat = torch.sigmoid(y_hat)

        return y_hat

def load_model(name):
    model = Model()
    model.load_state_dict(torch.load(name)['model_state_dict'])
    model.to(DEVICE)
    return model