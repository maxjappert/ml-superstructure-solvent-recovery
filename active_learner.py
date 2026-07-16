import os

import torch

from config import ACTIVE_LR, ACTIVE_WEIGHT_DECAY, ACTIVE_NUM_DATA_POOL, SEED
from create_dataset import create_dataset
from datasets import Dataset
from models import Model

name_input = '5_ensemble_20260715_141341.pt'

loaded = torch.load(name_input)

M = loaded['hparams']['M']

models = []
for i in range(M):
    models.append(Model())
    models[i].load_state_dict(loaded['model_state_dict'])

optimisers = [torch.optim.Adam(model.parameters(), lr=ACTIVE_LR, weight_decay=ACTIVE_WEIGHT_DECAY) for model in models]

dataset_train = Dataset('train')
dataset_val = Dataset('val')
dataset_test = Dataset('test')

name_output = name_input + 'post'

for generation in range(1000):
    # generate new data
    create_dataset(os.path.join(name_output, f'train{generation}'), ACTIVE_NUM_DATA_POOL, SEED)



    # select the subset of this data which maximises the acquisition function
