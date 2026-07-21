import json
import os
import random
import sys

import numpy as np
from torch.utils.data import DataLoader

from config import VAL_BATCH_SIZE, NUM_WORKERS
from datasets import Dataset
from train import train_ensemble


def main(filename):
    candidates = dict()

    candidates['batch_size'] = [128, 256, 512, 1024, 2048, 4096, 8192]
    candidates['learning_rate'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    candidates['weight_decay'] =  [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    candidates['dropout_rate'] =  [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    dataset_train = Dataset('train')
    dataset_val = Dataset('val')
    loader_val = DataLoader(dataset_val, batch_size=VAL_BATCH_SIZE, num_workers=NUM_WORKERS)

    results = dict()

    while True:

        batch_size = random.sample(candidates['batch_size'], 1)[0]
        learning_rate = random.sample(candidates['learning_rate'], 1)[0]
        weight_decay = random.sample(candidates['weight_decay'], 1)[0]
        dropout_rate = random.sample(candidates['dropout_rate'], 1)[0]

        trial = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'dropout_rate': dropout_rate,
        }

        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
        ensemble, train_losses, val_losses = train_ensemble(loader_train,
                                                            loader_val,
                                                            M=5,
                                                            num_epochs=5,
                                                            verbose=False,
                                                            lr=learning_rate,
                                                            weight_decay=weight_decay,
                                                            dropout_rate=dropout_rate)

        results[val_losses[-1].total.mean().item()] = trial

        with open(filename+'.json', "w") as f:
            json.dump(results, f)

        print(f'Size {len(results)}')
        print(results)



if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        with (open('hp_optimisation.json') as f):
            hps = json.load(f)

    sorted_float_keys = sorted([float(loss) for loss in hps.keys()])

    print('losses ranked: \n')

    for i in range(len(sorted_float_keys)):

        if sorted_float_keys[i] == float('inf'):
            continue

        print(f'Number {i+1} is val loss {sorted_float_keys[i]:.2} with {hps[str(sorted_float_keys[i])]}')
