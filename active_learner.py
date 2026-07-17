import os

import torch
from torch.utils.data import DataLoader, Subset

from config import ACTIVE_LR, ACTIVE_WEIGHT_DECAY, ACTIVE_NUM_DATA_POOL, SEED, ACTIVE_NUM_NEW_DATA, ACTIVE_BATCH_SIZE, \
    NUM_WORKERS, VAL_BATCH_SIZE, ACTIVE_NUM_EPOCHS, DEVICE
from create_dataset import create_dataset
from datasets import Dataset
from models import Model, get_ensemble_predictions, StreamComposition, new_ensemble, \
    get_ensemble_predictions_from_tensor

from models import load_ensemble
from train import train_ensemble


def get_stream(row):
    return StreamComposition(target_name=row['target_name'],
                             target_kgph=row['target_kgph'],
                             solvent2_name=row['solvent2_name'],
                             solvent2_kgph=row['solvent2_kgph'],
                             water_kgph=row['water_kgph'],
                             salt_name=row['salt_name'],
                             salt_kgph=row['salt_kgph'],
                             solids_kgph=row['solids_kgph'])

def acquisition_function(ensemble: list, datapool_set: Dataset, num_returned: int):

    rows_with_corresponding_epistemic_uncertainties = dict()

    # todo implement this with batch processing
    for idx in range(len(datapool_set)):
        # print(f'{idx+1}/{len(datapool_set)}')
        X, y = datapool_set[idx]
        prediction = get_ensemble_predictions_from_tensor(ensemble, X.unsqueeze(0))

        total_epistemic_uncertainty = (prediction.feasibility['epistemic']
                                       + prediction.recovery['epistemic']
                                       + prediction.purity['epistemic']
                                       + prediction.cost_per_kg['epistemic'])

        rows_with_corresponding_epistemic_uncertainties[idx] = total_epistemic_uncertainty

    sorted_keys = [pair[0] for pair in sorted(rows_with_corresponding_epistemic_uncertainties.items(), key=lambda item: item, reverse=True)]

    datapool_set.X = datapool_set.X[sorted_keys]
    datapool_set.y = datapool_set.y[sorted_keys]

    return datapool_set

name_input = '5_ensemble_20260715_141341.pt'

loaded = torch.load(name_input)

M = loaded['hparams']['M']

ensemble = load_ensemble(name_input)

optimisers = [torch.optim.Adam(model.parameters(), lr=ACTIVE_LR, weight_decay=ACTIVE_WEIGHT_DECAY) for model in ensemble]

dataset_train = Dataset('train_small')
dataset_val = Dataset('val_small')
val_loader = DataLoader(dataset_val, batch_size=VAL_BATCH_SIZE, num_workers=NUM_WORKERS)

name_output = name_input + '_post'

best_val_loss = float('inf')

for generation in range(1000):

    datapool_name = f'temp_datapool'
    # generate new data
    create_dataset(datapool_name, ACTIVE_NUM_DATA_POOL, SEED)

    datapool_set = Dataset(datapool_name)

    data_selected = acquisition_function(ensemble, datapool_set, ACTIVE_NUM_NEW_DATA)

    os.remove(os.path.join('data', datapool_name+'.csv'))

    dataset_train.append(data_selected)

    print(len(dataset_train))

    # throw away old weights
    ensemble, _, val_losses_list = train_ensemble(DataLoader(dataset_train, batch_size=ACTIVE_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
                   val_loader, num_epochs=ACTIVE_NUM_EPOCHS, verbose=True)

    val_loss = min([loss.total.mean().item() for loss in val_losses_list])
    print(val_loss)
