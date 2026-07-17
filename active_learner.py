import os

import torch
from torch.utils.data import DataLoader

from config import ACTIVE_LR, ACTIVE_WEIGHT_DECAY, ACTIVE_NUM_DATA_POOL, SEED, ACTIVE_NUM_NEW_DATA, ACTIVE_BATCH_SIZE, \
    NUM_WORKERS, VAL_BATCH_SIZE, ACTIVE_NUM_EPOCHS
from create_dataset import create_dataset
from datasets import Dataset
from models import Model, get_ensemble_predictions, StreamComposition, new_ensemble

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

def acquisition_function(models: list, datapool_set: Dataset, num_returned: int):

    rows_with_corresponding_epistemic_uncertainties = dict()

    for idx in range(len(datapool_set)):
        X, y = datapool_set[idx]
        prediction = get_ensemble_predictions(models, get_stream(X), X['temperature_C'],
                                 [X['solid_removal_idx', X['recovery_idx'], X['purification_idx'], X['refinement_idx']]])

        total_epistemic_uncertainty = (prediction.feasibility['epistemic']
                                       + prediction.recovery['epistemic']
                                       + prediction.purity['epistemic']
                                       + prediction.cost_per_kg['epistemic'])

        rows_with_corresponding_epistemic_uncertainties[idx] = total_epistemic_uncertainty

    sorted_keys = sorted(rows_with_corresponding_epistemic_uncertainties.items(), key=lambda item: item[1], reverse=True)


    return datapool_set[sorted_keys][:num_returned]

name_input = '5_ensemble_20260715_141341.pt'

loaded = torch.load(name_input)

M = loaded['hparams']['M']

ensemble = load_ensemble(name_input)

for i in range(M):
    ensemble.append(Model())
    ensemble[i].load_state_dict(loaded['model_state_dicts'][i])

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

    os.remove(os.path.join('data', datapool_name))

    dataset_train.append(data_selected)

    # throw away old weights
    ensemble, _, val_losses_list = train_ensemble(DataLoader(dataset_train, batch_size=ACTIVE_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
                   val_loader, num_epochs=ACTIVE_NUM_EPOCHS)

    val_loss = min(val_losses_list)
    print(val_loss)
