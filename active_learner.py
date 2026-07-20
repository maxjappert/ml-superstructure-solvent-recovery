import os
from threading import current_thread

import torch
from torch.utils.data import DataLoader, Subset

from config import ACTIVE_LR, ACTIVE_WEIGHT_DECAY, ACTIVE_NUM_DATA_POOL, SEED, ACTIVE_NUM_NEW_DATA, ACTIVE_BATCH_SIZE, \
    NUM_WORKERS, VAL_BATCH_SIZE, ACTIVE_NUM_EPOCHS, DEVICE
from create_dataset import create_dataset, create_dataset_parallel
from datasets import Dataset
from evaluate import evaluate_ensemble
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

def acquisition_function(ensemble: list, datapool_set: Dataset):

    rows_with_corresponding_epistemic_uncertainties = []

    loader = DataLoader(datapool_set, batch_size=VAL_BATCH_SIZE, num_workers=NUM_WORKERS)

    current_idx = 0

    for X, y in loader:
        # print(f'{idx+1}/{len(datapool_set)}')
        prediction = get_ensemble_predictions_from_tensor(ensemble, X) # ModelDistributionOutput

        total_epistemic_uncertainty = (prediction.feasibility['epistemic']
                                       + prediction.recovery['epistemic']
                                       + prediction.purity['epistemic']
                                       + prediction.cost_per_kg['epistemic'])

        rows_with_corresponding_epistemic_uncertainties.extend([(data_idx, total_epistemic_uncertainty[list_dx]) for list_dx, data_idx in enumerate(range(current_idx, current_idx + X.shape[0]))])

        current_idx += X.shape[0]

    sorted_keys = [pair[0] for pair in sorted(rows_with_corresponding_epistemic_uncertainties, key=lambda item: item[1], reverse=True)]

    datapool_set.X = datapool_set.X[sorted_keys]
    datapool_set.y = datapool_set.y[sorted_keys]

    return datapool_set[:ACTIVE_NUM_NEW_DATA]

def main():
    name_input = '5_ensemble_best_170726.pt'

    ensemble = load_ensemble(name_input)

    dataset_train = Dataset('train')
    dataset_val = Dataset('val')

    loader_val = DataLoader(dataset_val, batch_size=VAL_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    name_output = name_input + '_post'

    best_val_loss = float('inf')

    # print('Pre-active learning evaluation')

    # print(evaluate_ensemble(name_input, loader_val))

    for generation in range(1000):
        print(f'generation {generation+1}')

        datapool_name = f'temp_datapool'

        print('starting data pool generation')
        # dataframe = create_dataset(datapool_name, ACTIVE_NUM_DATA_POOL, SEED, return_df=True, save_to_file=False)
        dataframe = create_dataset_parallel(datapool_name, ACTIVE_NUM_DATA_POOL, SEED, return_df=True, save_to_file=False)
        # generate new data
        datapool_set = Dataset(datapool_name, df=dataframe)

        print('done')
        print('starting acquisition function')
        data_selected = acquisition_function(ensemble, datapool_set)
        print('done')

        dataset_train.append(data_selected[0], data_selected[1])

        print(f'new dataset length {len(dataset_train)}')

        print('starting training')
        # throw away old weightss
        ensemble, _, val_losses_list = train_ensemble(DataLoader(dataset_train, batch_size=ACTIVE_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
                       loader_val, num_epochs=ACTIVE_NUM_EPOCHS, verbose=True)
        print('done')

        val_loss = min([loss.total.mean().item() for loss in val_losses_list])
        print(f'best val loss {val_loss}')

if __name__ == '__main__':
    main()
