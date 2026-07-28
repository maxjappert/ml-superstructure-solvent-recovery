import json
import os
import sys
from threading import current_thread

import torch
from torch.utils.data import DataLoader, Subset

from config import ACTIVE_LR, ACTIVE_WEIGHT_DECAY, ACTIVE_NUM_DATA_POOL, SEED, \
    NUM_WORKERS, VAL_BATCH_SIZE, ACTIVE_NUM_EPOCHS, DEVICE, ACTIVE_NEW_DATA_FRAC, BATCH_SIZE, ACTIVE_BATCH_SIZE, \
    ACTIVE_EPSILON_EXPLORATION
from create_dataset import create_dataset, create_dataset_parallel
from datasets import Dataset
from evaluate import evaluate_ensemble_from_file, evaluate
from models import Model, get_ensemble_predictions, StreamComposition, new_ensemble, \
    get_ensemble_predictions_from_tensor

from models import load_ensemble
from train import train_ensemble
from utils import z_score


def get_stream(row):
    return StreamComposition(target_name=row['target_name'],
                             target_kgph=row['target_kgph'],
                             solvent2_name=row['solvent2_name'],
                             solvent2_kgph=row['solvent2_kgph'],
                             water_kgph=row['water_kgph'],
                             salt_name=row['salt_name'],
                             salt_kgph=row['salt_kgph'],
                             solids_kgph=row['solids_kgph'])

@torch.no_grad()
def acquisition_function(ensemble: list, datapool_set: Dataset, len_train_set:int, baseline_random=False):
    loader = DataLoader(datapool_set, batch_size=VAL_BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    total_epistemic_uncertainties = []
    total_aleatoric_uncertainties = []

    for X, y in loader:
        # print(f'{idx+1}/{len(datapool_set)}')
        prediction = get_ensemble_predictions_from_tensor(ensemble, X) # ModelDistributionOutput

        total_epistemic_uncertainty = (z_score(prediction.feasibility['epistemic'])
                                       + z_score(prediction.recovery['epistemic'])
                                       + z_score(prediction.purity['epistemic'])
                                       + z_score(prediction.cost_per_kg['epistemic'])).detach()

        total_aleatoric_uncertainty =(z_score(prediction.feasibility['aleatoric'])
                                       + z_score(prediction.recovery['aleatoric'])
                                       + z_score(prediction.purity['aleatoric'])
                                       + z_score(prediction.cost_per_kg['aleatoric'])).detach()

        total_epistemic_uncertainties.extend(list(total_epistemic_uncertainty))
        total_aleatoric_uncertainties.extend(list(total_aleatoric_uncertainty))

    total_epistemic_uncertainties = torch.Tensor(total_epistemic_uncertainties)
    total_aleatoric_uncertainties = torch.Tensor(total_aleatoric_uncertainties)


    n_new_data = int(len_train_set * ACTIVE_NEW_DATA_FRAC)
    top_epist_vals, top_epist_pos = torch.topk(total_epistemic_uncertainties, int(n_new_data * (1.0 - ACTIVE_EPSILON_EXPLORATION)))

    print(f'Selected epistemic uncertainty for this generation: {top_epist_vals.mean().item():4f} +- {top_epist_vals.std(correction=0).item():4f}')

    top_epist_X = datapool_set.X[list(top_epist_pos), :]
    top_epist_y = datapool_set.y[list(top_epist_pos), :]

    random_idxs = torch.randint(0, len(datapool_set) - 1, [int(n_new_data * ACTIVE_EPSILON_EXPLORATION), ])
    random_X = datapool_set.X[random_idxs]
    random_y = datapool_set.y[random_idxs]

    datapool_set.X = torch.cat((top_epist_X, random_X), dim=0)
    datapool_set.y = torch.cat((top_epist_y, random_y), dim=0)

    print(f'{len(top_epist_vals)} new data points created')

    return datapool_set

def main():
    name_input = '5_ensemble_20260727_142131.pt'

    ensemble = load_ensemble(name_input)

    dataset_train = Dataset('train_small')
    dataset_val = Dataset('val')

    len_original_training_data = len(dataset_train)

    loader_val = DataLoader(dataset_val, batch_size=VAL_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    name_output = name_input + '_post'

    print(f'active learning for {name_output} starting')

    print('Pre-active learning evaluation')
    val_losses = []
    val_loss_og = evaluate_ensemble_from_file(name_input, loader_val)
    print(val_loss_og)

    ensemble_old = ensemble
    dataset_train_old = dataset_train

    best_val_loss = float('inf')

    for generation in range(1000):
        print(f'generation {generation+1}')

        datapool_name = f'temp_datapool'

        print('starting data pool generation')
        # dataframe = create_dataset(datapool_name, ACTIVE_NUM_DATA_POOL, SEED, return_df=True, save_to_file=False)
        dataframe = create_dataset_parallel(datapool_name, ACTIVE_NUM_DATA_POOL, SEED + generation, return_df=True, save_to_file=False)
        # generate new data
        datapool_set = Dataset(datapool_name, df=dataframe)

        print('done')
        print('starting acquisition function')
        data_selected = acquisition_function(ensemble, datapool_set, len_original_training_data, baseline_random=True)
        print('done')

        dataset_train.append(data_selected.X, data_selected.y)

        print(f'new dataset length {len(dataset_train)}')

        print('starting training')
        # throw away old weights
        ensemble, _, val_losses_list = train_ensemble(DataLoader(dataset_train, batch_size=ACTIVE_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
                       loader_val, num_epochs=ACTIVE_NUM_EPOCHS, weight_decay=ACTIVE_WEIGHT_DECAY, lr=ACTIVE_LR, verbose=False)
        print('done')

        sorted_val_losses = sorted(val_losses_list, key=lambda x: x.total.mean().item())
        generation_best_val_loss = sorted_val_losses[0].total.mean().item()

        print(sorted_val_losses[0].detached_distribution_dict())

        if generation_best_val_loss > 5:
            print('Loss too high, model has diverged')
            sys.exit(-1)

        if generation_best_val_loss < best_val_loss:
            torch.save({'model_state_dicts': [model.state_dict() for model in ensemble]}, name_output+'.pt')
            best_val_loss = generation_best_val_loss
            print(f'{name_output} saved!')
            ensemble_old = ensemble
            dataset_train_old = dataset_train
            torch.save(data_selected, os.path.join('data', f'{name_output}_data_{generation + 1}.pt'))
        else:
            print('reverting to previous state as no improvement has been recognised')
            ensemble = ensemble_old
            dataset_train = dataset_train_old


        print(f'epoch best val loss {generation_best_val_loss}')
        val_losses.append(generation_best_val_loss)

        with open(f"active_learning_val_losses_{name_output}.json", "w") as f:
            json.dump(val_losses, f)

if __name__ == '__main__':
    main()
