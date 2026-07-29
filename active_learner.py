import copy
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
def acquisition_function(ensemble: list, datapool_set: Dataset, len_train_set: int) -> Dataset:
    # the validation loader
    loader = DataLoader(datapool_set, batch_size=VAL_BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    total_epistemic_regression = []
    total_epistemic_classification = []

    for X, y in loader:
        # run the validation batch through the ensemble and receive a ModelDistributionOutput
        prediction = get_ensemble_predictions_from_tensor(ensemble, X)

        # creates a per-datapoint list where only the feasible datapoints are set to True
        # this is used such that the unfeasible datapoints, whose other metrics have no consequence,
        # don't pollute the uncertainty ranking
        feasible_mask = y[:,0].squeeze().to(DEVICE)

        # first sum all the regression uncertainties
        epistemic_regression = (z_score(prediction.recovery['epistemic'] * feasible_mask)
                                + z_score(prediction.purity['epistemic'] * feasible_mask)
                                + z_score(prediction.cost_per_kg['epistemic'] * feasible_mask))

        epistemic_classification = z_score(prediction.feasibility['epistemic'])

        # concatenate to the list of uncertainties per data point
        total_epistemic_regression.extend(list(epistemic_regression))
        total_epistemic_classification.extend(list(epistemic_classification))

    # convert this list to a tensor
    total_epistemic_regression = torch.Tensor(total_epistemic_regression)
    total_epistemic_classification = torch.Tensor(total_epistemic_classification)

    # the number of acquired data is equal to a fraction of the original training set length
    n_new_data = int(len_train_set * ACTIVE_NEW_DATA_FRAC)

    # retrieve the top (1-eps) fraction of epistemically uncertain data
    top_epist_vals, top_epist_pos = torch.topk(total_epistemic_regression, int(n_new_data * (1.0 - ACTIVE_EPSILON_EXPLORATION)))

    print(f'Selected epistemic uncertainty for this generation: {top_epist_vals.mean().item():4f} +- {top_epist_vals.std(correction=0).item():4f}')

    top_epist_X = datapool_set.X[list(top_epist_pos), :]
    top_epist_y = datapool_set.y[list(top_epist_pos), :]

    # add an eps fraction of random data
    random_idxs = torch.randperm(len(datapool_set))[:int(n_new_data * ACTIVE_EPSILON_EXPLORATION)]
    random_X = datapool_set.X[random_idxs]
    random_y = datapool_set.y[random_idxs]

    datapool_set.X = torch.cat((top_epist_X, random_X), dim=0)
    datapool_set.y = torch.cat((top_epist_y, random_y), dim=0)

    print(f'{len(top_epist_vals)} new data points created')

    return datapool_set

def main():
    name_input = '5_ensemble_20260729_small.pt'

    ensemble = load_ensemble(name_input)

    dataset_train = Dataset('train_small')
    dataset_val = Dataset('val')

    len_original_training_data = len(dataset_train)

    loader_val = DataLoader(dataset_val, batch_size=VAL_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    name_output = name_input + '_post'

    print(f'active learning for {name_output} starting')

    # evaluate the baseline
    print('Pre-active learning evaluation')
    val_losses = []
    val_loss_og = evaluate_ensemble_from_file(name_input, loader_val, as_dict=False)
    print(val_loss_og.detached_distribution_dict())

    ensemble_old = ensemble
    dataset_train_old = copy.deepcopy(dataset_train)

    best_val_loss = val_loss_og.total.mean().item()

    # loop over the epochs
    for generation in range(1000):
        print(f'generation {generation+1}')

        datapool_name = f'temp_datapool'

        dataframe = create_dataset_parallel(datapool_name, ACTIVE_NUM_DATA_POOL, SEED + generation, return_df=True, save_to_file=False)
        datapool_set = Dataset(datapool_name, df=dataframe)

        data_selected = acquisition_function(ensemble, datapool_set, len_original_training_data)

        dataset_train.append(data_selected.X, data_selected.y)

        print(f'new dataset length {len(dataset_train)}')

        # throw away old weights
        ensemble, _, val_losses_list = train_ensemble(DataLoader(dataset_train, batch_size=ACTIVE_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
                       loader_val, num_epochs=ACTIVE_NUM_EPOCHS, weight_decay=ACTIVE_WEIGHT_DECAY, lr=ACTIVE_LR, verbose=False)

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
            dataset_train_old = copy.deepcopy(dataset_train)
            torch.save(data_selected, os.path.join('data', f'{name_output}_data_{generation + 1}.pt'))
        else:
            print('reverting to previous state as no improvement has been recognised')
            ensemble = ensemble_old
            dataset_train = dataset_train_old


        print(f'epoch best val loss {generation_best_val_loss}')
        val_losses.append(generation_best_val_loss)

        with open(f"active_learning_val_losses_{name_output}.json", "w") as f:
            json.dump(val_losses, f)

        print()

if __name__ == '__main__':
    main()
