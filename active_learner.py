import json
import os
from threading import current_thread

import torch
from torch.utils.data import DataLoader, Subset

from config import ACTIVE_LR, ACTIVE_WEIGHT_DECAY, ACTIVE_NUM_DATA_POOL, SEED, \
    NUM_WORKERS, VAL_BATCH_SIZE, ACTIVE_NUM_EPOCHS, DEVICE, ACTIVE_NEW_DATA_FRAC, BATCH_SIZE
from create_dataset import create_dataset, create_dataset_parallel
from datasets import Dataset
from evaluate import evaluate_ensemble_from_file, evaluate
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

@torch.no_grad()
def acquisition_function(ensemble: list, datapool_set: Dataset):
    selected_indices = []

    loader = DataLoader(datapool_set, batch_size=VAL_BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    current_idx = 0

    for X, y in loader:
        # print(f'{idx+1}/{len(datapool_set)}')
        prediction = get_ensemble_predictions_from_tensor(ensemble, X) # ModelDistributionOutput

        total_epistemic_uncertainty = (prediction.feasibility['epistemic']
                                       + prediction.recovery['epistemic']
                                       + prediction.purity['epistemic']
                                       + prediction.cost_per_kg['epistemic']).detach()

        top_vals, top_pos = torch.topk(total_epistemic_uncertainty, int(X.size(0)*ACTIVE_NEW_DATA_FRAC))

        selected_indices.extend(list(top_pos+current_idx) )

        current_idx += X.shape[0]

    datapool_set.X = datapool_set.X[selected_indices, :]
    datapool_set.y = datapool_set.y[selected_indices, :]

    print(f'{len(selected_indices)} new data points created')

    return datapool_set

def main():
    name_input = '5_ensemble_best_170726.pt'

    ensemble = load_ensemble(name_input)

    dataset_train = Dataset('train')
    dataset_val = Dataset('val')

    loader_val = DataLoader(dataset_val, batch_size=VAL_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    name_output = name_input + '_post'

    best_val_loss = float('inf')

    print(f'active learning for {name_output} starting')

    print('Pre-active learning evaluation')
    val_losses = []
    val_loss_og = evaluate_ensemble_from_file(name_input, loader_val)
    # val_losses.append(val_loss_og.total.mean().item())
    print(val_loss_og)

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
        data_selected = acquisition_function(ensemble, datapool_set)
        print('done')

        dataset_train.append(data_selected.X, data_selected.y)

        print(f'new dataset length {len(dataset_train)}')

        print('starting training')
        # throw away old weights
        ensemble, _, val_losses_list = train_ensemble(DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
                       loader_val, num_epochs=ACTIVE_NUM_EPOCHS, verbose=False)
        print('done')

        val_loss = min([loss.total.mean().item() for loss in val_losses_list])

        if val_loss < best_val_loss:
            torch.save({'model_state_dicts': [model.state_dict() for model in ensemble]}, name_output+'.pt')
            best_val_loss = val_loss

        print(f'val losses {val_losses_list} with best val loss {val_loss}')
        val_losses.append(val_loss)

        if generation % 10 == 0:
            torch.save(dataset_train, os.path.join('data', f'{name_output}_data_{generation+1}.pt'))
            with open("active_learning_val_losses.json", "w") as f:
                json.dump(val_losses, f)

if __name__ == '__main__':
    main()
