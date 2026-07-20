import datetime
import sys

import torch
from torch.utils.data import DataLoader

import config
from config import DEVICE, SEED, BATCH_SIZE, NUM_WORKERS, EPOCHS

from datasets import Dataset
from evaluate import evaluate
from models import get_losses, LossBreakdown, Model, new_ensemble, transfer_ensemble_losses
from utils import compare_dicts_numerical, compare_dicts_strings

torch.manual_seed(SEED)


def train_ensemble(train_loader, val_loader, M=5, num_epochs=EPOCHS, verbose=True):
    ensemble = new_ensemble(M)
    optimisers = [torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY) for model in ensemble]

    if verbose:
        checkpoint_filename = f"{M}_ensemble_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pt"

    train_losses_list = []
    val_losses_list = []

    if verbose:
        print(f'started training {checkpoint_filename}')

    best_val = float("inf")
    for epoch in range(num_epochs):
        if verbose:
            print(f'Epoch {epoch + 1}/{num_epochs}')

        epoch_losses_train = []
        epoch_losses_val = []

        for model_id in range(M):
            train_losses = train_epoch(ensemble[model_id], train_loader, optimisers[model_id])
            val_losses = evaluate(ensemble[model_id], val_loader)

            epoch_losses_train.append(train_losses)
            epoch_losses_val.append(val_losses)

        epoch_losses_breakdown_train = transfer_ensemble_losses(epoch_losses_train, len(train_loader.dataset))
        train_losses_list.append(epoch_losses_train)
        epoch_losses_breakdown_val = transfer_ensemble_losses(epoch_losses_val, len(val_loader.dataset))
        val_losses_list.append(epoch_losses_breakdown_val)

        if verbose:
            print(compare_dicts_strings(epoch_losses_breakdown_train.detached_distribution_dict(),
                                        epoch_losses_breakdown_val.detached_distribution_dict(),
                            'Train', 'Validation'))

        mean_val_loss = torch.mean(epoch_losses_breakdown_val.total).item()

        if verbose:
            if mean_val_loss < best_val:
                print('yay new best mean!')
                best_val = mean_val_loss
                torch.save({'model_state_dicts': [model.state_dict() for model in ensemble],
                            'optimiser_state_dict': [optimiser.state_dict() for optimiser in optimisers],
                            'epoch': epoch,
                            'hparams': {'seed': config.SEED, 'lr': config.LR, 'bs': config.BATCH_SIZE, 'M': M},
                            'val_loss': best_val}, checkpoint_filename,
                           )

    return ensemble, train_losses_list, val_losses_list


def train_single(train_loader, val_loader):
    model = Model().to(DEVICE)

    # if model_name is not None:
    #     model.load_state_dict(torch.load('fractions_20260708_091906.pt')['model_state_dict'])

    optimiser = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    checkpoint_filename = f"single_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pt"

    train_losses_list = []
    val_losses_list = []

    print(f'started training {checkpoint_filename}')

    best_val = float("inf")
    for epoch in range(EPOCHS):
        train_losses = train_epoch(model, train_loader, optimiser)
        val_losses = evaluate(model, val_loader)
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print(compare_dicts_numerical(train_losses.detached_and_normalised_dict(len(train_loader.dataset)),
                                      val_losses.detached_and_normalised_dict(len(val_loader.dataset)),
                            'Train', 'Validation'))

        if val_losses.total.item() < best_val:
            best_val = val_losses.total.item()
            torch.save({'model_state_dict': model.state_dict(),
                        'optimiser_state_dict': optimiser.state_dict(),
                        'epoch': epoch,
                        'hparams': {'seed': config.SEED, 'lr': config.LR, 'bs': config.BATCH_SIZE},
                        'val_loss': best_val}, checkpoint_filename,
                       )

        train_losses_list.append(train_losses.detached_and_normalised(len(train_loader.dataset)))
        val_losses_list.append(val_losses.detached_and_normalised(len(val_loader.dataset)))


def train_epoch(model, loader, optimizer,):
    model.train()
    total_losses = LossBreakdown.zeros()

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        losses = get_losses(model, x, y)

        # todo a bodge but keeps outliers at bay (wherever they might come from)
        if losses.cost_per_kg_nll > 1000:
            continue

        losses.total.backward()
        optimizer.step()

        total_losses.add(losses)

    return total_losses


def main():
    train_set = Dataset('train')
    val_set = Dataset('val')
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, BATCH_SIZE, num_workers=NUM_WORKERS)
    # train_ensemble(train_loader, val_loader)
    # train_single(train_loader, val_loader)

    training_type = sys.argv[1]
    if training_type == 'ensemble':
        train_ensemble(train_loader, val_loader)
    elif training_type == 'single':
        train_single(train_loader, val_loader)
    else:
        print('Unknown training type')
        sys.exit(-1)


if __name__ == '__main__':
    main()