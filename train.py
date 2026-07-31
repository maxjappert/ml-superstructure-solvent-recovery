import copy
import datetime
import sys

import torch
from torch.func import functional_call, stack_module_state
from torch.utils.data import DataLoader

import config
from config import DEVICE, SEED, BATCH_SIZE, NUM_WORKERS, EPOCHS

from dataset_torch import Dataset
from evaluate import evaluate
from models import get_losses, LossBreakdown, Model, new_ensemble, transfer_ensemble_losses
from utils import compare_dicts_strings

torch.manual_seed(SEED)

# --- global performance switches --------------------------------------------
# Use TF32 tensor cores for all fp32 matmuls (Ada supports this natively).
torch.set_float32_matmul_precision('high')

AMP_DTYPE = torch.bfloat16       # native on RTX 4080; no GradScaler needed
USE_COMPILE = True               # torch.compile the hot path
OUTLIER_NLL_THRESHOLD = 1000.0   # replaces the host-synced `continue` bodge


# --- data --------------------------------------------------------------------

def make_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        # constant batch shape -> no torch.compile recompiles on the last batch
        drop_last=shuffle,
    )


# --- ensemble (vectorised with torch.func) -----------------------------------

def _load_stacked_into_ensemble(ensemble, params, buffers):
    """Copy the stacked (M, ...) parameters back into the M nn.Modules,
    so `evaluate()` and checkpointing keep working unchanged."""
    with torch.no_grad():
        merged = {**params, **buffers}
        for i, model in enumerate(ensemble):
            model.load_state_dict({name: stacked[i] for name, stacked in merged.items()})


def train_ensemble(train_loader, val_loader, M=5, num_epochs=EPOCHS,
                   verbose=True, lr=config.LR, dropout_rate=config.DROPOUT_RATE,
                   weight_decay=config.WEIGHT_DECAY):
    ensemble = new_ensemble(M, dropout_rate=dropout_rate)

    # Architecture template on the meta device (holds no real weights).
    base_model = copy.deepcopy(ensemble[0]).to('meta')
    params, buffers = stack_module_state(ensemble)
    for p in params.values():
        p.requires_grad_(True)

    # One optimiser over the stacked (M, ...) parameter tensors. Each model's
    # slice still gets its own independent AdamW statistics, because Adam's
    # moment estimates are elementwise.
    optimiser = torch.optim.AdamW(params.values(), lr=lr,
                                  weight_decay=weight_decay)

    def per_model_loss_dict(p, b, x, y):
        # NOTE: get_losses must only *call* the model, i.e. treat its first
        # argument as a plain callable `forward(x)`. If it currently touches
        # Module attributes (.train(), .parameters(), ...), hoist that out.
        forward = lambda inp: functional_call(base_model, (p, b), (inp,))
        losses = get_losses(forward, x, y)
        # LossBreakdown is a frozen dataclass (not a registered pytree), so
        # hand vmap a plain dict of scalars via its shallow field iteration.
        return {name: value for name, value in losses}

    try:
        vmapped_losses = torch.vmap(per_model_loss_dict, in_dims=(0, 0, None, None),
                                randomness='different')
    except ValueError:
        return ensemble, [], []

    counter_no_improvement = 0

    if USE_COMPILE:
        vmapped_losses = torch.compile(vmapped_losses, dynamic=False)

    checkpoint_filename = f"{M}_ensemble_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    if verbose:
        print(f'started training {checkpoint_filename}')

    train_losses_list = []
    val_losses_list = []
    best_val = float("inf")

    warmup_epochs = config.WARMUP_EPOCHS
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimiser,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimiser, start_factor=0.01, total_iters=warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS - warmup_epochs,
                                                       eta_min=config.LR * 0.01),
        ],
        milestones=[warmup_epochs],
    )

    best_ensemble = ensemble
    for epoch in range(num_epochs):
        if verbose:
            print(f'Epoch {epoch + 1}/{num_epochs}')

        # ---- train: one fused pass updates all M members per batch ----------
        running = None
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)

            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
                loss_dict = vmapped_losses(params, buffers, x, y)  # each: (M,)

            # GPU-side outlier guard, no host sync: zero out this batch's loss
            # for any member whose cost NLL blew up.
            keep = (loss_dict['cost_per_kg_nll'] < OUTLIER_NLL_THRESHOLD).float()
            (loss_dict['total'] * keep).sum().backward()
            optimiser.step()

            detached = {k: v.detach() for k, v in loss_dict.items()}
            running = detached if running is None else \
                {k: running[k] + detached[k] for k in running}

        # Reconstruct per-model LossBreakdowns for the existing reporting code.
        epoch_losses_train = [LossBreakdown(**{k: v[i] for k, v in running.items()})
                              for i in range(M)]

        # ---- validate: copy stacked weights back into the modules ------------
        _load_stacked_into_ensemble(ensemble, params, buffers)
        epoch_losses_val = [evaluate(model, val_loader) for model in ensemble]

        epoch_losses_breakdown_train = transfer_ensemble_losses(
            epoch_losses_train, len(train_loader.dataset))
        train_losses_list.append(epoch_losses_train)
        epoch_losses_breakdown_val = transfer_ensemble_losses(
            epoch_losses_val, len(val_loader.dataset))
        val_losses_list.append(epoch_losses_breakdown_val)

        if verbose:
            print(compare_dicts_strings(
                epoch_losses_breakdown_train.detached_distribution_dict(),
                epoch_losses_breakdown_val.detached_distribution_dict(),
                'Train', 'Validation'))

        mean_val_loss = torch.mean(epoch_losses_breakdown_val.total).item()

        # NB: checkpointing used to live inside `if verbose:`, so silent runs
        # never saved anything. Moved out.
        if mean_val_loss < best_val:
            counter_no_improvement = 0
            best_val = mean_val_loss
            best_ensemble = ensemble
            if verbose:
                print('yay new best mean!')
                torch.save({'model_state_dicts': [m.state_dict() for m in ensemble],
                            'optimiser_state_dict': optimiser.state_dict(),
                            'epoch': epoch,
                            'hparams': {'seed': config.SEED, 'lr': config.LR,
                                        'bs': config.BATCH_SIZE, 'M': M},
                            'val_loss': best_val}, checkpoint_filename)
        else:
            counter_no_improvement += 1

        # if the ensemble hasn't improved in ten iterations, we consider it to have converged
        if counter_no_improvement == 10:
            print('Converged as no validation improvement has been recognised in ten epochs.')
            break

        scheduler.step()

    return best_ensemble, train_losses_list, val_losses_list


def train_epoch(model, loader, optimizer):
    model.train()
    total_losses = LossBreakdown.zeros()

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
            losses = get_losses(model, x, y)

        # GPU-side outlier guard: multiply the loss by 0 instead of
        # `if losses.cost_per_kg_nll > 1000: continue`, which forced a
        # CPU<->GPU synchronisation on every single batch.
        keep = (losses.cost_per_kg_nll < OUTLIER_NLL_THRESHOLD).float()
        (losses.total * keep).backward()
        optimizer.step()

        total_losses.add(losses.detached())

    return total_losses


# --- entry point --------------------------------------------------------------

def main():
    train_set = Dataset('train_large')
    val_set = Dataset('val')
    train_loader = make_loader(train_set, shuffle=True)
    val_loader = make_loader(val_set, shuffle=False)

    train_ensemble(train_loader, val_loader)


if __name__ == '__main__':
    main()