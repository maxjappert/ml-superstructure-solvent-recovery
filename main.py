import datetime

import torch
from torch.utils.data import DataLoader

import config
from config import SEED, BATCH_SIZE, DEVICE, EPOCHS, LR, NUM_WORKERS
from datasets import Dataset
from evaluate import evaluate
from models import Model, LossBreakdown
from train import train_epoch
from utils import plot_training, compare_dicts_numerical

torch.manual_seed(SEED)
train_set = Dataset('train')
val_set = Dataset('val')
test_set = Dataset('test')
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_set, BATCH_SIZE)
test_loader = DataLoader(test_set, BATCH_SIZE)

model = Model().to(DEVICE)
# loads model for further training
# model.load_state_dict(torch.load('fractions_20260708_091906.pt')['model_state_dict'])
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)

# todo create M = 5 models
checkpoint_filename = f"{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pt"

train_losses_list = []
val_losses_list = []

print(f'started training {checkpoint_filename}')

best_val = float("inf")
for epoch in range(EPOCHS):
    train_losses = train_epoch(model, train_loader, optimizer)
    val_losses = evaluate(model, val_loader)
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print(compare_dicts_numerical(train_losses.detached_and_normalised_dict(len(train_set)),
                                  val_losses.detached_and_normalised_dict(len(val_set)),
                        'Train', 'Validation'))

    if val_losses.total.item() < best_val:
        best_val = val_losses.total.item()
        torch.save({'model_state_dict': model.state_dict(),
                    'optimiser_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'hparams': {'seed': config.SEED, 'lr': config.LR, 'bs': config.BATCH_SIZE},
                    'val_loss': best_val}, checkpoint_filename,
                    )

    train_losses_list.append(train_losses.detached_and_normalised(len(train_set)))
    val_losses_list.append(val_losses.detached_and_normalised(len(val_set) ))

checkpoint = torch.load(checkpoint_filename)
model.load_state_dict(checkpoint['model_state_dict'])
test_loss = evaluate(model, test_loader)
print(f"test | loss {test_loss:.4f}")
