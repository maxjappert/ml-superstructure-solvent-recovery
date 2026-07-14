import datetime

import torch
from torch.utils.data import DataLoader

import config
from config import SEED, BATCH_SIZE, DEVICE, EPOCHS, LR, NUM_WORKERS
from datasets import Dataset
from evaluate import evaluate
from models import Model
from train import train_epoch
from utils import plot_training

torch.manual_seed(SEED)
train_set = Dataset('train')
val_set = Dataset('val')
test_set = Dataset('test')
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_set, BATCH_SIZE)
test_loader = DataLoader(test_set, BATCH_SIZE)

print()

model = Model().to(DEVICE)
# loads model for further training
# model.load_state_dict(torch.load('fractions_20260708_091906.pt')['model_state_dict'])
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)

# todo create M = 5 models
checkpoint_filename = f"{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pt"

train_losses = []
val_losses = []

print(f'started training {checkpoint_filename}')

best_val = float("inf")
for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss, acc = evaluate(model, val_loader)
    print(f"epoch {epoch:2d} | train {train_loss:.4f} | val {val_loss:.4f} | acc {acc:.4f}")
    if val_loss < best_val:
        best_val = val_loss
        torch.save({'model_state_dict': model.state_dict(),
                    'optimiser_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'hparams': {'seed': config.SEED, 'lr': config.LR, 'bs': config.BATCH_SIZE},
                    'val_loss': best_val}, checkpoint_filename,
                    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

checkpoint = torch.load(checkpoint_filename)
model.load_state_dict(checkpoint['model_state_dict'])
test_loss = evaluate(model, test_loader)
print(f"test | loss {test_loss:.4f}")
