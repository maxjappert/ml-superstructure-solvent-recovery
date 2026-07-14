import torch

SEED = 42
EPOCHS = 1000
BATCH_SIZE = 512
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0

loss_scalar_fractions = 1
loss_scalar_cost = 1