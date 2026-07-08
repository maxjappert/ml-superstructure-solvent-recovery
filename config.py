import torch

SEED = 42
EPOCHS = 1000
BATCH_SIZE = 256
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'mps')
NUM_WORKERS = 4
