import torch

SEED = 42
EPOCHS = 200
BATCH_SIZE = 512
LR = 1e-2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'mps')
NUM_WORKERS = 4
