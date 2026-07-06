import torch

SEED = 42
EPOCHS = 200
BATCH_SIZE = 512
LR = 1e-2
DEVICE = torch.device('mps') # torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
