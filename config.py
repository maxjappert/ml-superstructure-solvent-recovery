import torch

SEED = 42
EPOCHS = 1000
BATCH_SIZE = 256
LR = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4

ACTIVE_LR = 1e-4
ACTIVE_WEIGHT_DECAY = 1e-4
ACTIVE_NUM_DATA_POOL = 100000
ACTIVE_NUM_NEW_DATA = 10000

loss_scalar_fractions = 1
loss_scalar_cost = 1

PRED_METRICS = {
    'feasibility': 0,
    'recovery': 1,
    'purity': 2,
    'cost_per_kg': 3
}