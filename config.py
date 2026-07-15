import torch

SEED = 42
EPOCHS = 1000
BATCH_SIZE = 4192
LR = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0

loss_scalar_fractions = 1
loss_scalar_cost = 1

PRED_METRICS = {
    'feasibility': 0,
    'recovery': 1,
    'purity': 2,
    'cost_per_kg': 3
}