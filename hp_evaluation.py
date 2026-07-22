import json
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

hps = dict()

for i in range(1, 18):
    with open(f'hp_optimisation{i}.json') as f:
        hps = hps | json.load(f)


hps_proper = dict()

for key in hps.keys():
    hps_proper[float(key)] = hps[key]

sorted_float_keys = sorted([float(loss) for loss in hps.keys()])

print('losses ranked: \n')

for i in range(len(sorted_float_keys)):

    if sorted_float_keys[i] == float('inf'):
        continue

    print(f'Number {i + 1} is val loss {sorted_float_keys[i]:.4} with {hps[str(sorted_float_keys[i])]}')

