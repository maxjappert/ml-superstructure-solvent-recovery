import pandas as pd
import torch
from fontTools.designspaceLib.statNames import StatNames


class Dataset(torch.utils.data.Dataset):
    def __init__(self, type: str, output, normalise=True):
        df = pd.read_csv(f'data/{type}.csv')
        input_df = df[['target_kgph',
                         'solvent2_kgph',
                         'water_kgph',
                         'salt_kgph',
                         'solids_kgph',
                         'temperature_C',
                         'target_mw',
                         'target_density',
                         'target_tb',
                         'target_hvap',
                         'target_cp',
                         'target_logP', # octanol-water partition coefficient
                         'target_log_alpha_solvent2',
                         'target_log_alpha_water',
                         'solvent2_mw',
                         'solvent2_density',
                         'solvent2_tb',
                         'solvent2_hvap',
                         'solvent2_cp',
                         'solvent2_logP',
                         'solvent2_alpha',
                         'solid_removal_idx',
                         'recovery_idx',
                         'purification_idx',
                         'refinement_idx',]]
        #]

        if output == 'feasibility':
            output_df = df[['feasible']]
        elif output == 'fractions':
            output_df = df[[ 'target_purity',
                             'target_recovery']]
        elif output == 'cost':
            output_df = df[['cost_usd_per_kg_recovered',
                             'cost_usd_per_year']]
        else:
            print('wrong output form')
            exit(0)

        self.X = torch.tensor(input_df.values, dtype=torch.float32)
        self.y = torch.tensor(output_df.values, dtype=torch.float32)

        self.standardiser_X = Standardizer(self.X)
        self.standardiser_y = Standardizer(self.y)

        self.normalise = normalise

        if self.normalise:
            self.X = self.standardiser_X.transform(self.X)
            if output == 'cost':
                self.y = self.standardiser_y.transform(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Standardizer:
    def __init__(self, data: torch.Tensor):
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True) + 1e-8  # avoid div-by-zero

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean
