import pandas as pd
import torch
from fontTools.designspaceLib.statNames import StatNames


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_type):
        df = pd.read_csv(f'data/{dataset_type}.csv')
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
                         'solid_removal_idx',
                         'recovery_idx',
                         'purification_idx',
                         'refinement_idx',]]
        #]

        output_df = df[['feasible',
                        'target_recovery',
                        'target_purity',
                        'cost_usd_per_kg_recovered']]

        self.X = torch.tensor(input_df.values, dtype=torch.float32)
        self.y = torch.tensor(output_df.values, dtype=torch.float32)

        self.standardiser_X = Standardiser(self.X)

        # self.standardiser_y = Standardizer(self.y[:,3:5])

        self.X = self.standardiser_X.transform(self.X)
        # self.y[:,3:5] = self.standardiser_y.transform(self.y[:,3:5])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Standardiser:
    def __init__(self, data: torch.Tensor):
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True) + 1e-8  # avoid div-by-zero

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean
