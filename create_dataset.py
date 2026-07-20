import os
import csv
import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from solvent_recovery import list_solvents, list_salts, compute
from solvent_recovery.properties import get_solvent_props, get_water_props, get_salt_props, get_solids_props, \
    get_extractant_props
from solvent_recovery.units import log_alphas_pairwise

COLUMNS = ['target_name',
           'solvent2_name',
           'salt_name',
           'target_kgph',
           'solvent2_kgph',
           'water_kgph',
           'salt_kgph',
           'solids_kgph',
           'target_volume',
           'solvent2_volume',
           'water_volume',
           'salt_volume',
           'solids_volume',
           'temperature_C',
           'target_mw',
           'target_density',
           'target_tb',
           'target_hvap',
           'target_cp',
           'target_logP',  # octanol-water partition coefficient
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
           'refinement_idx',
           'feasible',
           'cost_usd_per_kg_recovered',
           'cost_usd_per_year',
           'target_purity',
           'target_recovery']

def create_dataset_parallel(type: str, size: int, seed: int,
                            skip_prob=0.5, n_workers=20,
                            save_to_file=True, return_df=False):
    chunk = size // n_workers
    sizes = [chunk] * n_workers
    sizes[-1] += size - chunk * n_workers  # remainder

    with ProcessPoolExecutor(n_workers) as ex:
        futures = [
            ex.submit(create_dataset, type, s, seed + i * 100003,  # distinct seeds
                      skip_prob, save_to_file=False, return_df=True)
            for i, s in enumerate(sizes)
        ]
        df = pd.concat([f.result() for f in futures], ignore_index=True)

    if save_to_file:
        os.makedirs('data', exist_ok=True)
        df.to_csv(os.path.join('data', f'{type}.csv'), index=False)
    if return_df:
        return df


def create_dataset(type: str, size: int, seed: int, skip_prob=0.5,
                   save_to_file: bool = True, return_df: bool = False):
    """Generate the dataset.

    save_to_file: write data/{type}.csv exactly as before.
    return_df:    return the same data as a pandas DataFrame.
    Both can be True at the same time.
    """
    rows = []

    solvents = list_solvents()
    num_solvents = len(solvents)
    salts = list_salts()
    rng = random.Random(seed)

    # a random starting solvent
    solvent_iterator = rng.randint(0, num_solvents - 1)
    total_iterator = 0

    while total_iterator < size:
        # latin hypercube sampling
        solvent_target_name = solvents[solvent_iterator % num_solvents]

        for solids_removal_idx in range(4):
            for recovery_idx in range(4):
                for purification_idx in range(4):
                    for refinement_idx in range(5):

                        # we don't need every single combination for every single solvent,
                        # we just need a representative overview over the entire state space
                        if rng.random() < skip_prob:
                            continue

                        names = {
                            'target': solvent_target_name,
                            'solvent2': rng.choice([s for s in solvents if s != solvent_target_name]),
                            'salt': rng.choice(salts)
                        }

                        temperature_C = rng.randint(15, 60)

                        idxs = {
                            'solids_removal': solids_removal_idx,
                            'recovery': recovery_idx,
                            'purification': purification_idx,
                            'refinement': refinement_idx,
                        }

                        props = {
                            "target": get_solvent_props(names['target']),
                            "solvent2": get_solvent_props(names['solvent2']),
                            "water": get_water_props(),
                            "salt": get_salt_props(names['salt']),
                            "solids": get_solids_props(),
                            "extractant": get_extractant_props(),
                        }

                        stream_kgph = {
                            "target": rng.randint(50, 2000),
                            "solvent2": rng.randint(0, 1000),
                            "water": rng.randint(0, 1500) if rng.random() < 0.8 else 0,
                            "salt": rng.randint(0, 200) if rng.random() < 0.8 else 0,
                            "solids": rng.randint(0, 100) if rng.random() < 0.8 else 0
                        }

                        n_components = 1
                        if stream_kgph['solvent2'] > 0:
                            n_components += 1
                        if stream_kgph['salt'] > 0:
                            n_components += 1
                        if stream_kgph['water'] > 0:
                            n_components += 1
                        if stream_kgph['solids'] > 0:
                            n_components += 1

                        volumetric_flows = {
                            "target": stream_kgph['target'] / props['target'].rho,
                            "solvent2": stream_kgph['solvent2'] / props['solvent2'].rho,
                            "water": stream_kgph['water'] / props['water'].rho,
                            "salt": stream_kgph['salt'] / props['salt'].rho,
                            "solids": stream_kgph['solids'] / props['solids'].rho
                        }

                        fractions = {
                            "target": stream_kgph['target'] / sum(stream_kgph.values()),
                            "solvent2": stream_kgph['solvent2'] / sum(stream_kgph.values()),
                            "water": stream_kgph['water'] / sum(stream_kgph.values()),
                            "salt": stream_kgph['salt'] / sum(stream_kgph.values()),
                            "solids": stream_kgph['solids'] / sum(stream_kgph.values()),
                        }

                        assert 0.99 < sum(fractions.values()) < 1.01

                        r = compute(
                            solvent_target_name=names['target'], solvent2_name=names['solvent2'],
                            salt_name=names['salt'],
                            temperature_C=temperature_C,
                            solvent_target_kgph=stream_kgph['target'],
                            solvent2_kgph=stream_kgph['solvent2'],
                            water_kgph=stream_kgph['water'],
                            salt_kgph=stream_kgph['salt'],
                            solids_kgph=stream_kgph['solids'],
                            idx_solids_removal=idxs['solids_removal'],
                            idx_recovery=idxs['recovery'],
                            idx_purification=idxs['purification'],
                            idx_refinement=idxs['refinement'],
                        )

                        # todo bodge bodge bodge
                        if r.cost_usd_per_kg_recovered > 100:
                            # print('row skipped because the cost is too high')
                            continue

                        feasible = not math.isnan(r.cost_usd_per_kg_recovered)

                        log_alphas = log_alphas_pairwise(stream_kgph, props, temperature_C + 273.15)

                        rows.append([names['target'],
                                     names['solvent2'],
                                     names['salt'],
                                     stream_kgph['target'],
                                     stream_kgph['solvent2'],
                                     stream_kgph['water'],
                                     stream_kgph['salt'],
                                     stream_kgph['solids'],
                                     volumetric_flows['target'],
                                     volumetric_flows['solvent2'],
                                     volumetric_flows['water'],
                                     volumetric_flows['salt'],
                                     volumetric_flows['solids'],
                                     temperature_C,
                                     props['target'].MW,
                                     props['target'].rho,
                                     props['target'].Tb,  # in kelvin, we could convert
                                     props['target'].Hvap,
                                     props['target'].Cp,
                                     props['target'].logP,
                                     log_alphas['target']['solvent2'],
                                     log_alphas['target']['water'],
                                     props['solvent2'].MW,
                                     props['solvent2'].rho,
                                     props['solvent2'].Tb,
                                     props['solvent2'].Hvap,
                                     props['solvent2'].Cp,
                                     props['solvent2'].logP,
                                     idxs['solids_removal'],
                                     idxs['recovery'],
                                     idxs['purification'],
                                     idxs['refinement'],
                                     int(feasible),
                                     r.cost_usd_per_kg_recovered if feasible else 0,
                                     # NaN implies an infeasible solution, in the output we'll encode this as -1
                                     r.cost_usd_per_year if feasible else 0,
                                     r.target_purity if feasible else 0,
                                     r.target_recovery if feasible else 0])

                        total_iterator += 1

                        if total_iterator % 500000 == 0:
                            print(f'{total_iterator}/{size}')

        solvent_iterator += 1

    if save_to_file:
        os.makedirs('data', exist_ok=True)
        with open(os.path.join('data', f'{type}.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            writer.writerows(rows)

    if return_df:
        return pd.DataFrame(rows, columns=COLUMNS)


def main():
    type = sys.argv[1]
    size = int(sys.argv[2])
    seed = int(sys.argv[3])

    # create_dataset(type, size, seed)
    create_dataset_parallel(type, size, seed, n_workers=20)

if __name__ == '__main__':
    main()
