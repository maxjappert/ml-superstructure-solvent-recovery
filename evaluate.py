import math
import random

import torch
import torch.nn.functional as F
from torch.nn import Module

import models
from config import DEVICE, loss_scalar_fractions, loss_scalar_cost
from datasets import Dataset
from models import Model
from solvent_recovery import compute
from solvent_recovery.properties import get_solvent_props, get_water_props, get_salt_props, get_solids_props, \
    get_extractant_props
from solvent_recovery.units import _alphas, _log_alphas_pairwise


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss_total, correct, total_loss_feasibility, total_loss_recovery, total_loss_purity, total_loss_cost_per_kg, total_loss_cost_per_year = 0, 0, 0, 0, 0, 0, 0
    total_correct = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        losses = models.get_losses(model, x, y)

        total_loss_total += losses['total'].item() * len(x)
        total_loss_feasibility += losses['feasibility'].item() * len(x)
        total_loss_recovery += losses['recovery'].item() * len(x)
        total_loss_purity += losses['purity'].item() * len(x)
        total_loss_cost_per_kg += losses['cost_per_kg'].item() * len(x)
        # total_loss_cost_per_year = loss_cost_per_year.item() * len(x)

        total_correct += losses['num_correct'].item()

    return {
        'total loss': total_loss_total / len(loader.dataset),
        'feasibility loss': total_loss_feasibility / len(loader.dataset),
        'feasibility accuracy': total_correct / len(loader.dataset),
        'recovery loss': total_loss_recovery / len(loader.dataset),
        'purity loss': total_loss_purity / len(loader.dataset),
        'cost per kg loss': total_loss_cost_per_kg / len(loader.dataset),
        # 'cost per year loss': total_loss_cost_per_year / len(loader.dataset)
    }

    return (total_loss_total / len(loader.dataset),
            total_loss_feasibility / len(loader.dataset),
            total_correct / len(loader.dataset),
            total_loss_recovery / len(loader.dataset),
            total_loss_purity / len(loader.dataset))

@torch.no_grad()
def manual_eval(model,
                props,
                dataset,
                solvent_target_flow,
                solvent2_flow,
                water_flow,
                salt_flow,
                solids_flow,
                solid_removal_idxs,
                recovery_idxs,
                purification_idxs,
                refinement_idxs,
                temperature_C=25):

    stream_kgph = {
        "target": solvent_target_flow,
        "solvent2": solvent2_flow,
        "water": water_flow,
        "salt": salt_flow,
        "solids": solids_flow
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

    model_outputs = torch.zeros((4, 4, 4, 5, 4))
    ground_truths = torch.zeros((4, 4, 4, 5, 4))

    for solid_removal_idx in solid_removal_idxs:
        for recovery_idx in recovery_idxs:
            for purification_idx in purification_idxs:
                for refinement_idx in refinement_idxs:
                    r = compute(
                        solvent_target_name=props['target'].name,
                        solvent2_name=props['solvent2'].name,
                        salt_name=props['salt'].name,
                        temperature_C=temperature_C,
                        solvent_target_kgph=stream_kgph['target'],
                        solvent2_kgph=stream_kgph['solvent2'],
                        water_kgph=stream_kgph['water'],
                        salt_kgph=stream_kgph['salt'],
                        solids_kgph=stream_kgph['solids'],
                        idx_solids_removal=solid_removal_idx,
                        idx_recovery=recovery_idx,
                        idx_purification=purification_idx,
                        idx_refinement=refinement_idx,
                    )

                    ground_truths[solid_removal_idx,recovery_idx,purification_idx,refinement_idx, 0] = not math.isnan(r.cost_usd_per_kg_recovered)
                    ground_truths[solid_removal_idx, recovery_idx, purification_idx, refinement_idx, 1] = r.target_recovery
                    ground_truths[solid_removal_idx, recovery_idx, purification_idx, refinement_idx, 2] = r.target_purity
                    ground_truths[solid_removal_idx, recovery_idx, purification_idx, refinement_idx, 3] = r.cost_usd_per_kg_recovered

                    log_alphas = _log_alphas_pairwise(stream_kgph, props, temperature_C + 273.15)

                    tensor_input = torch.tensor([stream_kgph['target'],
                                             stream_kgph['solvent2'],
                                             stream_kgph['water'],
                                             stream_kgph['salt'],
                                             stream_kgph['solids'],
                                             temperature_C,
                                             props['target'].MW,
                                             props['target'].rho,
                                             props['target'].Tb, # in kelvin, we could convert
                                             props['target'].Hvap,
                                             props['target'].Cp,
                                             props['target'].logP,
                                             log_alphas['target']['solvent2'],
                                             log_alphas['target']['water'],
                                             props['solvent2'].MW,
                                             props['solvent2'].rho,
                                             props['solvent2'].Tb, props['solvent2'].Hvap,
                                             props['solvent2'].Cp,
                                             props['solvent2'].logP,
                                             solid_removal_idx,
                                             recovery_idx,
                                             purification_idx,
                                             refinement_idx])

                    tensor_input = dataset.standardiser_X.transform(tensor_input)
                    model_output = model(tensor_input)

                    model_outputs[solid_removal_idx, recovery_idx,purification_idx,refinement_idx, 0] = model_output['feasibility'].item() > 0.5
                    model_outputs[solid_removal_idx, recovery_idx, purification_idx, refinement_idx, 1] = model_output['recovery_mu'].item()
                    model_outputs[solid_removal_idx, recovery_idx, purification_idx, refinement_idx, 2] = model_output['purity_mu'].item()
                    model_outputs[solid_removal_idx, recovery_idx, purification_idx, refinement_idx, 3] = model_output['cost_per_kg_mu_z'].item()


    return {
        'predicted': model_outputs,
        'true': ground_truths,
    }


def main():
    output = manual_eval('first_good.pt',
                '2-methyltetrahydrofuran',
                'acetone',
                'sodium bicarbonate',
                1000,
                300,
                0,
                0,
                [0],
                [3], [3], [2], 25)

    print(output)

if __name__ == '__main__':
    main()
