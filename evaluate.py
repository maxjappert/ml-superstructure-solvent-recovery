import math

import torch
import torch.nn.functional as F
from torch.nn import Module

from config import DEVICE
from datasets import Dataset
from models import Model
from solvent_recovery import compute
from solvent_recovery.properties import get_solvent_props, get_water_props, get_salt_props, get_solids_props, \
    get_extractant_props
from solvent_recovery.units import _alphas


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        # total_loss += F.mse_loss(logits, y).item() * len(x)
        total_loss += F.binary_cross_entropy_with_logits(logits, y).item() * len(x)
    return total_loss / len(loader.dataset)

def manual_eval(model_name,
                solvent_target_name,
                solvent2_name,
                salt_name,
                solvent_target_flow,
                solvent2_flow,
                water_flow,
                salt_flow,
                solids_flow,
                solid_removal_idxs,
                recovery_idxs,
                purification_idxs,
                refinement_idxs,
                temperature_C=25,
                ground_truth=True):
    names = {
        'target': solvent_target_name,
        'solvent2': solvent2_name,
        'salt': salt_name
    }

    props = {
        "target": get_solvent_props(solvent_target_name),
        "solvent2": get_solvent_props(solvent2_name),
        "water": get_water_props(),
        "salt": get_salt_props(salt_name),
        "solids": get_solids_props(),
        "extractant": get_extractant_props(),
    }

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

    alphas = _alphas(stream_kgph, props, temperature_C + 273.15)

    if not alphas.keys().__contains__('solvent2'):
        alphas['solvent2'] = 0

    model = Model()
    model.load_state_dict(torch.load(model_name)['model_state_dict'])
    model.eval()

    dataset = Dataset('train')

    # model_outputs = torch.zeros((len(solid_removal_idxs), len(recovery_idxs), len(purification_idxs), len(refinement_idxs)))
    # ground_truths = torch.zeros((len(solid_removal_idxs), len(recovery_idxs), len(purification_idxs), len(refinement_idxs)))

    model_outputs = torch.zeros((4, 4, 4, 4))
    ground_truths = torch.zeros((4, 4, 4, 4))

    for solid_removal_idx in solid_removal_idxs:
        for recovery_idx in recovery_idxs:
            for purification_idx in purification_idxs:
                for refinement_idx in refinement_idxs:
                    if ground_truth:
                        r = compute(
                            solvent_target_name=names['target'], solvent2_name=names['solvent2'],
                            salt_name=names['salt'],
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

                        ground_truths[solid_removal_idx,recovery_idx,purification_idx,refinement_idx] = not math.isnan(r.cost_usd_per_kg_recovered)

                    tensor_input = torch.tensor([stream_kgph['target'],
                                             stream_kgph['solvent2'],
                                             stream_kgph['water'],
                                             stream_kgph['salt'],
                                             stream_kgph['solids'],
                                             volumetric_flows['target'],
                                             volumetric_flows['solvent2'],
                                             volumetric_flows['water'],
                                             volumetric_flows['salt'],
                                             volumetric_flows['solids'],
                                             alphas['water'] if stream_kgph['water'] > 0 else 0,
                                             temperature_C,
                                             props['target'].MW,
                                             props['target'].rho,
                                             props['target'].Tb, # in kelvin, we could convert
                                             props['target'].Hvap,
                                             props['target'].Cp,
                                             props['target'].logP,
                                             alphas['target'],
                                             props['solvent2'].MW,
                                             props['solvent2'].rho,
                                             props['solvent2'].Tb, props['solvent2'].Hvap,
                                             props['solvent2'].Cp,
                                             props['solvent2'].logP,
                                             alphas['solvent2'], # the T_ref argument is expected in Kelvin
                                             solid_removal_idx,
                                             recovery_idx,
                                             purification_idx,
                                             refinement_idx])

                    tensor_input = dataset.standardiser_X.transform(tensor_input)

                    model_output = model(tensor_input)
                    model_output = torch.sigmoid(model_output).item()
                    model_output = model_output > 0.5
                    model_outputs[solid_removal_idx,recovery_idx,purification_idx,refinement_idx] = model_output

    if ground_truth:
        return {
            'predicted': model_outputs,
            'true': ground_truths,
        }
    else:
        return {
            'predicted': model_outputs
        }

def main():
    output = manual_eval('best_06-07-26_feasibility.pt',
                '2-methyltetrahydrofuran',
                'acetone',
                'sodium bicarbonate',
                1000,
                300,
                0,
                0,
                0,
                3, 3, 2, 1)

    print(output)

if __name__ == '__main__':
    main()
