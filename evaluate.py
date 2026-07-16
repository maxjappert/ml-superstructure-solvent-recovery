import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn import Module
from torch.utils.data import DataLoader

import models
from config import DEVICE, loss_scalar_fractions, loss_scalar_cost, BATCH_SIZE
from datasets import Dataset
from models import Model, LossBreakdown
from solvent_recovery import compute
from solvent_recovery.properties import get_solvent_props, get_water_props, get_salt_props, get_solids_props, \
    get_extractant_props
from solvent_recovery.units import _alphas, log_alphas_pairwise

@torch.no_grad()
def evaluate(model, loader, plots=False, model_name=None):
    model.eval()

    total_losses = LossBreakdown.zeros()

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        losses = models.get_losses(model, x, y)

        total_losses.total += losses.total
        total_losses.feasibility_bce += losses.feasibility_bce
        total_losses.feasibility_brier += losses.feasibility_brier
        total_losses.recovery_nll += losses.recovery_nll
        total_losses.recovery_rmse += losses.recovery_rmse
        total_losses.purity_nll += losses.purity_nll
        total_losses.purity_rmse += losses.purity_rmse
        total_losses.cost_per_kg_nll += losses.cost_per_kg_nll
        total_losses.cost_per_kg_rmse += losses.cost_per_kg_rmse
        total_losses.num_correct += losses.num_correct

    if plots:
        if model_name is None:
            print('error! no model name specified.')
        create_calibration_plot_binary_classification(model, loader, model_name)

    return total_losses


@torch.no_grad()
def create_regression_calibration_plot(model: Model, dataset: Dataset, model_name: str, prediction: str, num_bins=20):
    model.eval().to(DEVICE)

    # todo: allow for selecting the different regression outputs

    x, y = dataset.X.to(DEVICE), dataset.y.to(DEVICE)

    y_hat = model(x)

    x, y = x.cpu(), y.cpu()

    mu = y_hat.recovery_mu.cpu()
    var = y_hat.recovery_logvar.exp().cpu()

    # take the observed outcome y and shove it through the model's own predicted cdf
    pit = torch.distributions.Normal(mu, torch.sqrt(var)).cdf(y[:,1])

    p = np.linspace(0.05, 0.95, num_bins)
    p_hat = []

    for level in p:
        # The z-value from the equations for confidence intervals
        z = torch.distributions.Normal(0., 1.).icdf(torch.tensor(level))
        # Equation 3 from Kuleshov et al (2018)
        p_hat.append((y[:,1] <= (mu + z * torch.sqrt(var))).float().mean())

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], 'k--', label='perfect')  # dotted diagonal
    plt.scatter(p, p_hat, s=20, alpha=0.7, label='model')
    plt.xlabel('True probability $p$')
    plt.ylabel('Observed probability $\\hat{p}$')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('plots', model_name, f'{prediction}_calibration'), dpi=150)
    plt.close()


    # p_hat = np.array([(pit <= p).sum() / y.size()[0] for p in pit])
    # levels = np.linspace(0.05, 0.95, 19)
    # observed = [(pit <= p).sum() / y.size()[0] for p in levels]
    # plt.scatter(levels, observed)
    # plt.show()


@torch.no_grad()
def create_calibration_plot_binary_classification(model, dataset, model_name, n_bins=40):
    model.eval().to(DEVICE)

    x, y = dataset.X.to(DEVICE), dataset.y.to(DEVICE)

    y_hat = model(x)

    x, y = x.cpu(), y.cpu()

    probs = torch.sigmoid(y_hat.feasibility_logit).detach().cpu()

    bins_y = []
    bins_y_hat = []
    bins_mean_predicted = []
    boundaries = np.linspace(0, 1, n_bins+1)

    for i in range(n_bins):
        bins_y.append([])
        bins_y_hat.append([])
        for j in range(len(probs)):
            if boundaries[i] <= probs[j] < boundaries[i+1]:
                bins_y[i].append( y[j,0].item())
                bins_y_hat[i].append(probs[j])


    empirical_fraction_of_positives = []

    for i in range(n_bins):
        bins_mean_predicted.append(np.mean(bins_y_hat[i]))
        empirical_fraction_of_positives.append(np.array(bins_y[i]).sum() / len(bins_y[i]))

    os.makedirs('plots', exist_ok=True)  # no error if it already exists
    os.makedirs(os.path.join('plots', model_name), exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], 'k--', label='perfect')  # dotted diagonal
    plt.scatter(bins_mean_predicted, empirical_fraction_of_positives, s=20, alpha=0.7, label='model')
    plt.xlabel('mean predicted probability')
    plt.ylabel('observed fraction positive')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('plots', model_name, 'feasibility_calibration'), dpi=150)
    plt.close()



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

                    log_alphas = log_alphas_pairwise(stream_kgph, props, temperature_C + 273.15)

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
    # output = manual_eval('first_good.pt',
    #             '2-methyltetrahydrofuran',
    #             'acetone',
    #             'sodium bicarbonate',
    #             1000,
    #             300,
    #             0,
    #             0,
    #             [0],
    #             [3], [3], [2], 25)
    #
    # print(output)

    model = Model()
    name = '20260715_152407.pt'
    model.load_state_dict(torch.load(name)['model_state_dict'])
    dataset = Dataset('val')

    create_regression_calibration_plot(model, dataset, name, 'recovery')
    # create_calibration_plot_binary_classification(model, dataset, name)


if __name__ == '__main__':
    main()
