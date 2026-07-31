import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import EpsImagePlugin
from matplotlib import pyplot as plt
from torch.nn import Module
from torch.utils.data import DataLoader

import models
from config import DEVICE, loss_scalar_fractions, loss_scalar_cost, BATCH_SIZE, VAL_BATCH_SIZE, FLAGSHIP_MODEL_NAME, \
    PRED_METRICS, SCALING_RECOVERY, SCALING_PURITY, SCALING_COST, SCALING_FEASIBILITY, eps
from dataset_torch import Dataset
from models import Model, LossBreakdown, get_ensemble_predictions, StreamComposition, print_model_output_comparison, \
    load_ensemble, ModelDistributionOutput, get_losses, transfer_ensemble_losses
from solvent_recovery_data_generator import compute
from solvent_recovery_data_generator.properties import get_solvent_props, get_water_props, get_salt_props, get_solids_props, \
    get_extractant_props
from solvent_recovery_data_generator.units import _alphas, log_alphas_pairwise


@torch.no_grad()
def evaluate(model, loader, plots=False, model_name=None):
    model.eval()

    total_losses = LossBreakdown.zeros()

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        losses = models.get_losses(model, x, y)

        total_losses.add(losses)

    if plots:
        if model_name is None:
            print('error! no model name specified.')
        create_calibration_plot_binary_classification(model, loader, model_name)

    return total_losses

@torch.no_grad()
def evaluate_ensemble_from_file(ensemble_name, loader, as_dict=True):
    ensemble = load_ensemble(ensemble_name)

    losses = []

    for model in ensemble:
        losses.append(evaluate(model, loader))

    transferred_losses = transfer_ensemble_losses(losses, len(loader.dataset))

    if as_dict:
        return transferred_losses.detached_distribution_dict()
    else:
        return transferred_losses

@torch.no_grad()
def create_regression_calibration_plot(ensemble_name, dataset: Dataset, output_type, num_bins=80):
    ensemble = load_ensemble(ensemble_name)

    # dataset_calib = Dataset('calibration')
    # x_calib, y_calib = dataset_calib.X.to(DEVICE), dataset_calib.y.to(DEVICE)
    # y_hat_calib = models.get_ensemble_predictions_from_tensor(ensemble, x_calib)
    #
    # calib_ratio = (torch.pow(y_calib[:, PRED_METRICS[output_type]] - y_hat_calib.recovery['dist'].mean, 2) / y_hat_calib.recovery['dist'].variance.to(DEVICE)).mean()
    #
    # print(calib_ratio)

    x, y = dataset.X.to(DEVICE), dataset.y.to(DEVICE)

    y_hat_scaled = models.get_ensemble_predictions_from_tensor(ensemble, x, scaling=True)
    y_hat_unscaled = models.get_ensemble_predictions_from_tensor(ensemble, x, scaling=False)

    x, y = x.cpu(), y.cpu()

    p = np.linspace(0.01, 0.99, num_bins)
    p_hat_scaled = []
    p_hat_unscaled = []

    scale_param = 0

    for level in p:
        # The z-value from the equations for confidence intervals
        z = torch.distributions.Normal(0., 1.).icdf(torch.tensor(level))
        # Equation 3 from Kuleshov et al (2018)
        if output_type == 'recovery':
            p_hat_scaled.append((y[:,1] <= (y_hat_scaled.recovery['dist'].mean.cpu() + z * y_hat_scaled.recovery['dist'].stddev.cpu())).float().mean())
            p_hat_unscaled.append((y[:,1] <= (y_hat_unscaled.recovery['dist'].mean.cpu() + z * y_hat_unscaled.recovery['dist'].stddev.cpu())).float().mean())
            scale_param = SCALING_RECOVERY
        elif output_type == 'purity':
            p_hat_scaled.append((y[:,2] <= (y_hat_scaled.purity['dist'].mean.cpu() + z * y_hat_scaled.purity['dist'].stddev.cpu())).float().mean())
            p_hat_unscaled.append((y[:,2] <= (y_hat_unscaled.purity['dist'].mean.cpu() + z * y_hat_unscaled.purity['dist'].stddev.cpu())).float().mean())
            scale_param = SCALING_PURITY
        elif output_type == 'cost_per_kg':
            p_hat_scaled.append((y[:,3] <= (y_hat_scaled.cost_per_kg['dist'].mean.cpu() + z * y_hat_scaled.cost_per_kg['dist'].stddev.cpu())).float().mean())
            p_hat_unscaled.append((y[:,3] <= (y_hat_unscaled.cost_per_kg['dist'].mean.cpu() + z * y_hat_unscaled.cost_per_kg['dist'].stddev.cpu())).float().mean())
            scale_param = SCALING_COST


    os.makedirs(os.path.join('plots', ensemble_name), exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], 'k--', label='perfect')  # dotted diagonal
    plt.scatter(p, p_hat_scaled, s=20, alpha=0.7, label='$\\mathcal{N}(\\hat{\\mu}, s^2 \\cdot \\hat{\\sigma}^2)$')
    plt.scatter(p, p_hat_unscaled, s=20, alpha=0.7, label='$\\mathcal{N}(\\hat{\\mu},\\hat{\\sigma}^2)$')
    plt.xlabel('Nominal probability $p$')
    plt.ylabel('Observed probability $\\hat{p}$')
    plt.title(f'{output_type} calibration with $s^2 = {scale_param.item():.2f}$')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('plots', ensemble_name, f'{output_type}_calibration'), dpi=150)
    plt.show()
    plt.close()


    # p_hat = np.array([(pit <= p).sum() / y.size()[0] for p in pit])
    # levels = np.linspace(0.05, 0.95, 19)
    # observed = [(pit <= p).sum() / y.size()[0] for p in levels]
    # plt.scatter(levels, observed)
    # plt.show()


@torch.no_grad()
def create_calibration_plot_binary_classification(ensemble_name, dataset, n_bins=40):
    x, y = dataset.X.to(DEVICE), dataset.y.to(DEVICE)

    y_hat_unscaled = models.get_ensemble_predictions_from_tensor(load_ensemble(ensemble_name), x, scaling=False)
    y_hat_scaled = models.get_ensemble_predictions_from_tensor(load_ensemble(ensemble_name), x, scaling=True)
    probs_unscaled = y_hat_unscaled.feasibility['dist'].probs.detach().cpu()
    probs_scaled = y_hat_scaled.feasibility['dist'].probs.detach().cpu()
    x, y = x.cpu(), y.cpu()

    def calibration_bins(probs, targets, boundaries):
        probs = np.asarray(probs).ravel()
        targets = np.asarray(targets).ravel()
        # bin index per sample; right edge inclusive for the last bin
        idx = np.clip(np.digitize(probs, boundaries) - 1, 0, len(boundaries) - 2)
        mean_pred, frac_pos = [], []
        for i in range(len(boundaries) - 1):
            mask = idx == i
            if mask.sum() == 0:
                continue  # skip empty bins instead of producing NaN
            mean_pred.append(probs[mask].mean())
            frac_pos.append(targets[mask].mean())
        return mean_pred, frac_pos

    boundaries = np.linspace(0, 1, n_bins + 1)
    y0 = y[:, 0].numpy()
    mp_u, fp_u = calibration_bins(probs_unscaled.numpy(), y0, boundaries)
    mp_s, fp_s = calibration_bins(probs_scaled.numpy(), y0, boundaries)

    os.makedirs('plots', exist_ok=True)  # no error if it already exists
    os.makedirs(os.path.join('plots', ensemble_name), exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], 'k--', label='perfect')  # dotted diagonal
    plt.scatter(mp_u, fp_u, s=20, alpha=0.7, label='model w/o scaling')
    plt.scatter(mp_s, fp_s, s=20, alpha=0.7, label='model w/ scaling')
    plt.xlabel('mean predicted probability $\\hat{p}$')
    plt.ylabel('observed positive fraction')
    plt.title(f'feasibility calibration with $\\tau = {SCALING_FEASIBILITY.item():.1f}$')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('plots', ensemble_name, 'feasibility_calibration'), dpi=150)
    plt.show()
    plt.close()

@torch.no_grad()
def manual_eval(models_name,
                stream: models.StreamComposition,
                solid_removal_idx,
                recovery_idx,
                purification_idx,
                refinement_idx,
                temperature_C=25):

    r = compute(
        solvent_target_name=stream.target_solvent['props'].name,
        solvent2_name=stream.solvent2['props'].name,
        salt_name=stream.salt['props'].name,
        temperature_C=temperature_C,
        solvent_target_kgph=stream.target_solvent['kgph'],
        solvent2_kgph=stream.solvent2['kgph'],
        water_kgph=stream.water['kgph'],
        salt_kgph=stream.salt['kgph'],
        solids_kgph=stream.solids['kgph'],
        idx_solids_removal=solid_removal_idx,
        idx_recovery=recovery_idx,
        idx_purification=purification_idx,
        idx_refinement=refinement_idx,
    )

    ground_truth = models.ModelDistributionOutput(feasibility=r.feasible,
                                                  recovery=r.target_recovery,
                                                  purity=r.target_purity,
                                                  cost_per_kg=r.cost_usd_per_kg_recovered)

    model_list = load_ensemble(models_name)
    return {
        'predicted': get_ensemble_predictions(model_list,
                                              stream,
                                              temperature_C,
                                              [solid_removal_idx, recovery_idx, purification_idx, refinement_idx]),
        'true': ground_truth
    }


def main():

    dataset_calibration = Dataset('calibration')

    create_regression_calibration_plot(FLAGSHIP_MODEL_NAME, dataset_calibration, 'recovery')
    create_regression_calibration_plot(FLAGSHIP_MODEL_NAME, dataset_calibration, 'purity')
    create_regression_calibration_plot(FLAGSHIP_MODEL_NAME, dataset_calibration, 'cost_per_kg')
    create_calibration_plot_binary_classification(FLAGSHIP_MODEL_NAME, dataset_calibration, n_bins=30)

    stream = StreamComposition(target_name='2-methyltetrahydrofuran',
                               target_kgph=34,
                               solvent2_name='acetone',
                               solvent2_kgph=0,
                               salt_name='sodium bicarbonate',
                               salt_kgph=0,
                               water_kgph=0,
                               solids_kgph=0)

    print(get_ensemble_predictions(load_ensemble(FLAGSHIP_MODEL_NAME),
                             stream,
                             23,
                             [0, 0, 0, 0]))

    test_loader = DataLoader(Dataset('test'), batch_size=VAL_BATCH_SIZE)
    print(manual_eval(FLAGSHIP_MODEL_NAME, stream, 2, 1, 0, 0))

    print(evaluate_ensemble_from_file(FLAGSHIP_MODEL_NAME, test_loader))

if __name__ == '__main__':
    main()
