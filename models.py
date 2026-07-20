from dataclasses import dataclass, asdict, fields
from typing import Iterator, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from config import DEVICE, loss_scalar_fractions, loss_scalar_cost, eps
from datasets import Dataset
from solvent_recovery.properties import get_solvent_props, get_salt_props, get_water_props, get_solids_props
from solvent_recovery.units import log_alphas_pairwise


@dataclass(frozen=False, slots=True)
class ModelOutput:
    feasibility_logit: Tensor
    recovery_mu: Tensor
    recovery_logvar: Tensor
    purity_mu: Tensor
    purity_logvar: Tensor
    cost_per_kg_mu: Tensor
    cost_per_kg_logvar: Tensor

@dataclass(frozen=False, slots=True)
class ModelDistributionOutput:
    feasibility: Union[dict, bool, float]
    recovery: Union[dict, float]
    purity: Union[dict, float]
    cost_per_kg: Union[dict, float]

def load_ensemble(checkpoint_name: str) -> list[Model]:
    checkpoint = torch.load(checkpoint_name)
    M = checkpoint['hparams']['M']

    model_list = []

    for i in range(M):
        model_list.append(Model())
        model_list[i].load_state_dict(checkpoint['model_state_dicts'][i])
        model_list[i].eval()
        model_list[i].to(DEVICE)

    return model_list

def print_model_output_comparison(y_hat: ModelDistributionOutput, y: ModelDistributionOutput):
    recovery = y_hat.recovery['dist']
    purity = y_hat.purity['dist']
    cost_per_kg = y_hat.cost_per_kg['dist']

    output = (f'Predicted feasibility with {(y_hat.feasibility['dist'].probs * 100):.2f}% probability. Ground truth {y.feasibility}.\n'
              f'Predicted recovery is {(recovery.mean * 100):.2f}% while the true is {(y.recovery*100):.2f}%. \n'
              f'Predicted purity is {(purity.mean * 100):.2f}% while the true is {(y.purity * 100):.2f}%. \n'
              f'Predicted cost per kg is {cost_per_kg.mean:.2f} while the true is {y.cost_per_kg :.2f}.')

    return output

#@dataclass(frozen=False, slots=True)
class StreamComposition:
    def __init__(self, target_name, target_kgph, solvent2_name, solvent2_kgph, salt_name, salt_kgph, water_kgph, solids_kgph):
        self.target_solvent = {'props': get_solvent_props(target_name),
                               'kgph': target_kgph}
        self.solvent2 = {'props': get_solvent_props(solvent2_name),
                         'kgph': solvent2_kgph}
        self.salt = {'props': get_salt_props(salt_name),
                     'kgph': salt_kgph}
        self.water = {'props': get_water_props(),
                      'kgph': water_kgph}
        self.solids = {'props': get_solids_props(),
                       'kgph': solids_kgph}


    def get_kgph_dict(self) -> dict:
        return {'target': self.target_solvent['kgph'],
                     'solvent2': self.solvent2['kgph'],
                     'salt': self.salt['kgph'],
                     'water': self.water['kgph'],
                     'solids': self.solids['kgph']}

    def get_props_dict(self) -> dict:
        return {'target': self.target_solvent['props'],
                      'solvent2': self.solvent2['props'],
                      'salt': self.salt['props'],
                      'water': self.water['props'],
                      'solids': self.solids['props']}

@dataclass(frozen=False, slots=True)
class LossBreakdown:
    """Per-batch loss components for the multi-task recovery model."""

    total: Tensor

    feasibility_bce: Tensor
    feasibility_brier: Tensor

    num_correct: Tensor
    recovery_nll: Tensor
    recovery_rmse: Tensor
    purity_nll: Tensor
    purity_rmse: Tensor
    cost_per_kg_nll: Tensor
    cost_per_kg_rmse: Tensor

    @classmethod
    def zeros(cls) -> "LossBreakdown":
        return cls(**{f.name: torch.zeros([]).to(DEVICE) for f in fields(cls)})

    @classmethod
    def empty(cls) -> "LossBreakdown":
        return cls(**{f.name: torch.Tensor().unsqueeze(0).to(DEVICE) for f in fields(cls)})

    def add(self, losses: LossBreakdown):
        self.total += losses.total.detach()
        self.feasibility_bce += losses.feasibility_bce.detach()
        self.feasibility_brier += losses.feasibility_brier.detach()
        self.recovery_nll += losses.recovery_nll.detach()
        self.recovery_rmse += losses.recovery_rmse.detach()
        self.purity_nll += losses.purity_nll.detach()
        self.purity_rmse += losses.purity_rmse.detach()
        self.cost_per_kg_nll += losses.cost_per_kg_nll.detach()
        self.cost_per_kg_rmse += losses.cost_per_kg_rmse.detach()
        self.num_correct += losses.num_correct.detach()

        return self

    def div_by(self, denominator):
        self.total /= denominator
        self.feasibility_bce /= denominator
        self.feasibility_brier /= denominator
        self.recovery_nll /= denominator
        self.recovery_rmse /= denominator
        self.purity_nll /= denominator
        self.purity_rmse /= denominator
        self.cost_per_kg_nll /= denominator
        self.cost_per_kg_rmse /= denominator
        self.num_correct /= denominator

        return self

    @classmethod
    def from_shape(cls, shape):
        return LossBreakdown(**{f.name: torch.zeros(shape).to(DEVICE) for f in fields(cls)})

    def as_dict(self) -> dict[str, Loss]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def detached_and_normalised_dict(self, normaliser) -> dict[str, float]:
        return {
            f.name: (v.detach().item() / normaliser if isinstance(v := getattr(self, f.name), Tensor) else float(v) / normaliser)
            for f in fields(self)
        }

    def detached_distribution_dict(self):
        '''
        Requires the object to have been passed through the transfer_ensemble_losses(...) function in order to obtain multiple values in each field.
        '''
        return {
            f.name: (f'{torch.mean(v).item().__round__(3)} +- {torch.std(v).item().__round__(3)}' if isinstance(v := getattr(self, f.name), Tensor) else float(v))
            for f in fields(self)
        }

    def detached_and_normalised(self, normaliser) -> LossBreakdown:
        return LossBreakdown(**{f.name: getattr(self, f.name).item() / normaliser for f in fields(self)})

    def __iter__(self):
        return iter(self.as_dict().items())


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        dropout_rate = 0.2

        self.net_shared = nn.Sequential(nn.Linear(24, 128),
                                 nn.ReLU(),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(128, 256),
                                 nn.ReLU(),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(256, 512),
                                 nn.ReLU())

        self.net_feasibility = nn.Sequential(nn.Linear(512, 256),
                                             nn.ReLU(),
                                             nn.Dropout(dropout_rate),
                                             nn.Linear(256, 128),
                                             nn.ReLU(),
                                             nn.Dropout(dropout_rate),
                                             nn.Linear(128, 1),)

        self.net_fractions = nn.Sequential(nn.Linear(512, 256),
                                             nn.ReLU(),
                                             nn.Dropout(dropout_rate),
                                             nn.Linear(256, 128),
                                             nn.ReLU(),
                                             nn.Dropout(dropout_rate),
                                             nn.Linear(128, 4))

        self.net_cost = nn.Sequential(nn.Linear(512, 256),
                                             nn.ReLU(),
                                             nn.Dropout(dropout_rate),
                                             nn.Linear(256, 128),
                                             nn.ReLU(),
                                             nn.Dropout(dropout_rate),
                                             nn.Linear(128, 2))




    def forward(self, x):
        hidden = self.net_shared(x)
        y_hat_feasibility = self.net_feasibility(hidden)
        y_hat_fractions = self.net_fractions(hidden)
        y_hat_cost = self.net_cost(hidden)

        return ModelOutput(feasibility_logit=y_hat_feasibility.squeeze(),
                           recovery_mu=y_hat_fractions[:,0],
                           recovery_logvar=torch.clamp(y_hat_fractions[:,1], min=-10, max=10),
                           purity_mu=y_hat_fractions[:,2],
                           purity_logvar=torch.clamp(y_hat_fractions[:,3], min=-10, max=10),
                           cost_per_kg_mu=y_hat_cost[:,0],
                           cost_per_kg_logvar=torch.clamp(y_hat_cost[:,1], min=-10, max=10))


def load_model(name):
    model = Model()
    model.load_state_dict(torch.load(name)['model_state_dict'])
    model.to(DEVICE)
    return model

def new_ensemble(M):
    return [Model().to(DEVICE) for _ in range(M)]

def brier_score(logits, labels):
    probs = torch.sigmoid(logits).squeeze()
    return ((probs - labels.float()) ** 2).mean()

Loss = Union[Tensor, float]

def get_losses(model, x, y) -> LossBreakdown:
    y_hat = model(x)

    feasibility_bce = F.binary_cross_entropy_with_logits(y_hat.feasibility_logit, y[:, 0])
    feasibility_brier = brier_score(y_hat.feasibility_logit, y[:, 0])

    feasible_mask = y[:, 0] == 1

    recovery_nll = loss_scalar_fractions * F.gaussian_nll_loss(y_hat.recovery_mu[feasible_mask], y[feasible_mask, 1],
                                                                y_hat.recovery_logvar[feasible_mask].exp(), reduction='mean')
    recovery_rmse = torch.sqrt(F.mse_loss(y_hat.recovery_mu[feasible_mask], y[feasible_mask, 1]))

    purity_nll = loss_scalar_fractions * F.gaussian_nll_loss(y_hat.purity_mu[feasible_mask], y[feasible_mask, 2], y_hat.purity_logvar[feasible_mask].exp(),
                                                              reduction='mean')
    purity_rmse = torch.sqrt(F.mse_loss(y_hat.purity_mu[feasible_mask], y[feasible_mask, 2]))

    cost_per_kg_nll = loss_scalar_cost * F.gaussian_nll_loss(y_hat.cost_per_kg_mu[feasible_mask], y[feasible_mask, 3],
                                                              y_hat.cost_per_kg_logvar[feasible_mask].exp(), reduction='mean')
    cost_per_kg_rmse = torch.sqrt(F.mse_loss(y_hat.cost_per_kg_mu[feasible_mask], y[feasible_mask, 3]))

    loss_total = feasibility_bce + recovery_nll + purity_nll + cost_per_kg_nll  # + loss_cost_per_year

    if loss_total == float('inf'):
        ...

    preds = (torch.sigmoid(y_hat.feasibility_logit) > 0.5)
    num_correct = (preds == y[:, 0]).sum()

    losses = LossBreakdown(
        total=loss_total * len(x),
        feasibility_bce=feasibility_bce * len(x),
        feasibility_brier=feasibility_brier * len(x),
        num_correct=num_correct,
        recovery_nll=recovery_nll * len(x),
        recovery_rmse=recovery_rmse * len(x),
        purity_nll=purity_nll * len(x),
        purity_rmse=purity_rmse * len(x),
        cost_per_kg_nll=cost_per_kg_nll * len(x),
        cost_per_kg_rmse=cost_per_kg_rmse * len(x),
    )

    return losses

def bernoulli_entropy(p):
    return -p * torch.log(p + eps) - (1 - p) * torch.log(1 - p + eps)

def get_ensemble_predictions_from_tensor(ensemble, input_tensor):
    M = len(ensemble)

    raw_outputs = []

    for model in ensemble:
        raw_outputs.append(model(input_tensor.to(DEVICE)))

    ps = []

    for i in range(M):
        ps.append(torch.sigmoid(raw_outputs[i].feasibility_logit).detach())

    p_mean = torch.Tensor(ps).mean()
    feasibility_total_uncertainty = bernoulli_entropy(p_mean)
    feasibility_aleatoric = torch.Tensor([bernoulli_entropy(p) for p in ps]).mean()
    feasibility_epistemic = feasibility_total_uncertainty - feasibility_aleatoric

    mus = []
    vars = []
    var_plus_mu_squareds = []
    for i in range(M):
        mus.append(raw_outputs[i].recovery_mu.detach())
        vars.append(raw_outputs[i].recovery_logvar.exp().detach())
        var_plus_mu_squareds.append(raw_outputs[i].recovery_logvar.exp().detach() + torch.pow(raw_outputs[i].recovery_mu.detach(), 2))

    recovery_mu = torch.Tensor(mus).mean()
    recovery_var = torch.Tensor(var_plus_mu_squareds).mean() - torch.pow(recovery_mu, 2)
    recovery_dist = torch.distributions.Normal(recovery_mu, recovery_var.sqrt())

    recovery_epistemic = torch.var(torch.Tensor(mus), unbiased=False)
    recovery_aleatoric = torch.mean(torch.Tensor(vars))

    mus = []
    vars = []
    var_plus_mu_squareds = []
    for i in range(M):
        mus.append(raw_outputs[i].purity_mu.detach())
        vars.append(raw_outputs[i].purity_logvar.exp().detach())
        var_plus_mu_squareds.append(raw_outputs[i].purity_logvar.exp().detach() + torch.pow(raw_outputs[i].purity_mu.detach(), 2))

    purity_mu = torch.Tensor(mus).mean()
    purity_var = torch.Tensor(var_plus_mu_squareds).mean() - torch.pow(purity_mu, 2)
    purity_dist = torch.distributions.Normal(purity_mu, purity_var.sqrt())

    purity_epistemic = torch.var(torch.Tensor(mus), unbiased=False)
    purity_aleatoric = torch.mean(torch.Tensor(vars))

    mus = []
    vars = []
    var_plus_mu_squareds = []
    for i in range(M):
        mus.append(raw_outputs[i].cost_per_kg_mu.detach())
        vars.append(raw_outputs[i].cost_per_kg_logvar.exp().detach())
        var_plus_mu_squareds.append(raw_outputs[i].cost_per_kg_logvar.exp().detach() + torch.pow(raw_outputs[i].cost_per_kg_mu.detach(), 2))

    cost_per_kg_mu = torch.Tensor(mus).mean()
    cost_per_kg_var = torch.Tensor(var_plus_mu_squareds).mean() - torch.pow(cost_per_kg_mu, 2)
    cost_per_kg_dist = torch.distributions.Normal(cost_per_kg_mu, cost_per_kg_var.sqrt())

    cost_per_kg_epistemic = torch.var(torch.Tensor(mus), unbiased=False)
    cost_per_kg_aleatoric = torch.mean(torch.Tensor(vars))

    return ModelDistributionOutput(feasibility={'dist': torch.distributions.Bernoulli(p_mean),
                                                'epistemic': feasibility_epistemic,
                                                'aleatoric':feasibility_aleatoric},
                                   recovery={'dist': recovery_dist, 'epistemic': recovery_epistemic, 'aleatoric': recovery_aleatoric},
                                   purity={'dist': purity_dist, 'epistemic': purity_epistemic, 'aleatoric': purity_aleatoric},
                                   cost_per_kg={'dist': cost_per_kg_dist, 'epistemic': cost_per_kg_epistemic, 'aleatoric': cost_per_kg_aleatoric},)

def convert_data_to_input_tensor(stream: StreamComposition, temperature_C: float, superstructure_idxs, data_name='train'):
    log_alphas = log_alphas_pairwise(stream.get_kgph_dict(), stream.get_props_dict(), temperature_C + 273.15)

    input_tensor = torch.tensor([stream.target_solvent['kgph'],
                                 stream.solvent2['kgph'],
                                 stream.water['kgph'],
                                 stream.salt['kgph'],
                                 stream.solids['kgph'],
                                 temperature_C,
                                 stream.target_solvent['props'].MW,
                                 stream.target_solvent['props'].rho,
                                 stream.target_solvent['props'].Tb,
                                 stream.target_solvent['props'].Hvap,
                                 stream.target_solvent['props'].Cp,
                                 stream.target_solvent['props'].logP,
                                 log_alphas['target']['solvent2'],
                                 log_alphas['target']['water'],
                                 stream.solvent2['props'].MW,
                                 stream.solvent2['props'].rho,
                                 stream.solvent2['props'].Tb,
                                 stream.solvent2['props'].Hvap,
                                 stream.solvent2['props'].Cp,
                                 stream.solvent2['props'].logP,
                                 superstructure_idxs[0],
                                superstructure_idxs[1],
                                superstructure_idxs[2],
                                superstructure_idxs[3]])

    return Dataset(data_name).standardiser_X.transform(input_tensor)


def get_ensemble_predictions(ensemble: list, stream: StreamComposition, temperature_C: float, superstructure_idxs, data_name='train'):
    input_tensor = convert_data_to_input_tensor(stream, temperature_C, superstructure_idxs, data_name=data_name)

    return get_ensemble_predictions_from_tensor(ensemble, input_tensor)

def get_single_prediction(model, stream: StreamComposition, temperature_C: float, superstructure_idxs, data_name='train'):
    log_alphas = log_alphas_pairwise(stream.get_kgph_dict(), stream.get_props_dict(), temperature_C + 273.15)

    input_tensor = torch.tensor([stream.target_solvent['kgph'],
                                 stream.solvent2['kgph'],
                                 stream.water['kgph'],
                                 stream.salt['kgph'],
                                 stream.solids['kgph'],
                                 temperature_C,
                                 stream.target_solvent['props'].MW,
                                 stream.target_solvent['props'].rho,
                                 stream.target_solvent['props'].Tb,
                                 stream.target_solvent['props'].Hvap,
                                 stream.target_solvent['props'].Cp,
                                 stream.target_solvent['props'].logP,
                                 log_alphas['target']['solvent2'],
                                 log_alphas['target']['water'],
                                 stream.solvent2['props'].MW,
                                 stream.solvent2['props'].rho,
                                 stream.solvent2['props'].Tb,
                                 stream.solvent2['props'].Hvap,
                                 stream.solvent2['props'].Cp,
                                 stream.solvent2['props'].logP,
                                 superstructure_idxs[0],
                                superstructure_idxs[1],
                                superstructure_idxs[2],
                                superstructure_idxs[3]]).to('cpu')

    input_tensor = Dataset(data_name).standardiser_X.transform(input_tensor)

    return model(input_tensor)


def transfer_ensemble_losses(losses: list[LossBreakdown], normaliser: int) -> LossBreakdown:
    '''
    Takes a list of LossBreakdowns, one per model, and converts into a LossBreakdown object, where all fields
    contain the losses for each model. The respective first first axes describe the models.
    '''
    loss_breakdown = LossBreakdown.from_shape(len(losses))

    for model_id in range(len(losses)):
        loss_breakdown.total[model_id] = losses[model_id].total.item() / normaliser
        loss_breakdown.feasibility_bce[model_id] = losses[model_id].feasibility_bce.item() / normaliser
        loss_breakdown.feasibility_brier[model_id] = losses[model_id].feasibility_brier.item() / normaliser
        loss_breakdown.recovery_nll[model_id] = losses[model_id].recovery_nll.item() / normaliser
        loss_breakdown.recovery_rmse[model_id] = losses[model_id].recovery_rmse.item() / normaliser
        loss_breakdown.purity_nll[model_id] = losses[model_id].purity_nll.item() / normaliser
        loss_breakdown.purity_rmse[model_id] = losses[model_id].purity_rmse.item() / normaliser
        loss_breakdown.cost_per_kg_nll[model_id] = losses[model_id].cost_per_kg_nll.item() / normaliser
        loss_breakdown.cost_per_kg_rmse[model_id] = losses[model_id].cost_per_kg_rmse.item() / normaliser
        loss_breakdown.num_correct[model_id] = losses[model_id].num_correct.item() / normaliser

    return loss_breakdown
