from dataclasses import dataclass, asdict, fields
from typing import Iterator, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from config import DEVICE, loss_scalar_fractions, loss_scalar_cost

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

    def detached_and_normalised_summary_dict(self):
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
                           recovery_logvar=y_hat_fractions[:,1],
                           purity_mu=y_hat_fractions[:,2],
                           purity_logvar=y_hat_fractions[:,3],
                           cost_per_kg_mu=y_hat_cost[:,0],
                           cost_per_kg_logvar=y_hat_cost[:,1])


def load_model(name):
    model = Model()
    model.load_state_dict(torch.load(name)['model_state_dict'])
    model.to(DEVICE)
    return model


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
