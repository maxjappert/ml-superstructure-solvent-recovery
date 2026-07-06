"""Python re-implementation of the Chea et al. (2020) solvent-recovery
superstructure framework (Ind. Eng. Chem. Res. 59, 5931-5944), reduced to a
deterministic fixed-path evaluator for ML state-space generation."""

from .framework import (compute, ComputeResult, incineration_cost,
                        SOLIDS_REMOVAL_OPTIONS, RECOVERY_OPTIONS,
                        PURIFICATION_OPTIONS, REFINEMENT_OPTIONS)
from .solvents import list_solvents, list_salts

__all__ = [
    "compute", "ComputeResult", "incineration_cost",
    "list_solvents", "list_salts",
    "SOLIDS_REMOVAL_OPTIONS", "RECOVERY_OPTIONS",
    "PURIFICATION_OPTIONS", "REFINEMENT_OPTIONS",
]
__version__ = "0.1.0"
