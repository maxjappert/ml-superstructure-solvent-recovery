"""Superstructure pathway evaluation (re-implementation of Chea et al. 2020).

compute() simulates ONE fixed path through the four-stage superstructure
(solids removal -> recovery -> purification -> refinement) for a waste stream
of {target solvent, second solvent, water, dissolved salt, suspended solids}
and returns the annualized cost, the achieved target purity, and the target
recovery. Index 0 in any stage is a bypass.

Stage index maps:
    solids removal : 0 bypass | 1 sedimentation | 2 centrifugation | 3 filtration
    recovery       : 0 bypass | 1 distillation  | 2 pervaporation  | 3 ATPE
    purification   : 0 bypass | 1 distillation  | 2 pervaporation  | 3 ultrafiltration
    refinement     : 0 bypass | 1 distillation  | 2 pervaporation  | 3 ultrafiltration | 4 microfiltration
"""
from __future__ import annotations

from typing import NamedTuple

from . import units as U
from .costing import annualized_cost
from .properties import (get_solvent_props, get_salt_props, get_solids_props,
                         get_water_props, get_extractant_props)

NAN = float("nan")

SOLIDS_REMOVAL_OPTIONS = {0: "BYP", 1: "SDM", 2: "CNF", 3: "FLT"}
RECOVERY_OPTIONS = {0: "BYP", 1: "DST", 2: "PVP", 3: "ATPE"}
PURIFICATION_OPTIONS = {0: "BYP", 1: "DST", 2: "PVP", 3: "UF"}
REFINEMENT_OPTIONS = {0: "BYP", 1: "DST", 2: "PVP", 3: "UF", 4: "MF"}


class ComputeResult(NamedTuple):
    cost_usd_per_year: float          # total annualized cost [$/yr]
    target_purity: float              # mass fraction of target in product [-]
    target_recovery: float            # recovered target / fed target [-]
    cost_usd_per_kg_recovered: float  # [$/kg target recovered]
    feasible: bool
    path: tuple                       # e.g. ('BYP', 'PVP', 'UF', 'BYP')
    cost_breakdown: dict              # capital/labor/utilities/... [$ated /yr]


def _infeasible(path, note, breakdown=None) -> ComputeResult:
    return ComputeResult(NAN, NAN, 0.0, NAN, False, path,
                         breakdown or {"note": note})


def compute(solvent_target_name: str,
            solvent2_name: str,
            salt_name: str,
            temperature_C: int,
            solvent_target_kgph: int,
            solvent2_kgph: int,
            water_kgph: int,
            salt_kgph: int,
            solids_kgph: int,
            idx_solids_removal: int,
            idx_recovery: int,
            idx_purification: int,
            idx_refinement: int) -> ComputeResult:
    """Evaluate one waste stream + one path through the superstructure.

    Returns a ComputeResult; the first three fields are
    (annualized cost [$/yr], target purity [-], target recovery [-]).
    Infeasible paths return (nan, nan, 0.0) with feasible=False.
    """
    # ---- validate indices
    for idx, table, nm in ((idx_solids_removal, SOLIDS_REMOVAL_OPTIONS, "solids removal"),
                           (idx_recovery, RECOVERY_OPTIONS, "recovery"),
                           (idx_purification, PURIFICATION_OPTIONS, "purification"),
                           (idx_refinement, REFINEMENT_OPTIONS, "refinement")):
        if idx not in table:
            raise ValueError(f"invalid {nm} index {idx}; valid: {sorted(table)}")
    path = (SOLIDS_REMOVAL_OPTIONS[idx_solids_removal],
            RECOVERY_OPTIONS[idx_recovery],
            PURIFICATION_OPTIONS[idx_purification],
            REFINEMENT_OPTIONS[idx_refinement])

    if solvent_target_kgph <= 0:
        raise ValueError("solvent_target_kgph must be > 0")
    for v, nm in ((solvent2_kgph, "solvent2_kgph"), (water_kgph, "water_kgph"),
                  (salt_kgph, "salt_kgph"), (solids_kgph, "solids_kgph")):
        if v < 0:
            raise ValueError(f"{nm} must be >= 0")

    props = {
        "target": get_solvent_props(solvent_target_name),
        "solvent2": get_solvent_props(solvent2_name),
        "water": get_water_props(),
        "salt": get_salt_props(salt_name),
        "solids": get_solids_props(),
        "extractant": get_extractant_props(),
    }

    stream = {
        "target": float(solvent_target_kgph),
        "solvent2": float(solvent2_kgph),
        "water": float(water_kgph),
        "salt": float(salt_kgph),
        "solids": float(solids_kgph),
        "extractant": 0.0,
    }
    fed_target = stream["target"]
    T = float(temperature_C)

    unit_results = []

    def run(tech: str, stage: str, s):
        if tech == "BYP":
            return s, None
        if tech == "SDM":
            prod, _, r = U.sedimentation(s, props, stage)
        elif tech == "CNF":
            prod, _, r = U.centrifugation(s, props, stage)
        elif tech == "FLT":
            prod, _, r = U.filtration(s, props, stage)
        elif tech == "DST":
            prod, _, r = U.distillation(s, props, stage, T)
        elif tech == "PVP":
            prod, _, r = U.pervaporation(s, props, stage, T)
        elif tech == "ATPE":
            prod, _, r = U.atpe(s, props, stage)
        elif tech == "UF":
            prod, _, r = U.ultrafiltration(s, props, stage)
        elif tech == "MF":
            prod, _, r = U.microfiltration(s, props, stage)
        else:  # pragma: no cover
            raise ValueError(tech)
        return prod, r

    for tech, stage in zip(path, ("solids_removal", "recovery",
                                  "purification", "refinement")):
        stream, res = run(tech, stage, stream)
        if res is not None:
            if not res.feasible:
                return _infeasible(path, f"{tech} ({stage}): {res.note}")
            unit_results.append(res)
            if stream["target"] <= U.TINY:
                return _infeasible(path, f"{tech} ({stage}): target lost")

    breakdown = annualized_cost(unit_results)
    cost = breakdown["total"]
    mtot = U.total_mass(stream)
    purity = stream["target"] / mtot if mtot > 0 else NAN
    recovery = stream["target"] / fed_target
    rec_kg_per_year = stream["target"] * 7920.0
    cost_per_kg = cost / rec_kg_per_year if rec_kg_per_year > 0 else NAN
    breakdown["units"] = [
        dict(tech=u.tech, stage=u.stage, Qc=u.Qc, Cc=u.Cc, PW=u.PW,
             Mstm=u.Mstm, Mcw=u.Mcw) for u in unit_results
    ]
    return ComputeResult(cost, purity, recovery, cost_per_kg, True, path,
                         breakdown)


def incineration_cost(total_kgph: float, organic_fraction: float = 1.0,
                      heating_value_MJ_per_kg: float = 30.0) -> float:
    """Annualized incineration baseline [$/yr] (SI, simplified).

    Uses the SI's fuel/air/energy-credit structure with a generic heating
    value for the organic fraction instead of the Dulong elemental formula.
    """
    m = total_kgph / 3600.0                       # kg/s
    Q = heating_value_MJ_per_kg * organic_fraction
    m_fuel = Q * m / 38.9                         # kg/s fuel
    m_air = 4.35 * 2.0 * m                        # kg/s (approx. O demand)
    e_con = Q * m                                 # MJ/s
    e_net = (0.35 - 1.0) * e_con                  # MJ/s (35% recovered)
    sec_yr = 3600 * 24 * 340
    fuel = m_fuel * 0.81 * sec_yr
    energy = e_net * (0.10 / 3.6) * sec_yr        # $ credit if negative cost
    air = 0.0004 / 1.2 * m_air * sec_yr
    cap = 0.967e6 * (total_kgph / 1e5) ** 0.67
    labor = 0.1 * (total_kgph / 1e5) * 30.0 * 24 * 340
    return fuel + air + cap + labor + max(-energy, 0.0)
