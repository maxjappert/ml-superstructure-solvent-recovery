"""Costing per Chea et al. (2020), SI section B/C (Tables C.1, C.2).

Total annualized cost [$/yr]:
    CC_tot = CC_capital + CC_labor + CC_utility + CC_membrane + CC_overhead
with
    Cc_i      = C0_i * (Qc_i / Q0_i)^0.67          (equipment purchase cost)
    CC_cap    = 1.66 * CRF * BMC * sum_i Cc_i       (CRF=0.11, BMC=5.4)
    N_lbr_i   = Nlabr_i * Qc_i / Q0_i               (laborers)
    CC_labor  = 30 $/h * 7920 h/yr * sum_i N_lbr_i
    CC_util   = (sum PW_i * 0.1 $/kWh + sum Mstm_i * 0.012 $/kg
                 + sum Mcw_i * 5e-5 $/kg) * 7920
    CC_memb   = 7920/2000 * sum_i CPM_i * Qc_i      (membrane replacement)
    CC_consum = 7920 * sum_i cons_i                  (e.g. ATPE hexane + salt)
    CC_ovhd   = 2.78 * CC_labor
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# global economic parameters (SI)
CRF = 0.11              # capital recovery factor (25 yr)
BMC_MULT = 5.4          # bare module cost multiplier
CAP_FACTOR = 1.66       # extra installed-cost factor in SI eq. for CC_AC
NC = 0.67               # cost scaling exponent ("2/3 rule")
TANN = 7920.0           # operating hours per year (330 d x 24 h)
C_LABOR = 30.0          # $/laborer/h
C_ELEC = 0.1            # $/kWh
C_STEAM = 0.012         # $/kg
C_CW = 5e-5             # $/kg cooling water
REP_TIME = 2000.0       # h, membrane/consumable replacement period

# per-technology standard capacities/costs (Table C.1 + GAMS code)
#   tech: (Q0, C0 [$], Nlabr [laborers/h at Q0], Wsp [kW per unit Qc-flow], CPM [$/m2])
TECH_COST = {
    "SDM":  dict(Q0=2500.0,   C0=1.128e6, Nlabr=0.1, Wsp=0.0,  CPM=0.0),
    "CNF":  dict(Q0=60000.0,  C0=0.66e6,  Nlabr=0.1, Wsp=19.2, CPM=0.0),
    "FLT":  dict(Q0=80.0,     C0=0.039e6, Nlabr=0.5, Wsp=0.1,  CPM=400.0),
    "MF":   dict(Q0=80.0,     C0=0.75e6,  Nlabr=1.0, Wsp=0.1,  CPM=400.0),
    "UF":   dict(Q0=80.0,     C0=0.938e6, Nlabr=1.0, Wsp=0.2,  CPM=981.0),
    "PVP":  dict(Q0=80.0,     C0=0.261e6, Nlabr=1.0, Wsp=0.33, CPM=1000.0),
    "DST":  dict(Q0=22.58,    C0=0.082e6, Nlabr=1.0, Wsp=0.0,  CPM=0.0),
    "ATPE": dict(Q0=185.0,    C0=0.362e6, Nlabr=1.0, Wsp=0.5,  CPM=0.0),
    "INCN": dict(Q0=100000.0, C0=0.967e6, Nlabr=0.1, Wsp=0.0,  CPM=0.0),
}

PRICE_HEXANE = 2.0   # $/kg (ATPE consumable)
PRICE_SALT = 0.6     # $/kg (ATPE consumable)


@dataclass
class UnitResult:
    """Design/duty summary of one unit operation."""
    tech: str                 # key into TECH_COST
    stage: str
    Qc: float = 0.0           # costing capacity (m2, m3, m3/h ... per Table C.1)
    PW: float = 0.0           # electrical power [kW]
    Mstm: float = 0.0         # steam [kg/h]
    Mcw: float = 0.0          # cooling water [kg/h]
    cons_per_h: float = 0.0   # non-membrane consumables [$/h]
    feasible: bool = True
    note: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def Cc(self) -> float:
        # Cost of technologies [SI p.6]: Cc_i/C0_i = (Qc_i/Q0_i)^nc,
        # nc = 0.67 ("2/3 rule") [GAMS]; C0, Q0 from SI Table C.1
        p = TECH_COST[self.tech]
        if self.Qc <= 0:
            return 0.0
        return p["C0"] * (self.Qc / p["Q0"]) ** NC

    @property
    def Nlbr(self) -> float:
        # Labor requirement [SI p.6]: Nlb_i * Q0_i = Nlabr_i * Qc_i
        p = TECH_COST[self.tech]
        return p["Nlabr"] * self.Qc / p["Q0"]


def annualized_cost(units: List[UnitResult]) -> Dict[str, float]:
    """Aggregate annualized cost [$/yr]; SI section B, "Costing" (pp. 6-7).
    (The SI expresses these in million $/yr; here plain $/yr.)"""
    # Annualized capital [SI p.6]: CC_AC = 1.66 * CRF * BMC * sum_i Cc_i
    cap = CAP_FACTOR * CRF * BMC_MULT * sum(u.Cc for u in units)
    # Labor cost [SI p.6]: CC_LB = Clbr * Tann * sum_i Nlbr_i
    labor = C_LABOR * TANN * sum(u.Nlbr for u in units)
    # Utility cost [SI p.7]:
    #   CC_UC = (sum PW_i*C_elec + sum Mstm_i*C_stm) * Tann
    # [DEV] plus the cooling-water term Mcw_i*C_cw: the SI equation omits it
    # although SI Table C.2 prices cooling water; included here (tiny).
    util = (sum(u.PW for u in units) * C_ELEC
            + sum(u.Mstm for u in units) * C_STEAM
            + sum(u.Mcw for u in units) * C_CW) * TANN
    # Membrane cost [SI p.7]: CC_MC = Tann * sum_i1 CPM_i1*Qc_i1 / Rep_time
    memb = (TANN / REP_TIME) * sum(TECH_COST[u.tech]["CPM"] * u.Qc for u in units)
    # Consumable costs [SI p.6]: Cons_i = (tau_ann/theta_rep)*pi_rep*Qc_i;
    # realized here as $/h streams (ATPE hexane/salt make-up)
    consum = TANN * sum(u.cons_per_h for u in units)
    # Other cost [SI p.7]: CC_OC = 2.78 * CC_LB
    overhead = 2.78 * labor
    # Total [SI p.7]: CC_TC = CC_AC + CC_UC + CC_MC + CC_OC + CC_LB
    total = cap + labor + util + memb + consum + overhead
    return {
        "capital": cap, "labor": labor, "utilities": util,
        "membranes": memb, "consumables": consum, "overhead": overhead,
        "total": total,
    }
