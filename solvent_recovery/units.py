"""Unit-operation models (SI of Chea et al. 2020, sections B/C/F).

A stream is a dict {component_key: mass flow kg/h}. Component keys used by the
framework: 'target', 'solvent2', 'water', 'salt', 'solids', 'extractant'.

Each unit returns (product_stream, waste_stream, UnitResult). The product
stream is the one that carries the target solvent onward.

Where the paper hard-codes case-specific separation parameters (membrane
retention factors, ATPE partition coefficients), this module estimates them
from pure-component properties so that outputs vary smoothly over arbitrary
solvent pairs (see README, "Generalized separation parameters").

Equation provenance convention used in the comments below:
  [SI p.X]  -- equation implemented as printed in the Supporting Information
               of Chea et al. 2020 (page numbers of ie9b06725_si_001.pdf)
  [GAMS]    -- value/equation taken from the SI's GAMS listings (section F),
               used when text and code disagree (the GAMS code produced the
               published results)
  [DEV]     -- deliberate deviation/generalization; rationale given inline

SI models NOT implemented (not part of this superstructure): Dryer (SI p.16,
case-study-2 salt-drying loop) and the full Dulong-formula incineration
(SI p.20-21; a simplified version lives in framework.incineration_cost).
The SI's concentration-factor bound constraints (e.g. 1.01 <= CF_UF <= 35)
are not enforced [DEV]: CF is not a free variable here -- it is implied by
the retention factors -- so the bounds would only clip pathological cases.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

from .costing import UnitResult, PRICE_HEXANE, PRICE_SALT
from .properties import ComponentProps

Stream = Dict[str, float]
Props = Dict[str, ComponentProps]

TINY = 1e-9
G = 9.81                # m/s2
MU_L = 0.0012           # Pa*s, liquid viscosity (SI value)
DP_SOLID = 4e-5         # m, solid particle diameter (SI case study 2)
HSTM_PVP = 2116.0       # kJ/kg latent heat of steam (SI case study 1)
HSTM_DST = 2257.92      # kJ/kg latent heat of steam at 1 bar (SI)
CP_W = 4.184            # kJ/kg/K
DT_CW = 10.0            # K cooling water rise (25->35 ... SI uses 20->30)
U_VAP = 10800.0         # m/h vapor velocity in column (SI)
HETP = 0.457            # m per actual stage (SI: 1.5 ft)
STAGE_EFF = 0.8         # stage efficiency (SI)
ALPHA_MIN = 1.05        # minimum relative volatility for distillation (paper)

# membrane fluxes zeta [m3/m2/h] (SI C.6/C.9; PVP per case-study-1 GAMS code)
ZETA = {"UF": 0.0856, "PVP": 0.055, "FLT": 0.2, "MF": 0.2}
ATPE_MAKEUP = 0.05      # fraction of circulating hexane/salt purchased as make-up
                        # (the paper's superstructure recycles both)


# --------------------------------------------------------------------------- helpers
def _sigmoid(x: float) -> float:
    if x > 40:
        return 1.0
    if x < -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def vol_flow(stream: Stream, props: Props) -> float:
    """Volumetric flow [m3/h]."""
    return sum(m / props[k].rho for k, m in stream.items() if m > TINY)


def total_mass(stream: Stream) -> float:
    return sum(m for m in stream.values() if m > 0)


def _liquid_keys(stream: Stream, props: Props):
    return [k for k, m in stream.items() if m > TINY and props[k].volatile]


def _empty_like(stream: Stream) -> Stream:
    return {k: 0.0 for k in stream}


def _membrane_split(stream: Stream, retention: Dict[str, float]
                    ) -> Tuple[Stream, Stream]:
    """Retention-factor equation [SI p.8, p.9, p.19]:
        xi_k,i = M_Jretentate,k / M_Jin,k
    i.e. retention[k] is the fraction of component k staying in the
    RETENTATE; mass balance M_in = M_perm + M_ret [SI p.5, B.6] holds
    exactly by construction."""
    perm, ret = _empty_like(stream), _empty_like(stream)
    for k, m in stream.items():
        r = retention.get(k, 1.0)
        ret[k] = m * r
        perm[k] = m * (1.0 - r)
    return perm, ret


def _membrane_area(stream: Stream, props: Props, permeate: Stream,
                   zeta: float) -> float:
    """Flux balance [SI p.8 (UF), p.9 (PVP), p.19 (FLT)]:
        zeta_i * Qc_i = [sum_k V_feed,k] * (1 - 1/CF_i)
    with the concentration factor CF_i = V_feed/V_retentate [SI p.8]. Since
    V_feed*(1 - 1/CF) == V_permeate, the area follows directly from the
    permeate volumetric flow; CF is implied, not a free variable."""
    vperm = vol_flow(permeate, props)
    return max(vperm / zeta, 1e-6)


def _settling_velocity(props: Props, stream: Stream) -> float:
    """Stokes settling velocity [SI p.14]:
        U_S = g * Dp^2 * (rho_s - rho_L) / (18 * mu)
    with Dp = 4e-5 m and mu = 0.0012 N s/m2 [GAMS, case study 2].
    [DEV] rho_L is the volume-averaged density of the actual liquid phase
    (SI uses water); floor of 50 kg/m3 on the density difference avoids a
    zero/negative settling velocity for near-buoyant solids."""
    liq = _liquid_keys(stream, props)
    mliq = sum(stream[k] for k in liq)
    if mliq > TINY:
        rho_l = mliq / sum(stream[k] / props[k].rho for k in liq)
    else:
        rho_l = 1000.0
    drho = max(props["solids"].rho - rho_l, 50.0)
    return G * DP_SOLID ** 2 * drho / (18.0 * MU_L)


# --------------------------------------------------------------------------- solids removal
def sedimentation(stream: Stream, props: Props, stage: str):
    """Gravity sedimentation [SI p.14-15]."""
    # Efficiency equations [GAMS, F.2]: Eff*M_in,solid = M_sludge,solid and
    # Eff*M_in,liquid = M_clarified,liquid, with Eff_SDM = 0.7 [SI C.5]
    # (i.e. 70% of solids captured; 30% of every liquid lost to the sludge).
    eff = 0.70
    clar, sludge = _empty_like(stream), _empty_like(stream)
    for k, m in stream.items():
        if k == "solids":
            sludge[k] = eff * m
            clar[k] = (1 - eff) * m
        else:
            clar[k] = eff * m
            sludge[k] = (1 - eff) * m
    us = _settling_velocity(props, stream)          # [SI p.14] Stokes
    # Surface overflow rate: SOR = Eff * U_S [GAMS: SOR_SDM=Eff_SDM*Ug_SDM].
    # NOTE: the SI text (p.15) prints SOR = U_S / eta -- the SI contradicts
    # itself; we follow the GAMS listing that produced the published results.
    sor = eff * us                                  # m/s
    # Tank area [SI p.15]: A_SDM = V_feed / SOR  (3600: m/s -> m/h)
    area = vol_flow(stream, props) / (sor * 3600.0)
    # Costing capacity [SI p.15]: Qc_SDM = A_SDM; PW_SDM = 0 [GAMS]
    res = UnitResult("SDM", stage, Qc=area)
    return clar, sludge, res


def centrifugation(stream: Stream, props: Props, stage: str):
    """Centrifugation [SI p.18]."""
    # Efficiency equations [GAMS, F.2]: Eff_CNT*M_feed,solid = M_solids-out
    # and Eff_CNT*M_feed,liquid = M_liquid-out, with Eff_CNT = 1 [GAMS].
    liq, solids_out = _empty_like(stream), _empty_like(stream)
    for k, m in stream.items():
        if k == "solids":
            solids_out[k] = m
        else:
            liq[k] = m
    v = vol_flow(stream, props)                     # m3/h
    us = _settling_velocity(props, stream)          # m/s [SI p.14]
    # Sigma factor equation [SI p.18]: Qc_CNF * U_CNF = V_feed.
    # [DEV] The SI never assigns a value to U_CNF; we close the model with
    # sigma theory, U = 2 * u_g (throughput per unit sigma area = twice the
    # gravitational settling velocity), converted to m/h.
    U = 2.0 * us * 3600.0                           # m/h
    sigma = v / max(U, 1e-9)                        # m2
    # Power [SI p.18]: PW = Wsp_CNF * V_feed, Wsp = 19.2 kW/(m3/h) [SI C.1]
    pw = 19.2 * v
    # Cooling duty [SI p.18]: Mcw*cp_w*(Tcw_out-Tcw_in) = 0.4*PW ("power
    # dissipation to heat is about 40%"). [DEV] SI omits the kJ/s -> kJ/h
    # factor; 3600 restores unit consistency (kg/h of cooling water).
    mcw = 0.4 * pw * 3600.0 / (CP_W * DT_CW)
    res = UnitResult("CNF", stage, Qc=sigma, PW=pw, Mcw=mcw)
    return liq, solids_out, res


def filtration(stream: Stream, props: Props, stage: str):
    """Filtration [SI p.19]."""
    # Retention factors [SI C.9]: solids/salt-hydrate 100%, liquids 10%
    # (case study 2 uses 10% for every liquid organic). [DEV] dissolved salt
    # is treated as liquid-borne here (10%), not as the SI's solid hydrate.
    retention = {k: 0.10 for k in stream}
    retention["solids"] = 1.0
    filtrate, cake = _membrane_split(stream, retention)      # [SI p.19]
    # Flux balance [SI p.19] via _membrane_area; zeta_FLT = 0.2 [SI C.9]
    area = _membrane_area(stream, props, filtrate, ZETA["FLT"])
    pw = 0.1 * area              # PW = Wsp*Qc [SI p.19], Wsp = 0.1 [SI C.1]
    res = UnitResult("FLT", stage, Qc=area, PW=pw)
    return filtrate, cake, res


# --------------------------------------------------------------------------- distillation
def _alphas(stream: Stream, props: Props, T_ref: float) -> Dict[str, float]:
    """Relative volatilities alpha_k [SI B.3 parameter].
    [DEV] The SI specifies alpha as an input parameter per case study (e.g.
    alpha_IPA = 1.68 [GAMS]; DME/EME via the boiling-point correlation of
    SI section D). Here alpha_k = Psat_k(T_ref)/Psat_ref(T_ref) (ideal
    Raoult ratio, reference = least volatile component present) so that any
    solvent pair is covered."""
    vols = _liquid_keys(stream, props)
    ps = {k: props[k].Psat(T_ref) for k in vols}
    pmin = min(ps.values())
    return {k: ps[k] / max(pmin, 1e-9) for k in vols}


def distillation(stream: Stream, props: Props, stage: str, T_feed: float):
    """Fenske / Underwood / Gilliland-style shortcut column (SI eqs.).

    Deterministic choices (paper defaults): saturated liquid feed (q=1),
    R = 1.3 Rmin, N = Nmin/0.6, distillate LK:HK mole spec 92:8,
    LK recovery 99%. Split point = adjacent volatility gap next to the
    target with the larger relative volatility ratio.
    """
    vols = _liquid_keys(stream, props)
    nonvols = [k for k in stream if k not in vols]
    if "target" not in vols:
        res = UnitResult("DST", stage, feasible=False,
                         note="no volatile target in feed")
        return stream, _empty_like(stream), res

    mtot = sum(stream[k] for k in vols)
    T_bub = sum(stream[k] * props[k].Tb for k in vols) / mtot  # K
    alpha = _alphas(stream, props, T_bub)
    order = sorted(vols, key=lambda k: -alpha[k])

    if len(vols) == 1:
        # simple evaporation of the single volatile from non-volatiles
        top, bot = _empty_like(stream), _empty_like(stream)
        k = vols[0]
        top[k] = 0.99 * stream[k]
        bot[k] = 0.01 * stream[k]
        for j in nonvols:
            bot[j] = stream[j]
        R = 1.3
        return _dst_finish(stream, top, bot, props, stage, T_feed, T_bub, R,
                           Nact=4.0, target_in_top=True)

    i = order.index("target")
    cands = []
    if i > 0:
        cands.append((i - 1, alpha[order[i - 1]] / alpha[order[i]]))
    if i < len(order) - 1:
        cands.append((i, alpha[order[i]] / alpha[order[i + 1]]))
    j, aratio = max(cands, key=lambda c: c[1])
    LK, HK = order[j], order[j + 1]
    target_in_top = i <= j

    if aratio < ALPHA_MIN:
        res = UnitResult("DST", stage, feasible=False,
                         note=f"relative volatility {aratio:.3f} < {ALPHA_MIN}")
        return stream, _empty_like(stream), res

    # Molar flows [SI p.11]: F_j,k = M_j,k / MW_k
    F = {k: stream[k] / props[k].MW for k in vols}      # kmol/h
    # Component splits. SI p.11 "Constraints on recovery": components more
    # volatile than the LK have Xm_top ~ full recovery, components heavier
    # than the HK have Xm_top = 0; implemented as fixed sharp recoveries
    # (0.995 / 0.001) [DEV: SI states them as mole-fraction exclusion
    # constraints, equivalent at shortcut fidelity].
    rec_top = {}
    for idx, k in enumerate(order):
        if idx < j:
            rec_top[k] = 0.995
        elif k == LK:
            rec_top[k] = 0.99   # LK recovery (sharp-split assumption)
        elif k == HK:
            rec_top[k] = 0.0    # set below
        else:
            rec_top[k] = 0.001
    # Distillate spec [SI p.11 "Distillate recovery constraints"]:
    #   Xm_top,LK = 0.92 and Xm_top,HK = 0.08  (mole basis)
    # => D_HK = (0.08/0.92) * D_light-ends, capped at 50% of the HK feed.
    D_LK = rec_top[LK] * F[LK]
    D_lights = sum(rec_top[k] * F[k] for k in order[:j])
    D_HK = min((0.08 / 0.92) * (D_LK + D_lights), 0.5 * F[HK])
    rec_top[HK] = D_HK / max(F[HK], TINY)
    rec_top[HK] = min(max(rec_top[HK], 1e-4), 0.5)

    top, bot = _empty_like(stream), _empty_like(stream)
    for k in vols:
        top[k] = rec_top[k] * stream[k]
        bot[k] = stream[k] - top[k]
    for k in nonvols:
        bot[k] = stream[k]

    # Fenske equation [SI p.11]:
    #   Nmin * log(alpha) = log[(Xm_top,LK*Xm_bot,HK)/(Xm_top,HK*Xm_bot,LK)]
    # written in the equivalent recovery form r/(1-r).
    rl, rh = rec_top[LK], rec_top[HK]
    Nmin = math.log((rl / (1 - rl)) * ((1 - rh) / rh)) / math.log(aratio)
    Nmin = max(Nmin, 1.0)       # bound Nmin >= 1 [SI p.13: Nmin >= y_DST]

    # Underwood's variable [SI p.11], saturated-liquid feed q = 1 [GAMS]:
    #   0 = sum_k alpha_k * Xm_feed,k / (alpha_k - Uv)
    # volatilities taken relative to the HK
    a = {k: alpha[k] / alpha[HK] for k in vols}
    Ftot = sum(F.values())
    z = {k: F[k] / Ftot for k in vols}
    D = {k: rec_top[k] * F[k] for k in vols}
    Dtot = sum(D.values())
    xD = {k: D[k] / Dtot for k in vols}

    lo, hi = 1.0 + 1e-6, a[LK] - 1e-6

    def f(theta):
        return sum(a[k] * z[k] / (a[k] - theta) for k in vols)

    theta = 0.5 * (lo + hi)
    if hi > lo:
        flo, fhi = f(lo), f(hi)
        for _ in range(200):
            theta = 0.5 * (lo + hi)
            ft = f(theta)
            if abs(ft) < 1e-10:
                break
            if (ft > 0) == (flo > 0):
                lo, flo = theta, ft
            else:
                hi, fhi = theta, ft
    # Minimum reflux [SI p.11]: Rmin = sum_k alpha_k*Xm_top,k/(alpha_k-Uv) - 1
    Rmin = sum(a[k] * xD[k] / (a[k] - theta) for k in vols) - 1.0
    Rmin = max(Rmin, 0.1)
    # Actual reflux [SI p.12]: R = 1.3 * Rmin (stated assumption);
    # lower bound 1.01 from the SI's variable bounds (Rmin >= 1.01*y_DST)
    R = max(1.3 * Rmin, 1.01)
    # Number of stages [SI p.12]: 0.6*N = Nmin ("suggested by Towler");
    # actual stages: Nact * eta_stage = N, eta_stage = 0.8 [GAMS]
    Nact = (Nmin / 0.6) / STAGE_EFF
    return _dst_finish(stream, top, bot, props, stage, T_feed, T_bub, R,
                       Nact=Nact, target_in_top=target_in_top)


def _dst_finish(feed, top, bot, props, stage, T_feed, T_bub, R, Nact,
                target_in_top):
    vols = _liquid_keys(feed, props)
    D_mass = total_mass(top)
    # Internal flows [SI p.12]: Liq = R * sum(M_top); Vap = Liq + sum(M_top)
    Liq = R * D_mass
    Vap = Liq + D_mass                                   # kg/h
    # Column diameter [SI p.12]: D = sqrt(4*Vap/(pi*u_vap)), u_vap = 10800
    # m/h [GAMS]. [DEV] The SI divides a MASS flow by a velocity (implicit
    # vapor density of 1); here Vap is first converted to a volumetric flow
    # with an ideal-gas vapor density at the bubble point for unit
    # consistency. Lower bound D >= 0.6 m from the GAMS variable bounds.
    Fmol_top = sum(top[k] / props[k].MW for k in vols if top[k] > TINY)
    MW_top = D_mass / max(Fmol_top, TINY)                # kg/kmol
    rho_vap = 101325.0 * MW_top / (8314.0 * T_bub)       # kg/m3
    v_vap = Vap / max(rho_vap, 1e-6)                     # m3/h
    Dcol = max(math.sqrt(4.0 * v_vap / (math.pi * U_VAP)), 0.6)
    # Column height [SI p.12]: H = HETP * Nact (HETP = 1.5 ft [GAMS],
    # kept in metres here: 0.457 m)
    H = HETP * Nact
    # Costing variable [SI p.12]: Qc_DST = (pi/4) * D^2 * H  (volume, m3)
    Vol = math.pi / 4.0 * Dcol ** 2 * H

    # Feed preheat to saturation [SI p.12]:
    #   QS = sum_k M_k * Cp_k * (Tsat - Tamb)
    # [DEV] SI evaluates per-component Tsat over the distillate; here the
    # feed is heated from T_feed (user input, SI: Tamb = 20 C) to the
    # mixture bubble point T_bub.
    QS = sum(feed[k] * props[k].Cp * max(T_bub - (T_feed + 273.15), 0.0)
             for k in vols)                              # kJ/h
    # Reboiler duty [SI p.12]: QH = (1+R) * sum_top(F*MW*lambda_vap)
    QH = (1 + R) * sum(top[k] * props[k].Hvap for k in vols)
    # Condenser duty [SI p.12]: QC = R * sum_top(F*MW*lambda_vap)
    QC = R * sum(top[k] * props[k].Hvap for k in vols)
    # Steam [SI p.13]: Mstm * lambda_stm = QS + QH (lambda = 2257.92 kJ/kg)
    Mstm = (QS + QH) / HSTM_DST
    # Cooling water [SI p.13]: Mcw * Cp_w * (Tcw_out - Tcw_in) = QC
    Mcw = QC / (CP_W * DT_CW)
    res = UnitResult("DST", stage, Qc=Vol, Mstm=Mstm, Mcw=Mcw,
                     extras={"R": R, "Nact": Nact, "Dcol": Dcol})
    if target_in_top:
        return top, bot, res
    return bot, top, res


# --------------------------------------------------------------------------- pervaporation
def pervaporation(stream: Stream, props: Props, stage: str, T_feed: float):
    """Organophilic pervaporation (SI C.7/C.10). Retention factors are
    estimated from vapor pressures: volatile components permeate, heavy
    ones are retained; water floor-retained at 0.90 (both case studies)."""
    vols = _liquid_keys(stream, props)
    if "target" not in vols:
        res = UnitResult("PVP", stage, feasible=False,
                         note="no volatile target in feed")
        return stream, _empty_like(stream), res

    T = T_feed + 273.15
    mtot = sum(stream[k] for k in vols)
    # pivot pressure: midpoint of the volatility range of components that are
    # actually present (> 0.1% of the stream); a single-volatile stream is
    # simply concentrated (target permeates)
    lp = {k: math.log10(max(props[k].Psat(T), 1e-3)) for k in vols}
    major = [k for k in vols if stream[k] > 1e-3 * mtot]
    logp = 0.5 * (max(lp[k] for k in major) + min(lp[k] for k in major))
    single = len(major) < 2
    retention = {}
    for k in stream:
        if k not in vols:
            retention[k] = 1.0
            continue
        if single:
            r = 0.05 if k == "target" else 0.97
        else:
            s = _sigmoid(15.0 * (lp[k] - logp))
            r = 0.97 - 0.94 * s
        if k == "water" and not (single and k == "target"):
            r = max(r, 0.90)                     # hydrophobic membrane (paper)
        retention[k] = min(max(r, 0.03), 0.97)

    # Retention-factor split [SI p.9] and flux balance [SI p.9]:
    #   zeta_PVP * Qc = V_feed * (1 - 1/CF); zeta = 0.055 [GAMS, F.1]
    # [DEV] Retention values themselves are property-estimated (sigmoid
    # above) instead of the SI's per-case constants (C.7: IPA 0.05, water
    # 0.90; C.10: DME/EME 0.05, toluene 0.97).
    perm, ret = _membrane_split(stream, retention)
    area = _membrane_area(stream, props, perm, ZETA["PVP"])
    # Power [SI p.9]: PW = Wsp_PVP * Qc, Wsp = 0.33 [SI C.1]
    pw = 0.33 * area
    # Heat for vaporization [SI p.9]:
    #   Mstm * lambda_stm = sum_permeate(M_j,k * lambda_vap,k)
    # lambda_stm = 2116 kJ/kg [GAMS, F.1 'Hstm']
    Mstm = sum(perm[k] * props[k].Hvap for k in vols) / HSTM_PVP
    res = UnitResult("PVP", stage, Qc=area, PW=pw, Mstm=Mstm,
                     extras={"retention": retention})
    # product = side with more target
    if perm["target"] >= ret["target"]:
        return perm, ret, res
    return ret, perm, res


# --------------------------------------------------------------------------- UF / MF
def ultrafiltration(stream: Stream, props: Props, stage: str):
    """Solvent-selective UF (SI C.6/C.11): the target passes, water and the
    ATPE extractant are strongly retained (paper values); the co-solvent is
    retained according to molecular size relative to the target."""
    retention = {}
    vm_t = props["target"].Vm
    for k, m in stream.items():
        if k == "target":
            retention[k] = 0.001
        elif k == "water":
            retention[k] = 0.998
        elif k == "extractant":
            retention[k] = 0.999
        elif k in ("salt", "solids"):
            retention[k] = 1.0
        else:  # solvent2: size exclusion vs. target
            r = 0.02 + 0.96 * _sigmoid(12.0 * math.log(props[k].Vm / vm_t))
            retention[k] = min(max(r, 0.02), 0.98)
    # Retention split + flux balance [SI p.8]; zeta_UF = 0.0856 [SI C.6].
    # Fixed retentions mirror SI C.6 (target ~0 / water 0.998 / hexane 0.999
    # / salt 1.0); the co-solvent value is the [DEV] size sigmoid above
    # (SI C.11 uses a per-case constant, e.g. EME 0.97).
    perm, ret = _membrane_split(stream, retention)
    area = _membrane_area(stream, props, perm, ZETA["UF"])
    pw = 0.2 * area              # PW = Wsp*Qc [SI p.8], Wsp = 0.2 [SI C.1]
    res = UnitResult("UF", stage, Qc=area, PW=pw,
                     extras={"retention": retention})
    return perm, ret, res


def microfiltration(stream: Stream, props: Props, stage: str):
    """Microfiltration. The SI provides no dedicated MF model equations --
    only its costing row (SI Table C.1: Q0 = 80 m2, C0 = $0.75M, Wsp = 0.1,
    consumable $400/m2). [DEV] Modeled with the generic membrane equations
    (retention split + flux balance, as FLT [SI p.19]) with polishing-grade
    retentions: solids 1.0, salt 0.999, liquids 5% holdup loss."""
    retention = {k: 0.05 for k in stream}
    retention["solids"] = 1.0
    retention["salt"] = 0.999
    perm, ret = _membrane_split(stream, retention)
    area = _membrane_area(stream, props, perm, ZETA["MF"])
    pw = 0.1 * area
    res = UnitResult("MF", stage, Qc=area, PW=pw)
    return perm, ret, res


# --------------------------------------------------------------------------- ATPE
def atpe(stream: Stream, props: Props, stage: str):
    """Aqueous two-phase extraction with n-hexane + NaCl (SI C.4).

    Partition coefficients are estimated from logP (octanol-water as a proxy
    for the hexane-water pair); water is fixed at Kp = 0.05 (paper). The
    number of stages is set so the target reaches 90% extraction (paper's
    recovery constraint); all other components then follow the Kremser
    equation at that stage count. Added hexane = liquid feed mass; added
    salt = 0.2 x aqueous mass (SI Frc_Salt).
    """
    liq = _liquid_keys(stream, props)
    if "target" not in liq:
        res = UnitResult("ATPE", stage, feasible=False,
                         note="no liquid target in feed")
        return stream, _empty_like(stream), res

    # Hexane and salt dosage. [DEV] In the SI GAMS these are free decision
    # variables (M5, M6); deterministic closure here: hexane 1:1 with the
    # liquid feed, salt = 0.2 x aqueous mass (Frc_Salt = 0.2 [GAMS, F.1]).
    m_feed_liq = sum(stream[k] for k in liq)
    m_hex = m_feed_liq
    m_salt_add = max(0.2 * (stream.get("water", 0.0) + TINY), 0.05 * m_feed_liq)

    def kp(k):
        if k == "water":
            return 0.05
        return min(max(10.0 ** props[k].logP, 1e-3), 45.0)

    # Extraction factor [SI p.17]: EF_k = kappaP_k * M_Hexane / M_salt
    # [DEV] kappaP from logP (SI C.4 fixes kappaP_IPA = 8, kappaP_water =
    # 0.05; the water value is retained as-is).
    EF = {k: kp(k) * m_hex / m_salt_add for k in liq}
    ef_t = EF["target"]
    if ef_t <= 1.02:
        res = UnitResult("ATPE", stage, feasible=False,
                         note=f"extraction factor {ef_t:.2f} <= 1 for target")
        return stream, _empty_like(stream), res

    # Number of stages [SI p.17, Kremser form]:
    #   (EF - 1)/(EF^(NAE+1) - 1) = (M_feed,k - M_top,k)/M_feed,k
    # solved for NAE at the paper's 90% target-recovery requirement
    # [GAMS, F.1: M11_IPA >= 0.9*M2_IPA] => raffinate fraction phi = 0.1.
    Np1 = math.log(1.0 + 10.0 * (ef_t - 1.0)) / math.log(ef_t)
    Np1 = max(Np1, 2.0)

    def extracted(ef):
        # same Kremser equation applied to every other component at the
        # stage count fixed by the target [SI p.17]
        if abs(ef - 1.0) < 1e-6:
            return 1.0 - 1.0 / Np1
        phi = (ef - 1.0) / (ef ** Np1 - 1.0)
        return min(max(1.0 - phi, 0.0), 1.0)

    top, bot = _empty_like(stream), _empty_like(stream)
    for k in liq:
        e = extracted(EF[k])
        top[k] = e * stream[k]
        bot[k] = stream[k] - top[k]
    for k in ("salt", "solids"):
        bot[k] = bot.get(k, 0.0) + stream.get(k, 0.0)

    # SI p.17 "Solubility Equations" (psi = 0.005, SI C.4):
    #   M_Jbp,Hexane = psi_Hex-bp * M_Jbp,salt   (hexane lost to bottom phase)
    #   M_Jtp,salt   = psi_salt-tp * M_Jtp,Hexane (salt contaminating top phase)
    salt_bottom = bot.get("salt", 0.0) + m_salt_add
    hex_bottom = min(0.005 * salt_bottom, m_hex)
    top["extractant"] = m_hex - hex_bottom
    top["salt"] = top.get("salt", 0.0) + 0.005 * top["extractant"]

    # Size of unit [SI p.17]: Qc = V_feed + V_polymer/hexane + V_salt (m3/h)
    v_feed = vol_flow(stream, props)
    Qc = v_feed + m_hex / props["extractant"].rho + m_salt_add / 2160.0
    # Power [SI p.17]: PW = Wsp_ATPE * Qc, Wsp = 0.5 [SI C.1]
    pw = 0.5 * Qc
    # Cooling duty [SI p.17]: Mcw = 3600*PW / (cp*(Tcw_out - Tcw_in))
    mcw = pw * 3600.0 / (CP_W * DT_CW)
    # Consumables: hexane $2/kg, salt $0.6/kg [SI Table C.1, note a].
    # [DEV] charged as 5% make-up because the paper's flowsheet recycles
    # hexane (UF retentate) and salt (SDM underflow); SI GAMS charges the
    # full circulating amount but co-optimizes recycle streams.
    cons = ATPE_MAKEUP * (PRICE_HEXANE * m_hex + PRICE_SALT * m_salt_add)  # $/h
    res = UnitResult("ATPE", stage, Qc=Qc, PW=pw, Mcw=mcw, cons_per_h=cons,
                     extras={"stages": Np1 - 1.0, "EF": EF,
                             "hexane_kgph": m_hex, "salt_kgph": m_salt_add})
    return top, bot, res
