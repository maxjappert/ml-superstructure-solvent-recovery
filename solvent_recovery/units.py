"""Unit-operation models (SI of Chea et al. 2020, sections B/C/F).

A stream is a dict {component_key: mass flow kg/h}. Component keys used by the
framework: 'target', 'solvent2', 'water', 'salt', 'solids', 'extractant'.

Each unit returns (product_stream, waste_stream, UnitResult). The product
stream is the one that carries the target solvent onward.

Where the paper hard-codes case-specific separation parameters (membrane
retention factors, ATPE partition coefficients), this module estimates them
from pure-component properties so that outputs vary smoothly over arbitrary
solvent pairs (see README, "Generalized separation parameters").
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
    """retention = fraction of each component staying in the RETENTATE."""
    perm, ret = _empty_like(stream), _empty_like(stream)
    for k, m in stream.items():
        r = retention.get(k, 1.0)
        ret[k] = m * r
        perm[k] = m * (1.0 - r)
    return perm, ret


def _membrane_area(stream: Stream, props: Props, permeate: Stream,
                   zeta: float) -> float:
    """Flux balance (SI): zeta * A = V_feed * (1 - 1/CF) = V_permeate."""
    vperm = vol_flow(permeate, props)
    return max(vperm / zeta, 1e-6)


def _settling_velocity(props: Props, stream: Stream) -> float:
    """Stokes settling velocity of the solids [m/s] (SI, sedimentation)."""
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
    """Gravity sedimentation: 70% of solids to sludge; 70% of each liquid
    (and dissolved salt) reports to the clarified stream (SI, eff=0.7)."""
    eff = 0.70
    clar, sludge = _empty_like(stream), _empty_like(stream)
    for k, m in stream.items():
        if k == "solids":
            sludge[k] = eff * m
            clar[k] = (1 - eff) * m
        else:
            clar[k] = eff * m
            sludge[k] = (1 - eff) * m
    us = _settling_velocity(props, stream)
    sor = eff * us                                  # m/s
    area = vol_flow(stream, props) / (sor * 3600.0)
    res = UnitResult("SDM", stage, Qc=area)
    return clar, sludge, res


def centrifugation(stream: Stream, props: Props, stage: str):
    """Centrifuge: complete solids removal (SI case 2, eff=1). Sigma factor
    Qc = V_dot / U with U = 2 * settling velocity."""
    liq, solids_out = _empty_like(stream), _empty_like(stream)
    for k, m in stream.items():
        if k == "solids":
            solids_out[k] = m
        else:
            liq[k] = m
    v = vol_flow(stream, props)                     # m3/h
    us = _settling_velocity(props, stream)          # m/s
    U = 2.0 * us * 3600.0                           # m/h
    sigma = v / max(U, 1e-9)                        # m2
    pw = 19.2 * v                                   # kW (Wsp * volumetric feed)
    mcw = 0.4 * pw * 3600.0 / (CP_W * DT_CW) / 1000.0  # 40% dissipation, kg/h
    res = UnitResult("CNF", stage, Qc=sigma, PW=pw, Mcw=mcw)
    return liq, solids_out, res


def filtration(stream: Stream, props: Props, stage: str):
    """Dead-end filtration: solids fully retained in cake; 10% of every
    liquid (and dissolved salt) is lost to the cake (SI C.9)."""
    retention = {k: 0.10 for k in stream}
    retention["solids"] = 1.0
    filtrate, cake = _membrane_split(stream, retention)
    area = _membrane_area(stream, props, filtrate, ZETA["FLT"])
    pw = 0.1 * area
    res = UnitResult("FLT", stage, Qc=area, PW=pw)
    return filtrate, cake, res


# --------------------------------------------------------------------------- distillation
def _alphas(stream: Stream, props: Props, T_ref: float) -> Dict[str, float]:
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

    # molar feed and split
    F = {k: stream[k] / props[k].MW for k in vols}      # kmol/h
    rec_top = {}
    for idx, k in enumerate(order):
        if idx < j:
            rec_top[k] = 0.995
        elif k == LK:
            rec_top[k] = 0.99
        elif k == HK:
            rec_top[k] = 0.0    # set below
        else:
            rec_top[k] = 0.001
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

    # Fenske
    rl, rh = rec_top[LK], rec_top[HK]
    Nmin = math.log((rl / (1 - rl)) * ((1 - rh) / rh)) / math.log(aratio)
    Nmin = max(Nmin, 1.0)

    # Underwood (q = 1) with volatilities relative to HK
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
    Rmin = sum(a[k] * xD[k] / (a[k] - theta) for k in vols) - 1.0
    Rmin = max(Rmin, 0.1)
    R = max(1.3 * Rmin, 1.01)
    Nact = (Nmin / 0.6) / STAGE_EFF
    return _dst_finish(stream, top, bot, props, stage, T_feed, T_bub, R,
                       Nact=Nact, target_in_top=target_in_top)


def _dst_finish(feed, top, bot, props, stage, T_feed, T_bub, R, Nact,
                target_in_top):
    vols = _liquid_keys(feed, props)
    D_mass = total_mass(top)
    Liq = R * D_mass
    Vap = Liq + D_mass                                   # kg/h
    # vapor density (ideal gas at bubble point, 1 atm)
    Fmol_top = sum(top[k] / props[k].MW for k in vols if top[k] > TINY)
    MW_top = D_mass / max(Fmol_top, TINY)                # kg/kmol
    rho_vap = 101325.0 * MW_top / (8314.0 * T_bub)       # kg/m3
    v_vap = Vap / max(rho_vap, 1e-6)                     # m3/h
    Dcol = max(math.sqrt(4.0 * v_vap / (math.pi * U_VAP)), 0.6)
    H = HETP * Nact
    Vol = math.pi / 4.0 * Dcol ** 2 * H                  # m3 (costing capacity)

    QS = sum(feed[k] * props[k].Cp * max(T_bub - (T_feed + 273.15), 0.0)
             for k in vols)                              # kJ/h
    QH = (1 + R) * sum(top[k] * props[k].Hvap for k in vols)
    QC = R * sum(top[k] * props[k].Hvap for k in vols)
    Mstm = (QS + QH) / HSTM_DST
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

    perm, ret = _membrane_split(stream, retention)
    area = _membrane_area(stream, props, perm, ZETA["PVP"])
    pw = 0.33 * area
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
    perm, ret = _membrane_split(stream, retention)
    area = _membrane_area(stream, props, perm, ZETA["UF"])
    pw = 0.2 * area
    res = UnitResult("UF", stage, Qc=area, PW=pw,
                     extras={"retention": retention})
    return perm, ret, res


def microfiltration(stream: Stream, props: Props, stage: str):
    """Polishing MF: retains particulates and (as a cake/gel layer) salt;
    liquids pass with 5% holdup loss."""
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

    m_feed_liq = sum(stream[k] for k in liq)
    m_hex = m_feed_liq                              # 1:1 solvent-to-feed
    m_salt_add = max(0.2 * (stream.get("water", 0.0) + TINY), 0.05 * m_feed_liq)

    def kp(k):
        if k == "water":
            return 0.05
        return min(max(10.0 ** props[k].logP, 1e-3), 45.0)

    EF = {k: kp(k) * m_hex / m_salt_add for k in liq}
    ef_t = EF["target"]
    if ef_t <= 1.02:
        res = UnitResult("ATPE", stage, feasible=False,
                         note=f"extraction factor {ef_t:.2f} <= 1 for target")
        return stream, _empty_like(stream), res

    # stages for 90% target extraction: phi = (EF-1)/(EF^{N+1}-1) = 0.1
    Np1 = math.log(1.0 + 10.0 * (ef_t - 1.0)) / math.log(ef_t)
    Np1 = max(Np1, 2.0)

    def extracted(ef):
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

    # cross-phase solubility (SI: 0.005)
    top["extractant"] = m_hex * (1.0 - 0.005)
    top["salt"] = top.get("salt", 0.0) + 0.005 * m_hex

    v_feed = vol_flow(stream, props)
    Qc = v_feed + m_hex / props["extractant"].rho + m_salt_add / 2160.0
    pw = 0.5 * Qc
    mcw = pw * 3600.0 / (CP_W * DT_CW) / 1000.0
    # hexane/salt are recycled in the paper's superstructure -> make-up only
    cons = ATPE_MAKEUP * (PRICE_HEXANE * m_hex + PRICE_SALT * m_salt_add)  # $/h
    res = UnitResult("ATPE", stage, Qc=Qc, PW=pw, Mcw=mcw, cons_per_h=cons,
                     extras={"stages": Np1 - 1.0, "EF": EF,
                             "hexane_kgph": m_hex, "salt_kgph": m_salt_add})
    return top, bot, res
