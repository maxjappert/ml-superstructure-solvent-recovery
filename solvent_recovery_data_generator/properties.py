"""Chemical property resolution.

Primary source: the `thermo` / `chemicals` libraries (looked up by CAS).
Fallback: the bundled literature table in solvents.py (so the framework also
runs without those libraries installed).

Every component is reduced to the property set the Chea et al. (2020)
framework actually needs:
    MW [g/mol], rho [kg/m3], Cp [kJ/kg/K], Hvap [kJ/kg], Tb [K],
    Psat(T) [Pa], logP [-], Vm [m3/kmol] (molar volume, for size heuristics)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional

from .solvents import (SOLVENT_DATA, ALIASES, SALT_DATA, SALT_ALIASES)

R_GAS = 8.314  # J/mol/K

try:  # optional dependency
    from thermo import Chemical  # type: ignore
    HAVE_THERMO = True
except Exception:  # pragma: no cover
    Chemical = None
    HAVE_THERMO = False

try:  # optional dependency
    from chemicals.identifiers import CAS_from_any  # type: ignore
    HAVE_CHEMICALS = True
except Exception:  # pragma: no cover
    CAS_from_any = None
    HAVE_CHEMICALS = False


@dataclass
class ComponentProps:
    name: str
    CAS: str
    MW: float                 # g/mol
    rho: float                # kg/m3 (liquid, or solid for salt/solids)
    Cp: float                 # kJ/kg/K
    Hvap: float               # kJ/kg
    Tb: float                 # K (nan for non-volatiles)
    logP: float               # octanol-water partition coefficient (log10)
    volatile: bool = True
    _psat: Optional[Callable[[float], float]] = field(default=None, repr=False)

    @property
    def Vm(self) -> float:
        """Molar volume in m3/kmol."""
        return self.MW / self.rho  # (g/mol)/(kg/m3) == m3/kmol

    def Psat(self, T: float) -> float:
        """Saturation pressure [Pa] at T [K]."""
        if not self.volatile:
            return 0.0
        if self._psat is not None:
            try:
                p = self._psat(T)
                if p and p > 0:
                    return p
            except Exception:
                pass
        # Clausius-Clapeyron estimate anchored at (Tb, 1 atm)
        dHm = self.Hvap * self.MW  # J/mol (kJ/kg * g/mol = J/mol)
        return 101325.0 * math.exp(-(dHm / R_GAS) * (1.0 / T - 1.0 / self.Tb))


def _canonical(name: str, table: dict, aliases: dict) -> Optional[str]:
    key = name.strip().lower()
    if key in table:
        return key
    if key in aliases:
        return aliases[key]
    for k, row in table.items():  # allow CAS numbers directly
        if row[0] == key:
            return k
    return None


def _from_thermo(cas: str, name: str, fallback: Optional[tuple]) -> ComponentProps:
    """Build props from thermo, patching holes with the fallback row."""
    c = Chemical(cas, T=298.15)
    fb = fallback  # (CAS, MW, rho, Cp, Hvap, Tb, logP) or None

    def pick(val, idx):
        if val is not None and (not isinstance(val, float) or math.isfinite(val)):
            return val
        return fb[idx] if fb is not None else None

    MW = pick(c.MW, 1)
    rho = pick(c.rhol, 2)
    Cpl = c.Cpl  # J/mol/K
    Cp = Cpl / MW if (Cpl and MW) else (fb[3] if fb else None)
    Tb = c.Tb
    Hvap = None
    try:
        hv = c.EnthalpyVaporization(Tb) if Tb else c.Hvap
        if hv:
            Hvap = hv / MW  # J/mol -> kJ/kg
    except Exception:
        pass
    if Hvap is None and fb is not None:
        Hvap = fb[4]
    if Tb is None and fb is not None:
        Tb = fb[5] + 273.15
    logP = getattr(c, "logP", None)
    if logP is None and fb is not None:
        logP = fb[6]
    if logP is None:
        logP = 0.0

    psat = None
    try:
        vp = c.VaporPressure
        psat = lambda T: vp(T)  # noqa: E731
    except Exception:
        pass

    missing = [v for v in (MW, rho, Cp, Hvap, Tb) if v is None]
    if missing:
        raise ValueError(f"Could not resolve all properties for '{name}' ({cas})")
    return ComponentProps(name=name, CAS=cas, MW=MW, rho=rho, Cp=Cp,
                          Hvap=Hvap, Tb=Tb, logP=logP, _psat=psat)


def _from_table(name_key: str) -> ComponentProps:
    cas, MW, rho, Cp, Hvap, Tb, logP = SOLVENT_DATA[name_key]
    return ComponentProps(name=name_key, CAS=cas, MW=MW, rho=rho, Cp=Cp,
                          Hvap=Hvap, Tb=Tb + 273.15, logP=logP)


@lru_cache(maxsize=256)
def get_solvent_props(name: str) -> ComponentProps:
    """Resolve a solvent by name (or CAS) to its property set."""
    key = _canonical(name, SOLVENT_DATA, ALIASES)
    if key is not None:
        fb = SOLVENT_DATA[key]
        if HAVE_THERMO:
            try:
                return _from_thermo(fb[0], key, fb)
            except Exception:
                pass
        return _from_table(key)
    # not in curated list: try to resolve via chemicals/thermo directly
    if HAVE_CHEMICALS and HAVE_THERMO:
        try:
            cas = CAS_from_any(name)
            return _from_thermo(cas, name.strip().lower(), None)
        except Exception as exc:
            raise ValueError(
                f"Unknown solvent '{name}' (also failed thermo lookup: {exc}). "
                f"Use one of solvent_recovery.list_solvents() or a CAS number."
            ) from None
    raise ValueError(
        f"Unknown solvent '{name}'. Use one of solvent_recovery.list_solvents(), "
        f"or install the 'thermo' and 'chemicals' packages to resolve arbitrary names."
    )


@lru_cache(maxsize=64)
def get_salt_props(name: str) -> ComponentProps:
    """Resolve a (non-volatile, dissolved) salt by name."""
    key = _canonical(name, SALT_DATA, SALT_ALIASES)
    if key is None:
        if HAVE_CHEMICALS:
            try:
                cas = CAS_from_any(name)
                for k, (kcas, mw, rho) in SALT_DATA.items():
                    if kcas == cas:
                        key = k
                        break
            except Exception:
                pass
    if key is None:
        raise ValueError(
            f"Unknown salt '{name}'. Use one of solvent_recovery.list_salts()."
        )
    cas, MW, rho = SALT_DATA[key]
    return ComponentProps(name=key, CAS=cas, MW=MW, rho=rho, Cp=0.9,
                          Hvap=0.0, Tb=float("nan"), logP=-4.0, volatile=False)


def get_solids_props() -> ComponentProps:
    """Generic suspended sediment (paper: Dp = 40 um, rho ~ salt-like solid)."""
    return ComponentProps(name="solids", CAS="", MW=200.0, rho=2160.0, Cp=0.9,
                          Hvap=0.0, Tb=float("nan"), logP=-4.0, volatile=False)


def get_water_props() -> ComponentProps:
    return get_solvent_props("water")


def get_extractant_props() -> ComponentProps:
    """ATPE extraction solvent -- n-hexane, as in the paper."""
    return get_solvent_props("n-hexane")
