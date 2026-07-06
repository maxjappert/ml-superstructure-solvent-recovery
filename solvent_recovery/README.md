# solvent_recovery — Python re-implementation of the Chea et al. (2020) superstructure framework

Re-implementation of *Evaluation of Solvent Recovery Options for Economic
Feasibility through a Superstructure-Based Optimization Framework* (Chea,
Lehr, Stengel, Savelski, Slater, Yenkie — Ind. Eng. Chem. Res. 2020, 59,
5931−5944) and its Supporting Information, reduced to a **deterministic
fixed-path evaluator**: instead of solving the MINLP, you pass the path
through the superstructure explicitly and get the cost/purity/recovery of
that path. This is exactly what you want for generating ML training data over
the (waste composition × pathway) state space.

## Usage

```python
from solvent_recovery import compute, list_solvents, list_salts

r = compute(
    solvent_target_name="isopropanol",   # or CAS, e.g. "67-63-0"
    solvent2_name="ethanol",
    salt_name="sodium chloride",
    temperature_C=25,
    solvent_target_kgph=510,
    solvent2_kgph=0,
    water_kgph=490,
    salt_kgph=0,
    solids_kgph=0,
    idx_solids_removal=0,   # bypass
    idx_recovery=2,         # pervaporation
    idx_purification=3,     # ultrafiltration
    idx_refinement=0,       # bypass
)
cost, purity, recovery = r[:3]
# r also carries: cost_usd_per_kg_recovered, feasible, path, cost_breakdown
```

Run `python test_compute.py` for a paper-vs-model check, a full path sweep,
and a random-sampling demo.

## Stage index maps (0 = bypass)

| idx | solids removal | recovery | purification | refinement |
|-----|----------------|----------|--------------|------------|
| 0   | bypass         | bypass   | bypass       | bypass     |
| 1   | sedimentation  | distillation | distillation | distillation |
| 2   | centrifugation | pervaporation | pervaporation | pervaporation |
| 3   | filtration     | ATPE (hexane+salt) | ultrafiltration | ultrafiltration |
| 4   | —              | —        | —            | microfiltration |

## Outputs

* `cost_usd_per_year` — total annualized cost (capital + labor + utilities +
  membranes + consumables + overhead), SI section B costing:
  `Cc = C0·(Qc/Q0)^0.67`, `CC_cap = 1.66·CRF·BMC·ΣCc` (CRF = 0.11, BMC = 5.4),
  labor $30/h, 7920 h/yr, overhead = 2.78 × labor, membrane replacement every
  2000 h, utility prices from SI Table C.2.
* `target_purity` — mass fraction of the target solvent in the final product
  stream (added ATPE hexane counts as impurity until removed downstream).
* `target_recovery` — target mass in product / target mass in the feed.
* Infeasible path (e.g. distillation at relative volatility < 1.05, ATPE with
  extraction factor ≤ 1, target fully lost) → `(nan, nan, 0.0)`,
  `feasible=False`, and a reason in `cost_breakdown["note"]`. No purity or
  recovery constraints are *enforced* — achieved values are reported so the
  ML model can learn the trade-offs.

## Chemical properties

`properties.py` resolves each component to {MW, ρ, Cp, ΔHvap, Tb, Psat(T),
logP, molar volume}:

1. **Preferred:** the `thermo`/`chemicals` libraries by CAS
   (`pip install thermo chemicals`). Any name resolvable by
   `chemicals.identifiers.CAS_from_any` then works, not only the curated list.
2. **Fallback:** a bundled literature table for ~40 common solvents and 9
   salts (`solvents.py`), with Psat from Clausius–Clapeyron anchored at
   (Tb, 1 atm). The framework therefore runs (with slightly coarser
   properties) even without those libraries.

`list_solvents()` / `list_salts()` return the accepted names; aliases like
"IPA", "DCM", "MEK", "hexane" are understood.

## Unit models (SI equations, deterministic defaults)

* **Distillation** — Fenske (Nmin), Underwood (Rmin, q=1), R = 1.3·Rmin,
  N = Nmin/0.6, stage efficiency 0.8, HETP 1.5 ft, vapor velocity 10800 m/h,
  distillate spec 92:8 mol LK:HK, LK recovery 99%. α from pure-component
  vapor pressures at the feed bubble point. The split is made at the
  volatility gap adjacent to the target with the larger α-ratio; salt/solids
  go to the bottoms. Infeasible if α < 1.05 (paper's rule). Steam/cooling
  duties per SI; feed preheat from `temperature_C`.
* **Pervaporation** — flux 0.055 m³/m²h (case-study-1 GAMS value; this is
  what reproduces the published Table 3 cost), retention from a sigmoid in
  log10 Psat around the midpoint of the stream's volatility range
  (reproduces the paper's 0.05/0.90–0.97 pattern), water floor-retained at
  0.90 (hydrophobic membrane, both case studies). Steam duty = permeate
  vaporization.
* **Ultrafiltration** — flux 0.0856 m³/m²h; target passes (0.001), water
  0.998 and hexane 0.999 retained (paper values); the co-solvent is retained
  by a sigmoid in molecular size (molar volume) relative to the target.
* **ATPE** — hexane + salt system; partition coefficients from logP (water
  fixed at Kp = 0.05, paper); stage count set by the paper's 90% extraction
  requirement via the Kremser equation, other components follow at that
  stage count; 0.5% cross-phase solubility; hexane = 1:1 with feed, salt =
  0.2 × aqueous mass; **consumables charged as 5% make-up** because the
  paper's flowsheet recycles hexane and salt.
* **Sedimentation** — Stokes settling (Dp = 40 µm), efficiency 0.7 (70% of
  solids removed, 30% of liquid lost to sludge), area = Q/(SOR).
* **Centrifugation** — complete solids removal, sigma-factor sizing
  (Σ = Q/2u_g), 19.2 kW per m³/h, 40% power dissipated to cooling water.
* **Filtration / microfiltration** — cake retention of solids, 10% / 5%
  liquid holdup loss, area from flux balance.
* **Incineration** — `incineration_cost()` gives the disposal baseline
  (simplified: generic 30 MJ/kg heating value instead of the Dulong
  elemental formula).

## Generalized separation parameters (deviation from the paper)

The paper hard-codes case-specific retention/partition coefficients. To make
`compute()` meaningful over arbitrary solvent pairs, these are replaced by
smooth property-based estimates (volatility-sigmoid for PVP, size-sigmoid for
UF, logP for ATPE), calibrated so the two published case studies are
reproduced to well within the paper's own ±30% uncertainty band:

| Quantity (case study 1, 1000 kg/h, 51% IPA) | Paper | This code |
|---|---|---|
| PVP→UF annualized cost | $0.524M/yr | $0.54M/yr |
| DST→PVP annualized cost | $0.862M/yr | $0.52M/yr |
| Incineration | $8.1M/yr | $4.9M/yr |

## Known limitations

* No azeotrope detection: α is computed from pure-component Psat ratios, so
  azeotropic pairs look easier to distill than they are (the paper handles
  this by manual superstructure construction).
* Single generic solids pseudo-component (40 µm, 2160 kg/m³); dissolved salt
  passes the solids-removal stage and is removed by DST bottoms / membranes /
  ATPE instead.
* The case-study-2 anhydrous-salt drying trick (salt as desiccant binding
  water) is not modeled; salt in this framework is an inert dissolved
  contaminant.
* Recycle streams are not simulated; ATPE recycling is approximated by the
  5% make-up charge.
* Paper's stated result uncertainty is ±30%; treat absolute costs
  accordingly. Relative comparisons across paths/compositions (what an ML
  model learns) are the meaningful signal.
