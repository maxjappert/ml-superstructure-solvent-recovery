"""Test / demo driver for the solvent_recovery framework.

Run:  python test_generate_data.py

Sections:
  1. Case-study-1-like scenario (IPA + water, PVP->UF path) vs. paper
  2. All-path sweep for one waste stream (what the MINLP would enumerate)
  3. Random state-space sampling, as intended for ML training data
"""
import itertools
import random

from solvent_recovery import (compute, incineration_cost, list_solvents,
                              list_salts, SOLIDS_REMOVAL_OPTIONS,
                              RECOVERY_OPTIONS, PURIFICATION_OPTIONS,
                              REFINEMENT_OPTIONS)


def show(label, r):
    if r.feasible:
        print(f"  {label:34s} path={'-'.join(r.path):22s} "
              f"cost=${r.cost_usd_per_year/1e6:8.3f}M/yr  "
              f"purity={r.target_purity:6.1%}  recovery={r.target_recovery:6.1%}  "
              f"(${r.cost_usd_per_kg_recovered:.3f}/kg)")
    else:
        print(f"  {label:34s} path={'-'.join(r.path):22s} INFEASIBLE "
              f"({r.cost_breakdown.get('note', '')})")


# ------------------------------------------------------------------ 1. paper-like check
print("=" * 100)
print("1) Case-study-1-like: 1000 kg/h waste, 51% IPA / 49% water (Chea et al. Table 3)")
print("   Paper: PVP-UF $0.524M/yr, DST-PVP $0.862M/yr, incineration $8.1M/yr")
print("=" * 100)

common = dict(solvent_target_name="isopropanol", solvent2_name="ethanol",
              salt_name="sodium chloride", temperature_C=25,
              solvent_target_kgph=510, solvent2_kgph=0, water_kgph=490,
              salt_kgph=0, solids_kgph=0)

show("PVP -> UF", compute(**common, idx_solids_removal=0, idx_recovery=2,
                          idx_purification=3, idx_refinement=0))
show("DST -> PVP", compute(**common, idx_solids_removal=0, idx_recovery=1,
                           idx_purification=2, idx_refinement=0))
show("ATPE -> UF", compute(**common, idx_solids_removal=0, idx_recovery=3,
                           idx_purification=3, idx_refinement=0))
print(f"  incineration baseline: ${incineration_cost(1000, 0.51)/1e6:.2f}M/yr")

# ------------------------------------------------------------------ 2. full path sweep
print()
print("=" * 100)
print("2) All-path sweep: acetone (target) + toluene + water + NaCl + sediment")
print("=" * 100)
stream = dict(solvent_target_name="acetone", solvent2_name="toluene",
              salt_name="sodium chloride", temperature_C=30,
              solvent_target_kgph=400, solvent2_kgph=150, water_kgph=380,
              salt_kgph=40, solids_kgph=30)

results = []
for i0, i1, i2, i3 in itertools.product(SOLIDS_REMOVAL_OPTIONS,
                                        RECOVERY_OPTIONS,
                                        PURIFICATION_OPTIONS,
                                        REFINEMENT_OPTIONS):
    r = compute(**stream, idx_solids_removal=i0, idx_recovery=i1,
                idx_purification=i2, idx_refinement=i3)
    results.append(((i0, i1, i2, i3), r))

feas = [(idx, r) for idx, r in results if r.feasible]
print(f"  {len(results)} paths evaluated, {len(feas)} feasible")
good = [x for x in feas if x[1].target_purity > 0.95 and x[1].target_recovery > 0.80]
good.sort(key=lambda x: x[1].cost_usd_per_year)
print(f"  {len(good)} paths reach >95% purity and >80% recovery; 5 cheapest:")
for idx, r in good[:5]:
    show(f"  idx={idx}", r)

# ------------------------------------------------------------------ 3. random sampling
print()
print("=" * 100)
print("3) Random state-space sampling (ML training data generation)")
print("=" * 100)
rng = random.Random(42)
solvents = list_solvents()
salts = list_salts()
n_ok = n_bad = 0
rows = []
for _ in range(300):
    tgt = rng.choice(solvents)
    s2 = rng.choice([s for s in solvents if s != tgt])
    if tgt == "water":
        continue
    r = compute(
        solvent_target_name=tgt, solvent2_name=s2,
        salt_name=rng.choice(salts),
        temperature_C=rng.randint(15, 60),
        solvent_target_kgph=rng.randint(50, 2000),
        solvent2_kgph=rng.randint(0, 1000),
        water_kgph=rng.randint(0, 1500),
        salt_kgph=rng.randint(0, 200),
        solids_kgph=rng.randint(0, 100),
        idx_solids_removal=rng.randint(0, 3),
        idx_recovery=rng.randint(0, 3),
        idx_purification=rng.randint(0, 3),
        idx_refinement=rng.randint(0, 4),
    )
    if r.feasible:
        n_ok += 1
        rows.append(r)
    else:
        n_bad += 1

print(f"  300 random samples -> {n_ok} feasible, {n_bad} infeasible")
costs = sorted(x.cost_usd_per_year for x in rows)
purs = sorted(x.target_purity for x in rows)
recs = sorted(x.target_recovery for x in rows)
med = lambda v: v[len(v) // 2]
print(f"  cost   $/yr : min={costs[0]:,.0f}  median={med(costs):,.0f}  max={costs[-1]:,.0f}")
print(f"  purity      : min={purs[0]:.3f}  median={med(purs):.3f}  max={purs[-1]:.3f}")
print(f"  recovery    : min={recs[0]:.3f}  median={med(recs):.3f}  max={recs[-1]:.3f}")

print()
print("Example single call:")
r = compute("ethanol", "ethyl acetate", "sodium chloride", 25,
            600, 200, 300, 20, 10, 2, 1, 2, 3)
print(f"  compute('ethanol','ethyl acetate','sodium chloride',25, 600,200,300,20,10, 2,1,2,3)")
show("CNF -> DST -> PVP -> UF", r)
