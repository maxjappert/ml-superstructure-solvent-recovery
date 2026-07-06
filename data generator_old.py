import torch
from thermo import Chemical

from models import Model
from datasets import Dataset


def generate_input(solvent_target_name: str,
                   solvent2_name: str,
                   salt_name,
                   temperature_C: int,
                   solvent_target_kgph: int,
                   solvent2_kgph: int,
                   water_kgph: int,
                   salt_kgph: int,
                   solids_kgph: int,
                   idx_solids_removal: int,
                   idx_recovery: int,
                   idx_purification: int,
                   idx_refinement: int):

    T_K = temperature_C + 273.15

    total_flow_kgph = solvent_target_kgph + solvent2_kgph + water_kgph + salt_kgph + solids_kgph

    n_components = 1
    if solvent2_kgph > 0:
        n_components += 1
    if salt_kgph > 0:
        n_components += 1
    if water_kgph > 0:
        n_components += 1
    if solids_kgph > 0:
        n_components += 1

    solids_frac = solids_kgph / total_flow_kgph
    salt_frac = salt_kgph / total_flow_kgph
    water_frac = water_kgph / total_flow_kgph
    target_frac = solvent_target_kgph / total_flow_kgph
    target_mw =  Chemical(solvent_target_name, T=T_K).MW
    target_density = Chemical(solvent_target_name, T=T_K).rhol
    target_tb_C = Chemical(solvent_target_name, T=T_K).Tb - 273.15
    target_hvap = Chemical(solvent_target_name, T=T_K).Hvap / Chemical(solvent_target_name, T=T_K).MW # converted to kJ/kg
    target_cp = Chemical(solvent_target_name, T=T_K).Cpl / 1000 # converted to kH/(kg*K)
    target_logp = Chemical(solvent_target_name, T=T_K).logP
    if Chemical(solvent_target_name, T=T_K).logP is None: print('error') # todo, coverage not ideal
    target_water_miscible = int(target_logp < 0.5) # todo heuristic
    solvent_tb_min = min(Chemical(solvent_target_name, T=T_K).Tb, Chemical(solvent2_name, T=T_K).Tb) - 273.15
    solvent_tb_max = max(Chemical(solvent_target_name, T=T_K).Tb, Chemical(solvent2_name, T=T_K).Tb) - 273.15

    boiling_points = [Chemical(solvent_target_name, T=T_K).Tb, Chemical(solvent2_name, T=T_K).Tb, Chemical('water', T=T_K).Tb]

    def tb_celsius(name, T_K=298.15):
        """Normal boiling point [degC] from just the compound name.
        thermo resolves the name to a CAS number and returns Tb in Kelvin."""
        tb_K = Chemical(name, T=T_K).Tb  # normal boiling point [K]
        if tb_K is None:
            raise ValueError(f"no boiling point found for {name!r}")
        return tb_K - 273.15

    def relative_volatility_pair(tb1_C, tb2_C):
        """Edgeworth-Johnstone correlation (Chea SI p.27). alpha >= 1."""
        t1, t2 = sorted((tb1_C, tb2_C))  # t1 = lighter (lower boiler)
        Tmean = (t1 + t2) / 2 + 273.15  # mean boiling point [K]
        log_alpha = (t2 - t1) / Tmean * (3.99 + 0.001939 * Tmean)
        return 10.0 ** log_alpha

    def alpha_min_max(solvent_names, has_water):
        """alpha_min / alpha_max over all VOLATILE species (solvents + water)."""
        volatile = list(solvent_names) + (["water"] if has_water else [])
        tbs = [tb_celsius(n) for n in volatile]
        alphas = [relative_volatility_pair(tbs[i], tbs[j])
                  for i in range(len(tbs)) for j in range(i + 1, len(tbs))]
        if not alphas:  # 0 or 1 volatile species
            return 1.0, 1.0
        return min(alphas), max(alphas)

    alpha_min, alpha_max = alpha_min_max([solvent_target_name, solvent2_name], water_frac > 0)

    logp_min = min(target_logp, Chemical(solvent2_name, T=T_K).logP)
    logp_max = max(target_logp, Chemical(solvent2_name, T=T_K).logP)

    target_volumetric_flow_m3ph = (total_flow_kgph*target_frac) / target_density
    solution2_frac = 1 - target_frac - water_frac - salt_frac - solids_frac
    assert 0 <= solution2_frac < 1
    solution2_volumetric_flow_m3ph = (total_flow_kgph*solution2_frac) / Chemical(solvent2_name, T=T_K).rhol
    salt_volumetric_flow_m3ph = (total_flow_kgph*salt_frac) / 2160 # todo, this is just the density of NaCl
    water_volumetric_flow_m3ph = (total_flow_kgph*water_frac) / Chemical('water', T=T_K).rhol

    solid_density_assumption_kgpm3 = 1500 # todo
    solids_volumetric_flow_m3ph = (total_flow_kgph*solids_frac) / solid_density_assumption_kgpm3
    volumetric_flow_m3ph = (target_volumetric_flow_m3ph +
                            solution2_volumetric_flow_m3ph +
                            salt_volumetric_flow_m3ph +
                            water_volumetric_flow_m3ph +
                            solids_volumetric_flow_m3ph)

    return torch.Tensor([total_flow_kgph,
                           volumetric_flow_m3ph,
                           temperature_C,
                           n_components,
                           solids_frac,
                           salt_frac,
                           water_frac,
                           target_frac,
                           target_mw,
                           target_density,
                           target_tb_C,
                           target_hvap,
                           target_cp,
                           target_logp,
                           target_water_miscible,
                           solvent_tb_min,
                           solvent_tb_max,
                           alpha_min,
                           alpha_max,
                           logp_min,
                           logp_max,
                           idx_solids_removal,
                           idx_recovery,
                           idx_purification,
                           idx_refinement])

tensor_input = generate_input('benzene',
                              'isopropanol',
                              'NaCl',
                              20,
                              1000,
                              436,
                              320,
                              0,
                              0,
                              0, 2, 1, 0)

model = Model()
model.load_state_dict(torch.load("best_20260703_164646.pt")['model_state_dict'])
model.eval()
dataset = Dataset('val')
print()
print(dataset.standardiser_y.inverse_transform(model(dataset.standardiser_X.transform(tensor_input))))

print(dataset.standardiser_y.inverse_transform(model(dataset[18][0])))