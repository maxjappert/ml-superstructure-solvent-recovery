"""Curated list of common solvents (name -> CAS) plus fallback property data.

Fallback data columns:
    CAS, MW [g/mol], rho_liq [kg/m3], Cp_liq [kJ/kg/K], Hvap [kJ/kg],
    Tb [degC], logP (octanol-water)

Properties are taken from standard literature values. When the `thermo` /
`chemicals` libraries are installed they take precedence (see properties.py);
this table is only the offline fallback and the registry of supported names.
"""

# name: (CAS, MW, rho, Cp, Hvap, Tb, logP)
SOLVENT_DATA = {
    "methanol":            ("67-56-1",   32.04,  792, 2.53, 1100.0,  64.7, -0.77),
    "ethanol":             ("64-17-5",   46.07,  789, 2.44,  841.0,  78.4, -0.31),
    "1-propanol":          ("71-23-8",   60.10,  803, 2.39,  686.0,  97.2,  0.25),
    "isopropanol":         ("67-63-0",   60.10,  786, 2.32,  664.0,  82.5,  0.05),
    "1-butanol":           ("71-36-3",   74.12,  810, 2.39,  582.0, 117.7,  0.88),
    "tert-butanol":        ("75-65-0",   74.12,  786, 3.04,  527.0,  82.4,  0.35),
    "acetone":             ("67-64-1",   58.08,  790, 2.16,  518.0,  56.1, -0.24),
    "2-butanone":          ("78-93-3",   72.11,  805, 2.20,  443.0,  79.6,  0.29),
    "mibk":                ("108-10-1", 100.16,  802, 2.09,  358.0, 116.5,  1.31),
    "ethyl acetate":       ("141-78-6",  88.11,  902, 1.94,  366.0,  77.1,  0.73),
    "isopropyl acetate":   ("108-21-4", 102.13,  872, 1.99,  331.0,  88.6,  1.02),
    "butyl acetate":       ("123-86-4", 116.16,  882, 1.94,  309.0, 126.1,  1.78),
    "thf":                 ("109-99-9",  72.11,  889, 1.72,  410.0,  66.0,  0.46),
    "2-methyltetrahydrofuran": ("96-47-9", 86.13, 854, 1.78, 375.0,  80.2,  1.10),
    # "1,4-dioxane":         ("123-91-1",  88.11, 1033, 1.74,  406.0, 101.1, -0.27),
    "dichloromethane":     ("75-09-2",   84.93, 1327, 1.19,  330.0,  39.6,  1.25),
    "chloroform":          ("67-66-3",  119.38, 1489, 0.96,  247.0,  61.2,  1.97),
    "toluene":             ("108-88-3",  92.14,  876, 1.71,  401.6, 110.6,  2.73),
    "benzene":             ("71-43-2",   78.11,  876, 1.74,  394.0,  80.1,  2.13),
    "p-xylene":            ("106-42-3", 106.17,  861, 1.72,  340.0, 138.4,  3.15),
    "n-hexane":            ("110-54-3",  86.18,  655, 2.26,  335.0,  68.7,  3.76),
    "n-heptane":           ("142-82-5", 100.20,  684, 2.24,  318.0,  98.4,  4.50),
    "cyclohexane":         ("110-82-7",  84.16,  779, 1.84,  358.0,  80.7,  3.44),
    "n-pentane":           ("109-66-0",  72.15,  626, 2.32,  358.0,  36.1,  3.39),
    "acetonitrile":        ("75-05-8",   41.05,  786, 2.23,  727.0,  81.6, -0.34),
    "dmf":                 ("68-12-2",   73.09,  944, 2.06,  578.0, 153.0, -1.01),
    "dmso":                ("67-68-5",   78.13, 1101, 1.96,  677.0, 189.0, -1.35),
    "dmac":                ("127-19-5",  87.12,  940, 2.06,  500.0, 165.1, -0.77),
    "nmp":                 ("872-50-4",  99.13, 1028, 1.79,  493.0, 202.0, -0.38),
    "pyridine":            ("110-86-1",  79.10,  982, 1.70,  449.0, 115.2,  0.65),
    "diethyl ether":       ("60-29-7",   74.12,  713, 2.34,  358.0,  34.6,  0.89),
    "mtbe":                ("1634-04-4", 88.15,  740, 2.16,  337.0,  55.2,  0.94),
    # "1,2-dimethoxyethane": ("110-71-4",  90.12,  867, 1.42,  418.6,  85.0, -0.21),
    "ethylene glycol":     ("107-21-1",  62.07, 1113, 2.36,  846.0, 197.3, -1.36),
    "glycerol":            ("56-81-5",   92.09, 1261, 2.43,  976.0, 290.0, -1.76),
    "acetic acid":         ("64-19-7",   60.05, 1049, 2.05,  390.0, 117.9, -0.17),
    "anisole":             ("100-66-3", 108.14,  995, 1.72,  360.0, 153.7,  2.11),
    "chlorobenzene":       ("108-90-7", 112.56, 1106, 1.34,  325.0, 131.7,  2.84),
    "nitromethane":        ("75-52-5",   61.04, 1137, 1.74,  560.0, 101.2, -0.33),
    "water":               ("7732-18-5", 18.02, 1000, 4.18, 2260.0, 100.0, -1.38),
}

# common aliases -> canonical name
ALIASES = {
    "ipa": "isopropanol",
    "2-propanol": "isopropanol",
    "propan-2-ol": "isopropanol",
    "isopropyl alcohol": "isopropanol",
    "mek": "2-butanone",
    "methyl ethyl ketone": "2-butanone",
    "butanone": "2-butanone",
    "methyl isobutyl ketone": "mibk",
    "4-methyl-2-pentanone": "mibk",
    "tetrahydrofuran": "thf",
    "2-methf": "2-methyltetrahydrofuran",
    "2-methyl-thf": "2-methyltetrahydrofuran",
    "dioxane": "1,4-dioxane",
    "dcm": "dichloromethane",
    "methylene chloride": "dichloromethane",
    "hexane": "n-hexane",
    "heptane": "n-heptane",
    "pentane": "n-pentane",
    "xylene": "p-xylene",
    "n,n-dimethylformamide": "dmf",
    "dimethylformamide": "dmf",
    "dimethyl sulfoxide": "dmso",
    "n,n-dimethylacetamide": "dmac",
    "dimethylacetamide": "dmac",
    "n-methyl-2-pyrrolidone": "nmp",
    "n-methylpyrrolidone": "nmp",
    "ether": "diethyl ether",
    "methyl tert-butyl ether": "mtbe",
    "tert-butyl methyl ether": "mtbe",
    "dme": "1,2-dimethoxyethane",
    "dimethoxyethane": "1,2-dimethoxyethane",
    "glyme": "1,2-dimethoxyethane",
    "monoethylene glycol": "ethylene glycol",
    "meg": "ethylene glycol",
    "glycerin": "glycerol",
    "etoh": "ethanol",
    "meoh": "methanol",
    "acn": "acetonitrile",
    "2-butanol": "1-butanol",   # closest listed butanol
    "t-butanol": "tert-butanol",
    "tba": "tert-butanol",
}

# salts: name -> (CAS, MW, rho_solid [kg/m3])
SALT_DATA = {
    "sodium chloride":    ("7647-14-5", 58.44, 2160),
    "sodium sulfate":     ("7757-82-6", 142.04, 2664),
    "calcium chloride":   ("10043-52-4", 110.98, 2150),
    "magnesium sulfate":  ("7487-88-9", 120.37, 2660),
    "potassium chloride": ("7447-40-7", 74.55, 1984),
    "sodium carbonate":   ("497-19-8", 105.99, 2540),
    "potassium carbonate":("584-08-7", 138.21, 2430),
    "sodium bicarbonate": ("144-55-8", 84.01, 2200),
    "ammonium sulfate":   ("7783-20-2", 132.14, 1769),
}

SALT_ALIASES = {
    "nacl": "sodium chloride",
    "salt": "sodium chloride",
    "table salt": "sodium chloride",
    "na2so4": "sodium sulfate",
    "cacl2": "calcium chloride",
    "mgso4": "magnesium sulfate",
    "epsom salt": "magnesium sulfate",
    "kcl": "potassium chloride",
    "na2co3": "sodium carbonate",
    "soda ash": "sodium carbonate",
    "k2co3": "potassium carbonate",
    "potash": "potassium carbonate",
    "nahco3": "sodium bicarbonate",
    "(nh4)2so4": "ammonium sulfate",
}


def list_solvents():
    """Names accepted by compute() for the two solvent slots."""
    return sorted(SOLVENT_DATA.keys())


def list_salts():
    """Names accepted by compute() for the salt slot."""
    return sorted(SALT_DATA.keys())
