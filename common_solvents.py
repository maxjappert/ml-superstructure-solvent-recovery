"""
common_solvents.py
==================

A curated list of the most common process / laboratory solvents, chosen to line
up with the ICH Q3C residual-solvent classes and the CHEM21 solvent-selection
guide.  All names resolve in the `thermo` / `chemicals` libraries; CAS numbers
are provided as a fallback for when name lookup is ambiguous or fails.

Usage
-----
    from common_solvents import COMMON_SOLVENTS, SOLVENT_CAS, SOLVENT_FAMILIES

    for name in COMMON_SOLVENTS:
        c = Chemical(name)          # or Chemical(SOLVENT_CAS[name]) as fallback
        ...
"""

# --- flat list (import this) -----------------------------------------------
COMMON_SOLVENTS = [
    # alcohols
    "methanol",
    "ethanol",
    "isopropanol",
    "n-butanol",
    # ketones
    "acetone",
    "methyl ethyl ketone",
    "methyl isobutyl ketone",
    # esters
    "ethyl acetate",
    "isopropyl acetate",
    # ethers
    "diethyl ether",
    "tetrahydrofuran",
    "2-methyltetrahydrofuran",
    "methyl tert-butyl ether",
    "1,4-dioxane",
    # aromatics
    "toluene",
    "xylene",
    "benzene",
    # alkanes / cyclo
    "n-hexane",
    "n-heptane",
    "cyclohexane",
    # chlorinated
    "dichloromethane",
    "chloroform",
    # dipolar aprotic
    "acetonitrile",
    "n,n-dimethylformamide",
    "dimethyl sulfoxide",
    "n-methyl-2-pyrrolidone",
]

# --- CAS fallback (use if a name doesn't resolve) --------------------------
SOLVENT_CAS = {
    "methanol": "67-56-1",
    "ethanol": "64-17-5",
    "isopropanol": "67-63-0",
    "n-butanol": "71-36-3",
    "acetone": "67-64-1",
    "methyl ethyl ketone": "78-93-3",
    "methyl isobutyl ketone": "108-10-1",
    "ethyl acetate": "141-78-6",
    "isopropyl acetate": "108-21-4",
    "diethyl ether": "60-29-7",
    "tetrahydrofuran": "109-99-9",
    "2-methyltetrahydrofuran": "96-47-9",
    "methyl tert-butyl ether": "1634-04-4",
    "1,4-dioxane": "123-91-1",
    "toluene": "108-88-3",
    "xylene": "1330-20-7",
    "benzene": "71-43-2",
    "n-hexane": "110-54-3",
    "n-heptane": "142-82-5",
    "cyclohexane": "110-82-7",
    "dichloromethane": "75-09-2",
    "chloroform": "67-66-3",
    "acetonitrile": "75-05-8",
    "n,n-dimethylformamide": "68-12-2",
    "dimethyl sulfoxide": "67-68-5",
    "n-methyl-2-pyrrolidone": "872-50-4",
}

# --- grouped by chemical family (optional, for stratified sampling) --------
SOLVENT_FAMILIES = {
    "alcohol": ["methanol", "ethanol", "isopropanol", "n-butanol"],
    "ketone": ["acetone", "methyl ethyl ketone", "methyl isobutyl ketone"],
    "ester": ["ethyl acetate", "isopropyl acetate"],
    "ether": ["diethyl ether", "tetrahydrofuran", "2-methyltetrahydrofuran",
              "methyl tert-butyl ether", "1,4-dioxane"],
    "aromatic": ["toluene", "xylene", "benzene"],
    "alkane": ["n-hexane", "n-heptane", "cyclohexane"],
    "chlorinated": ["dichloromethane", "chloroform"],
    "dipolar_aprotic": ["acetonitrile", "n,n-dimethylformamide",
                        "dimethyl sulfoxide", "n-methyl-2-pyrrolidone"],
}

if __name__ == "__main__":
    print(f"{len(COMMON_SOLVENTS)} common solvents:")
    for s in COMMON_SOLVENTS:
        print(f"  {s:26} CAS {SOLVENT_CAS[s]}")
