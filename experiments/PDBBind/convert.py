import os
from pathlib import Path

import pandas as pd

from experiments.utils import RUNS

# Convert output of DataSAIL to one file that DeepDTA can use

path = Path("experiments") / "PDBBind"
for tool, techniques in [("deepchem", ["Fingerprint", "MinMax", "Scaffold", "Weight", "Butina"])]:
    for tech in techniques:
        for run in range(RUNS):
            train = pd.read_csv(path / tool / tech / f"split_{run}" / "train.csv")
            test = pd.read_csv(path / tool / tech / f"split_{run}" / "test.csv")
            train["split"] = "train"
            test["split"] = "test"
            df = pd.concat([train, test])
            df.rename(columns={"Ligand": "ligands", "Target": "proteins", "y": "affinity"}, inplace=True)
            df = df[df.apply(lambda x: len(x["ligands"]) <= 200 and len(x["proteins"]) <= 2000, axis=1)]
            df.to_csv(path / tool / tech / f"split_{run}" / "pdbbind.csv", index=False)
