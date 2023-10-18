import os
from pathlib import Path

import pandas as pd

from experiments.utils import RUNS

path = Path("data_scip_improved")
cleaned = Path("pdbbind")
for tech in ["R", "ICSe", "ICSf", "ICD", "CCSe", "CCSf", "CCD"]:
    for run in range(RUNS):
        train = pd.read_csv(path / tech / f"split_{run}" / "train.csv")
        test = pd.read_csv(path / tech / f"split_{run}" / "test.csv")
        train["split"] = "train"
        test["split"] = "test"
        df = pd.concat([train, test])
        df.rename(columns={"Ligand": "ligands", "Target": "proteins", "y": "affinity"}, inplace=True)
        df = df[df.apply(lambda x: len(x["ligands"]) <= 200 and len(x["proteins"]) <= 2000, axis=1)]
        os.makedirs(cleaned / tech / f"split_{run}", exist_ok=True)
        df.to_csv(cleaned / tech / f"split_{run}" / "pdbbind.csv", index=False)
