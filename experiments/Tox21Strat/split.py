from pathlib import Path

import deepchem as dc
import pandas as pd
from rdkit import Chem
from joblib import Parallel, delayed
from tqdm import tqdm
from rdkit import rdBase

from datasail.sail import datasail
from experiments.utils import RUNS, dc2pd

blocker = rdBase.BlockLogs()

def check_smiles(smiles):
    if Chem.MolFromSmiles(smiles) is None:
        print(smiles)
        return None
    return smiles


def tox2pd(ds):
    df = ds.to_dataframe()
    name_map = dict([(f"y{i + 1}", task) for i, task in enumerate(ds.tasks)] + [("y", ds.tasks[0]), ("X", "SMILES")])
    df.rename(columns=name_map, inplace=True)
    df["ID"] = [f"Comp{i + 1:06d}" for i in range(len(df))]
    df["SMILES"] = df["SMILES"].apply(check_smiles)
    df = df[df["SMILES"].notna()]
    df["SR-ARE"] = pd.to_numeric(df["SR-ARE"], downcast="integer")
    return df[["ID", "SMILES", "SR-ARE"]]


def split_w_datasail(df, delta, epsilon):
    base = Path('experiments') / 'Tox21Strat' / 'datasail' / "de_ablation" / f"d_{delta}_e_{epsilon}"

    try:
        e_splits, _, _ = datasail(
            techniques=["C1e"],
            splits=[8, 2],
            names=["train", "test"],
            runs=RUNS,
            delta=delta,
            epsilon=epsilon,
            solver="SCIP",
            e_type="M",
            e_data=dict(df[["ID", "SMILES"]].values.tolist()),
            e_strat=dict(df[["ID", "SR-ARE"]].values.tolist()),
        )

        for run in range(RUNS):
            path = base / f"split_{run}"
            path.mkdir(parents=True, exist_ok=True)
            train = list(df["ID"].apply(lambda x: e_splits["C1e"][run].get(x, "") == "train"))
            test = list(df["ID"].apply(lambda x: e_splits["C1e"][run].get(x, "") == "test"))
            df[train].to_csv(path / "train.csv", index=False)
            df[test].to_csv(path / "test.csv", index=False)
    except:
        pass


def split_w_deepchem(df):
    base = Path('experiments') / 'Tox21Strat' / 'deepchem'
    ds = dc.data.DiskDataset.from_numpy(
        X=df["SMILES"].values.reshape(-1, 1),
        y=df["SR-ARE"].values.reshape(-1, 1),
        ids=df["ID"].values.reshape(-1, 1)
    )
    for run in range(RUNS):
        path = base / f"split_{run}"
        path.mkdir(parents=True, exist_ok=True)
        train_set, test_set = dc.splits.SingletaskStratifiedSplitter(task_number=0).train_test_split(ds, frac_train=0.8)
        dc2pd(train_set, "tox21").to_csv(path / "train.csv", index=False)
        dc2pd(test_set, "tox21").to_csv(path / "test.csv", index=False)
        ds = ds.complete_shuffle()


tox = dc.molnet.load_tox21(featurizer=dc.feat.DummyFeaturizer(), splitter=None)[1][0]
tox = tox2pd(tox)
vals = []
for e in [0.3, 0.25, 0.2, 0.15, 0.1, 0.05]:
    for d in [0.3, 0.25, 0.2, 0.15, 0.1, 0.05]:
        vals.append((tox, d, e))

Parallel(n_jobs=12)(delayed(split_w_datasail)(*args) for args in tqdm(vals))
# split_w_deepchem(tox)
