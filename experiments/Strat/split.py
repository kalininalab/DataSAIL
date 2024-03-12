import sys
from pathlib import Path

import deepchem as dc
import pandas as pd

from datasail.sail import datasail
from experiments.utils import RUNS, dc2pd, smiles2mol, save_datasail_splits


def tox2pd(ds: dc.data.DiskDataset) -> pd.DataFrame:
    """
    Convert the Tox21 dataset to a DataFrame.

    Args:
        ds: DeepChem dataset

    Returns:
        pd.DataFrame: DataFrame of the Tox21 dataset
    """
    df = ds.to_dataframe()
    name_map = dict([(f"y{i + 1}", task) for i, task in enumerate(ds.tasks)] + [("y", ds.tasks[0]), ("X", "SMILES")])
    df.rename(columns=name_map, inplace=True)
    df["ID"] = [f"Comp{i + 1:06d}" for i in range(len(df))]
    df["mol"] = df["SMILES"].apply(smiles2mol)
    df = df[df["mol"].notna()]
    df["SR-ARE"] = pd.to_numeric(df["SR-ARE"], downcast="integer")
    return df[["ID", "SMILES", "SR-ARE"]]


def split_w_datasail(full_base: Path, df: pd.DataFrame, delta: float, epsilon: float, solver: str = "GUROBI") -> None:
    """
    Split the Tox21 dataset using DataSAIL with Stratification.

    Args:
        full_base: Path to the base directory
        df: DataFrame of the Tox21 dataset
        delta: Delta value for DataSAIL
        epsilon: Epsilon value for DataSAIL
        solver: Solver to use for DataSAIL
    """
    e_splits, _, _ = datasail(
        techniques=["C1e"],
        splits=[8, 2],
        names=["train", "test"],
        runs=RUNS,
        delta=delta,
        epsilon=epsilon,
        solver=solver,
        e_type="M",
        e_data=dict(df[["ID", "SMILES"]].values.tolist()),
        e_strat=dict(df[["ID", "SR-ARE"]].values.tolist()),
        max_sec=1000,
    )

    save_datasail_splits(full_base / 'datasail', df, "ID", [(f"d_{delta}_e_{epsilon}", "C1e")], e_splits)


def split_w_deepchem(full_base: Path, df: pd.DataFrame) -> None:
    """
    Split the Tox21 dataset using DeepChem with Stratification.

    Args:
        full_base: Path to the base directory
        df: DataFrame of the Tox21 dataset
    """
    ds = dc.data.DiskDataset.from_numpy(
        X=df["SMILES"].values.reshape(-1, 1),
        y=df["SR-ARE"].values.reshape(-1, 1),
        ids=df["ID"].values.reshape(-1, 1)
    )
    for run in range(RUNS):
        path = full_base / 'deepchem' / f"split_{run}"
        path.mkdir(parents=True, exist_ok=True)
        train_set, test_set = dc.splits.SingletaskStratifiedSplitter(task_number=0).train_test_split(ds, frac_train=0.8)
        dc2pd(train_set, "tox21").to_csv(path / "train.csv", index=False)
        dc2pd(test_set, "tox21").to_csv(path / "test.csv", index=False)
        ds = ds.complete_shuffle()


def main(path):
    """
    Split the Tox21 dataset using DataSAIL and DeepChem with Stratification. In addition, splits with different delta
    and epsilon values are calculated for the ablation study.

    Args:
        path: Path to the base directory
    """
    tox = dc.molnet.load_tox21(featurizer=dc.feat.DummyFeaturizer(), splitter=None)[1][0]
    tox = tox2pd(tox)
    for e in [0.3, 0.25, 0.2, 0.15, 0.1, 0.05]:
        for d in [0.3, 0.25, 0.2, 0.15, 0.1, 0.05]:
            print(f"Splitting Tox21 SR-ARE with delta={d} and epsilon={e}")
            split_w_datasail(path, tox, d, e)

    split_w_deepchem(path, tox)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
