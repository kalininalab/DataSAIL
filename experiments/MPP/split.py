import os

#num_threads = "128"
#os.environ["OPENBLAS_NUM_THREADS"] = num_threads
#os.environ["GOTO_NUM_THREADS"] = num_threads
#os.environ["OMP_NUM_THREADS"] = num_threads


import sys
from pathlib import Path
import time as T
from typing import List

import deepchem as dc
import pandas as pd
import lohi_splitter as lohi

from datasail.sail import datasail
from experiments.utils import SPLITTERS, DATASETS, RUNS, dc2pd, save_datasail_splits, TECHNIQUES


def prep_moleculenet(name) -> pd.DataFrame:
    """
    Prepare the MoleculeNet dataset for splitting.

    Args:
        name: Name of the dataset

    Returns:
        pd.DataFrame: Featurized dataframe
    """
    dataset = DATASETS[name][0](featurizer=dc.feat.DummyFeaturizer(), splitter=None)[1][0]
    return dc2pd(dataset, name)


def split_w_datasail(base_path: Path, name: str, techniques: List[str], solver: str = "GUROBI") -> None:
    """
    Split a MoleculeNet dataset using DataSAIL.

    Args:
        base_path: Path to the base directory
        name: Name of the dataset
        techniques: List of techniques to use
        solver: Solver to use for DataSAIL
    """
    base_path.mkdir(parents=True, exist_ok=True)
    # if (base_path / "time.txt").exists():
    #     print("DataSAIL skipping", name)
    #     return

    # with open(base_path / "time.txt", "w") as time:
    #     print("Start", file=time)

    df = prep_moleculenet(name)
    start = T.time()
    e_splits, _, _ = datasail(
        techniques=techniques,
        splits=[8, 2],
        names=["train", "test"],
        runs=1,  # 5,
        solver=solver,
        e_type="M",
        e_data=dict(df[["ID", "SMILES"]].values.tolist()),
        max_sec=1000,
        epsilon=0.1,
    )
    with open(base_path / "time2.txt", "a") as time:
        print(techniques[0], T.time() - start, file=time)

    save_datasail_splits(base_path, df, "ID", [(t, t) for t in techniques], e_splits=e_splits)


def split_w_deepchem(base_path, name, techniques):
    """
    Split a MoleculeNet dataset using DeepChem.

    Args:
        base_path: Path to the base directory
        name: Name of the dataset
        techniques: List of techniques to use
    """
    base_path.mkdir(parents=True, exist_ok=True)
    if (base_path / "time.txt").exists():
        print("DeepChem skipping", name)
        return

    df = prep_moleculenet(name)
    dataset = dc.data.DiskDataset.from_numpy(X=df["ID"], ids=df["SMILES"])
    with open(base_path / "time.txt", "w") as time:
        print("Start", file=time)

    for run in range(RUNS):
        for tech in techniques:
            try:
                path = base_path / tech / f"split_{run}"
                path.mkdir(parents=True, exist_ok=True)

                start = T.time()
                train_set, test_set = SPLITTERS[tech].train_test_split(dataset, frac_train=0.8)
                with open(base_path / "time.txt", "a") as time:
                    print(tech, T.time() - start, file=time)

                df[df["ID"].isin(set(train_set.X))].to_csv(path / "train.csv", index=False)
                df[df["ID"].isin(set(test_set.X))].to_csv(path / "test.csv", index=False)
            except Exception as e:
                print("=" * 80 + f"\n{e}\n" + "=" * 80)
        dataset = dataset.complete_shuffle()


def split_w_lohi(base_path: Path, name: str) -> None:
    """
    Split a MoleculeNet dataset using LoHi.

    Args:
        base_path: Path to the base directory
        name: Name of the dataset
    """
    base_path.mkdir(parents=True, exist_ok=True)
    if (base_path / "time.txt").exists():
        print("LoHi skipping", name)
        return

    df = prep_moleculenet(name)
    dataset = dc.data.DiskDataset.from_numpy(X=df["ID"], ids=df["SMILES"])
    df.set_index("ID", inplace=True, drop=False)
    with open(base_path / "time.txt", "w") as time:
        print("Start", file=time)

    for run in range(RUNS):
        try:
            path = base_path / "lohi" / f"split_{run}"
            path.mkdir(parents=True, exist_ok=True)

            start = T.time()
            train_test_partition = lohi.hi_train_test_split(
                smiles=dataset.ids.tolist(),
                similarity_threshold=0.4,
                train_min_frac=0.7,
                test_min_frac=0.1,
                coarsening_threshold=0.6,
                max_mip_gap=0.1,
                verbose=False,
            )
            with open(base_path / "time.txt", "a") as time:
                print("lohi", T.time() - start, file=time)

            df.loc[dataset.X[train_test_partition[0]]].to_csv(path / "train.csv", index=False)
            df.loc[dataset.X[train_test_partition[1]]].to_csv(path / "test.csv", index=False)
        except Exception as e:
            print("=" * 80 + f"\n{e}\n" + "=" * 80)
        dataset = dataset.complete_shuffle()


def split_all(path):
    """
    Split all MoleculeNet datasets using different techniques.

    Args:
        path: Path to the base directory
    """
    for name in DATASETS:
        if name.lower() == "pcba":
            continue
        print("Dataset:", name)
        split_w_datasail(path / "datasail" / name, name, techniques=TECHNIQUES["datasail"])
        split_w_deepchem(path / "deepchem" / name, name, techniques=TECHNIQUES["deepchem"])
        split_w_lohi(path / "lohi" / name, name)


def split(full_path, name, solver="GUROBI"):
    """
    Split the MoleculeNet datasets using different techniques.
    """
    split_w_datasail(full_path / "datasail" / name, name, techniques=["I1e"], solver=solver)
    # split_w_deepchem(full_path / "deepchem" / name, name, techniques=SPLITTERS.keys())
    # split_w_lohi(full_path / "lohi" / name, name)


def specific():
    for run in range(RUNS):
        for name in DATASETS.keys():
            if name.lower() in {"pcba"}:
                continue
            split_w_datasail(Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "DataSAIL" / "v10" / "MPP" / "datasail_new" / name, name, ["I1e"])
            split_w_datasail(Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "DataSAIL" / "v10" / "MPP" / "datasail_new" / name, name, ["C1e"])


if __name__ == '__main__':
    specific()
    exit(0)
    # split_w_datasail(Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "DataSAIL" / "v10" / "MPP" / "datasail_test" / "qm8", "qm8", ["C1e"])
    # exit(0)
    if len(sys.argv) == 1:
        specific()
    elif len(sys.argv) == 2:
        split_all(Path(sys.argv[1]))
    elif len(sys.argv) == 3:
        split(Path(sys.argv[1]), sys.argv[2])
    elif len(sys.argv) >= 4:
        split(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
