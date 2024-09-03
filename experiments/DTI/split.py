import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from deepchem.data import DiskDataset
import lohi_splitter as lohi

from datasail.sail import datasail
from experiments.utils import load_lp_pdbbind, SPLITTERS, RUNS, TECHNIQUES, save_datasail_splits


def split_w_datasail(base_path: Path, techniques: List[str], solver: str = "GUROBI") -> None:
    """
    Split the LP_PDBBind dataset using DataSAIL.

    Args:
        base_path: Path to the base directory
        techniques: List of techniques to use
        solver: Solver to use for DataSAIL
    """
    base = base_path / "datasail"
    base.mkdir(parents=True, exist_ok=True)
    df = load_lp_pdbbind()

    e_splits, f_splits, inter_splits = datasail(
        techniques=techniques,
        splits=[8, 2],
        names=["train", "test"],
        runs=RUNS,
        solver=solver,
        inter=[(x[0], x[0]) for x in df[["ids"]].values.tolist()],
        e_type="M",
        e_data=dict(df[["ids", "smiles"]].values.tolist()),
        f_type="P",
        f_data=dict(df[["ids", "seq"]].values.tolist()),
        f_sim="mmseqs",
        verbose="I",
        max_sec=1000,
        epsilon=0.1,
    )

    save_datasail_splits(base, df, "ids", [(t, t) for t in techniques], inter_splits=inter_splits)


def split_w_deepchem(base_path: Path, techniques: List[str]) -> None:
    """
    Split the LP_PDBBind dataset using DeepChem.

    Args:
        base_path: Path to the base directory
        techniques: List of techniques to use
    """
    base = base_path / "deepchem"
    base.mkdir(parents=True, exist_ok=True)
    df = load_lp_pdbbind()
    ds = DiskDataset.from_numpy(X=np.zeros(len(df)), ids=df["smiles"].tolist(), y=df["value"].tolist())

    for run in range(RUNS):
        for tech in techniques:
            try:
                path = base / tech / f"split_{run}"
                os.makedirs(path, exist_ok=True)

                train_set, test_set = SPLITTERS[tech].train_test_split(ds, frac_train=0.8)

                df[df["smiles"].isin(set(train_set.ids))].to_csv(path / "train.csv", index=False)
                df[df["smiles"].isin(set(test_set.ids))].to_csv(path / "test.csv", index=False)
            except Exception as e:
                print("=" * 80 + f"\n{e}\n" + "=" * 80)
        ds = ds.complete_shuffle()


def split_w_lohi(base_path: Path) -> None:
    """
    Split the LP_PDBBind dataset using LoHi.

    Args:
        base_path: Path to the base directory
    """
    base = base_path / "lohi"
    base.mkdir(parents=True, exist_ok=True)
    df = load_lp_pdbbind()

    for run in range(RUNS):
        try:
            path = base / "lohi" / f"split_{run}"
            os.makedirs(path, exist_ok=True)

            with open(path / "start.txt", "w") as start:
                print("Start", file=start)

            train_test_partition = lohi.hi_train_test_split(
                smiles=list(df["smiles"]),
                similarity_threshold=0.4,
                train_min_frac=0.7,
                test_min_frac=0.1,
                coarsening_threshold=0.4,
                max_mip_gap=0.1,
                verbose=False,
            )

            df.iloc[train_test_partition[0]].to_csv(path / "train.csv", index=False)
            df.iloc[train_test_partition[1]].to_csv(path / "test.csv", index=False)
        except Exception as e:
            print("=" * 80 + f"\n{e}\n" + "=" * 80)
        df = df.sample(frac=1)


def split_w_graphpart(base_path: Path) -> None:
    """
    Split the LP_PDBBind dataset using GraphPart.

    Args:
        base_path: Path to the base directory
    """
    base = base_path / "graphpart"
    base.mkdir(parents=True, exist_ok=True)
    df = load_lp_pdbbind()

    for run in range(RUNS):
        try:
            path = base / "graphpart" / f"split_{run}"
            os.makedirs(path, exist_ok=True)

            with open(path / "seqs.fasta", "w") as out:
                for _, row in df.iterrows():
                    print(f">{row['ids']}\n{row['seq']}", file=out)

            cmd = f"cd {os.path.abspath(path)} && graphpart mmseqs2 -ff seqs.fasta -th 0.3 -te 0.15"
            os.system(cmd)
            os.remove(path / "seqs.fasta")

            split = pd.read_csv(path / "graphpart_result.csv")
            train_ids = set(split[split["cluster"] < 0.5]["AC"])
            test_ids = set(split[split["cluster"] > 0.5]["AC"])

            df[df["ids"].isin(train_ids)].to_csv(path / "train.csv", index=False)
            df[df["ids"].isin(test_ids)].to_csv(path / "test.csv", index=False)
        except Exception as e:
            print("=" * 80 + f"\n{e}\n" + "=" * 80)
        df = df.sample(frac=1)


def main(path):
    split_w_datasail(path, TECHNIQUES["datasail"])
    split_w_deepchem(path, TECHNIQUES["deepchem"])
    split_w_lohi(path)
    split_w_graphpart(path)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
