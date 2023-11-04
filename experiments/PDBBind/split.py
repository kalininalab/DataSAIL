import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from deepchem.data import DiskDataset
import lohi_splitter as lohi

from datasail.sail import datasail
from experiments.utils import load_lp_pdbbind, SPLITTERS, RUNS, telegram

count = 0


def split_to_dataset(df, assignment, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    train = df["ids"].apply(lambda x: assignment.get((x, x), "") == "train")
    test = df["ids"].apply(lambda x: assignment.get((x, x), "") == "test")
    df[train].to_csv(target_dir / "train.csv")
    df[test].to_csv(target_dir / "test.csv")


def split_w_datasail():
    base = Path("experiments") / "PDBBind" / "datasail"
    df = load_lp_pdbbind()

    e_splits, f_splits, inter_splits = datasail(
        techniques=["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"],
        splits=[8, 2],
        names=["train", "test"],
        runs=RUNS,
        solver="SCIP",
        inter=[(x[0], x[0]) for x in df[["ids"]].values.tolist()],
        e_type="M",
        e_data=dict(df[["ids", "Ligand"]].values.tolist()),
        f_type="P",
        f_data=dict(df[["ids", "Target"]].values.tolist()),
        f_sim="mmseqs",
        verbose="I",
        max_sec=1000,
        epsilon=0.05,
    )

    for technique in inter_splits:
        for run in range(len(inter_splits[technique])):
            split_to_dataset(df, inter_splits[technique][run], base / technique / f"split_{run}")


def split_w_deepchem():
    base = Path("experiments") / "PDBBind" / "deepchem"
    df = load_lp_pdbbind()
    ds = DiskDataset.from_numpy(X=np.zeros(len(df)), ids=df["Ligand"].tolist(),
                                y=df["y"].tolist())  # , ids=df["ids"].tolist())

    for run in range(RUNS):
        for tech in SPLITTERS:
            try:
                path = base / tech / f"split_{run}"
                os.makedirs(path, exist_ok=True)

                with open(path / "start.txt", "w") as start:
                    print("Start", file=start)

                train_set, test_set = SPLITTERS[tech].train_test_split(ds, frac_train=0.8)

                df[df["Ligand"].isin(set(train_set.ids))].to_csv(path / "train.csv", index=False)
                df[df["Ligand"].isin(set(test_set.ids))].to_csv(path / "test.csv", index=False)
                global count
                count += 1
                telegram(
                    f"[PDBBind {count} / 25] Splitting finished for PDBBind - deepchem - {tech} - Run {run + 1} / 5")
            except Exception as e:
                print("=" * 80 + f"\n{e}\n" + "=" * 80)
        ds = ds.complete_shuffle()


def split_w_lohi():
    base = Path("experiments") / "PDBBind" / "lohi"
    df = load_lp_pdbbind()

    for run in range(RUNS):
        try:
            path = base / "lohi" / f"split_{run}"
            os.makedirs(path, exist_ok=True)

            with open(path / "start.txt", "w") as start:
                print("Start", file=start)

            train_test_partition = lohi.hi_train_test_split(
                smiles=list(df["Ligand"]),
                similarity_threshold=0.4,
                train_min_frac=0.7,
                test_min_frac=0.1,
                coarsening_threshold=0.4,
                max_mip_gap=0.1,
                verbose=False,
            )

            df.iloc[train_test_partition[0]].to_csv(path / "train.csv", index=False)
            df.iloc[train_test_partition[1]].to_csv(path / "test.csv", index=False)
            global count
            count += 1
            telegram(f"[PDBBind {count} / 35] Splitting finished for PDBBind - lohi - Run {run + 1} / 5")
        except Exception as e:
            print("=" * 80 + f"\n{e}\n" + "=" * 80)
        df = df.sample(frac=1)


def split_w_graphpart():
    base = Path("experiments") / "PDBBind" / "graphpart"
    df = load_lp_pdbbind()

    for run in range(RUNS):
        try:
            path = base / "graphpart" / f"split_{run}"
            os.makedirs(path, exist_ok=True)

            with open(path / "start.txt", "w") as start:
                print("Start", file=start)

            with open(path / "seqs.fasta", "w") as out:
                for _, row in df.iterrows():
                    print(f">{row['ids']}\n{row['Target']}", file=out)

            cmd = f"cd {os.path.abspath(path)} && graphpart mmseqs2 -ff seqs.fasta -th 0.3 -te 0.15"
            print(cmd)
            os.system(cmd)  # > log.txt")

            split = pd.read_csv("graphpart_result.csv")
            train_ids = set(split[split["cluster"] < 0.5]["AC"])
            test_ids = set(split[split["cluster"] > 0.5]["AC"])

            df[df["ids"].isin(train_ids)].to_csv(path / "train.csv", index=False)
            df[df["ids"].isin(test_ids)].to_csv(path / "test.csv", index=False)
            global count
            count += 1
            telegram(f"[PDBBind {count} / 35] Training finished for PDBBind - graphpart - Run {run + 1} / 5")
        except Exception as e:
            print("=" * 80 + f"\n{e}\n" + "=" * 80)
        df = df.sample(frac=1)


def main():
    # split_w_datasail()
    split_w_deepchem()
    # split_w_lohi()
    # split_w_graphpart()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == "datasail":
            split_w_datasail()
        elif sys.argv[1] == "deepchem":
            split_w_deepchem()
        elif sys.argv[1] == "lohi":
            split_w_lohi()
        elif sys.argv[1] == "graphpart":
            split_w_graphpart()
        else:
            print("Unknown splitter:", sys.argv[1])
    else:
        main()
