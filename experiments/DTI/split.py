import os
import sys
from pathlib import Path
import time as T

import numpy as np
import pandas as pd
from deepchem.data import DiskDataset
import lohi_splitter as lohi

from datasail.sail import datasail
from experiments.utils import load_lp_pdbbind, SPLITTERS, RUNS, telegram

count = 0
total_number = 1 * 14 * 5  # num_datasets * num_techs * num_runs


def message(tool, tech, run):
    global count
    count += 1
    telegram(f"[DTI Splitting {count}/{total_number}] {tool} - {tech} - {run + 1}/5")


def split_to_dataset(df, assignment, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for label in ["train", "test"]:
        sub = df["ids"].apply(lambda x: assignment.get((x, x), "") == label)
        df[sub].to_csv(target_dir / f"{label}.csv")


def split_w_datasail(full_base):
    base = full_base / "DTI" / "datasail"
    base.mkdir(parents=True, exist_ok=True)
    df = load_lp_pdbbind()

    start = T.time()
    e_splits, f_splits, inter_splits = datasail(
        techniques=["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"],
        splits=[8, 2],
        names=["train", "test"],
        runs=RUNS,
        solver="GUROBI",
        inter=[(x[0], x[0]) for x in df[["ids"]].values.tolist()],
        e_type="M",
        e_data=dict(df[["ids", "Ligand"]].values.tolist()),
        f_type="P",
        f_data=dict(df[["ids", "Target"]].values.tolist()),
        f_sim="mmseqs",
        verbose="I",
        max_sec=1000,
        epsilon=0.1,
    )
    with open(base / "time.txt", "w") as time:
        print("Start", T.time() - start, sep="\n", file=time)

    for technique in inter_splits:
        for run in range(len(inter_splits[technique])):
            split_to_dataset(df, inter_splits[technique][run], base / technique / f"split_{run}")
    message("DataSAIL", "all 7", 4)


def split_w_deepchem(full_base):
    base = full_base / "DTI" / "deepchem"
    base.mkdir(parents=True, exist_ok=True)
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
                message("DeepChem", tech, run)
            except Exception as e:
                print("=" * 80 + f"\n{e}\n" + "=" * 80)
        ds = ds.complete_shuffle()


def split_w_lohi(full_base):
    base = full_base / "DTI" / "lohi"
    base.mkdir(parents=True, exist_ok=True)
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
            message("LoHi", "lohi", run)
        except Exception as e:
            print("=" * 80 + f"\n{e}\n" + "=" * 80)
        df = df.sample(frac=1)


def split_w_graphpart(full_base):
    base = full_base / "DTI" / "graphpart"
    base.mkdir(parents=True, exist_ok=True)
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
            os.system(cmd)

            split = pd.read_csv(path / "graphpart_result.csv")
            train_ids = set(split[split["cluster"] < 0.5]["AC"])
            test_ids = set(split[split["cluster"] > 0.5]["AC"])

            df[df["ids"].isin(train_ids)].to_csv(path / "train.csv", index=False)
            df[df["ids"].isin(test_ids)].to_csv(path / "test.csv", index=False)
            message("GraphPart", "graphpart", run)
        except Exception as e:
            print("=" * 80 + f"\n{e}\n" + "=" * 80)
        df = df.sample(frac=1)


def main(path):
    for tool in [split_w_datasail, split_w_deepchem, split_w_lohi, split_w_graphpart]:
        try:
            tool(path)
        except Exception as e:
            telegram(f"[DTI Splitting] {tool.__name__} failed with error: {e}")
    telegram("Finished splitting DTI")


# Path('/') / "scratch" / "SCRATCH_SAS" / "roman" / "DataSAIL" / "v03"
if __name__ == '__main__':
    if len(sys.argv) > 2:
        if sys.argv[2] == "datasail":
            split_w_datasail(Path(sys.argv[1]))
        elif sys.argv[2] == "deepchem":
            split_w_deepchem(Path(sys.argv[1]))
        elif sys.argv[2] == "lohi":
            split_w_lohi(Path(sys.argv[1]))
        elif sys.argv[2] == "graphpart":
            split_w_graphpart(Path(sys.argv[1]))
        else:
            print("Unknown splitter:", sys.argv[1])
    else:
        main(Path(sys.argv[1]))
