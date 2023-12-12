from pathlib import Path
from urllib.request import urlretrieve

import lohi_splitter
import pandas as pd
import deepchem as dc

from datasail.sail import datasail
from experiments.utils import RUNS, SPLITTERS, dc2pd


def load_data():
    data_path = Path("experiments") / "Biogen" / "data" / "ADME_public_set_3521.csv"
    if data_path.exists():
        return
    data_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(
        "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/ADME_public_set_3521.csv",
        data_path
    )
    df = pd.read_csv(data_path)
    df.columns = ["ID", "vID", "SMILES", "C", "HLM", "MDR1_MDCK_ER", "SOLUBILITY", "hPPB", "rPPB", "RLM"]
    for cat in ["HLM", "MDR1_MDCK_ER", "SOLUBILITY", "hPPB", "rPPB", "RLM"]:
        cat_df = df[["ID", "SMILES", cat]]
        cat_df = cat_df.dropna()
        cat_df.to_csv(data_path.parent / f"{cat}.csv", index=False)


def split_w_datasail(subset):
    base = Path('experiments') / 'Biogen' / 'datasail' / subset
    df = pd.read_csv(Path("experiments") / "Biogen" / "data" / f"{subset}.csv")
    try:
        e_splits, _, _ = datasail(
            techniques=["I1e", "C1e"],
            splits=[8, 2],
            names=["train", "test"],
            runs=RUNS,
            solver="SCIP",
            e_type="M",
            e_data=dict(df[["ID", "SMILES"]].values.tolist())
        )

        for tech in ["I1e", "C1e"]:
            for run in range(RUNS):
                path = base / tech / f"split_{run}"
                path.mkdir(parents=True, exist_ok=True)
                train = list(df["ID"].apply(lambda x: e_splits[tech][run].get(x, "") == "train"))
                test = list(df["ID"].apply(lambda x: e_splits[tech][run].get(x, "") == "test"))
                df[train].to_csv(path / "train.csv", index=False)
                df[test].to_csv(path / "test.csv", index=False)
    except Exception as e:
        print("=" * 80 + f"\n{e}\n" + "=" * 80)


def split_w_deepchem(subset):
    base = Path('experiments') / 'Biogen' / 'deepchem' / subset
    df = pd.read_csv(Path("experiments") / "Biogen" / "data" / f"{subset}.csv")
    dataset = dc.data.DiskDataset.from_numpy(X=df["ID"], ids=df["SMILES"])
    for run in range(RUNS):
        for tech in SPLITTERS:
            print("Run:", run, "Tech:", tech)
            try:
                path = base / tech / f"split_{run}"
                path.mkdir(parents=True, exist_ok=True)

                train_set, test_set = SPLITTERS[tech].train_test_split(dataset, frac_train=0.8)
                df[df["ID"].isin(set(train_set.X))].to_csv(path / "train.csv", index=False)
                df[df["ID"].isin(set(test_set.X))].to_csv(path / "test.csv", index=False)
            except Exception as e:
                print("=" * 80 + f"\n{e}\n" + "=" * 80)
        dataset = dataset.complete_shuffle()


def split_w_lohi(subset):
    base = Path('experiments') / 'Biogen' / 'lohi' / subset
    df = pd.read_csv(Path("experiments") / "Biogen" / "data" / f"{subset}.csv")
    for run in range(RUNS):
        for tech in SPLITTERS:
            print("Run:", run, "Tech:", tech)
            try:
                path = base / tech / f"split_{run}"
                path.mkdir(parents=True, exist_ok=True)

                train_test_partition = lohi_splitter.hi_train_test_split(
                    smiles=list(df["SMILES"]),
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


if __name__ == '__main__':
    # load_data()
    # split_w_datasail("HLM")
    split_w_deepchem("HLM")
    split_w_lohi("HLM")
