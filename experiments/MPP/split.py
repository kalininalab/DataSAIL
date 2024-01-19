import sys
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import deepchem as dc
from datasail.sail import datasail
import lohi_splitter as lohi

from experiments.utils import SPLITTERS, mpp_datasets, dc2pd, RUNS, telegram, biogen_datasets, embed_smiles

count = 0
total_number = 6 * 5 + 1


def message(tool, name, tech, run):
    global count
    count += 1
    telegram(f"[Splitting {count}/{total_number}] {tool} - {name} - {tech} - {run + 1}/5")


def prep_moleculenet(name):
    dataset = mpp_datasets[name][0](featurizer=dc.feat.DummyFeaturizer(), splitter=None)[1][0]
    return dc2pd(dataset, name)


def prep_biogen(name):
    biogen_path = Path("experiments") / "MPP" / "data"
    data_path = biogen_path / "ADME_public_set_3521.csv"
    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(
            "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/ADME_public_set_3521.csv",
            data_path
        )
        df = pd.read_csv(data_path)
        df.columns = ["ID", "vID", "SMILES", "C", "HLM", "MDR1_MDCK_ER", "SOLUBILITY", "hPPB", "rPPB", "RLM"]
        for cat in ["HLM", "MDR1_MDCK_ER", "SOLUBILITY", "hPPB", "rPPB", "RLM"]:
            cat_df = df[["ID", "SMILES", cat]]
            cat_df["ECFP4"] = cat_df["SMILES"].apply(embed_smiles)
            cat_df = cat_df.dropna()
            cat_df.to_csv(biogen_path / f"{cat}.csv", index=False)

    return pd.read_csv(biogen_path / f"{name}.csv")


def split_w_datasail(df, name, techniques):
    base = Path('experiments') / 'MPP' / 'datasail' / name

    e_splits, _, _ = datasail(
        techniques=techniques,
        splits=[8, 2],
        names=["train", "test"],
        runs=RUNS,
        solver="SCIP",
        e_type="M",
        e_data=dict(df[["ID", "SMILES"]].values.tolist())
    )

    for tech in techniques:
        for run in range(RUNS):
            path = base / tech / f"split_{run}"
            path.mkdir(parents=True, exist_ok=True)

            train = list(df["ID"].apply(lambda x: e_splits[tech][run].get(x, "") == "train"))
            test = list(df["ID"].apply(lambda x: e_splits[tech][run].get(x, "") == "test"))
            df[train].to_csv(path / "train.csv", index=False)
            df[test].to_csv(path / "test.csv", index=False)
    message("datasail", name, "Both", 4)


def split_w_deepchem(df, name, techniques):
    base = Path('experiments') / 'MPP' / 'deepchem' / name
    dataset = dc.data.DiskDataset.from_numpy(X=df["ID"], ids=df["SMILES"])

    for run in range(RUNS):
        for tech in techniques:
            try:
                path = base / tech / f"split_{run}"
                path.mkdir(parents=True, exist_ok=True)

                train_set, test_set = SPLITTERS[tech].train_test_split(dataset, frac_train=0.8)

                df[df["ID"].isin(set(train_set.X))].to_csv(path / "train.csv", index=False)
                df[df["ID"].isin(set(test_set.X))].to_csv(path / "test.csv", index=False)
                message("deepchem", name, tech, run)
            except Exception as e:
                print("=" * 80 + f"\n{e}\n" + "=" * 80)
        dataset = dataset.complete_shuffle()


def split_w_lohi(df, name):
    base = Path('experiments') / 'MPP' / 'lohi' / name
    dataset = dc.data.DiskDataset.from_numpy(X=df["ID"], ids=df["SMILES"])
    df.set_index("ID", inplace=True, drop=False)

    for run in range(RUNS):
        try:
            path = base / "lohi" / f"split_{run}"
            path.mkdir(parents=True, exist_ok=True)

            train_test_partition = lohi.hi_train_test_split(
                smiles=dataset.ids.tolist(),
                similarity_threshold=0.4,
                train_min_frac=0.7,
                test_min_frac=0.1,
                coarsening_threshold=0.6,
                max_mip_gap=0.1,
                verbose=False,
            )

            df.loc[dataset.X[train_test_partition[0]]].to_csv(path / "train.csv", index=False)
            df.loc[dataset.X[train_test_partition[1]]].to_csv(path / "test.csv", index=False)
            message("deepchem", name, "lohi", run)
        except Exception as e:
            print("=" * 80 + f"\n{e}\n" + "=" * 80)
        dataset = dataset.complete_shuffle()


def split_all():
    for name in mpp_datasets:
        print("Dataset:", name)
        df = prep_biogen(name) if name in biogen_datasets else prep_moleculenet(name)
        split_w_datasail(df, name, techniques=["I1e", "C1e"])
        split_w_deepchem(df, name, techniques=SPLITTERS.keys())
        split_w_lohi(df, name)


def main():
    if len(sys.argv) == 1:
        split_all()
    else:
        df = prep_biogen(sys.argv[1]) if sys.argv[1] in biogen_datasets else prep_moleculenet(sys.argv[1])
        split_w_datasail(df, sys.argv[1], techniques=["I1e", "C1e"])
        split_w_deepchem(df, sys.argv[1], techniques=SPLITTERS.keys())
        split_w_lohi(df, sys.argv[1])


if __name__ == '__main__':
    main()
