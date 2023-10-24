import os
from pathlib import Path

from rdkit import Chem
import deepchem as dc
from datasail.sail import datasail

from experiments.utils import splitters, mpp_datasets, dc2pd, RUNS


def split_w_datasail(name):
    base = Path('experiments') / 'MPP' / 'datasail' / 'sdata' / name
    dataset = mpp_datasets[name][0](featurizer=dc.feat.DummyFeaturizer(), splitter=None)[1][0]
    df = dc2pd(dataset, name)

    for tech in ["I1e", "C1e"]:
        for run in range(RUNS):
            try:
                path = base / tech / f"split_{run}"
                os.makedirs(path, exist_ok=True)

                with open(path / "start.txt", "w") as start:
                    print("Start", file=start)

                e_splits, _, _ = datasail(
                    techniques=[tech],
                    splits=[8, 2],
                    names=["train", "test"],
                    runs=RUNS,
                    solver="SCIP",
                    e_type="M",
                    e_data=dict(df[["ID", "SMILES"]].values.tolist())
                )

                train = list(df["ID"].apply(lambda x: e_splits[tech][run].get(x, "") == "train"))
                test = list(df["ID"].apply(lambda x: e_splits[tech][run].get(x, "") == "test"))
                df[train].to_csv(path / "train.csv", index=False)
                df[test].to_csv(path / "test.csv", index=False)
            except Exception as e:
                print("=" * 80 + f"\n{e}\n" + "=" * 80)


def split_w_deepchem(name):
    base = Path('experiments') / 'MPP' / 'deepchem' / 'sdata' / name

    dataset = mpp_datasets[name][0](featurizer=dc.feat.DummyFeaturizer(), splitter=None)[1][0]
    if name[:2] != "qm":
        valid_ids = [i for i, smiles in enumerate(list(dataset.to_dataframe()["X"])) if Chem.MolFromSmiles(smiles) is not None]
        dataset = dataset.select(valid_ids)
    
    for run in range(RUNS):
        for tech in splitters:
            try:
                path = base / tech / f"split_{run}"
                os.makedirs(path, exist_ok=True)

                with open(path / "start.txt", "w") as start:
                    print("Start", file=start)

                train_set, test_set = splitters[tech].train_test_split(dataset, frac_train=0.8)

                dc2pd(train_set, name).to_csv(path / "train.csv", index=False)
                dc2pd(test_set, name).to_csv(path / "test.csv", index=False)
            except Exception as e:
                print("=" * 80 + f"\n{e}\n" + "=" * 80)
        dataset = dataset.complete_shuffle()


def full_main():
    for ds_name in mpp_datasets:
        if ds_name in ["pdbbind", "pcba"]:
            continue
        split_w_datasail(ds_name)
        split_w_deepchem(ds_name)


def scnd_main():
    for ds_name in ["hiv", "bace", "bbbp", "tox21", "toxcast", "sider", "clintox"]:
        split_w_datasail(ds_name)
        split_w_deepchem(ds_name)


if __name__ == '__main__':
    scnd_main()
