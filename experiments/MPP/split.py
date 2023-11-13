import os
import sys
from pathlib import Path

from rdkit import Chem
import deepchem as dc
from datasail.sail import datasail
import lohi_splitter as lohi

from experiments.utils import SPLITTERS, mpp_datasets, dc2pd, RUNS, telegram

count = 0


def split_w_datasail(name):
    base = Path('experiments') / 'MPP' / 'datasail' / 'cdata' / name
    dataset = mpp_datasets[name][0](featurizer=dc.feat.DummyFeaturizer(), splitter=None)[1][0]
    df = dc2pd(dataset, name)
    print(df.shape)

    for tech in ["I1e"]:
        try:

            with open(base / tech / "start.txt", "w") as start:
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

            for run in range(RUNS):
                path = base / tech / f"split_{run}"
                os.makedirs(path, exist_ok=True)
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
        valid_ids = [i for i, smiles in enumerate(list(dataset.to_dataframe()["X"])) if
                     Chem.MolFromSmiles(smiles) is not None]
        dataset = dataset.select(valid_ids)

    for run in range(RUNS):
        for tech in SPLITTERS:
            try:
                path = base / tech / f"split_{run}"
                os.makedirs(path, exist_ok=True)

                with open(path / "start.txt", "w") as start:
                    print("Start", file=start)

                train_set, test_set = SPLITTERS[tech].train_test_split(dataset, frac_train=0.8)

                dc2pd(train_set, name).to_csv(path / "train.csv", index=False)
                dc2pd(test_set, name).to_csv(path / "test.csv", index=False)
            except Exception as e:
                print("=" * 80 + f"\n{e}\n" + "=" * 80)
        dataset = dataset.complete_shuffle()


def split_w_lohi(name):
    base = Path('experiments') / 'MPP' / 'lohi' / 'sdata' / name

    dataset = mpp_datasets[name][0](featurizer=dc.feat.DummyFeaturizer(), splitter=None)[1][0]
    if name[:2] != "qm":
        valid_ids = [i for i, smiles in enumerate(list(dataset.to_dataframe()["X"])) if
                     Chem.MolFromSmiles(smiles) is not None]
        dataset = dataset.select(valid_ids)

    for run in range(RUNS):
        try:
            path = base / "lohi" / f"split_{run}"
            os.makedirs(path, exist_ok=True)
            df = dc2pd(dataset, name)

            with open(path / "start.txt", "w") as start:
                print("Start", file=start)

            train_test_partition = lohi.hi_train_test_split(
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
            global count
            count += 1
            telegram(f"[MPP {count} / 70] Splitting finished for MPP - lohi - {name} - Run {run + 1} / 5")
        except Exception as e:
            print("=" * 80 + f"\n{e}\n" + "=" * 80)
        dataset = dataset.complete_shuffle()


def main():
    # split_w_datasail("muv")
    split_w_datasail("qm9")
    # for ds_name in sorted(list(mpp_datasets.keys()), key=lambda x: mpp_datasets[x][3]):
    #     if ds_name in ["pdbbind", "pcba"]:
    #         continue
    # split_w_datasail(ds_name)
    # split_w_deepchem(ds_name)
    # split_w_lohi(ds_name)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        split_w_lohi(sys.argv[1])
    else:
        main()
