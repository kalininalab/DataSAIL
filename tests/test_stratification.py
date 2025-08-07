from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datasail.sail import datasail


@pytest.mark.full
@pytest.mark.parametrize("tech", ["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"])
def test_stratification(tech):
    base = Path("data") / "pipeline"
    drugs = pd.read_csv(base / "drugs.tsv", sep="\t")
    drugs["strat"] = np.random.choice(["A", "B"], replace=True, size=len(drugs))
    prots = pd.read_csv(base / "seqs.tsv", sep="\t")
    prots["strat"] = np.random.choice(["A", "B"], replace=True, size=len(prots))
    inter = pd.read_csv(base / "inter.tsv", sep="\t")
    e_splits, f_splits, inter_splits = datasail(
        techniques=[tech],
        splits=[8, 2],
        names=["train", "test"],
        runs=1,
        solver="SCIP",
        inter=inter[["Drug_ID", "Target_ID"]].values.tolist(),
        e_type="M",
        e_data=dict(drugs[["Drug_ID", "SMILES"]].values.tolist()),
        e_strat=dict(drugs[["Drug_ID", "strat"]].values.tolist()),
        e_sim="ecfp",
        f_type="P",
        f_data=dict(prots[["ID", "seq"]].values.tolist()),
        f_strat=dict(prots[["ID", "strat"]].values.tolist()),
        f_sim="mmseqs",
        epsilon=0.2,
        delta=0.2,
    )
    assert e_splits is not None
    if tech in e_splits:
        check(drugs, "Drug_ID", e_splits, tech)

    assert f_splits is not None
    if tech in f_splits:
        check(prots, "ID", f_splits, tech)

    assert inter_splits is not None


@pytest.mark.parametrize("clustering", [True, False])
def test_multiclass(clustering: bool):
    strats = {"A": {1}, "B": {1, 2}, "C": {2, 1}, "D": {2}}
    kwargs = dict(
        techniques=["I1e"],
        splits=[8, 2],
        names=["train", "test"],
        runs=1,
        solver="SCIP",
        delta=0.45,
        e_type="O",
        e_data={"A": np.array([1]), "B": np.array([2]), "C": np.array([3]), "D": np.array([4])},
        e_strat=strats,
    )
    if clustering:
        kwargs["techniques"] = ["C1e"]
        kwargs["e_clusters"] = 4
        kwargs["e_sim"] = ("A", "B", "C", "D"), np.eye(4)
    e_splits, _, _ = datasail(**kwargs,)
    assert e_splits is not None and e_splits != {}
    e = e_splits["C1e" if clustering else "I1e"][0]
    train_strats, test_strats = [], []
    for x in "ABCD":
        if e[x] == "train":
            train_strats += list(strats[x])
        else:
            test_strats += list(strats[x])
    for strat in [train_strats, test_strats]:
        for c in {1, 2}:
            assert strat.count(c) >= 1


def test_clustered_glycans():
    data = {}
    df = pd.read_csv(Path("data") / "rw_data" / "taxonomy_Phylum.tsv", sep="\t")
    df.drop(columns=list(df.columns[1:-1][df.values[:, 1:-1].sum(axis=0) <= 10]), inplace=True)
    df = df[df.values[:, 1:-1].sum(axis=1) != 0]
    for i, (_, row) in enumerate(df.iterrows()):
        name = f"Gly{i:05d}"
        tmp = row[df.columns[1:-1]]
        data[name] = (row["IUPAC"], set(dict(tmp[tmp != 0]).keys()))

    e_splits, _, _ = datasail(
        techniques=["I1e"],
        splits=[8, 2],
        names=["train", "test"],
        runs=1,
        solver="SCIP",
        delta=0.15,
        epsilon=0.15,
        e_type="O",
        e_data={k: v[0] for k, v in data.items()},
        e_strat={k: v[1] for k, v in data.items()},
    )
    assert e_splits is not None and e_splits != {}
    train_strats, test_strats = defaultdict(int), defaultdict(int)
    for k, v in e_splits["I1e"][0].items():
        for class_ in data[k][1]:
            if v == "train":
                train_strats[class_] += 1
            else:
                test_strats[class_] += 1
    for class_ in df.columns[1:-1]:
        assert train_strats[class_] > 0
        assert test_strats[class_] > 0
        assert train_strats[class_] > test_strats[class_]


def check(df, key, splits, tech):
    factor = 0.5 if tech[1] == "2" else 1
    df["split"] = df[key].apply(lambda x: splits[tech][0][x])
    for split, frac in [("train", 0.8), ("test", 0.2)]:
        for cls in ["A", "B"]:
            assert len(df[(df["split"] == split) & (df["strat"] == cls)]) >= len(
                df[df["strat"] == cls]) * frac * 0.8 * factor
