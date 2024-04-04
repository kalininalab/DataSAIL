from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datasail.sail import datasail


@pytest.mark.parametrize("tech", ["I1e", "I1f", "I2", "C1e", "C1f", "C2"])
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


def check(df, key, splits, tech):
    factor = 0.5 if tech[1] == "2" else 1
    df["split"] = df[key].apply(lambda x: splits[tech][0][x])
    for split, frac in [("train", 0.8), ("test", 0.2)]:
        for cls in ["A", "B"]:
            assert len(df[(df["split"] == split) & (df["strat"] == cls)]) >= len(
                df[df["strat"] == cls]) * frac * 0.8 * factor
