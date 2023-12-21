from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datasail.sail import datasail


@pytest.mark.parametrize("technique", ["I1e", "I1f", "I2", "C1e", "C1f", "C2"])
def test_stratification(technique):
    base = Path("data") / "pipeline"
    drugs = pd.read_csv(base / "drugs.tsv", sep="\t")
    drugs["strat"] = np.random.choice(["A", "B"], replace=True, size=len(drugs))
    prots = pd.read_csv(base / "seqs.tsv", sep="\t")
    prots["strat"] = np.random.choice(["A", "B"], replace=True, size=len(prots))
    e_splits, f_splits, inter_splits = datasail(
        techniques=[technique],
        splits=[8, 2],
        names=["train", "test"],
        runs=1,
        solver="SCIP",
        e_type="M",
        e_data=dict(drugs[["Drug_ID", "SMILES"]].values.tolist()),
        e_strat=dict(drugs[["Drug_ID", "strat"]].values.tolist()),
        e_sim="ecfp",
        f_type="P",
        f_data=dict(prots[["ID", "seq"]].values.tolist()),
        f_strat=dict(prots[["ID", "strat"]].values.tolist()),
        f_sim="mmseqs",
    )
    assert e_splits is not None
    assert f_splits is not None
    assert inter_splits is not None
