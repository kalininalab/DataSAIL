import deepchem as dc
import numpy as np

from datasail.sail import datasail
from experiments.utils import dc2pd

df = dc2pd(dc.molnet.load_freesolv(dc.feat.DummyFeaturizer(), splitter=None)[1][0], "freesolv")
df["strat"] = np.random.choice(["A", "B"], replace=True, size=len(df))
print(df.head())

e_splits, _, _ = datasail(
    techniques=["I1e"],
    splits=[7, 2, 1],
    names=["train", "val", "test"],
    runs=1,
    solver="SCIP",
    e_type="M",
    e_data=dict(df[["ID", "SMILES"]].values.tolist()),
    e_strat=dict(df[["ID", "strat"]].values.tolist()),
)

print(e_splits["I1e"][0])
