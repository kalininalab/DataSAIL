import deepchem as dc
from rdkit import Chem

from datasail.sail import datasail

"""
MoleculeNet tasks:
-------------------
X - split
O - open
- - not considered
? - partially split
-------------------
X    7160  QM7
-    7210  QM7b
X   21786  QM8
X  133885  QM9
X    1128  ESOL
X     642  FreeSolv
X    4200  Lipophilicity
-  437929  PCBA
-   93087  MUV
-   41127  HIV
?   11908  PDBBind
X    1513  BACE
X    2039  BBBP
X    7831  Tox21
X    8575  ToxCast
X    1427  SIDER
X    1478  ClinTox
"""

dataset = dc.molnet.load_sider(featurizer=dc.feat.DummyFeaturizer(), splitter=None)[1][0]
df = dataset.to_dataframe()
df.rename(columns=dict([(f"y{i + 1}", task) for i, task in enumerate(dataset.tasks)] + [("X", "SMILES")]), inplace=True)
df["SMILES"] = df["SMILES"].apply(lambda mol: Chem.MolToSmiles(Chem.rdmolops.RemoveHs(Chem.MolFromSmiles(mol))))
df["ID"] = [f"Comp{i + 1:06d}" for i in range(len(df))]
df = df[["ID", "SMILES"] + dataset.tasks.tolist()]

e_splits, f_splits, inter_splits = datasail(
    techniques=["ICSe", "CCSe"],
    splits=[7, 2, 1],
    names=["train", "val", "test"],
    runs=1,
    solver="MOSEK",
    e_type="M",
    e_data=dict(df[["ID", "SMILES"]].values.tolist())
)
print(len(e_splits))
print(e_splits[0].keys())
