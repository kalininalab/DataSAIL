import os
from pathlib import Path

import deepchem as dc
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from datasail.sail import datasail

threetoone = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def load_pdbbind(set_name="general"):
    dataset = dc.molnet.load_pdbbind(featurizer=dc.feat.DummyFeaturizer(), splitter=None, set_name=set_name)
    df = dataset[1][0].to_dataframe()
    df.rename(columns={"X1": "Ligand", "X2": "Target"}, inplace=True)
    df = df[["ids", "Ligand", "Target", "y"]]
    df["Ligand"] = df["Ligand"].apply(mol2smiles)
    df["Target"] = df["Target"].apply(pdb_to_sequence)
    df.dropna(inplace=True)
    df.set_index("ids", inplace=True)
    df["ids"] = df.index
    return df


def mol2smiles(x):
    try:
        mol = Chem.MolFromMol2File(x.replace(".sdf", ".mol2"))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except:
        return None


def pdb_to_sequence(pdb_filename: str):
    try:
        pdb_filename = pdb_filename.replace("pocket", "protein")
        sequences = {}
        with open(pdb_filename, "r") as data:
            for line in data.readlines():
                if line[:4] == "ATOM" and line[12:16].strip() == "CA":
                    res = line[20:22].strip()
                    if res not in sequences:
                        sequences[res] = ""
                    sequences[res] += threetoone[line[17:20].strip()]
        longest = max(sequences.items(), key=lambda x: len(x[1]))
        return longest[1]
    except:
        return None


def split_to_dataset(df, assignment, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    train = df["ids"].apply(lambda x: assignment.get((x, x), "") == "train")
    test = df["ids"].apply(lambda x: assignment.get((x, x), "") == "test")
    df[train].to_csv(target_dir / "train.csv")
    df[test].to_csv(target_dir / "test.csv")


def main():
    df = load_lp_pdbbind()

    e_splits, f_splits, inter_splits = datasail(
        techniques=["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"],
        splits=[8, 2],
        names=["train", "test"],
        runs=1,
        solver="SCIP",
        inter=[(x[0], x[0]) for x in df[["ids"]].values.tolist()],
        e_type="M",
        e_data=dict(df[["ids", "Ligand"]].values.tolist()),
        f_type="P",
        f_data=dict(df[["ids", "Target"]].values.tolist()),
        f_sim="mmseqs",
        verbose="I",
        max_sec=1000,
        epsilon=0.05,
    )

    for technique in inter_splits:
        for run in range(len(inter_splits[technique])):
            split_to_dataset(df, inter_splits[technique][run], Path("experiments") / "PDBBind" / "lppdbbind" / technique / f"split_{run}")


def load_lp_pdbbind():
    df = pd.read_csv("/home/rjo21/Downloads/LP_PDBBind.csv")
    df.rename(columns={"Unnamed: 0": "ids", "smiles": "Ligand", "seq": "Target", "value": "y"}, inplace=True)
    df = df[["ids", "Ligand", "Target", "y"]]
    df.dropna(inplace=True)
    return df


def random(path):
    os.makedirs(path, exist_ok=True)
    # df = load_pdbbind("refined")
    df = load_lp_pdbbind()
    train_mask = np.random.choice([True, False], len(df), replace=True, p=[0.8, 0.2])
    test_mask = ~train_mask
    df[train_mask].to_csv(path / "train.csv", index=False)
    df[test_mask].to_csv(path / "test.csv", index=False)
    df["split"] = ["train" if x else "test" for x in train_mask]
    df.rename(columns={"Ligand": "ligands", "Target": "proteins", "y": "affinity"}, inplace=True)
    df = df[df.apply(lambda x: len(x["ligands"]) <= 200 and len(x["proteins"]) <= 2000, axis=1)]
    df.to_csv(path / "lp_pdbbind.csv", index=False)


if __name__ == '__main__':
    # extract(pickle.load(open("backup_scip.pkl", "rb"))[-1])
    main()
    # random(Path("experiments") / "PDBBind" / "random_lp")
