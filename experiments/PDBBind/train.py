import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import mean_squared_error

from experiments.MPP.train import models, message
from experiments.utils import RUNS, embed_aaseqs, telegram

count = 0
prot_embeds = {}


def embed_smiles(smile):
    if smile != smile or isinstance(smile, float) or len(smile) == 0:
        return None
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=480))


def embed_sequence(aaseq):
    global prot_embeds
    amino_acids_pattern = re.compile('[^ACDEFGHIKLMNPQRSTVWY]')
    aaseq = amino_acids_pattern.sub('G', aaseq)[:1022]
    if aaseq not in prot_embeds:
        print("Compute new ESM embed")
        prot_embeds[aaseq] = embed_aaseqs(aaseq)
    return list(prot_embeds[aaseq])


def prepare_sl_data():
    global prot_embeds

    root = Path("..") / "DataSAIL" / "experiments" / "PDBBind" / "data"
    root.mkdir(parents=True, exist_ok=True)

    data_path = root / "lppdbbind.csv"
    if data_path.exists():
        return pd.read_csv(data_path)

    df = pd.read_csv(Path("..") / "DataSAIL" / "experiments" / "PDBBind" / "LP_PDBBind.csv")
    df.rename({"Unnamed: 0": "ids"}, axis=1, inplace=True)
    df = df[["ids", "smiles", "seq", "value"]]

    embeds_path = Path("..") / "DataSAIL" / "experiments" / "PDBBind" / "data" / "prot_embeds_esm2_t12.pkl"
    if embeds_path.exists():
        with open(embeds_path, "rb") as f:
            prot_embeds = pickle.load(f)
    df["seq_feat"] = df["seq"].apply(lambda x: embed_sequence(x))
    with open(embeds_path, "wb") as f:
        pickle.dump(prot_embeds, f)

    df["smiles_feat"] = df["smiles"].apply(lambda x: embed_smiles(x))
    df.dropna(inplace=True)

    df["feat"] = df[['seq_feat', 'smiles_feat']].apply(lambda x: x["seq_feat"] + x["smiles_feat"], axis=1)
    df.to_csv(data_path, index=False)

    return pd.read_csv(data_path)


def train_sl_models(model, tool, techniques):
    df = prepare_sl_data()
    perf = {}
    for tech in techniques:
        for run in range(RUNS):
            print(tool, model, tech, run, "Start", sep=" - ")
            root = Path("..") / "DataSAIL" / "experiments" / "PDBBind" / tool / tech / f"split_{run}"
            X_train = np.array([rec[1:-1].split(", ") for rec in df[df["ids"].isin(pd.read_csv(root / "train.csv")["ids"])]["feat"].values], dtype=float)
            y_train = df[df["ids"].isin(pd.read_csv(root / "train.csv")["ids"])]["value"].to_numpy().reshape(-1, 1)
            X_test = np.array([rec[1:-1].split(", ") for rec in df[df["ids"].isin(pd.read_csv(root / "test.csv")["ids"])]["feat"].values], dtype=float)
            y_test = df[df["ids"].isin(pd.read_csv(root / "test.csv")["ids"])]["value"].to_numpy().reshape(-1, 1)

            if model.startswith("rf"):
                y_train = y_train.squeeze()
                y_test = y_test.squeeze()
                X_train = X_train.squeeze()
                X_test = X_test.squeeze()

            m = models[f"{model}-r"]
            m.fit(X_train, y_train)

            test_predictions = m.predict(X_test)
            test_perf = mean_squared_error(y_test, test_predictions)

            perf[f"{tech}_{run}"] = test_perf
            print(tool, model, tech, run, test_perf, sep=" - ")
            message(tool, "PDBBind", model, tech)
    pd.DataFrame(list(perf.items()), columns=["Name", "Perf"]).to_csv(Path("experiments") / "PDBBind" / f"{model}.csv", index=False)


def main():
    # for tool, techniques in [
    #     ("datasail", ["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"]),
    #     ("deepchem", ["Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"]),
    #     ("lohi", ["lohi"]),
    #     ("graphpart", ["graphpart"])
    # ]:
    #     for tech in techniques:
    for model in ["rf", "svm", "xgb", "mlp"]:
        try:
            train_sl_models(model, "datasail", ["I2"])
        except Exception as e:
            print("EXCEPTION", "datasail", model, "I2", sep=" - ")
            print(e)
            print("END")


if __name__ == '__main__':
    vals = [
        ("rf", "datasail", ["I2"]),
        ("xgb", "datasail", ["I2"]),
        ("mlp", "datasail", ["I2"])
    ]
    Parallel(n_jobs=3)(delayed(train_sl_models)(*args) for args in vals)
    # main()
