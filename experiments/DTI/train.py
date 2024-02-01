import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import mean_squared_error

from experiments.utils import RUNS, embed_aaseqs, telegram, models
from experiments.DTI.lppdbbind.model_retraining.deepdta.retrain_script import train_deepdta as deepdta

count = 0
total_number = 5 * 14


def message(tool, algo, tech):
    global count
    count += 1
    telegram(f"[Training {count}/{total_number}] {tool.split('_')[0]} - DTI - {algo} - {tech}")


def embed_smiles(smile, drug_embeds, n_bits=480):
    try:
        if smile not in drug_embeds:
            if smile != smile or isinstance(smile, float) or len(smile) == 0:
                drug_embeds[smile] = None
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                drug_embeds[smile] = None
            drug_embeds[smile] = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
    except:
        drug_embeds[smile] = None
    return drug_embeds[smile]


def embed_sequence(aaseq, prot_embeds):
    amino_acids_pattern = re.compile('[^ACDEFGHIKLMNPQRSTVWY]')
    aaseq = amino_acids_pattern.sub('G', aaseq)[:1022]
    if aaseq not in prot_embeds:
        prot_embeds[aaseq] = embed_aaseqs(aaseq)
    return list(prot_embeds[aaseq])


def prepare_sl_data(split_path, data_path) -> pd.DataFrame:
    if (prot_path := data_path / "prot_embeds.pkl").exists():
        with open(prot_path, "rb") as prots:
            prot_embeds = pickle.load(prots)
    else:
        prot_embeds = {}

    if (drug_path := data_path / "drug_embeds.pkl").exists():
        with open(drug_path, "rb") as drugs:
            drug_embeds = pickle.load(drugs)
    else:
        drug_embeds = {}

    df = pd.read_csv(split_path)
    df["seq_feat"] = df["Target"].apply(lambda x: embed_sequence(x, prot_embeds))
    df["smiles_feat"] = df["Ligand"].apply(lambda x: embed_smiles(x, drug_embeds))
    df["feat"] = df[['seq_feat', 'smiles_feat']].apply(lambda x: x["seq_feat"] + x["smiles_feat"], axis=1)

    with open(prot_path, "wb") as prots:
        pickle.dump(prot_embeds, prots)
    with open(drug_path, "wb") as drugs:
        pickle.dump(drug_embeds, drugs)

    df.dropna(inplace=True)

    return df


def train_sl_models(full_path: Path, model, tool, techniques):
    perf = {}
    for tech in techniques:
        for run in range(RUNS):
            print(tool, model, tech, run, "Start", sep=" - ")
            root = full_path / tool / tech / f"split_{run}"
            train_df = prepare_sl_data(root / "train.csv", full_path / "data")
            test_df = prepare_sl_data(root / "test.csv", full_path / "data")
            X_train = np.array([np.array(x) for x in train_df["feat"].values])
            y_train = train_df["y"].to_numpy().reshape(-1, 1)
            X_test = np.array([np.array(x) for x in test_df["feat"].values])
            y_test = test_df["y"].to_numpy().reshape(-1, 1)

            if model.startswith("rf"):
                y_train = y_train.squeeze()
                y_test = y_test.squeeze()
                X_train = X_train.squeeze()
                X_test = X_test.squeeze()

            m = models[f"{model}-r"]
            m.fit(X_train, y_train)

            test_predictions = m.predict(X_test)
            test_perf = mean_squared_error(y_test, test_predictions, squared=False)

            perf[f"{tech}_{run}"] = test_perf
        message(tool, model.upper(), tech)
    pd.DataFrame(list(perf.items()), columns=["Name", "Perf"]).to_csv(full_path / tool / f"{model}.csv", index=False)


def train_deepdta(full_path: Path, tool, techniques):
    perf = {}
    for tech in techniques:
        for run in range(RUNS):
            split_path = full_path / tool / tech / f"split_{run}"
            deepdta(
                (split_path / "train.csv", split_path / "test.csv"),
                split_path / "results",
                split_names=("train", "test"),
                column_map={"Target": "proteins", "Ligand": "ligands", "y": "affinity"},
            )

            df = pd.read_csv(split_path / "results" / "test_predictions.csv")
            labels, pred = df["Label"], df["Pred"]
            perf[f"{tech}_{run}"] = mean_squared_error(labels, pred, squared=False)
        message(tool, "deepdta", tech)
    pd.DataFrame(list(perf.items()), columns=["Name", "Perf"]).to_csv(full_path / tool / f"deepdta.csv", index=False)


def main(full_path):
    (full_path / "data").mkdir(exist_ok=True, parents=True)
    for tool, techniques in [
        ("datasail", ["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"]),
        ("deepchem", ["Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"]),
        ("lohi", ["lohi"]),
        ("graphpart", ["graphpart"])
    ]:
        try:
            train_deepdta(full_path, tool, techniques)
        except Exception as e:
            print("EXCEPTION", tool, "deepdta", sep=" - ")
            print(e)
            print("END")

        # for model in ["rf", "svm", "xgb", "mlp"]:
        #     try:
        #         train_sl_models(full_path, model, tool, techniques)
        #     except Exception as e:
        #         print("EXCEPTION", tool, model, sep=" - ")
        #         print(e)
        #         print("END")
    telegram("Finished DTI training")


if __name__ == '__main__':
    # train_deepdta(Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "DataSAIL" / "v03", "STEPc", ["R"])
    main(Path(sys.argv[1]))
