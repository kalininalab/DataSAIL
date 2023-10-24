import os
import subprocess
import sys

import deepchem as dc
import esm
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

RUNS = 5
MPP_EPOCHS = 50
USE_UMAP = False  # if False uses tSNE

mpp_datasets = {
    "qm7": [dc.molnet.load_qm7, "regression", "mae", 7160],
    "qm8": [dc.molnet.load_qm8, "regression", "mae", 21786],
    "qm9": [dc.molnet.load_qm9, "regression", "mae", 133885],
    "esol": [dc.molnet.load_delaney, "regression", "rmse", 1128],
    "freesolv": [dc.molnet.load_freesolv, "regression", "rmse", 642],
    "lipophilicity": [dc.molnet.load_lipo, "regression", "rmse", 4200],
    "pcba": [dc.molnet.load_pcba, "classification", "prc-auc", 327929],
    "muv": [dc.molnet.load_muv, "classification", "prc-auc", 93087],
    "hiv": [dc.molnet.load_hiv, "classification", "auc", 41127],
    "bace": [dc.molnet.load_bace_classification, "classification", "auc", 1513],
    "bbbp": [dc.molnet.load_bbbp, "classification", "auc", 2039],
    "tox21": [dc.molnet.load_tox21, "classification", "auc", 7831],
    "toxcast": [dc.molnet.load_toxcast, "classification", "auc", 8575],
    "sider": [dc.molnet.load_sider, "classification", "auc", 1427],
    "clintox": [dc.molnet.load_clintox, "classification", "auc", 1478],
}


splitters = {
    "Scaffold": dc.splits.ScaffoldSplitter(),
    "Weight": dc.splits.MolecularWeightSplitter(),
    "MinMax": dc.splits.MaxMinSplitter(),
    "Butina": dc.splits.ButinaSplitter(),
    "Fingerprint": dc.splits.FingerprintSplitter(),
}


num_layers = 12
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()


def embed_smiles(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        print("Failed")
        return [0] * 1024
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))


def embed_aaseqs(aaseq):
    batch_labels, batch_strs, batch_tokens = batch_converter([("query", aaseq)])
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[num_layers], return_contacts=True)
        token_representations = results["representations"][num_layers]

        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
        return sequence_representations[0].numpy()


def get_bounds(values, axis=0):
    mean = np.mean(values, axis=axis)
    return mean - np.std(values, axis=axis), mean + np.std(values, axis=axis)


def dc2pd(ds, ds_name):
    df = ds.to_dataframe()
    name_map = dict([(f"y{i + 1}", task) for i, task in enumerate(ds.tasks)] + [("y", ds.tasks[0]), ("X", "SMILES")])
    df.rename(columns=name_map, inplace=True)
    df["ID"] = [f"Comp{i + 1:06d}" for i in range(len(df))]
    if ds_name in ["qm7", "qm8", "qm9"]:
        df["SMILES"] = df["SMILES"].apply(lambda mol: Chem.MolToSmiles(Chem.rdmolops.RemoveHs(mol)))
    if mpp_datasets[ds_name][1][0] == "classification":
        df[ds.tasks.tolist()] = pd.to_numeric(df[ds.tasks.tolist()], downcast="integer")
        return df[["ID", "SMILES", "w"] + ds.tasks.tolist()]
    return df[["ID", "SMILES"] + ds.tasks.tolist()]


def telegram(message: str = "Hello World"):
    chat_id = "694905585"
    bot_id = "1141416729:AAFhKaONIFu3keTB6mjLfYEX_HtaYQDLLiY"
    try:
        subprocess.call([
            'curl',
            '--data', 'parse_mode=HTML',
            '--data', f'chat_id={chat_id}',
            '--data', f'text={message}',
            '--request', 'POST',
            f'https://api.telegram.org/bot{bot_id}/sendMessage'
        ], stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT)
    except Exception as e:
        print("Telegram notification failed. Error Message:", str(e), file=sys.stderr)
