import os.path
from pathlib import Path

import numpy as np
import pandas as pd
from DeepPurpose import DTI as models
from DeepPurpose.utils import data_process, generate_config

from experiments.utils import RUNS, telegram

count = 0


def train_model(folder, technique, run):
    target = "results_DDTA"
    if not os.path.exists(folder / target) or True:
        drug_encoding, target_encoding = "CNN", "CNN"
        train = pd.read_csv(folder / "train.csv")
        test = pd.read_csv(folder / "test.csv")
        train, _, _ = data_process(
            X_drug=list(train["Ligand"].values),
            X_target=list(train["Target"].values),
            y=list(train["y"].values),
            drug_encoding=drug_encoding,
            target_encoding=target_encoding,
            split_method="random",
            frac=[1, 0, 0],
        )
        test, _, _ = data_process(
            X_drug=list(test["Ligand"].values),
            X_target=list(test["Ligand"].values),
            y=list(test["y"].values),
            drug_encoding=drug_encoding,
            target_encoding=target_encoding,
            split_method="random",
            frac=[1, 0, 0],
        )

        config = generate_config(
            drug_encoding=drug_encoding,
            target_encoding=target_encoding,
            result_folder=folder / target,
            batch_size=256,
            train_epoch=100,

            # DeepDTA
            cls_hidden_dims=[1024, 1024, 512],
            cnn_drug_filters=[32, 64, 96],
            cnn_target_filters=[32, 64, 96],
            cnn_drug_kernels=[4, 6, 8],
            cnn_target_kernels=[4, 8, 12],
            LR=0.001,

            # transformer_emb_size_drug=64,
            # transformer_intermediate_size_drug=256,
            # transformer_num_attention_heads_drug=4,
            # transformer_n_layer_drug=2,
            # cls_hidden_dims=[1024, 256],
            # mlp_hidden_dims_drug=[1024, 128],
            # mlp_hidden_dims_target=[1024, 128],
            # input_dim_protein=1024,
        )
        net = models.model_initialize(**config)
        model_parameters = filter(lambda p: p.requires_grad, net.model.parameters())
        print("Number of parameters:", sum([np.prod(p.size()) for p in model_parameters]))
        # net.train(train, train, train)
        net.train(train, test, test)
        del train
        del test
        del config
        del net
        del model_parameters
    global count
    count += 1
    # telegram(f"[PDB {count} / 35] Training finished for PDBBind - {technique} - Run {run + 1}/5")


def read_val_rmse(folder):
    output = []
    with open(folder / "results" / "valid_markdowntable.txt", "r") as table:
        for line in table.readlines()[3:]:
            if line[0] == "|":
                output.append(np.sqrt(line.split("|")[2].strip()))


def main():
    for technique in ["R", "ICSe", "ICSf", "ICD", "CCSe", "CCSf", "CCD"]:
        for run in range(RUNS):
            print(f"Train {technique} - {run}")
            train_model(Path("experiments") / "PDBBind" / "data_scip_improved" / technique / f"split_{run}", technique, run)


if __name__ == '__main__':
    train_model(Path("experiments") / "PDBBind" / "random_lp", "R", 0)
    # main()
