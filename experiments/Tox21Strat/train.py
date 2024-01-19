import os
from pathlib import Path

import chemprop
import deepchem as dc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from experiments.utils import RUNS, MPP_EPOCHS, telegram, embed_smiles, dc2pd

models = {
    "rf": RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42),
    "svm": MultiOutputClassifier(LinearSVC(random_state=42)),
    "xgb": MultiOutputClassifier(GradientBoostingClassifier(random_state=42)),
    "mlp": MLPClassifier(hidden_layer_sizes=(512, 256, 64), random_state=42, max_iter=4 * MPP_EPOCHS),
}

count = 0


def message(tool, algo, run):
    global count
    count += 1
    telegram(f"[Training {count}/10] {tool} - Tox21Strat - {algo.upper()} - {run + 1}/5")


def train_chemprop(tool):
    dfs = {"val": pd.DataFrame({"rows": list(range(MPP_EPOCHS))}), "test": pd.DataFrame({"rows": [0]})}
    # store the results in training, validation, and test files
    base = Path("experiments") / "Tox21Strat" / tool
    for run in range(RUNS):
        try:
            path = base / f"split_{run}"

            # train the D-MPNN model
            arguments = [
                "--data_path", str(path / "train.csv"),
                "--separate_val_path", str(path / "test.csv"),
                "--separate_test_path", str(path / "train.csv"),
                "--dataset_type", "classification",
                "--save_dir", str(path),
                "--quiet", "--epochs", str(MPP_EPOCHS),
                "--smiles_columns", "SMILES",
                "--target_columns", "SR-ARE",
                "--metric", "auc",
            ]
            args = chemprop.args.TrainArgs().parse_args(arguments)
            chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
            del arguments
            del args

            # extract the data and save them in a CSV file
            tb_path = Path("experiments") / "Tox21Strat" / tool / f"split_{run}" / "fold_0" / "model_0"
            tb_file = tb_path / list(sorted(filter(
                lambda x: x.startswith("events"), os.listdir(tb_path)
            )))[-1]
            ea = EventAccumulator(str(tb_file))
            ea.Reload()
            for long, short in [("validation_", "val"), ("test_", "test")]:
                for metric in filter(lambda x: x.startswith(long), ea.Tags()["scalars"]):
                    dfs[short][f"{metric}_split_{run}"] = [e.value for e in ea.Scalars(metric)]
            del tb_file
            del ea
            message(tool, "mpnn", run)
        except Exception as e:
            print("[ERROR]", e)
    for split, df in dfs.items():
        save_path = Path("experiments") / "Tox21Strat" / tool / f"{split}_metrics.tsv"
        print("Saving:", df.shape, "to", save_path)
        df.to_csv(save_path, sep="\t", index=False)


def prepare_sl_data(name):
    data_path = Path("experiments") / "Tox21Strat" / f"{name}.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    dataset = dc.molnet.load_tox21(featurizer=dc.feat.DummyFeaturizer(), splitter=None)[1][0]
    df = dc2pd(dataset, name)
    df["ECFP4"] = df["SMILES"].apply(lambda x: embed_smiles(x))
    df.dropna(inplace=True)
    df.to_csv(data_path, index=False)
    return pd.read_csv(data_path)


def train_sl_models(model, tool):
    df = prepare_sl_data("tox21")
    perf = {}
    for run in range(RUNS):
        targets = [x for x in df.columns if x not in ["SMILES", "ECFP4", "ID"]]
        root = Path("experiments") / "Tox21Strat" / tool / f"split_{run}"
        X_train = np.array([rec[1:-1].split(", ") for rec in df[df["ID"].isin(pd.read_csv(root / "train.csv")["ID"])]["ECFP4"].values], dtype=int)
        y_train = df[df["ID"].isin(pd.read_csv(root / "train.csv")["ID"])][targets].to_numpy()
        X_test = np.array([rec[1:-1].split(", ") for rec in df[df["ID"].isin(pd.read_csv(root / "test.csv")["ID"])]["ECFP4"].values], dtype=int)
        y_test = df[df["ID"].isin(pd.read_csv(root / "test.csv")["ID"])][targets].to_numpy()

        if model.startswith("rf"):
            y_train = y_train.squeeze()
            y_test = y_test.squeeze()

        m = models[model]
        m.fit(X_train, y_train)

        if model.endswith("c") and not model.startswith("mlp") and not model.startswith("svm"):
            test_predictions = m.predict_proba(X_test)
        else:
            test_predictions = m.predict(X_test)

        test_perf = roc_auc_score(y_test, test_predictions)
        perf[f"{run}"] = test_perf
        message(tool, model[:-2], run)
    pd.DataFrame(perf, index=[0]).to_csv(Path("experiments") / "Tox21Strat" / tool / f"{model}.csv", index=False)


# for splitter in ["datasail", "deepchem"]:
#     train_chemprop(splitter)
#     for model in ["rf", "xgb", "mlp", "svm"]:
#         train_sl_models(model, splitter)

train_chemprop("deepchem")
