import os
import sys
from pathlib import Path

import chemprop
import numpy as np
from chemprop.train.metrics import prc_auc
import deepchem as dc
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVR, LinearSVC
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from experiments.MPP.split import prep_biogen
from experiments.utils import mpp_datasets, RUNS, MPP_EPOCHS, telegram, dc2pd, biogen_datasets, embed_smiles


models = {
    "rf-r": RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42),
    "svm-r": MultiOutputRegressor(LinearSVR(random_state=42)),
    "xgb-r": MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
    "mlp-r": MLPRegressor(hidden_layer_sizes=(512, 256, 64), random_state=42, max_iter=4 * MPP_EPOCHS),
    "rf-c": RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42),
    "svm-c": MultiOutputClassifier(LinearSVC(random_state=42)),
    "xgb-c": MultiOutputClassifier(GradientBoostingClassifier(random_state=42)),
    "mlp-c": MLPClassifier(hidden_layer_sizes=(512, 256, 64), random_state=42, max_iter=4 * MPP_EPOCHS),
}
metric = {
    "mae": mean_absolute_error,
    "rmse": lambda pred, truth: mean_squared_error(pred, truth, squared=False),
    "prc-auc": prc_auc,
    "auc": roc_auc_score,
}

count = 0
total_number = 3 * 4 * 14


def message(tool, name, algo, tech):
    global count
    count += 1
    telegram(f"[Training {count}/{total_number}] {tool.split('_')[0]} - {name} - {algo.upper()} - {tech}")


def clean_dfs(path):
    train_df = pd.read_csv(path / "train.csv")
    test_df = pd.read_csv(path / "test.csv")
    train_nunique = train_df.nunique()
    test_nunique = test_df.nunique()
    train_dropable = train_nunique[train_nunique == 1].index
    test_dropable = test_nunique[test_nunique == 1].index
    train_df.drop(train_dropable, axis=1, inplace=True)
    test_df.drop(train_dropable, axis=1, inplace=True)
    train_df.drop(test_dropable, axis=1, inplace=True)
    test_df.drop(test_dropable, axis=1, inplace=True)
    train_df = train_df.loc[:, ~train_df.columns.duplicated()]
    test_df = test_df.loc[:, ~test_df.columns.duplicated()]
    train_df.to_csv(path / "train.csv", index=False)
    test_df.to_csv(path / "test.csv", index=False)


def train_chemprop(tool, name, techniques):
    dfs = {"val": pd.DataFrame({"rows": list(range(RUNS))}), "test": pd.DataFrame({"rows": [0]})}
    # store the results in training, validation, and test files
    base = Path("experiments") / "MPP" / tool / name
    for tech in techniques:
        for run in range(RUNS):
            try:
                path = base / tech / f"split_{run}"
                clean_dfs(path)

                # train the D-MPNN model
                targets = [x for x in pd.read_csv(path / "train.csv").columns if x not in ["SMILES", "ECFP4", "ID"]]
                arguments = [
                    "--data_path", str(path / "train.csv"),
                    "--separate_val_path", str(path / "test.csv"),
                    "--separate_test_path", str(path / "train.csv"),
                    "--dataset_type", mpp_datasets[name][1],
                    "--save_dir", str(path),
                    "--quiet", "--epochs", str(MPP_EPOCHS),
                    "--smiles_columns", "SMILES",
                    "--target_columns", *targets,
                    "--metric", mpp_datasets[name][2],
                ]
                args = chemprop.args.TrainArgs().parse_args(arguments)
                chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
                del targets
                del arguments
                del args

                # extract the data and save them in a CSV file
                tb_path = Path("experiments") / "MPP" / tool / name / tech / f"split_{run}" / "fold_0" / "model_0"
                tb_file = tb_path / list(sorted(filter(
                    lambda x: x.startswith("events"), os.listdir(tb_path)
                )))[-1]
                ea = EventAccumulator(str(tb_file))
                ea.Reload()
                for long, short in [("validation_", "val"), ("test_", "test")]:
                    for metric in filter(lambda x: x.startswith(long), ea.Tags()["scalars"]):
                        dfs[short][f"{tech}_{metric}_split_{run}"] = [e.value for e in ea.Scalars(metric)]
                del tb_file
                del ea
            except Exception as e:
                print(e)
            message(tool, name, "mpnn", tech)
    for split, df in dfs.items():
        save_path = Path("experiments") / "MPP" / tool / name / f"{split}_metrics.tsv"
        print("Saving:", df.shape, "to", save_path)
        df.to_csv(save_path, sep="\t", index=False)


def prepare_sl_data(name):
    if name in biogen_datasets:
        return prep_biogen(name)
    data_path = Path("experiments") / "MPP" / "data" / f"{name}.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    dataset = mpp_datasets[name][0](featurizer=dc.feat.DummyFeaturizer(), splitter=None)[1][0]
    df = dc2pd(dataset, name)
    df["ECFP4"] = df["SMILES"].apply(lambda x: embed_smiles(x))
    df.dropna(inplace=True)
    df.to_csv(data_path, index=False)
    return pd.read_csv(data_path)


def train_sl_models(model, tool, name, techniques):
    df = prepare_sl_data(name)
    perf = {}
    for tech in techniques:
        for run in range(RUNS):
            targets = [x for x in df.columns if x not in ["SMILES", "ECFP4", "ID"]]
            root = Path("experiments") / "MPP" / tool / name / tech / f"split_{run}"
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

            if isinstance(test_predictions, list):
                test_perf = np.mean([metric[mpp_datasets[name][2]](y_test[:, i], test_predictions[i][:, 1]) for i in range(len(test_predictions))])
            else:
                test_perf = metric[mpp_datasets[name][2]](y_test, test_predictions)

            perf[f"{tech}_{run}"] = test_perf
            print(tool, name, model[:-2], tech, run, test_perf, sep=" - ")
        message(tool, name, model[:-2], tech)
    pd.DataFrame(perf, index=[0]).to_csv(Path("experiments") / "MPP" / tool / name / f"{model}.csv", index=False)


def train_all():
    for tool, techniques in [
        ("datasail_old", ["I1e", "C1e"]),
        ("deepchem_old", "Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"),
        ("lohi_old", ["lohi"])
    ]:
        for name in mpp_datasets:
            # train_chemprop(tool, name, techniques)
            for model in ["rf", "svm", "xgb", "mlp"]:
                try:
                    train_sl_models(f"{model}-{mpp_datasets[name][1][0]}", tool, name, techniques)
                except Exception as e:
                    print("EXCEPTION", tool, model, name, sep=" - ")
                    print(e)
                    print("END")


def main():
    if len(sys.argv) == 1:
        train_all()
    else:
        for tool, techniques in [
            ("datasail", ["I1e", "C1e"]),
            ("deepchem", ["Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"]),
            ("lohi", ["lohi"])
        ]:
            train_chemprop(tool, sys.argv[1], techniques)
            for model in ["rf", "svm", "xgb", "mlp"]:
                print(tool, "-", model, "-", sys.argv[1])
                train_sl_models(f"{model}-{mpp_datasets[sys.argv[1]][1][0]}", tool, sys.argv[1], techniques)


if __name__ == '__main__':
    main()
