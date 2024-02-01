import os
import pickle
import sys
from pathlib import Path

import chemprop
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from experiments.DTI.train import embed_smiles
from experiments.utils import mpp_datasets, RUNS, MPP_EPOCHS, telegram, metric, models

count = 0
total_number = 5 * 8 * 14


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


def train_chemprop(full_path, tool, name, techniques):
    dfs = {"val": {}, "test": {}}
    # store the results in training, validation, and test files
    base = full_path / tool / name
    # if (base / f"val_metrics.tsv").exists():
    #     telegram(f"Skipped - {tool} - {name} - D-MPNN")
    #     return
    for tech in techniques:
        # for run in range(RUNS):
        for run in range(1):
            try:
                path = base / tech / f"split_{run}"
                clean_dfs(path)

                # train the D-MPNN model
                targets = [x for x in pd.read_csv(path / "train.csv").columns if x not in ["SMILES", "ECFP4", "ID"]]
                arguments = [
                    "--data_path", str(path / "train.csv"),
                    "--separate_val_path", str(path / "test.csv"),
                    "--separate_test_path", str(path / "test.csv"),
                    "--dataset_type", mpp_datasets[name][1],
                    "--save_dir", str(path),
                    "--quiet", 
                    "--epochs", str(MPP_EPOCHS),
                    # "--epochs", str(3),
                    "--smiles_columns", "SMILES",
                    "--target_columns", *targets,
                    "--metric", mpp_datasets[name][2],
                    "--gpu", "0",
                ]
                args = chemprop.args.TrainArgs().parse_args(arguments)
                chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
                del targets
                del arguments
                del args

                # extract the data and save them in a CSV file
                tb_path = path / "fold_0" / "model_0"
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
    for split, perf in dfs.items():
        save_path = base / f"{split}_metrics.tsv"
        print("Saving to", save_path)
        pd.DataFrame(perf).to_csv(save_path, sep="\t", index=False)


def prepare_sl_data(split_path, data_path, name) -> pd.DataFrame:
    # if (drug_path := data_path / f"drug_embeds_{name}.pkl").exists():
    #     with open(drug_path, "rb") as drugs:
    #         drug_embeds = pickle.load(drugs)
    # else:
    drug_embeds = {}

    df = pd.read_csv(split_path)
    df["feat"] = df["SMILES"].apply(lambda x: embed_smiles(x, drug_embeds, n_bits=1024))
    df.dropna(inplace=True)

    with open(data_path / f"drug_embeds_{name}.pkl", "wb") as drugs:
        pickle.dump(drug_embeds, drugs)

    return df


def train_sl_models(full_path, model, tool, name, techniques):
    try:
        if (full_path / tool / name / f"{model}.csv").exists():
            telegram(f"Skipped - {tool} - {name} - {model}")
            return
        perf = {}
        for tech in techniques:
            for run in range(RUNS):
                print(tool, name, model[:-2], tech, run, "Start", sep=" - ")
                root = full_path / tool / name / tech / f"split_{run}"
                clean_dfs(root)

                train_df = prepare_sl_data(root / "train.csv", full_path / "data", name)
                test_df = prepare_sl_data(root / "test.csv", full_path / "data", name)
                targets = [x for x in train_df.columns if x not in ["SMILES", "feat", "ID"]]
                X_train = np.array([np.array(x) for x in train_df["feat"].to_numpy()])
                y_train = np.array([np.array(x) for x in train_df[targets].to_numpy()])
                X_test = np.array([np.array(x) for x in test_df["feat"].to_numpy()])
                y_test = np.array([np.array(x) for x in test_df[targets].to_numpy()])

                if model.startswith("rf"):
                    y_train = y_train.squeeze()
                    y_test = y_test.squeeze()

                m = models[model]
                m.fit(X_train, y_train)

                if model.endswith("c") and not model.startswith("mlp") and not model.startswith("svm"):
                    test_predictions = m.predict_proba(X_test)
                    if np.array(y_test).shape != np.array(test_predictions).shape:
                        test_predictions = np.array(test_predictions).argmax(axis=-1).T
                else:
                    test_predictions = m.predict(X_test)

                if isinstance(test_predictions, list):
                    test_perf = np.mean([metric[mpp_datasets[name][2]](y_test[:, i], test_predictions[i][:, 1]) for i in
                                         range(len(test_predictions))])
                else:
                    test_perf = metric[mpp_datasets[name][2]](y_test, test_predictions)

                perf[f"{tech}_{run}"] = test_perf
                print(tool, name, model[:-2], tech, run, test_perf, sep=" - ")
            message(tool, name, model[:-2], tech)
        pd.DataFrame.from_dict(perf, orient="index").to_csv(full_path / tool / name / f"{model}.csv", index=False)
    except Exception as e:
        print(e)


def train_sl_models_speed(full_path, model, tool, name, tech, run):
    target_path = full_path / tool / name / f"{model}_{tech}_{run}.txt"
    if target_path.exists():
        with open(target_path, "r") as out:
            if len(out.readlines()[0].strip()) > 2:
                return
    try:
        print(tool, name, model[:-2], tech, run, "Start", sep=" - ")
        root = full_path / tool / name / tech / f"split_{run}"
        clean_dfs(root)

        train_df = prepare_sl_data(root / "train.csv", full_path / "data", name)
        test_df = prepare_sl_data(root / "test.csv", full_path / "data", name)
        targets = [x for x in train_df.columns if x not in ["SMILES", "feat", "ID"]]
        X_train = np.array([np.array(x) for x in train_df["feat"].to_numpy()])
        y_train = np.array([np.array(x) for x in train_df[targets].to_numpy()])
        X_test = np.array([np.array(x) for x in test_df["feat"].to_numpy()])
        y_test = np.array([np.array(x) for x in test_df[targets].to_numpy()])

        if model.startswith("rf"):
            y_train = y_train.squeeze()
            y_test = y_test.squeeze()

        m = models[model]
        m.fit(X_train, y_train)

        if model.endswith("c") and not model.startswith("mlp") and not model.startswith("svm"):
            test_predictions = m.predict_proba(X_test)
            if np.array(y_test).shape != np.array(test_predictions).shape:
                test_predictions = np.array(test_predictions).argmax(axis=-1).T
        else:
            test_predictions = m.predict(X_test)

        if name == "muv":
            test_perf = np.mean([metric[mpp_datasets[name][2]](y_test[:, i], test_predictions[:, i]) for i in range(len(test_predictions[0]))])
        elif isinstance(test_predictions, list):
            test_perf = np.mean([metric[mpp_datasets[name][2]](y_test[:, i], test_predictions[i][:, 1]) for i in
                                 range(len(test_predictions))])
        else:
            test_perf = metric[mpp_datasets[name][2]](y_test, test_predictions)

        with open(target_path, "w") as out:
            print(test_perf, file=out)
    except Exception as e:
        print("EXCEPTION", tool, name, model[:-2], tech, run, sep="-")
        print(e)
        print("END")


def train_chemprop_speed(full_path, tool, name, tech, run):
    try:
        path = full_path / tool / name / tech / f"split_{run}"
        clean_dfs(path)

        # train the D-MPNN model
        targets = [x for x in pd.read_csv(path / "train.csv").columns if x not in ["SMILES", "ECFP4", "ID"]]
        arguments = [
            "--data_path", str(path / "train.csv"),
            "--separate_val_path", str(path / "test.csv"),
            "--separate_test_path", str(path / "test.csv"),
            "--dataset_type", mpp_datasets[name][1],
            "--save_dir", str(path),
            "--quiet",
            "--epochs", str(MPP_EPOCHS),
            "--smiles_columns", "SMILES",
            "--target_columns", *targets,
            "--metric", mpp_datasets[name][2],
            "--gpu", "0",
        ]
        args = chemprop.args.TrainArgs().parse_args(arguments)
        chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
    except Exception as e:
        print("EXCEPTION", tool, name, "D-MPNN", tech, run, sep="-")
        print(e)
        print("END")


def para_sl(full_path):
    tool2tech = {
        "datasail": ["I1e", "C1e"],
        "deepchem": ["Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"],
        "lohi": ["lohi"],
    }
    comb = [
        # ("qm9", "datasail"),
        # ("qm9", "lohi"),
        ("muv", "datasail"),
        ("muv", "deepchem"),
    ]
    fkts = []
    for run in range(RUNS):
        for name, tool in comb:
            for model in ["rf", "svm", "xgb", "mlp"]:
                for tech in tool2tech[tool]:
                    fkts.append((f"{model}-{mpp_datasets[name][1][0]}", tool, name, tech, run))
    fkts = [("rf-c", "datasail", "hiv", "I1e", 0), ("rf-c", "datasail", "hiv", "C1e", 0)] + fkts + [("rf-c", "datasail", "hiv", "I1e", i) for i in range(1, RUNS)] + [("rf-c", "datasail", "hiv", "C1e", i) for i in range(1, RUNS)]
    print("\n".join([str(x) for x in fkts]))
    Parallel(n_jobs=32)(delayed(train_sl_models_speed)(full_path, *fkt) for fkt in fkts)


def para_dl(full_path, tech=None):
    fkts = []
    for run in range(RUNS):
        if tech is None:
            for tech in ["Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"]:
                fkts.append(("deepchem", "muv", tech, run))
        else:
            fkts.append(("deepchem", "muv", tech, run))
    print("\n".join([str(x) for x in fkts]))
    if tech is None:
        Parallel(n_jobs=5)(delayed(train_chemprop_speed)(full_path, *fkt) for fkt in fkts)
    else:
        for fkt in fkts:
            train_chemprop_speed(full_path, *fkt)


def train_dataset(full_path, name):
    for tool, techniques in [
        ("datasail", ["I1e", "C1e"]),
        ("deepchem", ["Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"]),
        ("lohi", ["lohi"]),
    ]:
        try:
            train_chemprop(full_path, tool, name, techniques)
        except Exception as e:
            print("EXCEPTION", tool, name, sep=" - ")
            print(e)
            print("END")

        # for model in ["rf", "svm", "xgb", "mlp"]:
        #     try:
        #         train_sl_models(full_path, f"{model}-{mpp_datasets[name][1][0]}", tool, name, techniques)
        #     except Exception as e:
        #         print("EXCEPTION", tool, model, name, sep=" - ")
        #         print(e)
        #         print("END")
    telegram(f"Finished MPP training {name}")


def read_from_tf(full_path, tool, name, techniques):
    dfs = {"val": {}, "test": {}}
    base = full_path / tool / name
    for tech in techniques:
        for run in range(RUNS):
            path = base / tech / f"split_{run}"
            # extract the data and save them in a CSV file
            tb_path = path / "fold_0" / "model_0"
            tb_file = tb_path / list(sorted(filter(
                lambda x: x.startswith("events"), os.listdir(tb_path)
            ), key=lambda x: os.path.getsize(Path(tb_path) / x)))[-1]
            ea = EventAccumulator(str(tb_file))
            ea.Reload()
            for long, short in [("validation_", "val"), ("test_", "test")]:
                for metric in filter(lambda x: x.startswith(long), ea.Tags()["scalars"]):
                    dfs[short][f"{tech}_{metric}_split_{run}"] = [e.value for e in ea.Scalars(metric)]
    for split, perf in dfs.items():
        save_path = base / f"{split}_metrics.tsv"
        print("Saving to", save_path)
        pd.DataFrame(perf).to_csv(save_path, sep="\t", index=False)
        # print(pd.DataFrame(perf))


def main(full_path):
    (full_path / "data").mkdir(exist_ok=True, parents=True)
    # NAMES = ["qm7", "qm8", "qm9", "esol", "freesolv", "lipophilicity", "muv", "hiv", "bace", "bbbp", "tox21", "toxcast", "clintox"]  # "sider",
    # Parallel(n_jobs=12)(delayed(train_dataset)(full_path, name) for name in NAMES)
    # train_chemprop(full_path, "deepchem", "lipophilicity", ["Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"])
    # train_chemprop(full_path, "datasail", "esol", ["I1e", "C1e"])
    # train_dataset(full_path, "sider")


def brute_force(full_path, model, name, tech):
    model = f"{model}-{mpp_datasets[name][1][0]}"
    for run in range(RUNS):
        try:
            train_sl_models_speed(full_path, model, "deepchem", name, tech, run)
            telegram(f"Finished! {name} - {model} - {tech} - {run}")
            return
        except Exception as e:
            telegram(f"EXCEPTION {name} - {model} - {tech} - {run}\n{e}")


if __name__ == '__main__':
    # train_chemprop_speed(Path(sys.argv[1]), "deepchem", "qm8", "MaxMin", 0)  # jerry
    # train_chemprop_speed(Path(sys.argv[1]), "deepchem", "qm9", "MaxMin", 0)  # minnie
    # train_chemprop_speed(Path(sys.argv[1]), "deepchem", "hiv", "Butina", 0)
    # train_chemprop_speed(Path(sys.argv[1]), "deepchem", "hiv", "Fingerprint", 0)
    # train_chemprop_speed(Path(sys.argv[1]), "deepchem", "hiv", "MaxMin", 0)
    # train_chemprop_speed(Path(sys.argv[1]), "deepchem", "hiv", "Scaffold", 0)
    # train_chemprop_speed(Path(sys.argv[1]), "deepchem", "hiv", "Weight", 0)

    brute_force(full_path=Path(sys.argv[1]), model=sys.argv[2], name=sys.argv[3], tech=sys.argv[4])
    exit(0)
    # train_chemprop(Path(sys.argv[1]), "deepchem", "qm8", ["MaxMin"])
    # train_chemprop(Path(sys.argv[1]), "deepchem", "qm9", ["MaxMin"])
    # train_sl_models_speed(Path(sys.argv[1]), "mlp-c", "datasail", "muv", "I1e", 0)
    # exit(0)
    # read_from_tf(Path(sys.argv[1]), "datasail", "qm9", ["I1e", "C1e"])
    # train_chemprop(Path(sys.argv[1]), "datasail", "qm8", ["I1e", "C1e"])
    if sys.argv[2] == "jerry":
        train_chemprop(Path(sys.argv[1]), "lohi", "qm9", ["lohi"])
    elif sys.argv[2] == "minnie":
        para_sl(Path(sys.argv[1]))
    elif sys.argv[2] in ["Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"]:
        para_dl(Path(sys.argv[1]), sys.argv[2])
    elif sys.argv[2] == "mickey":
        para_dl(Path(sys.argv[1]))
    # if sys.argv[2] == "sl":
    #     run_sl(Path(sys.argv[1]))
    # else:
    #     run_dl(Path(sys.argv[1]))
