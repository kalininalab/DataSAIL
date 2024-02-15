import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import chemprop
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from experiments.utils import DATASETS, RUNS, MPP_EPOCHS, telegram, metric, models, TECHNIQUES, DRUG_TECHNIQUES, \
    embed_smiles


def clean_dfs(path: Path) -> None:
    """
    Clean the dataframes by removing duplicate columns and columns with only one unique value.

    Args:
        path: Path to the folder holding the dataframes
    """
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


def prepare_sl_data(split_path, data_path, name) -> pd.DataFrame:
    """
    Prepare the data for statistical learning.

    Args:
        split_path: Path to the split
        data_path: Path to the folder holding the data for embeddings
        name: Name of the dataset

    Returns:
        pd.DataFrame: Featurized dataframe
    """
    data_path.mkdir(parents=True, exist_ok=True)
    if (drug_path := data_path / f"drug_embeds_{name}.pkl").exists():
        with open(drug_path, "rb") as drugs:
            drug_embeds = pickle.load(drugs)
    else:
        drug_embeds = {}

    df = pd.read_csv(split_path, index_col=0)
    df["feat"] = df["SMILES"].apply(lambda x: embed_smiles(x, drug_embeds, n_bits=1024))
    df.dropna(inplace=True)

    with open(data_path / f"drug_embeds_{name}.pkl", "wb") as drugs:
        pickle.dump(drug_embeds, drugs)

    return df


def train_chemprop_run(base_path, name: str) -> float:
    """
    Train a single run of a split for a given technique and model.

    Args:
        base_path: Path to the folder holding the splits for all runs of this tool and technique
        name: Name of the dataset

    Returns:
        float: Performance of the model
    """
    # train the D-MPNN model
    targets = [x for x in pd.read_csv(base_path / "train.csv").columns if x not in ["SMILES", "ECFP4", "ID"]]
    arguments = [
        "--data_path", str(base_path / "train.csv"),
        "--separate_val_path", str(base_path / "test.csv"),
        "--separate_test_path", str(base_path / "test.csv"),
        "--dataset_type", DATASETS[name][1],
        "--save_dir", str(base_path),
        "--quiet",
        "--epochs", str(MPP_EPOCHS),
        "--smiles_columns", "SMILES",
        "--target_columns", *targets,
        "--metric", DATASETS[name][2],
        "--gpu", "0",
    ]
    args = chemprop.args.TrainArgs().parse_args(arguments)
    chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

    # extract the data and save them in a CSV file
    tb_path = base_path / "fold_0" / "model_0"
    tb_file = tb_path / list(sorted(filter(lambda x: x.startswith("events"), os.listdir(tb_path))))[-1]
    ea = EventAccumulator(str(tb_file))
    ea.Reload()
    perf = [next(iter(ea.Scalars(metric))).value for metric in filter(lambda x: x.startswith("test_"), ea.Tags()["scalars"])]
    if len(perf) > 0:
        return float(np.mean(perf))
    return 0.0


def train_run(run_path: Path, data_path: Path, name: str, model: str) -> float:
    """
    Train a single run of a split for a given technique and model.

    Args:
        run_path: Path to the folder holding the splits for all runs of this tool and technique
        data_path: Path to the data directory
        name: Name of the dataset
        model: Statistical Learning model to fit

    Returns:
        float: Performance of the model
    """
    clean_dfs(run_path)

    if model == "d-mpnn":
        return train_chemprop_run(run_path, name)

    model = model + "-" + DATASETS[name][1][0]
    train_df = prepare_sl_data(run_path / "train.csv", data_path, name)
    test_df = prepare_sl_data(run_path / "test.csv", data_path, name)
    targets = [x for x in train_df.columns if x not in ["SMILES", "feat", "ID"]]
    x_train, y_train = np.stack(train_df["feat"].values), np.stack(train_df[targets].values)
    x_test, y_test = np.stack(test_df["feat"].values), np.stack(test_df[targets].values)

    m = models[model]
    m.fit(x_train, y_train)

    test_predictions = m.predict(x_test)
    test_perf = metric[DATASETS[name][2]](y_test, test_predictions)

    return test_perf


def train_tech(base_path: Path, data_path, model: str, tech: str, name: str) -> dict:
    """
    Train a single model on all splits for a given technique.

    Args:
        base_path: Path to the folder holding the runs for this technique
        data_path: Path to the folder holding the data
        model: Statistical Learning model to fit
        tech: Technique to use for the splits
        name: Name of the dataset

    Returns:
        dict: Dictionary of run -> RMSE of the model on the test set
    """
    perf = {}
    for run in range(RUNS):
        perf[f"{tech}_{run}"] = train_run(base_path / f"split_{run}", data_path, name, model)
    return perf


def train_model(base_path: Path, data_path: Path, model: str, tool: str, name: str) -> pd.DataFrame:
    """
    Train all models for a given tool.

    Args:
        base_path: Path to the folder holding the runs for this tool
        data_path: Path to the folder holding the data
        model: Statistical Learning model to fit
        tool: Name of the tool
        name: Name of the dataset

    Returns:
        pd.DataFrame: Dataframe of the performance of the models
    """
    perf = {}
    for tech in set(TECHNIQUES[tool]).intersection(set(DRUG_TECHNIQUES)):
        perf.update(train_tech(base_path / tech, data_path, model, tech, name))
        # message(tool, name, model[:-2], tech)
    df = pd.DataFrame(list(perf.items()), columns=["name", "perf"])
    df["model"] = model
    df["tool"] = tool
    df["tech"] = df["name"].apply(lambda x: x.split("_")[0])
    df["run"] = df["name"].apply(lambda x: x.split("_")[1])
    df["dataset"] = name
    return df


def train_tool(full_path: Path, tool: str, name: str) -> None:
    """
    Train all models for a given tool.

    Args:
        full_path: Path to the folder holding the runs for this tool
        tool: Name of the tool
        name: Name of the dataset
    """
    dfs = []
    for model in list(set([x[:-2] for x in models.keys()])) + ["d-mpnn"]:
        dfs.append(train_model(full_path / tool / name, full_path / "data", model, tool, name))
    pd.concat(dfs).to_csv(full_path / tool / name / f"results.csv", index=False)


def train_dataset(full_path: Path, name: str) -> None:
    """
    Train all models for a given dataset.

    Args:
        full_path: Path to the folder holding the runs for this dataset
        name: Name of the dataset
    """
    for tool in ["datasail", "deepchem", "lohi"]:
        train_tool(full_path, tool, name)


def train(full_path: Path, name: Optional[str] = None) -> None:
    """
    Train all models for all tools and datasets.

    Args:
        full_path: Path to the folder holding the runs for all tools and datasets
    """
    if name is None:
        for name in DATASETS:
            train_dataset(full_path, name)
    else:
        train_dataset(full_path, name)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        train(Path(sys.argv[1]))
    elif len(sys.argv) == 3:
        train_dataset(Path(sys.argv[1]), sys.argv[2])
