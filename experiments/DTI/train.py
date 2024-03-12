import pickle
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from experiments.utils import RUNS, models, embed_sequence, embed_smiles, TECHNIQUES

try:
    from experiments.DTI.lppdbbind.model_retraining.deepdta.retrain_script import train_deepdta as deepdta
except ImportError as e:
    print("LP-PDBBind is not installed, please clone the repo to use DeepDTA for model training.", file=sys.stderr)
    raise e


def prepare_sl_data(file_path: Path, data_path: Path) -> pd.DataFrame:
    """
    Prepare, i.e., featurize, the data for a single split of the DTI dataset.

    Args:
        file_path: Path to the split file
        data_path: Path to the folder holding the embeddings for proteins and drugs

    Returns:
        pd.DataFrame: Featurized dataframe
    """
    data_path.mkdir(parents=True, exist_ok=True)
    # Load embeddings for proteins
    if (prot_path := data_path / "prot_embeds.pkl").exists():
        with open(prot_path, "rb") as prots:
            prot_embeds = pickle.load(prots)
    else:
        prot_embeds = {}

    # Load embeddings for drugs
    if (drug_path := data_path / "drug_embeds.pkl").exists():
        with open(drug_path, "rb") as drugs:
            drug_embeds = pickle.load(drugs)
    else:
        drug_embeds = {}

    # featurize the dataframe
    df = pd.read_csv(file_path)
    df["seq_feat"] = df["seq"].apply(lambda x: embed_sequence(x, prot_embeds))
    df["smiles_feat"] = df["smiles"].apply(lambda x: embed_smiles(x, drug_embeds))
    df.dropna(inplace=True)
    df["feat"] = df[['seq_feat', 'smiles_feat']].apply(lambda x: np.concatenate([x["seq_feat"], x["smiles_feat"]]), axis=1)

    # Save the embeddings
    with open(prot_path, "wb") as prots:
        pickle.dump(prot_embeds, prots)
    with open(drug_path, "wb") as drugs:
        pickle.dump(drug_embeds, drugs)

    # Clean and return the dataframe
    df.dropna(inplace=True)
    return df


def train_deepdta_run(run_path: Path) -> float:
    """
    Train a single run of a split for a given technique and model.

    Args:
        run_path: Path to the folder holding the splits for all runs of this tool and technique

    Returns:
        float: RMSE of the model on the test set
    """
    deepdta(
        (run_path / "train.csv", run_path / "test.csv"),
        run_path / "results",
        split_names=("train", "test"),
        column_map={"seq": "proteins", "smiles": "ligands", "value": "affinity"},
    )

    df = pd.read_csv(run_path / "results" / "test_predictions.csv")
    labels, pred = df["Label"], df["Pred"]
    print("Trained DeepDTA on", run_path)
    return mean_squared_error(labels, pred, squared=False)


def train_run(run_path: Path, base_path, model: Literal["rf", "svm", "xgb", "mlp"]) -> float:
    """
    Train a single run of a split for a given technique and model.

    Args:
        run_path: Path to the folder holding the splits for all runs of this tool and technique
        base_path: Path to the folder holding the embeddings for proteins and drugs
        model: Statistical Learning model to fit

    Returns:
        float: RMSE of the model on the test set
    """
    if model == "deepdta":
        return train_deepdta_run(run_path)
    train_df = prepare_sl_data(run_path / "train.csv", base_path / "data")
    test_df = prepare_sl_data(run_path / "test.csv", base_path / "data")
    x_train = np.stack(train_df["feat"].values)
    y_train = np.stack(train_df["value"].values).reshape(-1, 1)
    x_test = np.stack(test_df["feat"].values)
    y_test = np.stack(test_df["value"].values).reshape(-1, 1)

    m = models[f"{model}-r"]
    m.fit(x_train, y_train)

    test_predictions = m.predict(x_test)
    print("Trained", model, "on", run_path)
    return mean_squared_error(y_test, test_predictions, squared=False)


def train_tech(base_path: Path, model: Literal["rf", "svm", "xgb", "mlp", "deepdta"], tech: str) -> dict:
    """
    Train a single model on all splits for a given technique.

    Args:
        base_path: Path to the folder holding the runs for this technique
        model: Statistical Learning model to fit
        tech: Technique to use for the splits

    Returns:
        dict: Dictionary of run -> RMSE of the model on the test set
    """
    perf = {}
    for run in range(RUNS):
        perf[f"{tech}_{run}"] = train_run(base_path / tech / f"split_{run}", base_path.parent, model)
    return perf


def train_model(base_path: Path, model: Literal["rf", "svm", "xgb", "mlp", "deepdta"], tool: str) -> pd.DataFrame:
    """
    Train all models for a given tool.

    Args:
        base_path: Path to the folder holding the runs for this tool
        model: Statistical Learning model to fit
        tool: Tool to take techniques from for the splits

    Returns:
        pd.DataFrame: DataFrame of the results
    """
    perf = {}
    for tech in TECHNIQUES[tool]:
        perf.update(train_tech(base_path / tool, model, tech))
    df = pd.DataFrame(list(perf.items()), columns=["name", "perf"])
    df["model"] = model
    df["tool"] = tool
    df["tech"] = df["name"].apply(lambda x: x.split("_")[0])
    df["run"] = df["name"].apply(lambda x: x.split("_")[1])
    return df


def train_tool(full_path: Path, tool: Literal["datasail", "deepchem", "lohi", "graphpart"]) -> None:
    """
    Train all models for a given tool.

    Args:
        full_path: Path to the folder holding the runs for this tool
        tool: Tool to take techniques from for the splits
    """
    dfs = []
    for model in list(set([x[:-2] for x in models.keys()])) + ["deepdta"]:
        dfs.append(train_model(full_path, model, tool))
    pd.concat(dfs, ignore_index=True).to_csv(full_path / f"{tool}.csv", index=False)


def main(full_path: Path):
    """
    Train all models for all tools.

    Args:
        full_path: Path to the folder holding the runs for all tools
    """
    for tool in TECHNIQUES:
        train_tool(full_path, tool)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
