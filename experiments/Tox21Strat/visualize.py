from pathlib import Path

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from experiments.utils import embed_smiles, get_bounds

root = Path("experiments") / "Tox21Strat"


def plot_perf(ax):
    models = ["RF", "SVM", "XGB", "MLP", "D-MPNN"]
    values = [[] for _ in range(2)]

    for s, split in enumerate(["deepchem", "datasail"]):
        for model in models[:-1]:
            df = pd.read_csv(root / split / f"{model.lower()}.csv")
            values[s].append(df.mean(axis=1).values[0])
        values[s].append(pd.read_csv(root / split / "val_metrics.tsv", sep="\t").max(axis=0).values[1:].mean())
    df = pd.DataFrame(np.array(values).T, columns=["Stratified", "DataSAIL"], index=models)
    df.plot.bar(ax=ax, rot=0, ylabel="AUROC")


def plot_dmpnn_perf(ax):
    ds = pd.read_csv(root / "datasail" / "val_metrics.tsv", sep="\t").values[:, 1:].T
    dc = pd.read_csv(root / "deepchem" / "val_metrics.tsv", sep="\t").values[:, 1:].T

    ds_mean = np.mean(ds, axis=0)
    ds_bounds = get_bounds(ds)
    dc_mean = np.mean(dc, axis=0)
    dc_bounds = get_bounds(dc)

    x = np.arange(0.0, 50, 1)
    ax.fill_between(x, *dc_bounds, alpha=0.5)
    ax.plot(dc_mean, label="stratified")
    ax.fill_between(x, *ds_bounds, alpha=0.5)
    ax.plot(ds_mean, label="DataSAIL")
    ax.legend()
    ax.set_title(f"Performance comparison\n(ROC-AUC ↑)")


def embed():
    print("Embedding - read data ...")
    base = Path("experiments") / "Tox21Strat"
    dc_tr = pd.read_csv(base / "deepchem" / "split_0" / "train.csv")
    dc_te = pd.read_csv(base / "deepchem" / "split_0" / "test.csv")
    ds_tr = pd.read_csv(base / "datasail" / "split_0" / "train.csv")
    ds_te = pd.read_csv(base / "datasail" / "split_0" / "test.csv")

    print("Embedding - compute fingerprints ...")
    smiles = [(s, embed_smiles(s)) for s in set(list(dc_tr["SMILES"]) + list(dc_te["SMILES"]) +
                                                list(ds_tr["SMILES"]) + list(ds_te["SMILES"]))]
    ids, fps = zip(*smiles)

    print("Embedding - compute t-SNE ...")
    embedder = TSNE(n_components=2, learning_rate="auto", init="random", random_state=42)
    embeddings = embedder.fit_transform(np.array(fps))

    print("Embedding - relocate samples ...")
    embed_map = {idx: emb for idx, emb in zip(ids, embeddings)}
    return dc_tr["SMILES"].apply(lambda x: embed_map[x]), dc_te["SMILES"].apply(lambda x: embed_map[x]), \
        ds_tr["SMILES"].apply(lambda x: embed_map[x]), ds_te["SMILES"].apply(lambda x: embed_map[x])


def main():
    matplotlib.rc('font', **{'size': 16})
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    dc_tr, dc_te, ds_tr, ds_te = embed()
    axs[0].scatter(dc_tr.apply(lambda x: x[0]), dc_tr.apply(lambda x: x[1]), s=1)
    axs[0].scatter(dc_te.apply(lambda x: x[0]), dc_te.apply(lambda x: x[1]), s=1)
    axs[0].set_xticks([])
    axs[0].set_xticks([], minor=True)
    axs[0].set_yticks([])
    axs[0].set_yticks([], minor=True)
    axs[0].set_title("ECFP4 embeddings of\na stratified split")
    axs[1].scatter(ds_tr.apply(lambda x: x[0]), ds_tr.apply(lambda x: x[1]), s=1)
    axs[1].scatter(ds_te.apply(lambda x: x[0]), ds_te.apply(lambda x: x[1]), s=1)
    axs[1].set_xticks([])
    axs[1].set_xticks([], minor=True)
    axs[1].set_yticks([])
    axs[1].set_yticks([], minor=True)
    axs[1].set_title("ECFP4 embeddings of a\nstratified split with DataSAIL")
    # plot_perf(axs[2])
    plot_dmpnn_perf(axs[2])

    fig.tight_layout()
    plt.savefig("experiments/Tox21Strat/Tox21Strat.png")
    plt.show()


main()
