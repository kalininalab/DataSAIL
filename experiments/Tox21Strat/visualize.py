from pathlib import Path

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from experiments.utils import embed_smiles, get_bounds, set_subplot_label, colors

root = Path("experiments") / "Tox21Strat"


def plot_perf(ax):
    models = ["RF", "SVM", "XGB", "MLP", "D-MPNN"]
    values = [[] for _ in range(2)]

    for s, split in enumerate(["deepchem", "datasail"]):
        for model in models[:-1]:
            df = pd.read_csv(root / split / f"{model.lower()}.csv")
            values[s].append(df.mean(axis=1).values[0])
        values[s].append(pd.read_csv(root / split / "val_metrics.tsv", sep="\t").max(axis=0).values[1:].mean())
    df = pd.DataFrame(np.array(values).T, columns=["Stratified baseline", "DataSAIL split (S1 w/ classes)"], index=models)
    df.plot.bar(ax=ax, rot=0, ylabel="AUROC", color=[colors["r1d"], colors["s1d"]])
    ax.legend(loc="lower right")
    ax.set_title(f"Performance comparison")


def plot_dmpnn_perf(ax):
    ds = pd.read_csv(root / "datasail" / "val_metrics.tsv", sep="\t").values[:, 1:].T
    dc = pd.read_csv(root / "deepchem" / "val_metrics.tsv", sep="\t").values[:, 1:].T

    ds_mean = np.mean(ds, axis=0)
    ds_bounds = get_bounds(ds)
    dc_mean = np.mean(dc, axis=0)
    dc_bounds = get_bounds(dc)

    x = np.arange(0.0, 50, 1)
    ax.fill_between(x, *dc_bounds, alpha=0.5, color=colors["r1d"])
    ax.plot(dc_mean, label="Stratified", color=colors["r1d"])
    ax.fill_between(x, *ds_bounds, alpha=0.5, color=colors["s1d"])
    ax.plot(ds_mean, label="DataSAIL", color=colors["s1d"])
    ax.legend(loc="lower right")
    # ax.set_title(f"Performance comparison\n(ROC-AUC ↑)")


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
    return np.stack(dc_tr["SMILES"].apply(lambda x: embed_map[x])), np.stack(dc_te["SMILES"].apply(lambda x: embed_map[x])), \
        np.stack(ds_tr["SMILES"].apply(lambda x: embed_map[x])), np.stack(ds_te["SMILES"].apply(lambda x: embed_map[x]))


def plot_embeds(ax, train, test, title, legend=None):
    n_train = len(train)
    n_test = len(test)

    p = np.concatenate([train, test])
    c = np.array([colors["train"]] * n_train + [colors["test"]] * n_test)
    perm = np.random.permutation(len(p))
    ax.scatter(p[perm, 0], p[perm, 1], s=5, c=c[perm])
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        train_dot = Line2D([0], [0], marker='o', label="train", color=colors["train"], linestyle='None')
        test_dot = Line2D([0], [0], marker='o', label="test", color=colors["test"], linestyle='None')
        handles.extend([train_dot, test_dot])
        ax.legend(handles=handles, loc="lower right", markerscale=2)


def main():
    matplotlib.rc('font', **{'size': 16})
    fig = plt.figure(figsize=(20, 5.33))
    # gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1])
    # gs_left = gs[0].subgridspec(1, 2, hspace=0.17, wspace=0.05)
    # ax = [fig.add_subplot(gs_left[0]), fig.add_subplot(gs_left[1]), fig.add_subplot(gs[1])]
    gs = gridspec.GridSpec(1, 3, figure=fig)
    ax = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]

    dc_tr, dc_te, ds_tr, ds_te = embed()
    plot_embeds(ax[0], dc_tr, dc_te, "Stratified baseline", legend=True)
    set_subplot_label(ax[0], fig, "A")
    plot_embeds(ax[1], ds_tr, ds_te, "DataSAIL split (S1 w/ classes)")
    set_subplot_label(ax[1], fig, "B")
    plot_perf(ax[2])
    # plot_dmpnn_perf(ax[2])
    set_subplot_label(ax[2], fig, "C")

    fig.tight_layout()
    plt.savefig("Tox21Strat.png")
    plt.show()


if __name__ == '__main__':
    main()
