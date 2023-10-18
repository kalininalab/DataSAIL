import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.transforms as mtransforms

from experiments.PDBBind.visualize import embed_smiles, get_bounds
from experiments.utils import USE_UMAP

SPLITS = ["ICSe", "CCSe", "Butina", "Fingerprint", "MinMax", "Scaffold", "Weight"]
DATASETS = ["QM7", "QM8", "QM9", "ESOL", "FreeSolv", "Lipophilicity", "BACE", "BBBP", "Tox21", "ToxCast", "SIDER",
            "ClinTox"]
METRICS = ["MAE ↓"] * 3 + ["RMSE ↓"] * 3 + ["ROC-AUC ↑"] * 6


def plot_embeds():
    fig, axes = plt.subplots(
        len(DATASETS), len(SPLITS),
        figsize=(len(SPLITS) * 4, len(DATASETS) * 3),
    )
    data = [[None for _ in range(len(SPLITS))] for _ in range(len(DATASETS))]
    for i, dataset in enumerate(DATASETS):
        print(dataset)
        if os.path.exists(f"experiments/{'umap' if USE_UMAP else 'tsne'}/embeds_{dataset}.pkl"):
            print("Loaded", os.path.abspath(f"experiments/{'umap' if USE_UMAP else 'tsne'}/embeds_{dataset}.pkl"))
            data[i] = pickle.load(open(f"experiments/{'umap' if USE_UMAP else 'tsne'}/embeds_{dataset}.pkl", "rb"))
            continue
        smiles = set()
        for j, split in enumerate(SPLITS):
            if j < 2:
                filename = lambda x: f"experiments/MPP/datasail/cdata/{dataset.lower()}/{split}/split_0/{x}.csv"
            else:
                filename = lambda x: f"experiments/MPP/deepchem/cdata/{dataset.lower()}/{split}/split_0/{x}.csv"
            if not os.path.exists(filename("train")):
                continue
            try:
                train = pd.read_csv(filename("train"))
                test = pd.read_csv(filename("test"))
                data[i][j] = {"train": train["SMILES"].values, "test": test["SMILES"].values}
                smiles.update(train["SMILES"].values)
                smiles.update(test["SMILES"].values)
            except Exception as e:
                print(f"{dataset} - {split}: {e}")
        smiles = [(s, embed_smiles(s)) for s in smiles]
        if USE_UMAP:
            embedder = UMAP(n_components=2, random_state=42)
        else:
            embedder = TSNE(n_components=2, learning_rate="auto", init="random", random_state=42)
        embed = embedder.fit_transform(np.array([s[1] for s in smiles]))
        smiles = dict(list(zip([s[0] for s in smiles], embed)))
        for j, split in enumerate(SPLITS):
            if data[i][j] is None:
                continue
            data[i][j]["train"] = np.array([smiles[s] for s in data[i][j]["train"]])
            data[i][j]["test"] = np.array([smiles[s] for s in data[i][j]["test"]])
        pickle.dump(data[i], open(f"experiments/{'umap' if USE_UMAP else 'tsne'}/embeds_{dataset}.pkl", "wb"))

    for i, dataset in enumerate(DATASETS):
        for j, split in enumerate(SPLITS):
            axes[i, j].set_xticks([])
            axes[i, j].set_xticks([], minor=True)
            axes[i, j].set_yticks([])
            axes[i, j].set_yticks([], minor=True)
            try:
                if data[i][j] is None:
                    continue
                axes[i, j].scatter(*data[i][j]["train"].T, s=1)
                axes[i, j].scatter(*data[i][j]["test"].T, s=1)
            except Exception as e:
                pass

    for ax, split in zip(axes[0], ["I1", "C1"] + SPLITS[2:]):
        ax.set_title(split, fontsize=20)

    for ax, ds_name, metric in zip(axes[:, 0], DATASETS, METRICS):
        ax.set_ylabel(f"{ds_name}", rotation=90, fontsize=16)

    fig.tight_layout()
    plt.savefig(f"MPP_embeds_{'umap' if USE_UMAP else 'tsne'}.png")
    plt.show()


def plot_perf():
    fig, axes = plt.subplots(
        len(DATASETS), len(SPLITS),
        sharex='col', sharey='row',
        figsize=(len(SPLITS) * 4, len(DATASETS) * 3),
    )
    for i, dataset in enumerate(DATASETS):
        for j, split in enumerate(SPLITS):
            if j < 2:
                filename = f"experiments/MPP/datasail/cdata/{dataset.lower()}/val_metrics.tsv"
            else:
                filename = f"experiments/MPP/deepchem/cdata/{dataset.lower()}/val_metrics.tsv"
            if not os.path.exists(filename):
                continue
            try:
                table = pd.read_csv(filename, sep="\t")
                mask = [split in col for col in table.columns]
                mean = np.average(table[table.columns[mask]].values, axis=1)
                bounds = get_bounds(table[table.columns[mask]].values, axis=1)
                x = np.arange(0.0, 50, 1)
                axes[i, j].fill_between(x, *bounds, alpha=0.5)
                axes[i, j].plot(mean)
            except Exception as e:
                print(f"{dataset} - {split}: {e}")

    for ax, split in zip(axes[0], ["I1", "C1"] + SPLITS[2:]):
        ax.set_title(split, fontsize=20)

    for ax, ds_name, metric in zip(axes[:, 0], DATASETS, METRICS):
        ax.set_ylabel(f"{ds_name} ({metric})", rotation=90, fontsize=16)

    fig.tight_layout()
    plt.savefig("MPP_perf.png")
    plt.show()


def plot_perf_4x3():
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    handles, labels = [], []
    for i, dataset in enumerate(DATASETS):
        axes[i // 4, i % 4].set_title(dataset)
        axes[i // 4, i % 4].set_ylabel(METRICS[i], rotation=90)
        for j, split in enumerate(SPLITS):
            filename = f"experiments/MPP/{'datasail' if j < 2 else 'deepchem'}/cdata/{dataset.lower()}/val_metrics.tsv"
            if not os.path.exists(filename):
                continue
            try:
                table = pd.read_csv(filename, sep="\t")
                mask = [split in col for col in table.columns]
                mean = np.average(table[table.columns[mask]].values, axis=1)
                bounds = get_bounds(table[table.columns[mask]].values, axis=1)
                x = np.arange(0.0, 50, 1)
                # axes[i // 4, i % 4].fill_between(x, *bounds, alpha=0.5)
                h, = axes[i // 4, i % 4].plot(mean)
                if i == 0:
                    handles.append(h)
                    labels.append(split)
            except Exception as e:
                print(f"{dataset} - {split}: {e}")

    # axes[-1, -2].legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=3)
    axes[-1, -2].legend(
        handles=handles,
        labels=labels,
        bbox_to_anchor=(-1.65, -0.3, 3., .102),
        loc='lower center',
        ncol=7,
        mode="expand",
        borderaxespad=0.,
    )

    fig.tight_layout()
    plt.savefig("MPP_perf_4x3.png")
    plt.show()


def plot_double(names):
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for i, name in enumerate(names):
        for s, split in enumerate(SPLITS[:2]):
            filename = f"experiments/MPP/{'datasail' if i < 2 else 'deepchem'}/cdata/{name.lower()}/val_metrics.tsv"
            if not os.path.exists(filename):
                continue
            try:
                table = pd.read_csv(filename, sep="\t")
                mask = [split in col for col in table.columns]
                mean = np.average(table[table.columns[mask]].values, axis=1)
                bounds = get_bounds(table[table.columns[mask]].values, axis=1)
                x = np.arange(0.0, 50, 1)
                ax[i, 2].fill_between(x, *bounds, alpha=0.5)
                ax[i, 2].plot(mean, label='random' if s == 0 else 'cluster-based')
                ax[i, 2].set_title(f"Performance comparison ({'MAE ↓' if i == 0 else 'ROC-AUC ↑'})")
                if i == 0:
                    ax[i, 2].legend(loc=2)
            except Exception as e:
                print(f"{name} - {split}: {e}")

        data = pickle.load(open(f"experiments/{'umap' if USE_UMAP else 'tsne'}/embeds_{name}.pkl", "rb"))
        for t, tech in enumerate(SPLITS[:2]):
            ax[i, t].set_title(tech)
            ax[i, t].set_xticks([])
            ax[i, t].set_xticks([], minor=True)
            ax[i, t].set_yticks([])
            ax[i, t].set_yticks([], minor=True)
            try:
                ax[i, t].scatter(*data[t]["train"].T, s=1, label="train")
                ax[i, t].scatter(*data[t]["test"].T, s=1, label="test")
                ax[i, t].set_title(f"ECFP4 embeddings of\nthe {'random' if t == 0 else 'cluster-based'} split using t-SNE")
                if i == 1 and t == 0:
                    ax[i, t].legend(loc=4, markerscale=8)
                if t == 0:
                    ax[i, 0].text(
                        0.0,
                        1.0,
                        ["A", "B"][i],
                        transform=ax[i, 0].transAxes + mtransforms.ScaledTranslation(
                            -20 / 72,
                            7 / 72,
                            fig.dpi_scale_trans
                        ),
                        fontsize="x-large",
                        va="bottom",
                        fontfamily="serif",
                        fontweight="bold",
                    )
            except Exception as e:
                print(e)
    fig.tight_layout()
    plt.savefig(f"QM9_Tox21.png")
    plt.show()


if __name__ == '__main__':
    # plot_perf()
    plot_perf_4x3()
    # plot_double(["QM7", "Tox21"])
    # plot_embeds()
