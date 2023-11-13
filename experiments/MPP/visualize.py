import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.transforms as mtransforms

from experiments.utils import USE_UMAP, embed_smiles, get_bounds

SPLITS = ["I1e", "C1e", "lohi", "Butina", "Fingerprint", "MinMax", "Scaffold", "Weight"]
DATASETS = ["QM7", "QM8", "QM9", "ESOL", "FreeSolv", "Lipophilicity", "MUV", "HIV", "BACE", "BBBP", "Tox21", "ToxCast",
            "SIDER", "ClinTox"]
METRICS = ["MAE ↓"] * 3 + ["RMSE ↓"] * 3 + ["PRC-AUC ↑"] + ["ROC-AUC ↑"] * 7

def plot_embeds():
    fig, axes = plt.subplots(
        len(DATASETS), len(SPLITS),
        figsize=(len(SPLITS) * 4, len(DATASETS) * 3),
    )
    data = [[None for _ in range(len(SPLITS))] for _ in range(len(DATASETS))]
    for i, dataset in enumerate(DATASETS):
        os.makedirs(Path("experiments") / "MPP" / ('umap' if USE_UMAP else 'tsne'), exist_ok=True)
        embed_path = Path("experiments") / "MPP" / ('umap' if USE_UMAP else 'tsne') / f"embeds_{dataset}.pkl"
        if os.path.exists(embed_path):
            data[i] = pickle.load(open(embed_path, "rb"))
            continue
        smiles = set()
        for j, split in enumerate(SPLITS):
            base_filename = lambda x, y: Path("experiments") / "MPP" / y / "cdata" / dataset.lower() / split / \
                "split_0" / f"{x}.csv"
            if j == 2:
                filename = lambda x: base_filename(x, "lohi")
            elif j < 2:
                filename = lambda x: base_filename(x, "datasail")
            else:
                filename = lambda x: base_filename(x, "deepchem")
            print(filename("train"))
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
                print("Data is empty -", dataset, "-", split)
                continue
            data[i][j]["train"] = np.array([smiles[s] for s in data[i][j]["train"]])
            data[i][j]["test"] = np.array([smiles[s] for s in data[i][j]["test"]])
        pickle.dump(data[i], open(embed_path, "wb"))

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
        if split == "MinMax":
            split = "MaxMin"
        elif split == "lohi":
            split = "LoHi"
        ax.set_title(split, fontsize=30, fontweight="bold")

    for ax, ds_name, metric in zip(axes[:, 0], DATASETS, METRICS):
        ax.set_ylabel(f"{ds_name}", rotation=90, fontsize=30, fontweight="bold")

    fig.tight_layout()
    plt.savefig(Path("experiments") / "MPP" / f"MPP_embeds_{'umap' if USE_UMAP else 'tsne'}.png")
    plt.show()


def plot_perf():
    fig, axes = plt.subplots(
        len(DATASETS), len(SPLITS),
        sharex='col', sharey='row',
        figsize=(len(SPLITS) * 4, len(DATASETS) * 3),
    )
    for i, dataset in enumerate(DATASETS):
        for j, split in enumerate(SPLITS):
            base_filename = lambda x: Path("experiments") / "MPP" / x / "cdata" / dataset.lower() / "val_metrics.tsv"
            if j == 2:
                filename = str(base_filename("lohi"))
                filename.replace("cdata", "sdata")
            elif j < 2:
                filename = str(base_filename("datasail"))
            else:
                filename = str(base_filename("deepchem"))
            if not os.path.exists(filename):
                continue
            try:
                table = pd.read_csv(filename, sep="\t")
                mask = [(split.replace("1", "CS") if "1" in split else split) in col for col in table.columns]
                mean = np.average(table[table.columns[mask]].values, axis=1)
                bounds = get_bounds(table[table.columns[mask]].values, axis=1)
                x = np.arange(0.0, 50, 1)
                axes[i, j].fill_between(x, *bounds, alpha=0.5)
                axes[i, j].plot(mean)
            except Exception as e:
                print(f"{dataset} - {split}: {e}")

    for ax, split in zip(axes[0], ["I1", "C1", "LoHi"] + SPLITS[3:]):
        ax.set_title(split, fontsize=20)

    for ax, ds_name, metric in zip(axes[:, 0], DATASETS, METRICS):
        ax.set_ylabel(f"{ds_name} ({metric})", rotation=90, fontsize=16)

    fig.tight_layout()
    plt.savefig(Path("experiments") / "MPP" / "MPP_perf.png")
    plt.show()


def plot_perf_5x3():
    fig, axes = plt.subplots(3, 5, figsize=(20, 9))
    handles, labels = [], []
    for d, dataset in enumerate(DATASETS):
        i = d if d < 7 else d + 1
        print(i, "|", i // 5, "|", i % 5)
        axes[i // 5, i % 5].set_title(dataset, fontsize=15)
        axes[i // 5, i % 5].set_ylabel(METRICS[d], rotation=90, fontsize=10)
        for j, split in enumerate(SPLITS):
            base_name = lambda x: Path("experiments") / "MPP" / x / "cdata" / dataset.lower() / "val_metrics.tsv"
            if j < 2:
                filename = base_name("datasail")
            elif j == 2:
                filename = base_name("lohi")
            else:
                filename = base_name("deepchem")
            if not os.path.exists(filename):
                axes[i // 5, i % 5].plot([], [], visible=False)
                continue
            try:
                table = pd.read_csv(filename, sep="\t")
                f = lambda x: "ICSe" if x == "I1e" else ("CCSe" if x == "C1e" else split)
                mask = [f(split) in col for col in table.columns]
                mean = np.average(table[table.columns[mask]].values, axis=1)
                bounds = get_bounds(table[table.columns[mask]].values, axis=1)
                x = np.arange(0.0, 50, 1)
                axes[i // 5, i % 5].plot(mean)
            except Exception as e:
                print(f"{dataset} - {split}: {e}")
    for _ in range(len(SPLITS)):
        axes[1, 2].plot([], [], visible=False)
    axes[1, 2].legend(loc='center')
    names = list(map(lambda x: "LoHi" if x == "lohi" else ("MaxMin" if x == "MinMax" else x), SPLITS))
    legend = axes[1, 2].legend(names, loc="center", markerscale=10, fontsize=15)
    for h, handle in enumerate(legend.legend_handles):
        handle.set_visible(True)
    axes[1, 2].set_axis_off()

    fig.tight_layout()
    plt.savefig(Path("experiments") / "MPP" / "MPP_perf_5x3.png")
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
    plot_perf_5x3()
    # plot_double(["QM7", "Tox21"])
    # plot_embeds()
