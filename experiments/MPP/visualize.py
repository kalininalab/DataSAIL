import os

import matplotlib
import pandas as pd
import umap
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from experiments.utils import USE_UMAP, mpp_datasets


def plot_embed(train_mask, test_mask, embeds, tech, ax, legend=None):
    ax.scatter(embeds[train_mask, 0], embeds[train_mask, 1], s=4, label="train")
    ax.scatter(embeds[test_mask, 0], embeds[test_mask, 1], s=4, label="test")
    ax.set_xlabel(f"{'UMAP' if USE_UMAP else 'tSNE'} 1")
    ax.set_ylabel(f"{'UMAP' if USE_UMAP else 'tSNE'} 2")
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.set_yticks([], minor=True)
    if legend:
        ax.legend(loc=legend, markerscale=5)
    ax.set_title(f"ECFP4 embeddings of\n{tech}-based split")


def plot_perf(val_df, name, runs, ax):
    for split, lc, c in (("ICSe", "lightblue", "blue"), ("CCSe", "lightcoral", "darkred")):
        metric = np.array([
            val_df[col].values.tolist() for col in
            [f"{split}_validation_{mpp_datasets[name][2]}_split_{i}" for i in range(runs)]
        ])

        axis = 0
        x = np.arange(0.0, len(metric[0]), 1)

        ax.fill_between(x, np.min(metric, axis=axis), np.max(metric, axis=axis), color=lc, alpha=0.5)
        ax.plot(np.average(metric, axis=axis), color=c, label=("identity-based" if split[0] == "I" else "cluster-based"))
        ax.set_ylabel(mpp_datasets[name][2].upper())
        ax.set_xlabel("Epoch")
        ax.set_title("Performance Comparison")
        ax.legend()


def main(folder, name, letter=None):
    matplotlib.rc('font', **{'size': 16})
    val_df = pd.read_csv(os.path.join(folder, "val_metrics.tsv"), sep="\t")
    fps = {}
    with open(f"{folder}/{name.lower()}.csv", "r") as data:
        for line in data.readlines()[1:]:
            c, s = line.strip().split(",")[:2]
            mol = Chem.MolFromSmiles(s)
            if mol is None or s in fps:
                continue
            fps[s] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)

    ics_clustering, ccs_clustering = dict(), dict()
    with open(f"{folder}/ICSe/split_0/train.csv", "r") as data:
        for line in data.readlines()[1:]:
            s = line.strip().split(",")[1]
            if s in fps:
                ics_clustering[s] = "train"
    with open(f"{folder}/ICSe/split_0/test.csv", "r") as data:
        for line in data.readlines()[1:]:
            s = line.strip().split(",")[1]
            if s in fps:
                ics_clustering[s] = "test"
    with open(f"{folder}/CCSe/split_0/train.csv", "r") as data:
        for line in data.readlines()[1:]:
            s = line.strip().split(",")[1]
            if s in fps:
                ccs_clustering[s] = "train"
    with open(f"{folder}/CCSe/split_0/test.csv", "r") as data:
        for line in data.readlines()[1:]:
            s = line.strip().split(",")[1]
            if s in fps:
                ccs_clustering[s] = "test"

    for x in set(fps.keys()).difference(set(ccs_clustering.keys())):
        del fps[x]

    ics_clustering = np.array([a for _, a in sorted(ics_clustering.items())])
    ccs_clustering = np.array([a for _, a in sorted(ccs_clustering.items())])

    if USE_UMAP:
        embeds = umap.UMAP(random_state=42).fit_transform([a for _, a in sorted(fps.items())])
    else:
        embeds = TSNE(n_components=2, learning_rate="auto", init="random", random_state=42).fit_transform(
            np.array([a for _, a in sorted(fps.items())])
        )

    fig, axs = plt.subplots(1, 3)
    """
    if letter is not None:
        axs[0].text(
            0.0,
            1.0,
            letter,
            transform=axs[0].transAxes + mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans),
            fontsize="x-large",
            va="bottom",
            fontfamily="serif",
            fontweight="bold",
        )
    else:
        plt.suptitle(f"Visual analysis of the {name} dataset with DataSAIL, UMAP, and D-MPNN", fontsize="x-large")
    """
    plot_embed(ics_clustering == "train", ics_clustering == "test", embeds, "identity", axs[0], legend=4)
    plot_embed(ccs_clustering == "train", ccs_clustering == "test", embeds, "cluster", axs[1])
    plot_perf(val_df, name.lower(), 5, axs[2])
    fig.set_size_inches(20, 6)
    fig.tight_layout()
    plt.savefig(f"{name.lower()}_{'UMAP' if USE_UMAP else 'tSNE'}.png")
    plt.show()


if __name__ == '__main__':
    for n in [
        "BACE",
        # "BBBP",
        "ClinTox",
        "ESOL",
        "FreeSolv",
        "Lipophilicity",
        "QM7",
        "QM8",
        "QM9",
        "Tox21",
    ]:
        print(n)
        main(f"data/{n.lower()}", n)
