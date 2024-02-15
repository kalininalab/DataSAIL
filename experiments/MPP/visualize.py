import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from experiments.utils import DATASETS, COLORS, set_subplot_label, HSPACE, METRICS, embed, plot_embeds


def plot_double(full_path: Path, names: List[str]) -> None:
    """
    Plot the performance and t-SNE embeddings of two datasets.

    Args:
        full_path: Path to the base directory
        names: Names of the datasets
    """
    (full_path / "plots").mkdir(parents=True, exist_ok=True)

    matplotlib.rc('font', **{'size': 16})
    fig = plt.figure(figsize=(20, 10.67))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1])
    gs_left = gs[0].subgridspec(2, 2, hspace=HSPACE, wspace=0.1)
    gs_right = gs[1].subgridspec(2, 1, hspace=HSPACE)
    ax = [
        [fig.add_subplot(gs_left[0, 0]), fig.add_subplot(gs_left[0, 1]), fig.add_subplot(gs_right[0])],
        [fig.add_subplot(gs_left[1, 0]), fig.add_subplot(gs_left[1, 1]), fig.add_subplot(gs_right[1])],
    ]
    perf = {name: get_perf(full_path, name) for name in names}
    for i, name in enumerate(names):
        df = perf[name]
        df = df.loc[df["tech"].isin(["I1e", "C1e"]), ["tech", "model", "perf"]].groupby(["model", "tech"])["perf"] \
            .mean().reset_index().pivot(index="model", columns="tech", values="perf")
        df.rename(columns={"I1e": "Random baseline (I1)", "C1e": "DataSAIL split (S1)"}, inplace=True)
        df.plot.bar(
            ax=ax[i][2], rot=0,
            ylabel=METRICS[DATASETS[name.lower()][2]],
            color=[COLORS["r1d"], COLORS["s1d"]],
        ).legend(loc="lower right")
        ax[i][2].set_title("Performance comparison")
        ax[i][2].set_xlabel("ML Models")
        set_subplot_label(ax[i][2], fig, ["C", "F"][i])

        i_tr, i_te, c_tr, c_te = embed(full_path, name.lower())
        plot_embeds(ax[i][0], i_tr, i_te, "Random baseline (I1)", legend=True)
        set_subplot_label(ax[i][0], fig, "A")
        plot_embeds(ax[i][1], c_tr, c_te, "DataSAIL split (S1)")
        set_subplot_label(ax[i][1], fig, "B")
    plt.tight_layout()
    plt.savefig(full_path / "plots" / f"{names[0]}_{names[1]}.png")
    plt.show()


def heatmap_plot(full_path: Path):
    """
    Plot the performance of the models on the datasets as a heatmap.

    Args:
        full_path: Path to the base directory
    """
    (full_path / "plots").mkdir(parents=True, exist_ok=True)
    matplotlib.rc('font', **{'size': 16})
    models = ["RF", "SVM", "XGB", "MLP", "D-MPNN"]
    techniques = ["I1e", "C1e", "lohi", "Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"]

    dfs = {name: get_perf(full_path, name) for name in DATASETS if name.lower() != "pcba"}

    fig = plt.figure(figsize=(20, 25))
    cols, rows = 4, 4
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    axs = [fig.add_subplot(gs[i, j]) for i in range(rows) for j in range(cols)]

    for i, (name, df) in enumerate(dfs.items()):
        if DATASETS[name.lower()][1] == "classification":
            cmap = LinearSegmentedColormap.from_list("Custom", [COLORS["r1d"], COLORS["s1d"]], N=256)
            cmap.set_bad(color="white")
        else:
            cmap = LinearSegmentedColormap.from_list("Custom", [COLORS["train"], COLORS["test"]], N=256)
            cmap.set_bad(color="white")
        values = df[["tech", "model", "perf"]].groupby(["model", "tech"])["perf"] \
            .mean().reset_index().pivot(index="tech", columns="model", values="perf")
        values = np.array(values.reindex([
            "C1e", "I1e", "lohi", "Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"
        ])[["RF", "SVM", "XGB", "MLP", "d-mpnn"]], dtype=float)
        pic = axs[i].imshow(values, cmap=cmap, vmin=np.nanmin(values), vmax=np.nanmax(values))
        for b in range(len(models)):
            for a in range(len(techniques)):
                if np.isnan(values[a, b]):
                    continue
                label = f"{values[a, b]:.2f}"
                axs[i].text(b, a, label, ha='center', va='center')
        if i % cols == 0:
            axs[i].set_yticks(range(len(techniques)),
                              ["DataSAIL (S1)", "Rd. basel. (I1)", "LoHi", "DC - Butina", "DC - Fingerp.",
                               "DC - MaxMin", "DC - Scaffold", "DC - Weight"])
        else:
            axs[i].set_yticks([])
        axs[i].set_xticks(range(len(models)), models)
        axs[i].set_xlabel("ML Models")
        axs[i].set_title(name)

        cax = make_axes_locatable(axs[i]).append_axes('right', size='5%', pad=0.05)
        fig.colorbar(pic, cax=cax, orientation='vertical', label=METRICS[DATASETS[name.lower()][2]])

    for i in range(14, cols * rows):
        axs[i].set_axis_off()

    fig.tight_layout()
    plt.savefig(full_path / "plots" / f"MoleculeNet_comp.png", transparent=True)
    plt.show()


def get_perf(full_path: Path, name: str) -> pd.DataFrame:
    dfs = []
    for tool in ["datasail", "lohi", "deepchem"]:
        if name.lower() == "muv" and tool == "lohi":
            dfs.append(pd.DataFrame({
                "name": ["LoHi_0", "LoHi_0", "LoHi_0", "LoHi_0", "LoHi_0"],
                "perf": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "model": ["RF", "SVM", "XGB", "MLP", "d-mpnn"],
                "tool": ["lohi", "lohi", "lohi", "lohi", "lohi"],
                "tech": ["lohi", "lohi", "lohi", "lohi", "lohi"],
                "run": [0, 0, 0, 0, 0],
                "dataset": ["muv", "muv", "muv", "muv", "muv"],
            }))
            continue
        dfs.append(pd.read_csv(full_path / tool / name.lower() / "results.csv", index_col=None))
    return pd.concat(dfs)


if __name__ == '__main__':
    plot_double(Path(sys.argv[1]), ["QM8", "Tox21"])
    heatmap_plot(Path(sys.argv[1]))
