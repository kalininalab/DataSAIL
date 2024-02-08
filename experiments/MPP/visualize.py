import os
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
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from experiments.utils import RUNS, DATASETS, COLORS, set_subplot_label, HSPACE, METRICS, embed, plot_embeds


def plot_double(full_path: Path, names: List[str]) -> None:
    """
    Plot the performance and t-SNE embeddings of two datasets.

    Args:
        full_path: Path to the base directory
        names: Names of the datasets
    """
    matplotlib.rc('font', **{'size': 16})
    fig = plt.figure(figsize=(20, 10.67))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1])
    gs_left = gs[0].subgridspec(2, 2, hspace=HSPACE, wspace=0.1)
    gs_right = gs[1].subgridspec(2, 1, hspace=HSPACE)
    ax = [
        [fig.add_subplot(gs_left[0, 0]), fig.add_subplot(gs_left[0, 1]), fig.add_subplot(gs_right[0])],
        [fig.add_subplot(gs_left[1, 0]), fig.add_subplot(gs_left[1, 1]), fig.add_subplot(gs_right[1])],
    ]
    perf = {name: read_perf(full_path, name) for name in names}
    for i, name in enumerate(names):
        df = perf[name]
        df.rename(index={"I1e": "Random baseline (I1)", "C1e": "DataSAIL split (S1)"}, inplace=True)
        df[df.index.isin(["Random baseline (I1)", "DataSAIL split (S1)"])].T.plot.bar(
            ax=ax[i][2], rot=0,
            ylabel=METRICS[DATASETS.index(name)],
            color=[COLORS["r1d"], COLORS["s1d"]],
        ).legend(loc="lower right")
        ax[i][2].set_title("Performance comparison")
        ax[i][2].set_xlabel("ML Models")
        set_subplot_label(ax[i][2], fig, ["C", "F"][i])

        i_tr, i_te, c_tr, c_te = embed(full_path, name.lower())
        plot_embeds(ax[i][0], i_tr, i_te, "Random baseline", legend=True)
        set_subplot_label(ax[i][0], fig, "A")
        plot_embeds(ax[i][1], c_tr, c_te, "DataSAIL split (S1)")
        set_subplot_label(ax[i][1], fig, "B")
    plt.tight_layout()
    plt.savefig(full_path / f"{names[0]}_{names[1]}.png")
    plt.show()


def read_perf(full_path: Path, name: str) -> pd.DataFrame:
    """
    Read the performance of the models on the datasets.

    Args:
        full_path: Path to the base directory
        name: Name of the dataset

    Returns:
        pd.DataFrame: Dataframe of the performance of the models
    """
    models = ["RF", "SVM", "XGB", "MLP", "D-MPNN"]
    tools = ["datasail", "datasail", "lohi", "deepchem", "deepchem", "deepchem", "deepchem", "deepchem"]
    techniques = ["I1e", "C1e", "lohi", "Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"]
    offset = [0, 0, 1, 2, 2, 2, 2, 2]
    df = pd.DataFrame(columns=models, index=techniques)
    for i, (tool, tech) in enumerate(zip(tools, techniques)):
        base = full_path / tool / name.lower()
        for model in models:
            if model != "D-MPNN":
                model_name = f"{model.lower()}-{DATASETS[name.lower()][1][0]}"
                if (base / f"{model_name}.csv").exists():
                    with open(base / f"{model_name}.csv") as f:
                        idx = (i - offset[i]) * 5
                        df.at[tech, model] = np.mean([float(x) for x in f.readlines()][idx:idx + 5])
                else:
                    perf = []
                    for run in range(RUNS):
                        try:
                            if (base / f"{model_name}_{tech}_{run}.txt").exists():
                                with open(base / f"{model_name}_{tech}_{run}.txt") as f:
                                    if len(line := f.readlines()[0].strip()) > 2:
                                        perf.append(float(float(line)))
                        except:
                            pass
                    if len(perf) > 0:
                        df.at[tech, model] = np.mean(perf)
            elif (base / f"test_metrics.tsv").exists():
                try:
                    tmp = pd.read_csv(base / f"test_metrics.tsv", sep="\t")
                    cols = [c for c in tmp.columns if tech in c]
                    if len(cols) > 0:
                        df.at[tech, "D-MPNN"] = tmp[cols].values.mean()
                    else:
                        df.at[tech, "D-MPNN"] = check_tb(base, tech)
                except:
                    df.at[tech, "D-MPNN"] = check_tb(base, tech)
            else:
                df.at[tech, "D-MPNN"] = check_tb(base, tech)
    return df


def check_tb(base, tech):
    try:
        perfs = []
        for run in range(RUNS):
            path = base / tech / f"split_{run}" / "fold_0" / "model_0"
            files = list(sorted(filter(
                lambda x: x.startswith("events"), os.listdir(path)
            ), key=lambda x: os.path.getsize(Path(path) / x)))
            for tb_file in files:
                ea = EventAccumulator(str(path / tb_file))
                ea.Reload()
                broken = False
                for metric in filter(lambda x: x.startswith("test_"), ea.Tags()["scalars"]):
                    perf = [e.value for e in ea.Scalars(metric)]
                    if len(perf) > 0:
                        perfs.append(perf[-1])
                        broken = True
                        break
                if broken:
                    break
        if len(perfs) > 0:
            return np.mean(perfs)
    except:
        pass


def heatmap_plot(full_path: Path):
    """
    Plot the performance of the models on the datasets as a heatmap.

    Args:
        full_path: Path to the base directory
    """
    matplotlib.rc('font', **{'size': 16})
    models = ["RF", "SVM", "XGB", "MLP", "D-MPNN"]
    techniques = ["I1e", "C1e", "lohi", "Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"]

    dfs = {name: read_perf(full_path, name) for name in DATASETS}

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
        values = np.array(df.values, dtype=float)
        values = values[[1, 0, 2, 3, 4, 5, 6, 7], :]
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
        fig.colorbar(pic, cax=cax, orientation='vertical', label=METRICS[i])

    for i in range(14, cols * rows):
        axs[i].set_axis_off()

    fig.tight_layout()
    plt.savefig(full_path / f"MoleculeNet_comp.png", transparent=True)
    plt.show()


if __name__ == '__main__':
    plot_double(Path(sys.argv[1]), ["QM8", "Tox21"])
    # heatmap_plot(Path(sys.argv[1]))
