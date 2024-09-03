import sys
from pathlib import Path

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from experiments.utils import set_subplot_label, COLORS, embed, plot_embeds, plot_bars_2y


def plot_perf(base_path, ax):
    with open("strat.pkl", "rb") as f:
        leakage = pd.read_pickle(f)
    models = ["RF", "SVM", "XGB", "MLP", "D-MPNN"]
    df = pd.read_csv(base_path / "results.csv")
    values = df[["tool", "model", "perf"]].groupby(["model", "tool"])["perf"].mean().reset_index() \
        .pivot(index="model", columns="tool", values="perf")
    values = np.array(values.reindex(["rf", "svm", "xgb", "mlp", "d-mpnn"])[["datasail", "deepchem"]], dtype=float)
    df = pd.DataFrame(values, columns=["DataSAIL split (S1 w/ classes)", "Stratified baseline"], index=models)
    df.loc["IL"] = [np.average([x for x, _ in leakage[k]]) for k in ["datasail", "deepchem"]]
    il = plot_bars_2y(df.T, ax, color=[COLORS["s1d"], COLORS["r1d"]])
    ax.set_ylabel("AUROC (↑)")
    ax.set_xlabel("ML Models")
    ax.legend(loc="lower left")
    ax.set_title(f"Performance comparison")


def main(full_path):
    (plot_dir := (full_path / "plots")).mkdir(exist_ok=True, parents=True)
    matplotlib.rc('font', **{'size': 16})
    fig = plt.figure(figsize=(20, 5.33))
    gs = gridspec.GridSpec(1, 3, figure=fig)
    ax = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]

    dc_tr, dc_te, ds_tr, ds_te = embed(full_path)
    plot_embeds(ax[0], dc_tr, dc_te, "Stratified baseline", legend=True)
    set_subplot_label(ax[0], fig, "A")
    plot_embeds(ax[1], ds_tr, ds_te, "DataSAIL split (S1 w/ classes)")
    set_subplot_label(ax[1], fig, "B")
    plot_perf(full_path, ax[2])
    set_subplot_label(ax[2], fig, "C")

    fig.tight_layout()
    plt.savefig(plot_dir / "Strat.png")
    plt.show()


if __name__ == '__main__':
    main(Path(sys.argv[1]))
