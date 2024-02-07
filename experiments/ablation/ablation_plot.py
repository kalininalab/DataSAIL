import sys
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt, gridspec

from experiments.ablation.visualize_de import plot_de_ablation
from experiments.ablation.time import get_tool_times
from experiments.utils import set_subplot_label
from experiments.ablation.david import visualize


def plot_ablations(full_path):
    matplotlib.rc('font', **{'size': 18})
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 1, figure=fig)
    gs_upper = gs[0].subgridspec(1, 2)
    gs_lower = gs[1].subgridspec(1, 3, width_ratios=[1, 1.5, 0.25], wspace=0.4)
    ax = [fig.add_subplot(gs_upper[0]), fig.add_subplot(gs_upper[1]), fig.add_subplot(gs_lower[0]), fig.add_subplot(gs_lower[1])]

    visualize("tox21", list(range(10, 50, 5)) + list(range(50, 150, 10)) + list(range(150, 401, 50)), ["GUROBI", "MOSEK", "SCIP"], ax=(ax[0], ax[1]))
    set_subplot_label(ax[0], fig, "A")
    set_subplot_label(ax[1], fig, "B")

    plot_de_ablation(full_path, ax=ax[2], fig=fig)
    set_subplot_label(ax[2], fig, "C")

    get_tool_times(Path("experiments") / "MPP", ax=ax[3])
    set_subplot_label(ax[3], fig, "D")

    plt.tight_layout()
    plt.savefig("ablation.png")
    plt.show()


if __name__ == '__main__':
    plot_ablations(Path(sys.argv[1]))
