from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt, gridspec

from experiments.Tox21Strat.visualize_de import plot_de_ablation
from experiments.time import get_tool_times
from experiments.utils import set_subplot_label
from experiments.david import visualize


def plot_ablations():
    matplotlib.rc('font', **{'size': 18})
    fig = plt.figure(figsize=(26.25, 7))
    gs = gridspec.GridSpec(1, 3, figure=fig)
    ax = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
    
    get_tool_times(Path("experiments") / "MPP", ax=ax[0])
    set_subplot_label(ax[0], fig, "A")
    
    visualize("tox21", list(range(10, 50, 5)) + list(range(50, 150, 10)) + list(range(150, 401, 50)), ["GUROBI", "MOSEK", "SCIP"], ax=ax[1])
    set_subplot_label(ax[1], fig, "B")
    
    plot_de_ablation(ax=ax[2], fig=fig)
    set_subplot_label(ax[2], fig, "C")
    
    plt.tight_layout()
    plt.savefig("ablation.png")
    plt.show()


if __name__ == '__main__':
    plot_ablations()
