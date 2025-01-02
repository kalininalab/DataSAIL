import os
import pickle
from pathlib import Path
from typing import List

import matplotlib
import numpy as np
from matplotlib import pyplot as plt, gridspec

from experiments.utils import DATASETS, RUNS, COLORS

files = []

MARKERS = {
    "i1e": "o",
    "c1e": "P",
    "lohi": "X",
    "gurobi": "o",
    "mosek": "P",
    "scip": "X",
    "butina": "v",
    "fingerprint": "^",
    "maxmin": "<",
    "scaffold": ">",
    "weight": "D",
}


def get_single_time(path: Path) -> float:
    """
    Get the time it took to split the dataset for a single run.

    Args:
        path: Path to the splitting directory.

    Returns:
        The time it took to split the dataset.
    """
    if not os.path.exists(path / "train.csv"):
        return 0
    return os.path.getctime(path / "train.csv") - os.path.getctime(path / "start.txt")


def get_run_times(path: Path) -> List[float]:
    """
    Get the time it took to split the dataset for all runs.

    Args:
        path: Path to the technique directory.

    Returns:
        The time it took to split the dataset for all runs.
    """
    return [get_single_time(path / f"split_{run}") for run in range(RUNS)]


def get_tech_times(path: Path) -> List[List[float]]:
    """
    Get the time it took to split the dataset for all techniques.

    Args:
        path: Path to the dataset directory.

    Returns:
        The time it took to split the dataset for all techniques.
    """
    if "deepchem" in str(path):
        techniques = ["Scaffold", "Weight", "MinMax", "Butina", "Fingerprint"]
    elif "datasail" in str(path):
        techniques = ["I1e", "C1e"]
    elif "lohi" in str(path):
        techniques = ["lohi"]
    elif "graphpart" in str(path):
        techniques = ["graphpart"]
    else:
        raise ValueError(f"No known technique in path {str(path)}.")
    return [get_run_times(path / tech) for tech in techniques]


def get_dataset_times(path: Path) -> List[List[List[float]]]:
    """
    Get the time it took to split the dataset for all datasets.

    Args:
        path: Path to the dataset directory.

    Returns:
        The time it took to split the dataset for all datasets.
    """
    return [get_tech_times(path / ds_name) for ds_name in sorted(os.listdir(path), key=lambda x: DATASETS[x][3])]


def get_tool_times(path, ax=None) -> None:
    """
    Plot the time it took to split the dataset for all datasets and techniques.

    Args:
        path: Path to the dataset directory.
        ax: Axis to plot on.
    """
    if show := ax is None:
        matplotlib.rc('font', **{'size': 16})
        fig = plt.figure(figsize=(20, 10.67))
        gs = gridspec.GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0])
    pkl_path = Path("../..") / "DataSAIL" / "experiments" / "MPP" / "timing.pkl"
    if not os.path.exists(pkl_path):
        times = np.array(get_dataset_times(path / "datasail")), \
            np.array(get_dataset_times(path / "lohi")), \
            np.array(get_dataset_times(path / "deepchem"))
        pickle.dump(times, open(pkl_path, "wb"))
    else:
        times = pickle.load(open(pkl_path, "rb"))
    times = list(times)
    times[1] = np.concatenate([times[1], np.array([[[0, 0, 0, 0, 0]]])])
    timings = np.concatenate(times, axis=1)
    timings = timings[:, [1, 0, 2, 3, 4, 5, 6, 7]]
    # labels = ["I1e", "C1e", "LoHi", "Scaffold", "Weight", "MaxMin", "Butina", "Fingerprint"]
    labels = ["C1e", "I1e", "LoHi", "Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"]
    x = np.array(list(sorted([6160, 21786, 133885, 1128, 642, 4200, 93087, 41127, 1513, 2039, 7831, 8575, 1427, 1478])))
    for i, label in enumerate(labels):
        tmp = timings[:, i].mean(axis=1)
        tmp_x = x[tmp > 0]
        tmp = tmp[tmp > 0]
        ax.plot(tmp_x, tmp, label={"I1e": "Random (I1)", "C1e": "DataSAIL (S1)", "LoHi": "LoHi"}.get(label, "DC - " + label), color=COLORS[label.lower()], marker=MARKERS[label.lower()], markersize=9)
        if i == 2:
            ax.plot([tmp_x[-1], x[-1]], [tmp[-1], 22180], color=COLORS[label.lower()], marker=MARKERS[label.lower()], markersize=9, linestyle='dashed')
    ax.hlines(1, x[0], x[-1], linestyles="dashed", colors="black")
    ax.text(x[0], 1, "1 sec", verticalalignment="bottom", horizontalalignment="left")
    ax.hlines(60, x[0], x[-1], linestyles="dashed", colors="black")
    ax.text(x[0], 60, "1 min", verticalalignment="bottom", horizontalalignment="left")
    ax.hlines(3600, x[0], x[-1], linestyles="dashed", colors="black")
    ax.text(x[0], 3600, "1 h", verticalalignment="bottom", horizontalalignment="left")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("#Molecules in Dataset")
    ax.set_ylabel("Time for splitting [s] (↓)")
    ax.set_title("Runtime on MoleculeNet")
    if show:
        plt.tight_layout()
        plt.savefig("timing.png")
        plt.show()


if __name__ == '__main__':
    get_tool_times(Path("experiments") / "MPP")
