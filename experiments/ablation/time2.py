from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from experiments.utils import DATASETS, TECHNIQUES, RUNS


def read_times(base_path: Path, name, tool):
    data = {"tech": [], "run": [], "time": []}
    counter = {"lohi": 0, "Scaffold": 0, "Weight": 0, "MaxMin": 0, "Butina": 0, "Fingerprint": 0}
    with open(base_path / tool / name / "time2.txt") as f:
        for line in f.readlines()[1:]:
            tech, time = line.split()
            data["tech"].append(tech)
            data["time"].append(float(time))
            data["run"].append(counter[tech])
            counter[tech] += 1
    df = pd.DataFrame(data)
    df["name"] = name
    df["tool"] = tool
    return df


def plot_times(base_path: Path, ax=None, fig=None):
    if show := ax is None:
        fig, ax = plt.subplots()
    times = []
    # for tool in ["datasail", "deepchem", "lohi"]:
    for tool in ["deepchem", "lohi"]:
        # for name in ['qm8', 'bbbp', 'clintox', 'freesolv']:  # DATASETS.keys():
        for name in DATASETS.keys():
            if name == "pcba":
                continue
            times.append(read_times(base_path, name, tool))
    times = pd.concat(times)

    # names, sizes = zip(*list(sorted([(k, DATASETS[k][-1]) for k in ['qm8', 'bbbp', 'clintox', 'freesolv']], key=lambda x: x[1])))
    names = list(DATASETS.keys())
    names.remove("pcba")
    names, sizes = zip(*list(sorted([(k, DATASETS[k][-1]) for k in names], key=lambda x: x[1])))

    # values = times[["tech", "name", "time"]].groupby(["name", "tech"])["time"] \
    #     .mean().reset_index().pivot(index="tech", columns="name", values="time")
    # values = np.array(values.reindex([
    #     "C1e", "I1e", "lohi", "Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"
    # ])[names], dtype=float)

    for tech in [
        # "C1e", "I1e",
        # "lohi",
        "Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"
    ]:
        ax.plot(sizes, times[times["tech"] == tech].groupby("name")["time"].mean(), label=tech)

    ax.hlines(1, sizes[0], sizes[-1], linestyles="dashed", colors="black")
    ax.text(sizes[0], 1, "1 sec", verticalalignment="bottom", horizontalalignment="left")
    ax.hlines(60, sizes[0], sizes[-1], linestyles="dashed", colors="black")
    ax.text(sizes[0], 60, "1 min", verticalalignment="bottom", horizontalalignment="left")
    ax.hlines(3600, sizes[0], sizes[-1], linestyles="dashed", colors="black")
    ax.text(sizes[0], 3600, "1 h", verticalalignment="bottom", horizontalalignment="left")

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
        # plt.savefig("timing.png")
        plt.show()


def collect_times():
    base = Path("/scratch") / "SCRATCH_SAS" / "roman" / "DataSAIL" / "v03" / "MPP"
    for tool in ["deepchem", "lohi"]:
        for name in DATASETS.keys():
            if name == "pcba" or (name == "muv" and tool == "lohi"):
                continue
            with open(base / tool / name / "time2.txt", "w") as out:
                print("Start", file=out)
                for tech in TECHNIQUES[tool]:
                    for run in range(RUNS):
                        with open(base / tool / name / tech / f"split_{run}" / "time.txt") as f:
                            for line in f.readlines():
                                print(line, file=out, end="")


if __name__ == '__main__':
    # collect_times()
    plot_times(Path("/scratch") / "SCRATCH_SAS" / "roman" / "DataSAIL" / "v03" / "MPP")
