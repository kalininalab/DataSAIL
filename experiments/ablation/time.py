from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

from experiments.utils import DATASETS, COLORS

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

ax = None
base_path = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "DataSAIL" / "v10" / "MPP"

def read_times(base_path: Path, name, tool):
    data = {"tech": [], "run": [], "time": []}
    counter = {"I1e": 0, "C1e": 0, "lohi": 0, "Scaffold": 0, "Weight": 0, "MaxMin": 0, "Butina": 0, "Fingerprint": 0}
    if (file_path := base_path / tool / name / "time2.txt").exists():
        with open(file_path, "r") as f:
            for i, line in enumerate(f.readlines()):
                if i == 0 and tool != "datasail_new":
                    continue
                tech, time = line.split()
                data["tech"].append(tech)
                data["time"].append(float(time))
                data["run"].append(counter[tech])
                counter[tech] += 1
        df = pd.DataFrame(data)
        df["name"] = name
        df["tool"] = tool.split("_")[0]
        return df
    return None


def map_tech_name(name: str) -> str:
    if name.lower() == "c1e":
        return "DataSAIL (S1)"
    elif name.lower() == "i1e":
        return "Rd. Baseline (I1)"
    elif name.lower() == "lohi":
        return "LoHi"
    elif name.lower() == "maxmin":
        return "DC - MaxMin"
    return "DC - " + name[0].upper() + name[1:].lower()


def map_tech_color(name: str) -> str:
    if name.lower().endswith("scaffold"):
        return "train"
    elif name.lower().endswith("weight"):
        return "test"
    return name


def plot_times(base_path: Path, ax=None):
    if show := ax is None:
        matplotlib.rc('font', **{'size': 16})
        fig = plt.figure(figsize=(20, 10.67))
        gs = gridspec.GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0])
    times = []
    for tool in ["deepchem", "lohi", "datasail_new"]:
        for name in DATASETS.keys():
            if name == "pcba":
                continue
            if (res := read_times(base_path, name, tool)) is not None:
                times.append(res)
    times = pd.concat(times)
    
    names = list(DATASETS.keys())
    names.remove("pcba")
    names, sizes = zip(*list(sorted([(k, DATASETS[k][-1]) for k in names], key=lambda x: x[1])))

    for tech in [
        "C1e", "I1e",
        "lohi",
        "Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"
    ]:
        tmp = sorted(dict(times[times["tech"] == tech].groupby("name")["time"].mean()).items(), key=lambda x: names.index(x[0]))
        if tech == "lohi":
            ax.plot([sizes[names.index(x[0])] for x in tmp[:-1]], [x[1] for x in tmp[:-1]], label=map_tech_name(tech), color=COLORS[map_tech_color(tech.lower())], marker=MARKERS[tech.lower()])
            ax.scatter(sizes[names.index(tmp[-1][0])], tmp[-1][1], linestyle="dashed", color=COLORS[map_tech_color(tech.lower())], marker=MARKERS[tech.lower()])
        else:
            ax.plot([sizes[names.index(x[0])] for x in tmp], [x[1] for x in tmp], label=map_tech_name(tech), color=COLORS[map_tech_color(tech.lower())], marker=MARKERS[tech.lower()])
    
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
        (plot_path := base_path / "plots").mkdir(exist_ok=True, parents=True)
        plt.savefig(plot_path / "timing.png")
        plt.show()


if __name__ == '__main__':
    plot_times(Path("/scratch") / "SCRATCH_SAS" / "roman" / "DataSAIL" / "v10" / "MPP")