import os
import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from utils import mpp_datasets, RUNS

files = []


def get_single_time(path):
    return os.path.getctime(path / "train.csv") - os.path.getctime(path / "start.txt")


def get_run_times(path):
    return [get_single_time(path / f"split_{run}") for run in range(RUNS)]


def get_tech_times(path):
    techniques = ["Scaffold", "Weight", "MinMax", "Butina", "Fingerprint"] if "deepchem" in str(path) else ["ICSe", "CCSe"]
    return [get_run_times(path / tech) for tech in techniques]


def get_dataset_times(path):
    return [get_tech_times(path / ds_name) for ds_name in sorted(os.listdir(path), key=lambda x: mpp_datasets[x][3])]


def get_tool_times(path):
    if not os.path.exists("MPP_timing.pkl"):
        times = np.array(get_dataset_times(path / "datasail" / "sdata")), \
            np.array(get_dataset_times(path / "deepchem" / "sdata"))
        pickle.dump(times, open("MPP_timing.pkl", "wb"))
    else:
        times = pickle.load(open("MPP_timing.pkl", "rb"))
    timings = np.concatenate(times, axis=1)
    print(timings.shape)
    labels = ["ICSe", "CCSe", "Scaffold", "Weight", "MinMax", "Butina", "Fingerprint"]
    x = list(sorted([6160, 21786, 133885, 1128, 642, 4200, 93087, 41127, 1513, 2039, 7831, 8575, 1427, 1478]))
    for i in range(7):
        label = labels[i]
        if label == "ICSe":
            label = "I1"
        if label == "CCSe":
            label = "C1"
        plt.plot(x, timings[:, i].mean(axis=1), label=label)

    plt.legend(framealpha=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("#Molecules in Dataset")
    plt.ylabel("Time for splitting [sec]")
    plt.tight_layout()
    plt.savefig("stiming.png")
    plt.show()


def old():
    for tool in ["datasail", "deepchem"]:
        for ds_name in sorted(os.listdir(Path("MPP") / "deepchem" / "sdata"), key=lambda x: mpp_datasets[x][3]):
            path = Path("MPP") / tool / "sdata" / str(ds_name)
            if not os.path.exists(path):
                continue
            for tech in os.listdir(path):
                filepath = path / str(tech)
                files.append([str(filepath), os.path.getctime(filepath)])
    order = list(sorted(files, key=lambda x: x[1]))
    print("Total time:", (order[-1][1] - order[0][1]))
    times = [[None for _ in range(7)] for _ in range(12)]
    x = list(sorted([6160, 21786, 133885, 1128, 642, 4200, 93087, 41127, 1513, 2039, 7831, 8575, 1427, 1478]))
    techs = ["ICSe", "CCSe", "Scaffold", "Weight", "MinMax", "Butina", "Fingerprint"]
    tech2idx = {n: i for i, n in enumerate(techs)}
    ds2idx = {n: i for i, n in
              enumerate(sorted(os.listdir(Path("MPP") / "deepchem" / "sdata"), key=lambda x: mpp_datasets[x][3]))}
    for i in range(1, len(order)):
        time = order[i][1] - order[i - 1][1]
        order[i].append(time)
        parts = order[i][0].split("/")
        if parts[-1] == "start.txt":
            continue
        times[ds2idx[parts[3]]][tech2idx[parts[4][:-4]]] = time
    print(ds2idx)
    print("\n".join(["\t".join([f"{cell:.3f}" for cell in row]) for row in times]))
    np_times = np.array(times).T

    plt.hlines(1, x[0], x[-1], linestyles="dashed", colors="black")
    plt.hlines(60, x[0], x[-1], linestyles="dashed", colors="black")
    plt.hlines(3600, x[0], x[-1], linestyles="dashed", colors="black")

    for i in range(7):
        print(np_times[i])
        label = techs[i]
        if label == "ICSe":
            label = "I1"
        if label == "CCSe":
            label = "C1"
        plt.plot(x, np_times[i], label=label)

    plt.legend(framealpha=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("#Molecules in Dataset")
    plt.ylabel("Time for splitting [sec]")
    plt.tight_layout()
    plt.savefig("stiming.png")
    plt.show()
    print("\n".join(["\t".join([str(y) for y in x[::-1]]) for x in order]))
    print(times)


if __name__ == '__main__':
    get_tool_times(Path("MPP"))
