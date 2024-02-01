import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import deepchem as dc
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from datasail.reader.read_molecules import read_molecule_data
from experiments.david import eval, run_ecfp
from experiments.utils import RUNS, mpp_datasets, dc2pd, colors


def score_split(full_path, dataset, delta, epsilon):
    base = full_path / "datasail" / f"d_{delta}_e_{epsilon}"
    # base = Path("experiments") / "Tox21Strat" / "datasail" / "de_ablation" / f"d_{delta}_e_{epsilon}"
    vals = []
    for run in range(RUNS):
        print(f"\r{delta}, {epsilon}, {run}", end=" " * 10)
        train = set(pd.read_csv(base / f"split_{run}" / "train.csv")["ID"].tolist())
        tmp2 = np.array([1 if n in train else -1 for n in dataset.names]).reshape(-1, 1)
        vals.append(eval(tmp2, dataset.cluster_similarity))
    return np.mean(vals)


def read_quality(full_path):
    dataset = mpp_datasets["tox21"][0](featurizer=dc.feat.DummyFeaturizer(), splitter=None)[1][0]
    df = dc2pd(dataset, "tox21")
    dataset = read_molecule_data(dict(df[["ID", "SMILES"]].values.tolist()), sim="ecfp")
    dataset.cluster_names, dataset.cluster_map, dataset.cluster_similarity, dataset.cluster_weights = \
        run_ecfp(dataset)

    columns, rows = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05], [0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    df = pd.DataFrame(columns=columns, index=rows)
    for d in columns:
        for e in rows:
            if e <= d:
                df.at[e, d] = score_split(full_path, dataset, d, e)

    # df.at[0.3, 0.1] = 100
    return df


def read_times():
    data = {}
    with open("strat_timing.txt") as f:
        for line in f.readlines():
            d, e, t = [float(x) for x in line.split(",")]
            if (d, e) not in data:
                data[(d, e)] = []
            data[(d, e)].append(t)

    columns, rows = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05], [0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    df = pd.DataFrame(columns=columns, index=rows)
    for k, v in data.items():
        df.at[k[1], k[0]] = sum(v) / len(v)
    return df


def plot_de_ablation(full_path, ax=None, fig=None):
    if Path(full_path / "strat_data.pkl").exists():
        with open(full_path / "strat_data.pkl", "rb") as f:
            time, qual = pickle.load(f)
    else:
        # time = read_times()
        qual = read_quality(full_path)
        with open(full_path / "strat_data.pkl", "wb") as out:
            pickle.dump(({}, qual), out)
    if show := ax is None:
        matplotlib.rc('font', **{'size': 16})
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cmap = LinearSegmentedColormap.from_list("Custom", [colors["r1d"], colors["s1d"]], N=256)
    cmap.set_bad(color="white")
    q_values = np.array(qual.values, dtype=float)[::-1, :].T
    tmp = ax.imshow(q_values, cmap=cmap, vmin=np.nanmin(q_values), vmax=np.nanmax(q_values))
    ax.set_xticks(list(reversed(range(1, 6, 2))), [0.3, 0.2, 0.1])
    ax.set_yticks(list(range(0, 6, 2)), [0.3, 0.2, 0.1])
    ax.set_xlabel("$\epsilon$")
    ax.set_ylabel("$\delta$")
    ax.set_title("Effect of $\delta$ and $\epsilon$")
    fig.colorbar(tmp, cax=cax, orientation='vertical', label="$L(\pi)$ (↓)")

    if show:
        plt.tight_layout()
        plt.savefig("strat.png")
        plt.show()


if __name__ == '__main__':
    plot_de_ablation(Path(sys.argv[1]))
    # print(read_quality())
