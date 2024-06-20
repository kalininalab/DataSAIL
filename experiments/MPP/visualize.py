import sys
import pickle
from pathlib import Path
from typing import List
import traceback

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt, gridspec, cm, colors as mpl_colors
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import deepchem as dc

from datasail.reader.utils import DataSet
from experiments.ablation import david
from experiments.ablation.david import run_ecfp
from experiments.utils import DATASETS, COLORS, set_subplot_label, HSPACE, METRICS, embed, plot_embeds, dc2pd, RUNS, \
    TECHNIQUES, plot_bars_2y, DS2UPPER, create_heatmap


def compute_il(root, name, tools, techniques, mode="M"):
    """
    root = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "DataSAIL" / "v10" / "Strat"
    root = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "DataSAIL" / "v10" / "MPP"
    """
    dataset = DATASETS[name][0](featurizer=dc.feat.DummyFeaturizer(), splitter=None)[1][0]
    df = dc2pd(dataset, name)
    dataset = DataSet(
        names=df["ID"].tolist(),
        data=dict(df[["ID", "SMILES"]].values.tolist()),
        id_map={x: x for x in df["ID"].tolist()},
    )
    dataset.cluster_names, dataset.cluster_map, dataset.cluster_similarity, dataset.cluster_weights = run_ecfp(
        dataset
    )
    names = set(dataset.cluster_names)
    output = {}
    if mode == "S":
        output["datasail"] = []
        output["deepchem"] = []
        for run in range(5):
            for d in ["datasail", "deepchem"]:
                if d == "datasail":
                    base = root / d / "d_0.1_e_0.1" / f"split_{run}"
                else:
                    base = root / d / f"split_{run}"
                train_ids = set(pd.read_csv(base / "train.csv")["SMILES"].values)
                test_ids = set(pd.read_csv(base / "test.csv")["SMILES"].values)
                df["assi"] = df["SMILES"].apply(lambda x: 1 if x in train_ids else -1 if x in test_ids else 0)
                df.dropna(subset=["assi"], inplace=True)
                il, total = david.eval(
                    df["assi"].to_numpy().reshape(-1, 1),
                    dataset.cluster_similarity,
                    [dataset.cluster_weights[c] for c in dataset.cluster_names],
                )
                print(il, total)
                output[d].append((il, total))
    else:
        for tool in tools:
            if tool not in output:
                output[tool] = {}

            for technique in techniques:
                if technique not in TECHNIQUES[tool]:
                    continue
                if technique not in output[tool]:
                    output[tool][technique] = []
                for run in range(RUNS):
                    try:
                        base = root / tool / name / technique / f"split_{run}"
                        train_ids = pd.read_csv(base / "train.csv")["ID"]
                        test_ids = pd.read_csv(base / "test.csv")["ID"]
                        df["assi"] = df["ID"].apply(lambda x: 1 if x in train_ids.values else -1 if x in test_ids.values else 0 if x in names else None)
                        # df["assi"] = df["ID"].apply(lambda x: 1 if x in train_ids.values else -1 if x in test_ids.values else 0)
                        df.dropna(subset=["assi"], inplace=True)
                        il, total = david.eval(
                            df["assi"].to_numpy().reshape(-1, 1),
                            dataset.cluster_similarity,
                            [dataset.cluster_weights[c] for c in dataset.cluster_names],
                        )
                        output[tool][technique].append(il)
                    except Exception as e:
                        print("=" * 20, "EXCEPTION", "=" * 20)
                        print(name, tool, technique, run)
                        print(e)
                        print("-" * 51)
                        traceback.print_exc()
                        print("=" * 51)
                        # pass
    return output


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
    
    if (leak_path := full_path / "data" / "leakage.pkl").exists():
        with open(leak_path, "rb") as f:
            leakage = pickle.load(f)
    else:
        leakage = comp_all_il()
        with open(leak_path, "wb") as f:
            pickle.dump(leakage, f)
    
    for i, name in enumerate(names):
        il = {tech: leakage[name.lower()][tool][tech] for tool, techniques in [("datasail", ["I1e", "C1e"]), ("deepchem", TECHNIQUES["deepchem"]), ("lohi", TECHNIQUES["lohi"])] for tech in techniques}
        df = perf[name]
        df["model"] = df["model"].apply(lambda x: x.upper())
        df = df.loc[df["tech"].isin(["I1e", "Fingerprint", "lohi", "C1e"]), ["tech", "model", "perf"]].groupby(["model", "tech"])["perf"] \
            .mean().reset_index().pivot(index="model", columns="tech", values="perf")
        df.loc["Split"] = [np.average(il[tech]) for tech in df.columns]
        df = df.loc[["RF", "SVM", "XGB", "MLP", "D-MPNN", "Split"], ["C1e", "lohi", "Fingerprint", "I1e"]]
        df.rename(columns={"I1e": "Random baseline (I1)", "C1e": "DataSAIL split (S1)", "lohi": "LoHi"}, inplace=True)
        plot_bars_2y(df.T, ax[i][2], color=[COLORS["r1d"], COLORS["lohi"], COLORS["fingerprint"], COLORS["s1d"]])
        ax[i][2].set_ylabel(METRICS[DATASETS[name.lower()][2]])
        ax[i][2].legend(loc="lower left", framealpha=1)
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

    fig = plt.figure(figsize=(30, 30))
    cols, rows = 4, 4
    gs_main = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[85, 3], wspace=0.1)
    gs = gs_main[0].subgridspec(rows, cols, wspace=0.3, hspace=0.25)
    ax = fig.add_subplot(gs_main[1])
    
    if (leak_path := full_path / "data" / "leakage.pkl").exists():
        with open(leak_path, "rb") as f:
            leakage = pickle.load(f)
    else:
        leakage = comp_all_il()
        with open(leak_path, "wb") as f:
            pickle.dump(leakage, f)
            
    il = {
        name: {
            tech: leakage[name.lower()][tool][tech] for tool, techniques in [("datasail", ["I1e", "C1e"]), ("deepchem", TECHNIQUES["deepchem"]), ("lohi", TECHNIQUES["lohi"])] for tech in techniques
        } for name in leakage.keys()
    }
    max_leak = max([leak for name in il.keys() for tech in il[name].keys() for leak in il[name][tech]])
    leak_cmap = mpl_colors.LinearSegmentedColormap.from_list("cyan_magenta", ["cyan", "magenta"])

    def get_il(name, tech):
        if name in il and tech in il[name]:
            return np.average(il[name][tech])
        return 0
    
    for i, (name, df) in enumerate(dfs.items()):
        if DATASETS[name.lower()][1] == "classification":
            cmap = LinearSegmentedColormap.from_list("Custom", [COLORS["r1d"], COLORS["s1d"]], N=256)
            cmap.set_bad(color="white")
        else:
            cmap = LinearSegmentedColormap.from_list("Custom", [COLORS["train"], COLORS["test"]], N=256)
            cmap.set_bad(color="white")
        df["model"] = df["model"].apply(lambda x: x.upper())
        df = df[["tech", "model", "perf"]].groupby(["model", "tech"])["perf"] \
            .mean().reset_index().pivot(index="tech", columns="model", values="perf")
        df["Split"] = [get_il(name, tech) for tech in df.index]
        values = np.array(df.loc[
            ["C1e", "I1e", "lohi", "Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"],
            ["RF", "SVM", "XGB", "MLP", "D-MPNN", "Split"], 
        ], dtype=float)
        create_heatmap(values[:, :-1], values[:, -1:], cmap, leak_cmap, fig, gs[i // 4, i % 4], name, METRICS[DATASETS[name.lower()][2]], y_labels=(i % cols == 0), mode="MMB", max_val=max_leak, yticklabels=["DataSAIL (S1)", "Rd. basel. (I1)", "LoHi", "DC - Butina", "DC_Fingerp.", "DC - MinMax", "DC - Scaffold", "DC - Weight"])
    
    plt.colorbar(cm.ScalarMappable(mpl_colors.Normalize(0, max_leak), leak_cmap), cax=ax, label="$L(\pi)$ ↓")
    plt.savefig(full_path / "plots" / f"MoleculeNet_comp.png", transparent=True)
    plt.show()


def get_perf(full_path: Path, name: str) -> pd.DataFrame:
    dfs = []
    for tool in ["datasail", "lohi", "deepchem"]:
        if (name.lower() == "muv" and tool == "lohi"):
            dfs.append(pd.DataFrame({
                "name": ["LoHi_0", "LoHi_0", "LoHi_0", "LoHi_0", "LoHi_0"],
                "perf": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "model": ["RF", "SVM", "XGB", "MLP", "d-mpnn"],
                "tool": ["lohi", "lohi", "lohi", "lohi", "lohi"],
                "tech": ["lohi", "lohi", "lohi", "lohi", "lohi"],
                "run": [0, 0, 0, 0, 0],
                "dataset": ["muv", "muv", "muv", "muv", "muv"],
            }))
        else:
            dfs.append(pd.read_csv(full_path / tool / name.lower() / "results.csv", index_col=None))
    return pd.concat(dfs)


def comp_all_il(base: Path):
    if (pkl_path := base / "data" / "leakage.pkl").exists():
        with open(pkl_path, "rb") as f:
            output = pickle.load(f)
    else:
        output = {}
    
    for name in DATASETS:
        if name in output or name == "pcba":
            print(f"Leakage already exists for {name}")
            continue
        try:
            output[name] = compute_il(
                root=base,
                name=name.lower(),
                tools=["datasail", "deepchem", "lohi"],
                techniques=["I1e", "C1e"] + TECHNIQUES["deepchem"] + TECHNIQUES["lohi"],
                mode="M",
            )
            print(f"Leakage computed for {name}")
        except Exception as e:
            print(f"Computation failed for {name}")
            print(e)
    with open(pkl_path, "wb") as f:
        pickle.dump(output, f)
    return output


if __name__ == '__main__':
    # comp_all_il(Path(sys.argv[1]))
    plot_double(Path(sys.argv[1]), ["QM8", "Tox21"])
    # plot_double(Path(sys.argv[1]), ["FreeSolv", "ESOL"])
    # heatmap_plot(Path(sys.argv[1]))

