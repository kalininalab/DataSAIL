import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from umap import UMAP

from experiments.utils import USE_UMAP, embed_smiles, get_bounds, RUNS, mpp_datasets, colors, set_subplot_label, HSPACE

SPLITS = ["I1e", "C1e", "lohi", "Butina", "Fingerprint", "M", "Scaffold", "Weight"]
DATASETS = ["QM7", "QM8", "QM9", "ESOL", "FreeSolv", "Lipophilicity", "MUV", "HIV", "BACE", "BBBP", "Tox21", "ToxCast",
            "SIDER", "ClinTox"]
METRICS = ["MAE ↓"] * 3 + ["RMSE ↓"] * 3 + ["PRC-AUC ↑"] + ["ROC-AUC ↑"] * 7


def plot_embeds():
    fig, axes = plt.subplots(
        len(DATASETS), len(SPLITS),
        figsize=(len(SPLITS) * 4, len(DATASETS) * 3),
    )
    data = [[None for _ in range(len(SPLITS))] for _ in range(len(DATASETS))]
    for i, dataset in enumerate(DATASETS):
        os.makedirs(Path("experiments") / "MPP" / ('umap' if USE_UMAP else 'tsne'), exist_ok=True)
        embed_path = Path("experiments") / "MPP" / ('umap' if USE_UMAP else 'tsne') / f"embeds_{dataset}.pkl"
        if os.path.exists(embed_path):
            data[i] = pickle.load(open(embed_path, "rb"))
            continue
        smiles = set()
        for j, split in enumerate(SPLITS):
            base_filename = lambda x, y: Path("experiments") / "MPP" / y / "cdata" / dataset.lower() / split / \
                                         "split_0" / f"{x}.csv"
            if j == 2:
                filename = lambda x: base_filename(x, "lohi")
            elif j < 2:
                filename = lambda x: base_filename(x, "datasail")
            else:
                filename = lambda x: base_filename(x, "deepchem")
            print(filename("train"))
            if not os.path.exists(filename("train")):
                continue
            try:
                train = pd.read_csv(filename("train"))
                test = pd.read_csv(filename("test"))
                data[i][j] = {"train": train["SMILES"].values, "test": test["SMILES"].values}
                smiles.update(train["SMILES"].values)
                smiles.update(test["SMILES"].values)
            except Exception as e:
                print(f"{dataset} - {split}: {e}")
        smiles = [(s, embed_smiles(s)) for s in smiles]
        if USE_UMAP:
            embedder = UMAP(n_components=2, random_state=42)
        else:
            embedder = TSNE(n_components=2, learning_rate="auto", init="random", random_state=42)
        embed = embedder.fit_transform(np.array([s[1] for s in smiles]))
        smiles = dict(list(zip([s[0] for s in smiles], embed)))
        for j, split in enumerate(SPLITS):
            if data[i][j] is None:
                print("Data is empty -", dataset, "-", split)
                continue
            data[i][j]["train"] = np.array([smiles[s] for s in data[i][j]["train"]])
            data[i][j]["test"] = np.array([smiles[s] for s in data[i][j]["test"]])
        pickle.dump(data[i], open(embed_path, "wb"))

    for i, dataset in enumerate(DATASETS):
        for j, split in enumerate(SPLITS):
            axes[i, j].set_xticks([])
            axes[i, j].set_xticks([], minor=True)
            axes[i, j].set_yticks([])
            axes[i, j].set_yticks([], minor=True)
            try:
                if data[i][j] is None:
                    continue
                axes[i, j].scatter(*data[i][j]["train"].T, s=1)
                axes[i, j].scatter(*data[i][j]["test"].T, s=1)
            except Exception as e:
                pass

    for ax, split in zip(axes[0], ["I1", "C1"] + SPLITS[2:]):
        if split == "MinMax":
            split = "MaxMin"
        elif split == "lohi":
            split = "LoHi"
        ax.set_title(split, fontsize=30, fontweight="bold")

    for ax, ds_name, metric in zip(axes[:, 0], DATASETS, METRICS):
        ax.set_ylabel(f"{ds_name}", rotation=90, fontsize=30, fontweight="bold")

    fig.tight_layout()
    plt.savefig(Path("experiments") / "MPP" / f"MPP_embeds_{'umap' if USE_UMAP else 'tsne'}.png")
    plt.show()


def plot_perf():
    fig, axes = plt.subplots(
        len(DATASETS), len(SPLITS),
        sharex='col', sharey='row',
        figsize=(len(SPLITS) * 4, len(DATASETS) * 3),
    )
    for i, dataset in enumerate(DATASETS):
        for j, split in enumerate(SPLITS):
            base_filename = lambda x: Path("experiments") / "MPP" / x / "cdata" / dataset.lower() / "val_metrics.tsv"
            if j == 2:
                filename = str(base_filename("lohi"))
                filename.replace("cdata", "sdata")
            elif j < 2:
                filename = str(base_filename("datasail"))
            else:
                filename = str(base_filename("deepchem"))
            if not os.path.exists(filename):
                continue
            try:
                table = pd.read_csv(filename, sep="\t")
                mask = [(split.replace("1", "CS") if "1" in split else split) in col for col in table.columns]
                mean = np.average(table[table.columns[mask]].values, axis=1)
                bounds = get_bounds(table[table.columns[mask]].values, axis=1)
                x = np.arange(0.0, 50, 1)
                axes[i, j].fill_between(x, *bounds, alpha=0.5)
                axes[i, j].plot(mean)
            except Exception as e:
                print(f"{dataset} - {split}: {e}")

    for ax, split in zip(axes[0], ["I1", "C1", "LoHi"] + SPLITS[3:]):
        ax.set_title(split, fontsize=20)

    for ax, ds_name, metric in zip(axes[:, 0], DATASETS, METRICS):
        ax.set_ylabel(f"{ds_name} ({metric})", rotation=90, fontsize=16)

    fig.tight_layout()
    plt.savefig(Path("experiments") / "MPP" / "MPP_perf.png")
    plt.show()


def plot_perf_5x3():
    fig, axes = plt.subplots(3, 5, figsize=(20, 9))
    handles, labels = [], []
    for d, dataset in enumerate(DATASETS):
        i = d if d < 7 else d + 1
        print(i, "|", i // 5, "|", i % 5)
        axes[i // 5, i % 5].set_title(dataset, fontsize=15)
        axes[i // 5, i % 5].set_ylabel(METRICS[d], rotation=90, fontsize=10)
        for j, split in enumerate(SPLITS):
            base_name = lambda x: Path("experiments") / "MPP" / x / "cdata" / dataset.lower() / "val_metrics.tsv"
            if j < 2:
                filename = base_name("datasail")
            elif j == 2:
                filename = base_name("lohi")
            else:
                filename = base_name("deepchem")
            if not os.path.exists(filename):
                axes[i // 5, i % 5].plot([], [], visible=False)
                continue
            try:
                table = pd.read_csv(filename, sep="\t")
                f = lambda x: "ICSe" if x == "I1e" else ("CCSe" if x == "C1e" else split)
                mask = [f(split) in col for col in table.columns]
                mean = np.average(table[table.columns[mask]].values, axis=1)
                bounds = get_bounds(table[table.columns[mask]].values, axis=1)
                x = np.arange(0.0, 50, 1)
                axes[i // 5, i % 5].plot(mean)
            except Exception as e:
                print(f"{dataset} - {split}: {e}")
    for _ in range(len(SPLITS)):
        axes[1, 2].plot([], [], visible=False)
    axes[1, 2].legend(loc='center')
    names = list(map(lambda x: "LoHi" if x == "lohi" else ("MaxMin" if x == "MinMax" else x), SPLITS))
    legend = axes[1, 2].legend(names, loc="center", markerscale=10, fontsize=15)
    for h, handle in enumerate(legend.legend_handles):
        handle.set_visible(True)
    axes[1, 2].set_axis_off()

    fig.tight_layout()
    plt.savefig(Path("experiments") / "MPP" / "MPP_perf_5x3.png")
    plt.show()


def plot_double(names):
    matplotlib.rc('font', **{'size': 16})
    root = Path("..") / "DataSAIL" / "experiments" / "MPP"
    fig = plt.figure(figsize=(20, 10.67))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1])
    gs_left = gs[0].subgridspec(2, 2, hspace=HSPACE, wspace=0.1)
    gs_right = gs[1].subgridspec(2, 1, hspace=HSPACE)
    ax = [
        [fig.add_subplot(gs_left[0, 0]), fig.add_subplot(gs_left[0, 1]), fig.add_subplot(gs_right[0])],
        [fig.add_subplot(gs_left[1, 0]), fig.add_subplot(gs_left[1, 1]), fig.add_subplot(gs_right[1])],
    ]
    for i, name in enumerate(names):
        viz_sl([name], ax=ax[i][2])
        set_subplot_label(ax[i][2], fig, ["C", "F"][i])

        data = pickle.load(open(root / ('umap' if USE_UMAP else 'tsne') / f"embeds_{name}.pkl", "rb"))
        for t, tech in enumerate(SPLITS[:2]):
            ax[i][t].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            n_train = len(data[t]["train"])
            n_test = len(data[t]["test"])
            p = np.concatenate([data[t]["train"], data[t]["test"]])
            c = np.array([colors["train"]] * n_train + [colors["test"]] * n_test)
            perm = np.random.permutation(len(p))
            ax[i][t].scatter(p[perm, 0], p[perm, 1], s=5, c=c[perm])
            ax[i][t].set_xlabel("tSNE 1")
            ax[i][t].set_ylabel("tSNE 2")
            ax[i][t].set_title(f"{['QM8', 'Tox21'][i]} - {['Random baseline (I1)', 'DataSAIL split (S1)'][t]}")
            if i == 0 and t == 0:
                handles, labels = ax[i][t].get_legend_handles_labels()
                train_dot = Line2D([0], [0], marker='o', label="train", color=colors["train"], linestyle='None')
                test_dot = Line2D([0], [0], marker='o', label="test", color=colors["test"], linestyle='None')
                handles.extend([train_dot, test_dot])
                ax[i][t].legend(handles=handles, loc="lower right", markerscale=2)
            set_subplot_label(ax[i][t], fig, [["A", "B"], ["D", "E"]][i][t])
    plt.tight_layout()
    plt.savefig(f"QM8_Tox21.png")
    plt.show()


def plot_single(name):
    matplotlib.rc('font', **{'size': 16})
    index = DATASETS.index(name)
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.5])
    gs_left = gs[0].subgridspec(2, 1, hspace=0.3)
    gs_right = gs[1].subgridspec(1, 1)
    ax_rand = fig.add_subplot(gs_left[0])
    ax_cold = fig.add_subplot(gs_left[1])
    ax_full = fig.add_subplot(gs_right[0])

    for s, split in enumerate(SPLITS[:2]):
        filename = f"experiments/MPP/datasail/cdata/{name.lower()}/val_metrics.tsv"
        if not os.path.exists(filename):
            print(filename, "not found")
            continue
        try:
            table = pd.read_csv(filename, sep="\t")
            mask = [split.replace("1", "CS") in col for col in table.columns]
            mean = np.average(table[table.columns[mask]].values, axis=1)
            bounds = get_bounds(table[table.columns[mask]].values, axis=1)
            x = np.arange(0.0, 50, 1)
            ax_full.fill_between(x, *bounds, alpha=0.5)
            ax_full.plot(mean, label='random' if s == 0 else 'clustered')
            ax_full.set_title(f"Performance comparison ({METRICS[index]})")
        except Exception as e:
            print(f"{name} - {split}: {e}")
    ax_full.legend(loc=1, markerscale=5)

    data = pickle.load(open(f"experiments/MPP/{'umap' if USE_UMAP else 'tsne'}/embeds_{name}.pkl", "rb"))
    for t, tech in enumerate(SPLITS[:2]):
        ax = ax_rand if t == 0 else ax_cold
        ax.set_title(tech)
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        try:
            ax.scatter(*data[t]["train"].T, s=3, label="train")
            ax.scatter(*data[t]["test"].T, s=3, label="test")
            ax.set_title(
                f"ECFP4 embeddings of\nthe {'random' if t == 0 else 'cluster-based'} split using t-SNE")
            if t == 0:
                ax.legend(loc=4, markerscale=5)
        except Exception as e:
            print(e)
    fig.set_size_inches(14, 8)
    fig.tight_layout()
    plt.savefig(f"{name}.png", transparent=True)
    plt.show()


def viz_sl(names, ax=None):
    if show := ax is None:
        matplotlib.rc('font', **{'size': 16})
        fig = plt.figure(figsize=(20, 10.67))
        gs = gridspec.GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0])
    for name in names:
        root = Path("..") / "DataSAIL" / "experiments" / "MPP"
        models = ["RF", "SVM", "XGB", "MLP", "D-MPNN"]
        values = [[] for _ in range(2)]

        for i, split in enumerate(["I1e", "C1e"]):
            for model in models[:-1]:
                df = pd.read_csv(root / "datasail_old" / name.lower() / f"{model.lower()}-{mpp_datasets[name.lower()][1][0]}.csv")
                values[i].append(df[[f"{split}_0", f"{split}_1", f"{split}_2", f"{split}_3", f"{split}_4"]].values.mean(axis=1)[0])
            df = pd.read_csv(root / "datasail_old" / name.lower() / f"val_metrics.tsv", sep="\t")
            values[i].append(df[[c for c in df.columns if c.startswith(split[0])]].values.max(axis=0).mean())

        df = pd.DataFrame(np.array(values).T, columns=["Random baseline (I1)", "DataSAIL (S1)"], index=models)
        ax = df.plot.bar(ax=ax, rot=0, ylabel=METRICS[DATASETS.index(name)], ylim=(0.5, 0.9), color=[colors["r1d"], colors["s1d"]])
        ax.set_xlabel("ML Models")
        ax.set_title(f"{name} - Performance Comparison")
        if show:
            plt.tight_layout()
            plt.savefig(root / f"{name}.png")
            plt.show()


if __name__ == '__main__':
    # viz_sl(["Tox21"])
    # plot_single("Lipophilicity")
    # plot_perf()
    # plot_perf_5x3()
    plot_double(["QM8", "Tox21"])
    # plot_embeds()
