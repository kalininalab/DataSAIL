import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.manifold import TSNE
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
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


def embed(full_path, name):
    print("Embedding - read data ...")
    i_tr = pd.read_csv(full_path / "datasail" / name / "I1e" / "split_0" / "train.csv")
    i_te = pd.read_csv(full_path / "datasail" / name / "I1e" / "split_0" / "test.csv")
    c_tr = pd.read_csv(full_path / "datasail" / name / "C1e" / "split_0" / "train.csv")
    c_te = pd.read_csv(full_path / "datasail" / name / "C1e" / "split_0" / "test.csv")

    print("Embedding - compute fingerprints ...")
    smiles = [(s, embed_smiles(s)) for s in set(list(i_tr["SMILES"]) + list(i_te["SMILES"]) +
                                                list(c_tr["SMILES"]) + list(c_te["SMILES"]))]
    ids, fps = zip(*smiles)

    print("Embedding - compute t-SNE ...")
    embedder = TSNE(n_components=2, learning_rate="auto", init="random", random_state=42)
    embeddings = embedder.fit_transform(np.array(fps))

    print("Embedding - relocate samples ...")
    embed_map = {idx: emb for idx, emb in zip(ids, embeddings)}
    return np.stack(i_tr["SMILES"].apply(lambda x: embed_map[x])), np.stack(
        i_te["SMILES"].apply(lambda x: embed_map[x])), \
        np.stack(c_tr["SMILES"].apply(lambda x: embed_map[x])), np.stack(c_te["SMILES"].apply(lambda x: embed_map[x]))


def plot_embeds_2(ax, train, test, title, legend=None):
    n_train = len(train)
    n_test = len(test)

    p = np.concatenate([train, test])
    c = np.array([colors["train"]] * n_train + [colors["test"]] * n_test)
    perm = np.random.permutation(len(p))
    ax.scatter(p[perm, 0], p[perm, 1], s=5, c=c[perm])
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        train_dot = Line2D([0], [0], marker='o', label="train", color=colors["train"], linestyle='None')
        test_dot = Line2D([0], [0], marker='o', label="test", color=colors["test"], linestyle='None')
        handles.extend([train_dot, test_dot])
        ax.legend(handles=handles, loc="lower right", markerscale=2)


def plot_double(full_path, names):
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
    perf = {name: read_perf(full_path, name) for name in names}
    for i, name in enumerate(names):
        df = perf[name]
        df.rename(index={"I1e": "Random baseline (I1)", "C1e": "DataSAIL split (S1)"}, inplace=True)
        df[df.index.isin(["Random baseline (I1)", "DataSAIL split (S1)"])].T.plot.bar(
            ax=ax[i][2], rot=0,
            ylabel=METRICS[DATASETS.index(name)],
            color=[colors["r1d"], colors["s1d"]],
        ).legend(loc="lower right")
        set_subplot_label(ax[i][2], fig, ["C", "F"][i])

        i_tr, i_te, c_tr, c_te = embed(full_path, name.lower())
        plot_embeds_2(ax[i][0], i_tr, i_te, "Stratified baseline", legend=True)
        set_subplot_label(ax[i][0], fig, "A")
        plot_embeds_2(ax[i][1], c_tr, c_te, "DataSAIL split (S1 w/ classes)")
        set_subplot_label(ax[i][1], fig, "B")
    plt.tight_layout()
    plt.savefig(f"QM8_Tox21.png")
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
                df = pd.read_csv(
                    root / "datasail_old" / name.lower() / f"{model.lower()}-{mpp_datasets[name.lower()][1][0]}.csv")
                values[i].append(
                    df[[f"{split}_0", f"{split}_1", f"{split}_2", f"{split}_3", f"{split}_4"]].values.mean(axis=1)[0])
            df = pd.read_csv(root / "datasail_old" / name.lower() / f"val_metrics.tsv", sep="\t")
            values[i].append(df[[c for c in df.columns if c.startswith(split[0])]].values.max(axis=0).mean())

        df = pd.DataFrame(np.array(values).T, columns=["Random baseline (I1)", "DataSAIL (S1)"], index=models)
        ax = df.plot.bar(ax=ax, rot=0, ylabel=METRICS[DATASETS.index(name)], ylim=(0.5, 0.9),
                         color=[colors["r1d"], colors["s1d"]])
        ax.set_xlabel("ML Models")
        ax.set_title(f"{name} - Performance Comparison")
        if show:
            plt.tight_layout()
            plt.savefig(root / f"{name}.png")
            plt.show()


def read_perf(full_path, name):
    models = ["RF", "SVM", "XGB", "MLP", "D-MPNN"]
    tools = ["datasail", "datasail", "lohi", "deepchem", "deepchem", "deepchem", "deepchem", "deepchem"]
    techniques = ["I1e", "C1e", "lohi", "Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"]
    offset = [0, 0, 1, 2, 2, 2, 2, 2]
    df = pd.DataFrame(columns=models, index=techniques)
    for i, (tool, tech) in enumerate(zip(tools, techniques)):
        base = full_path / tool / name.lower()
        for model in models:
            if model != "D-MPNN":
                model_name = f"{model.lower()}-{mpp_datasets[name.lower()][1][0]}"
                if (base / f"{model_name}.csv").exists():
                    with open(base / f"{model_name}.csv") as f:
                        idx = (i - offset[i]) * 5
                        df.at[tech, model] = np.mean([float(x) for x in f.readlines()][idx:idx + 5])
                else:
                    perf = []
                    for run in range(RUNS):
                        try:
                            if (base / f"{model_name}_{tech}_{run}.txt").exists():
                                with open(base / f"{model_name}_{tech}_{run}.txt") as f:
                                    if len(line := f.readlines()[0].strip()) > 2:
                                        perf.append(float(float(line)))
                        except:
                            pass
                    if len(perf) > 0:
                        df.at[tech, model] = np.mean(perf)
            elif (base / f"test_metrics.tsv").exists():
                try:
                    tmp = pd.read_csv(base / f"test_metrics.tsv", sep="\t")
                    cols = [c for c in tmp.columns if tech in c]
                    if len(cols) > 0:
                        df.at[tech, "D-MPNN"] = tmp[cols].values.mean()
                    else:
                        df.at[tech, "D-MPNN"] = check_tb(base, tech)
                except:
                    df.at[tech, "D-MPNN"] = check_tb(base, tech)
            else:
                df.at[tech, "D-MPNN"] = check_tb(base, tech)
    return df


def check_tb(base, tech):
    try:
        perfs = []
        for run in range(RUNS):
            path = base / tech / f"split_{run}" / "fold_0" / "model_0"
            files = list(sorted(filter(
                lambda x: x.startswith("events"), os.listdir(path)
            ), key=lambda x: os.path.getsize(Path(path) / x)))
            for tb_file in files:
                ea = EventAccumulator(str(path / tb_file))
                ea.Reload()
                broken = False
                for metric in filter(lambda x: x.startswith("test_"), ea.Tags()["scalars"]):
                    perf = [e.value for e in ea.Scalars(metric)]
                    if len(perf) > 0:
                        perfs.append(perf[-1])
                        broken = True
                        break
                if broken:
                    break
        if len(perfs) > 0:
            return np.mean(perfs)
    except:
        pass


def heatmap_plot(full_path):
    matplotlib.rc('font', **{'size': 16})
    models = ["RF", "SVM", "XGB", "MLP", "D-MPNN"]
    techniques = ["I1e", "C1e", "lohi", "Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"]

    dfs = {name: read_perf(full_path, name) for name in DATASETS}

    fig = plt.figure(figsize=(20, 25))
    cols, rows = 4, 4
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    axs = [fig.add_subplot(gs[i, j]) for i in range(rows) for j in range(cols)]

    for i, (name, df) in enumerate(dfs.items()):
        if mpp_datasets[name.lower()][1] == "classification":
            cmap = LinearSegmentedColormap.from_list("Custom", [colors["r1d"], colors["s1d"]], N=256)
            cmap.set_bad(color="white")
        else:
            cmap = LinearSegmentedColormap.from_list("Custom", [colors["train"], colors["test"]], N=256)
            cmap.set_bad(color="white")
        values = np.array(df.values, dtype=float)
        values = values[[1, 0, 2, 3, 4, 5, 6, 7], :]
        pic = axs[i].imshow(values, cmap=cmap, vmin=np.nanmin(values), vmax=np.nanmax(values))
        for b in range(len(models)):
            for a in range(len(techniques)):
                if np.isnan(values[a, b]):
                    continue
                label = f"{values[a, b]:.2f}"
                axs[i].text(b, a, label, ha='center', va='center')
        if i % cols == 0:
            axs[i].set_yticks(range(len(techniques)),
                              ["DataSAIL (S1)", "Rd. basel. (I1)", "LoHi", "DC - Butina", "DC - Fingerp.",
                               "DC - MaxMin", "DC - Scaffold", "DC - Weight"])
        else:
            axs[i].set_yticks([])
        axs[i].set_xticks(range(len(models)), models)
        axs[i].set_xlabel("ML Models")
        axs[i].set_title(name)

        cax = make_axes_locatable(axs[i]).append_axes('right', size='5%', pad=0.05)
        fig.colorbar(pic, cax=cax, orientation='vertical', label=METRICS[i])

    for i in range(14, cols * rows):
        axs[i].set_axis_off()

    fig.tight_layout()
    plt.savefig(full_path / f"MoleculeNet_comp.png", transparent=True)
    plt.show()


if __name__ == '__main__':
    # viz_sl(["Tox21"])
    # plot_single("Lipophilicity")
    # plot_perf()
    # plot_perf_5x3()
    plot_double(Path(sys.argv[1]), ["QM8", "Tox21"])
    # plot_embeds()
    # heatmap_plot(Path(sys.argv[1]))
