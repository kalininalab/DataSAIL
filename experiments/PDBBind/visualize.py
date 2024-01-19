import os.path
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from matplotlib import gridspec
from sklearn.manifold import TSNE
from sympy.physics.control.control_plots import matplotlib, plt

from experiments.utils import RUNS, USE_UMAP, embed_smiles, embed_aaseqs, colors, set_subplot_label

LINES = {
    "Random": (colors["0d"], "solid"),
    "drug ID-based": (colors["r1d"], "solid"),
    "prot. ID-based": (colors["r1d"], "dashed"),
    "drug similarity-based": (colors["s1d"], "solid"),
    "prot. similarity-based": (colors["s1d"], "dashed"),
    "ID-based 2D": (colors["i2"], "solid"),
    "similarity-based 2D": (colors["c2"], "dashed"),
    "LoHi": (colors["lohi"], "solid"),
    "Butina": (colors["butina"], "solid"),
    "Fingerprint": (colors["fingerprint"], "solid"),
    "MaxMin": (colors["maxmin"], "solid"),
    "Scaffold": (colors["scaffold"], "solid"),
    "Weight": (colors["weight"], "solid"),
    "GraphPart": (colors["graphpart"], "solid"),
}


def read_log(path):
    output = []
    with open(path, "r") as data:
        for line in data.readlines():
            if "Validation Loss" in line:
                output.append(float(line.split(" ")[-1].strip()))
    return output


def read_single_data(name, path, encodings):
    data = {"folder": path, "train_ids": [], "test_ids": []}
    metric = []
    for r in range(RUNS):
        if r == 0 and name[-1] not in "2R":
            df = pd.read_csv(path / f"split_{r}" / "pdbbind.csv")
            train = df[df["split"] == "train"]
            test = df[df["split"] == "test"]
            smiles = {"train": train["ligands"].tolist(), "test": test["ligands"].tolist()}
            aaseqs = {"train": train["proteins"].tolist(), "test": test["proteins"].tolist()}

            if name in {"I1f", "C1f", "graphpart"}:
                embed_path = Path("experiments") / "PDBBind" / "prot_embeds.pkl"
                if os.path.exists(embed_path):
                    prot_embeds = pickle.load(open(embed_path, "rb"))
                else:
                    prot_embeds = {}
                print(len(prot_embeds), "protein embedding loaded")

                for split in ["train", "test"]:
                    for aa_seq in aaseqs[split]:
                        aa_seq = aa_seq.replace(":", "G")[:1022]
                        if aa_seq not in encodings["p_map"]:
                            if aa_seq not in prot_embeds:
                                prot_embeds[aa_seq] = embed_aaseqs(aa_seq)
                            encodings["p_map"][aa_seq] = len(encodings["prots"])
                            encodings["prots"].append(prot_embeds[aa_seq])
                        data[f"{split}_ids"].append(encodings["p_map"][aa_seq])
                pickle.dump(prot_embeds, open(embed_path, "wb"))

            else:
                embed_path = Path("experiments") / "PDBBind" / "drug_embeds.pkl"
                if os.path.exists(embed_path):
                    drug_embeds = pickle.load(open(embed_path, "rb"))
                else:
                    drug_embeds = {}
                print(len(drug_embeds), "drug embedding loaded")

                for split in ["train", "test"]:
                    for smile in smiles[split]:
                        if smile not in encodings["d_map"]:
                            if smile not in drug_embeds:
                                drug_embeds[smile] = embed_smiles(smile)
                            encodings["d_map"][smile] = len(encodings["drugs"])
                            encodings["drugs"].append(drug_embeds[smile])
                        data[f"{split}_ids"].append(encodings["d_map"][smile])
                pickle.dump(drug_embeds, open(embed_path, "wb"))

        results_path = path / f"split_{r}" / "results" / "training.log"
        if os.path.exists(results_path):
            metric.append(read_log(results_path)[-50:])

    data["metric"] = np.array((5, 50)) if len(metric) == 0 else np.array(metric)

    return data


def read_data():
    encodings = {
        "drugs": [],
        "prots": [],
        "d_map": {},
        "p_map": {},
    }
    techniques = [("datasail", "R"), ("datasail", "I1e"), ("datasail", "I1f"), ("datasail", "I2"), ("datasail", "C1e"),
                  ("datasail", "C1f"), ("datasail", "C2"), ("deepchem", "Butina"), ("deepchem", "MinMax"),
                  ("deepchem", "Fingerprint"), ("deepchem", "Scaffold"), ("deepchem", "Weight"), ("lohi", "lohi"),
                  ("graphpart", "graphpart")]
    data = {n: read_single_data(n, Path("experiments") / "PDBBind" / tool / n, encodings) for tool, n in techniques}

    if USE_UMAP:
        prot_umap, drug_umap = umap.UMAP(), umap.UMAP()
    else:
        prot_umap = TSNE(n_components=2, learning_rate="auto", init="random", random_state=42)
        drug_umap = TSNE(n_components=2, learning_rate="auto", init="random", random_state=42)

    d_trans = drug_umap.fit_transform(np.array(encodings["drugs"]))
    p_trans = prot_umap.fit_transform(np.array(encodings["prots"]))

    for d, trans, techniques in [("e", d_trans,
                                  ["I1e", "C1e", "Butina", "MinMax", "Fingerprint", "Scaffold", "Weight", "lohi"]),
                                 ("f", p_trans, ["I1f", "C1f", "graphpart"])]:
        for tech in techniques:
            for split in ["train", "test"]:
                data[tech][f"{split}_coord"] = trans[data[tech][f"{split}_ids"]]
    return data


def plot_embeds(ax, data, embed, tech, legend=None):
    ax.scatter(data["train_coord"][:, 0], data["train_coord"][:, 1], s=5, c=colors["train"], label="train")
    ax.scatter(data["test_coord"][:, 0], data["test_coord"][:, 1], s=5, c=colors["test"], label="test")
    # ax.set_xlabel(f"{'umap' if USE_UMAP else 'tsne'} 1")
    # ax.set_ylabel(f"{'umap' if USE_UMAP else 'tsne'} 2")
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    if legend:
        ax.legend(loc=legend, markerscale=3)
    # ax.set_title(f"{embed} embeddings of the\n{tech} split using t-SNE")


def plot_full(data):
    matplotlib.rc('font', **{'size': 16})
    axis = 0
    rand, icse, icsf, icd, ccse, ccsf, ccd = \
        data["R"], data["I1e"], data["I1f"], data["I2"], data["C1e"], data["C1f"], data["C2"]

    fig = plt.figure(figsize=(20, 8.2))
    gs = gridspec.GridSpec(1, 2, figure=fig)

    gs_left = gs[0].subgridspec(2, 2, hspace=0.22, wspace=0.05)
    gs_right = gs[1].subgridspec(2, 1, hspace=0.22)
    gs_right_top = gs_right[0].subgridspec(1, 2, wspace=0.05)
    ax_di = fig.add_subplot(gs_left[0, 0])
    ax_dc = fig.add_subplot(gs_left[0, 1])
    ax_pi = fig.add_subplot(gs_left[1, 0])
    ax_pc = fig.add_subplot(gs_left[1, 1])
    ax_full = [fig.add_subplot(gs_right_top[0])]
    ax_full += [fig.add_subplot(gs_right_top[1], sharey=ax_full[0]), fig.add_subplot(gs_right[1])]

    plot_embeds(ax_di, icse, "ECFP4", "drug ID-based", legend="lower right")
    set_subplot_label(ax_di, fig, "A")
    plot_embeds(ax_dc, ccse, "ECFP4", "drug similarity-based")
    set_subplot_label(ax_dc, fig, "B")
    plot_embeds(ax_pi, icsf, "ESM2-t12", "prot. ID-based")
    set_subplot_label(ax_pi, fig, "C")
    plot_embeds(ax_pc, ccsf, "ESM2-t12", "prot. similarity-based")
    set_subplot_label(ax_pc, fig, "D")

    # for tech, name in [(rand, "Random"), (icse, "drug ID-based"), (icsf, "prot. ID-based"), (icd, "ID-based 2D"),
    #                    (ccse, "drug similarity-based"), (ccsf, "prot. similarity-based"), (ccd, "similarity-based 2D")]:
    #       ax_full.fill_between(x, *get_bounds(tech["metric"], axis=axis), alpha=0.5)
    #       ax_full.plot(np.average(tech["metric"], axis=axis), label=name)
    #       c, s = LINES[name]
    #       ax_full.plot(smooth(get_mean(tech["metric"], axis=axis)), label=name, color=c, linestyle=s)
    #       set_subplot_label(ax_full, fig, "E")
    # ax_full.set_ylabel("MSE")
    # ax_full.set_xlabel("Epoch")
    # ax_full.set_title("Performance comparison (MSE ↓)")
    # ax_full.margins(x=0)
    # ax_full.legend(loc="lower left")

    root = Path("..") / "DataSAIL" / "experiments" / "PDBBind" / "datasail"
    for i, techniques in enumerate([
        [(icse, "drug ID-based", "I1e"), (ccse, "drug similarity-based", "C1e")],
        [(icsf, "prot. ID-based", "I1f"), (ccsf, "prot. similarity-based", "C1f")],
        [(rand, "Random", "R"), (icd, "ID-based 2D", "I2"), (ccd, "similarity-based 2D", "C2")]
    ]):
        models = ["RF", "SVM", "XGB", "MLP", "DeepDTA"]
        values = [[] for _ in range(len(techniques))]

        for s, (tech, name, t) in enumerate(techniques):
            for model in models[:-1]:
                try:
                    df = pd.read_csv(root / f"{model.lower()}.csv")
                    df["tech"] = df["Name"].apply(lambda x: x.split("_")[0])
                    values[s].append(df[['Perf', 'tech']].groupby("tech").mean().loc[t].values[0])
                except Exception as e:
                    print(e)
                    values[s].append(0)
                    pass
            values[s].append(tech["metric"].min(axis=1).mean())
            # values[s].append(pd.read_csv(root / split / "val_metrics.tsv", sep="\t").max(axis=0).values[1:].mean())
        df = pd.DataFrame(np.array(values).T, columns=[x[1] for x in techniques], index=models)
        c_map = {"I1f": "I1e", "C1f": "C1e", "R": "0d"}
        df.plot.bar(ax=ax_full[i], rot=0, ylabel="MSE", color=[colors[c_map.get(x[2], x[2]).lower()] for x in techniques])
        # ax_full[i].plot(smooth(get_mean(tech["metric"], axis=axis)), label=name, color=LINES[name][0],
        #                 linestyle=LINES[name][1])
        #  ax_full[i].set_ylabel("MSE")
        # ax_full[i].set_xlabel("Epoch")
        # ax_full[i].set_title("Performance comparison (MSE ↓)")
        # ax_full[i].margins(x=0)
        ax_full[i].legend(loc="lower right")
        set_subplot_label(ax_full[i], fig, ["E", "F", "G"][i])

    fig.tight_layout()
    plt.savefig(f"PDBBind_{'umap' if USE_UMAP else 'tsne'}.png", transparent=True)
    plt.show()


def smooth(data, window_size=5):
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2
    smoothed = []
    for i in range(len(data)):
        if i < half_window:
            smoothed.append(np.mean(data[:i + half_window + 1]))
        elif i > len(data) - half_window:
            smoothed.append(np.mean(data[i - half_window:]))
        else:
            smoothed.append(np.mean(list(sorted(data[i - half_window:i + half_window + 1]))[1:-1]))
    return smoothed


def get_mean(data, axis=0):
    min_index = np.argmin(data, axis=axis)
    max_index = np.argmax(data, axis=axis)

    mask = np.ones(data.shape)
    mask[min_index[0], :] = 0
    mask[max_index[0], :] = 0

    # return np.mean(np.ma.masked_array(data, np.array(mask, dtype=bool)), axis=axis)
    return np.mean(data, axis=axis)


def plot_cold_drug(data):
    i1e, c1e, lohi, butina, fingerprint, minmax, scaffold, weight = \
        data["I1e"], data["C1e"], data["lohi"], data["Butina"], data["Fingerprint"], data["MinMax"], data["Scaffold"], \
            data["Weight"]
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, figure=fig)

    gs_left = gs[0].subgridspec(3, 3, hspace=0.3)
    gs_right = gs[1].subgridspec(1, 1)
    ax_i1 = fig.add_subplot(gs_left[0, 0])
    ax_c1 = fig.add_subplot(gs_left[0, 1])
    axl = fig.add_subplot(gs_left[0, 2])
    ax_lh = fig.add_subplot(gs_left[1, 0])
    ax_bu = fig.add_subplot(gs_left[1, 1])
    ax_fi = fig.add_subplot(gs_left[1, 2])
    ax_mm = fig.add_subplot(gs_left[2, 0])
    ax_sc = fig.add_subplot(gs_left[2, 1])
    ax_we = fig.add_subplot(gs_left[2, 2])
    ax_full = fig.add_subplot(gs_right[0])

    plot_embeds(ax_i1, i1e, "ECFP", "drug identity", legend=4)
    plot_embeds(ax_c1, c1e, "ECFP", "drug cluster")
    plot_embeds(ax_lh, lohi, "ECFP", "LoHi")
    plot_embeds(ax_bu, butina, "ECFP", "Butina")
    plot_embeds(ax_fi, fingerprint, "ECFP", "Fingerprint")
    plot_embeds(ax_mm, minmax, "ECFP", "MaxMin")
    plot_embeds(ax_sc, scaffold, "ECFP", "Scaffold")
    plot_embeds(ax_we, weight, "ECFP", "Weight")

    for tech, name in [(i1e, "Drug ID-based"), (c1e, "Drug cluster-based"), (lohi, "LoHi"), (butina, "Butina"),
                       (fingerprint, "Fingerprint"), (minmax, "MaxMin"), (scaffold, "Scaffold"), (weight, "Weight")]:
        # ax_full.fill_between(x, *get_bounds(tech["metric"], axis=axis), alpha=0.5)
        # ax_full.plot(np.average(tech["metric"], axis=0), label=name)
        c, s = LINES[name]
        ax_full.plot(smooth(get_mean(tech["metric"], axis=0)), label=name, color=c, linestyle=s)
    ax_full.set_ylabel("MSE")
    ax_full.set_xlabel("Epoch")
    ax_full.set_title("Performance comparison")
    ax_full.margins(x=0)
    # ax_full.legend(loc=3)
    for _ in range(8):
        axl.plot([], [], visible=False)
    legend = axl.legend(["ID-based", "Cluster-based", "LoHi", "Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"],
                        loc="center right", markerscale=10, fontsize=15)
    for handle in legend.legend_handles:
        handle.set_visible(True)
    axl.set_axis_off()

    fig.set_size_inches(25, 12)
    fig.tight_layout()
    plt.savefig(Path("experiments") / "PDBBind" / f"PDBBind_CD_{'umap' if USE_UMAP else 'tsne'}.png")
    plt.show()


def plot_cold_prot(data):
    i1f, c1f, graphpart = data["I1f"], data["C1f"], data["graphpart"]
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, figure=fig)  # , width_ratios=[1, 2])

    gs_left = gs[0].subgridspec(2, 2, hspace=0.3)
    gs_right = gs[1].subgridspec(1, 1, hspace=0.3)
    ax_i1 = fig.add_subplot(gs_left[0, 0])
    ax_c1 = fig.add_subplot(gs_left[0, 1])
    ax_gp = fig.add_subplot(gs_left[1, 0])
    axl = fig.add_subplot(gs_left[1, 1])
    ax_full = fig.add_subplot(gs_right[0])

    plot_embeds(ax_i1, i1f, "ESM", "prot identity", legend=4)
    plot_embeds(ax_c1, c1f, "ESM", "prot cluster")
    plot_embeds(ax_gp, graphpart, "ESM", "GraphPart")

    for tech, name in [(i1f, "Prot ID-based"), (c1f, "Prot cluster-based"), (graphpart, "GraphPart")]:
        # ax_full.fill_between(x, *get_bounds(tech["metric"], axis=axis), alpha=0.5)
        # ax_full.plot(np.average(tech["metric"], axis=0), label=name)
        c, s = LINES[name]
        ax_full.plot(smooth(get_mean(tech["metric"], axis=0)), label=name, color=c, linestyle=s)
    ax_full.set_ylabel("MSE")
    ax_full.set_xlabel("Epoch")
    ax_full.set_title("Performance comparison")
    ax_full.margins(x=0)
    # ax_full.legend(loc=3)
    for _ in range(3):
        axl.plot([], [], visible=False)
    legend = axl.legend(["ID-based", "Cluster-based", "GraphPart"], loc="center right", markerscale=10, fontsize=15)
    for h, handle in enumerate(legend.legend_handles):
        handle.set_visible(True)
    axl.set_axis_off()

    fig.set_size_inches(20, 10)
    fig.tight_layout()
    plt.savefig(Path("experiments") / "PDBBind" / f"PDBBind_CT_{'umap' if USE_UMAP else 'tsne'}.png")
    plt.show()


def viz_sl():
    root = Path("experiments") / "PDBBind"
    models = ["RF", "SVM", "XGB", "MLP", "D-MPNN"]
    values = {
        "drug": [[] for _ in range(8)],
        "target": [[] for _ in range(3)],
        "both": [[] for _ in range(3)],
    }

    for s, tool, tech, mode in [
        (0, "datasail", "R", "both"),
        (0, "datasail", "I1e", "drug"),
        (0, "datasail", "I1f", "target"),
        (1, "datasail", "I2", "both"),
        (1, "datasail", "C1e", "drug"),
        (1, "datasail", "C1f", "target"),
        (2, "datasail", "C2", "both"),
        (2, "deepchem", "Butina", "drug"),
        (3, "deepchem", "MinMax", "drug"),
        (4, "deepchem", "Fingerprint", "drug"),
        (5, "deepchem", "Scaffold", "drug"),
        (6, "deepchem", "Weight", "drug"),
        (7, "lohi", "lohi", "drug"),
        (2, "graphpart", "graphpart", "target"),
    ]:
        for model in models[:-1]:
            df = pd.read_csv(root / tool / f"{model.lower()}.csv")
            df["run"] = df["Name"].apply(lambda x: int(x.split("_")[1]))
            df["tech"] = df["Name"].apply(lambda x: x.split("_")[0])
            values[mode][s].append(df[df["tech"] == tech]["Perf"].mean())
        vals = 0
        for run in range(RUNS):
            path = Path("experiments") / "PDBBind" / tool / tech / f"split_{run}"
            vals += max(read_log(path / "results" / "training.log"))
        values[mode][s].append(vals / 5.0)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    df = pd.DataFrame(np.array(values["drug"]).T,
                      columns=["random cold-drug", "similarity-based", "LoHi", "Butina", "Fingerprint", "MaxMin",
                               "Scaffold", "Weight"], index=models)
    df.plot.bar(ax=axs[0], rot=0, ylabel="MSE")
    df = pd.DataFrame(np.array(values["target"]).T, columns=["random cold-target", "similarity-based", "GraphPart"],
                      index=models)
    df.plot.bar(ax=axs[1], rot=0, ylabel="MSE")
    df = pd.DataFrame(np.array(values["both"]).T, columns=["Random", "identity-based", "similarity-based"],
                      index=models)
    df.plot.bar(ax=axs[2], rot=0, ylabel="MSE")
    fig.tight_layout()
    plt.savefig(root / "sl.png")
    plt.show()


def analyze():
    pkl_name = Path("..") / "DataSAIL" / "experiments" / "PDBBind" / f"{'umap' if USE_UMAP else 'tsne'}_embeds.pkl"
    if not os.path.exists(pkl_name):  # or True:
        data = read_data()
        with open(pkl_name, "wb") as out:
            pickle.dump(data, out)
    else:
        with open(pkl_name, "rb") as pickled_data:
            data = pickle.load(pickled_data)
    plot_full(data)
    # plot_cold_drug(data)
    # plot_cold_prot(data)


if __name__ == '__main__':
    analyze()
    # viz_sl()
