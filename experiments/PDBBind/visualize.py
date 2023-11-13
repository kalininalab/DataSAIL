import os.path
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from matplotlib import gridspec
from sklearn.manifold import TSNE
from sympy.physics.control.control_plots import matplotlib, plt

from experiments.utils import RUNS, USE_UMAP, embed_smiles, embed_aaseqs


LINES = {
    "Random": ("black", "solid"),
    "Drug ID-based": ("tab:blue", "solid"),
    "Prot ID-based": ("tab:blue", "dashed"),
    "Drug cluster-based": ("tab:orange", "solid"),
    "Prot cluster-based": ("tab:orange", "dashed"),
    "ID-based 2D": ("gold", "solid"),
    "cluster-based 2D": ("gold", "dashed"),
    "LoHi": ("tab:green", "solid"),
    "Butina": ("tab:red", "solid"),
    "Fingerprint": ("tab:purple", "solid"),
    "MaxMin": ("tab:brown", "solid"),
    "Scaffold": ("tab:pink", "solid"),
    "Weight": ("tab:gray", "solid"),
    "GraphPart": ("tab:cyan", "solid"),
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
    ax.scatter(data["train_coord"][:, 0], data["train_coord"][:, 1], s=1, c="blue", label="train", alpha=0.5)
    ax.scatter(data["test_coord"][:, 0], data["test_coord"][:, 1], s=1, c="orange", label="test", alpha=0.5)
    ax.set_xlabel(f"{'umap' if USE_UMAP else 'tsne'} 1")
    ax.set_ylabel(f"{'umap' if USE_UMAP else 'tsne'} 2")
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.set_yticks([], minor=True)
    if legend:
        ax.legend(loc=legend, markerscale=10)
    ax.set_title(f"{embed} embeddings of\n{tech} split")


def plot_full(data):
    matplotlib.rc('font', **{'size': 16})
    axis = 0
    rand, icse, icsf, icd, ccse, ccsf, ccd = \
        data["R"], data["I1e"], data["I1f"], data["I2"], data["C1e"], data["C1f"], data["C2"]
    x = np.arange(0.0, 50, 1)

    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, figure=fig)

    gs_left = gs[0].subgridspec(2, 2, hspace=0.3)
    gs_right = gs[1].subgridspec(1, 1)
    ax_di = fig.add_subplot(gs_left[0, 0])
    ax_dc = fig.add_subplot(gs_left[0, 1])
    ax_pi = fig.add_subplot(gs_left[1, 0])
    ax_pc = fig.add_subplot(gs_left[1, 1])
    ax_full = fig.add_subplot(gs_right[0])

    plot_embeds(ax_di, icse, "ECFP", "Drug ID-based", legend=4)
    plot_embeds(ax_dc, ccse, "ECFP", "Drug cluster-based")
    plot_embeds(ax_pi, icsf, "ESM", "Prot ID-based")
    plot_embeds(ax_pc, ccsf, "ESM", "Prot cluster-based")

    for tech, name in [(rand, "Random"), (icse, "Drug ID-based"), (icsf, "Prot ID-based"), (icd, "ID-based 2D"),
                       (ccse, "Drug cluster-based"), (ccsf, "Prot cluster-based"), (ccd, "cluster-based 2D")]:
        # ax_full.fill_between(x, *get_bounds(tech["metric"], axis=axis), alpha=0.5)
        # ax_full.plot(np.average(tech["metric"], axis=axis), label=name)
        c, s = LINES[name]
        ax_full.plot(smooth(get_mean(tech["metric"], axis=axis)), label=name, color=c, linestyle=s)
    ax_full.set_ylabel("MSE")
    ax_full.set_xlabel("Epoch")
    ax_full.set_title("Performance comparison")
    ax_full.margins(x=0)
    ax_full.legend(loc=3)

    fig.set_size_inches(20, 10)
    fig.tight_layout()
    plt.savefig(Path("experiments") / "PDBBind" / f"PDBBind_{'umap' if USE_UMAP else 'tsne'}.png")
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
            smoothed.append(np.mean(data[i - half_window:i + half_window + 1]))
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
    legend = axl.legend(["ID-based", "Cluster-based", "LoHi", "Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"], loc="center right", markerscale=10, fontsize=15)
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


def analyze():
    pkl_name = Path("experiments") / "PDBBind" / f"{'umap' if USE_UMAP else 'tsne'}_embeds.pkl"
    if not os.path.exists(pkl_name):  # or True:
        data = read_data()
        with open(pkl_name, "wb") as out:
            pickle.dump(data, out)
    else:
        with open(pkl_name, "rb") as pickled_data:
            data = pickle.load(pickled_data)
    plot_full(data)
    plot_cold_drug(data)
    plot_cold_prot(data)


if __name__ == '__main__':
    analyze()
