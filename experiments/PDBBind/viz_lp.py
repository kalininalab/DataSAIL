import os.path
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from matplotlib import gridspec
from sklearn.manifold import TSNE
from sympy.physics.control.control_plots import matplotlib, plt

from experiments.PDBBind.visualize import embed_smiles, embed_aaseqs, get_bounds, read_markdowntable
from experiments.utils import RUNS, USE_UMAP


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
        if r == 0 and name[-1] in "ef":
            df = pd.read_csv(path / f"split_{r}" / "pdbbind.csv")
            train = df[df["split"] == "train"]
            test = df[df["split"] == "test"]
            smiles = {"train": train["ligands"].tolist(), "test": test["ligands"].tolist()}
            aaseqs = {"train": train["proteins"].tolist(), "test": test["proteins"].tolist()}

            if name[-1] == "e" and os.path.exists("drug_embeds.pkl"):
                drug_embeds = pickle.load(open("drug_embeds.pkl", "rb"))
                print(len(drug_embeds), "drug embedding loaded")

                for split in ["train", "test"]:
                    for smile in smiles[split]:
                        if smile not in encodings["d_map"]:
                            if smile not in drug_embeds:
                                drug_embeds[smile] = embed_smiles(smile)
                            encodings["d_map"][smile] = len(encodings["drugs"])
                            encodings["drugs"].append(drug_embeds[smile])
                        data[f"{split}_ids"].append(encodings["d_map"][smile])
                pickle.dump(drug_embeds, open("drug_embeds.pkl", "wb"))

            elif name[-1] == "f" and os.path.exists("prot_embeds.pkl"):
                prot_embeds = pickle.load(open("prot_embeds.pkl", "rb"))
                print(len(prot_embeds), "protein embedding loaded")

                for split in ["train", "test"]:
                    for aa_seq in aaseqs[split]:
                        if aa_seq not in encodings["p_map"]:
                            if aa_seq not in prot_embeds:
                                prot_embeds[aa_seq] = embed_aaseqs(aa_seq)
                            encodings["p_map"][aa_seq] = len(encodings["prots"])
                            encodings["prots"].append(prot_embeds[aa_seq])
                        data[f"{split}_ids"].append(encodings["p_map"][aa_seq])
                pickle.dump(prot_embeds, open("prot_embeds.pkl", "wb"))

        results_path = path / f"split_{r}" / "results" / "training.log"
        if os.path.exists(results_path):
            metric.append(read_log(results_path))

    data["metric"] = np.array((5, 50)) if len(metric) == 0 else np.array(metric)

    return data


def read_data():
    encodings = {
        "drugs": [],
        "prots": [],
        "d_map": {},
        "p_map": {},
    }
    data = {n: read_single_data(n, Path("pdbbind") / n, encodings) for n in
            ["R", "ICSe", "ICSf", "ICD", "CCSe", "CCSf", "CCD"]}

    if USE_UMAP:
        prot_umap, drug_umap = umap.UMAP(), umap.UMAP()
    else:
        prot_umap = TSNE(n_components=2, learning_rate="auto", init="random", random_state=42)
        drug_umap = TSNE(n_components=2, learning_rate="auto", init="random", random_state=42)

    d_trans = drug_umap.fit_transform(np.array(encodings["drugs"]))
    p_trans = prot_umap.fit_transform(np.array(encodings["prots"]))

    for d, trans in [("e", d_trans), ("f", p_trans)]:
        for m in "IC":
            for split in ["train", "test"]:
                data[f"{m}CS{d}"][f"{split}_coord"] = trans[data[f"{m}CS{d}"][f"{split}_ids"]]
    return data


def plot_embeds(ax, data, mode, legend=None):
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
    ax.set_title(
        f"{'ECFP4' if mode[0] == 'e' else 'ESM'} embeddings of\n{'identity' if mode[1] == 'i' else 'cluster'}-based split")


def plot_full(data):
    matplotlib.rc('font', **{'size': 16})
    axis = 0
    rand, icse, icsf, icd, ccse, ccsf, ccd = data["R"], data["ICSe"], data["ICSf"], data["ICD"], data["CCSe"], data[
        "CCSf"], data["CCD"]
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

    plot_embeds(ax_di, icse, "ei", legend=4)
    plot_embeds(ax_dc, ccse, "ec")
    plot_embeds(ax_pi, icsf, "fi")
    plot_embeds(ax_pc, ccsf, "fc")

    for tech, name in [(rand, "Random"), (icse, "Drug ID-based"), (icsf, "Prot ID-based"), (icd, "ID-based 2D"),
                       (ccse, "Drug cluster-based"), (ccsf, "Prot cluster-based"), (ccd, "cluster-based 2D")]:
        # ax_full.fill_between(x, *get_bounds(tech["metric"], axis=axis), alpha=0.5)
        ax_full.plot(np.average(tech["metric"], axis=axis), label=name)
    ax_full.set_ylabel("MSE of RMSE-prediction")
    ax_full.set_xlabel("Epoch")
    ax_full.set_title("Performance comparison")
    ax_full.margins(x=0)
    ax_full.legend()

    fig.set_size_inches(20, 10)
    fig.tight_layout()
    plt.savefig(f"PDBBind_{'umap' if USE_UMAP else 'tsne'}.png")
    plt.show()


def analyze():
    pkl_name = f"read_lpdata_{'umap' if USE_UMAP else 'tsne'}.pkl"
    if not os.path.exists(pkl_name):  # or True:
        data = read_data()
        with open(pkl_name, "wb") as out:
            pickle.dump(data, out)
    else:
        with open(pkl_name, "rb") as pickled_data:
            data = pickle.load(pickled_data)
    plot_full(data)


if __name__ == '__main__':
    analyze()
