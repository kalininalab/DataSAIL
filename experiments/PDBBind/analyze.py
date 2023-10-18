import pickle
import os
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib import gridspec
import torch
import esm
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from experiments.PDBBind.visualize import embed_smiles

USE_UMAP = False
num_layers = 12
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()


def embed_aaseqs(aaseq):
    batch_labels, batch_strs, batch_tokens = batch_converter([("query", aaseq)])
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[num_layers], return_contacts=True)
        token_representations = results["representations"][num_layers]

        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
        return sequence_representations[0].numpy()


def read_markdowntable(path):
    values = []
    with open(path, "r") as table:
        for line in table.readlines()[3:-1]:
            values.append(float(line[12:20].strip()))
    return values


def read():
    data = {
        "rand": {"folder": Path("data_scip_improved") / "R"},
        "icse": {"folder": Path("data_scip_improved") / "ICSe"},
        "icsf": {"folder": Path("data_scip_improved") / "ICSf"},
        "icd": {"folder": Path("data_scip_improved") / "ICD"},
        "ccse": {"folder": Path("data_scip_improved") / "CCSe"},
        "ccsf": {"folder": Path("data_scip_improved") / "CCSf"},
        "ccd": {"folder": Path("data_scip_improved") / "CCD"}
    }
    prot_embeds, drug_embeds = dict(), dict()
    if USE_UMAP:
        prot_umap, drug_umap = umap.UMAP(), umap.UMAP()
    else:
        prot_umap = TSNE(n_components=2, learning_rate="auto", init="random", random_state=42)
        drug_umap = TSNE(n_components=2, learning_rate="auto", init="random", random_state=42)
    for mode, info in data.items():
        info["metric"] = []
        for r in range(5):
            print(info['folder'] / f"split_{r}")
            if r == 0:
                train = pd.read_csv(info['folder'] / f"split_{r}" / "train.csv")
                test = pd.read_csv(info['folder'] / f"split_{r}" / "test.csv")
                smiles = {"train": train["Ligand"].tolist(), "test": test["Ligand"].tolist()}
                aaseqs = {"train": train["Target"].tolist(), "test": test["Target"].tolist()}
                y = {"train": train["y"].tolist(), "test": test["y"].tolist()}
                # smiles, aaseqs, y = pickle.load(open(info['folder'] / f"split_{r}" / "data.pkl", "rb"))
                train_size = len(smiles["train"])

                embeds = []
                if mode[-1] == "e" and os.path.exists("drug_embeds.pkl"):
                    drug_embeds = pickle.load(open("drug_embeds.pkl", "rb"))
                    print(len(drug_embeds), "drug embedding loaded")
                elif mode[-1] == "f" and os.path.exists("prot_embeds.pkl"):
                    prot_embeds = pickle.load(open("prot_embeds.pkl", "rb"))
                    print(len(prot_embeds), "protein embedding loaded")
                print("#Embeds:", len(embeds))

                for split in ["train", "test"]:
                    if mode[-1] == "e":
                        for i, smile in enumerate(smiles[split]):
                            print(f"\r{mode}|{split} - {i}/{len(smiles[split])}", end="\r")
                            if smile not in drug_embeds:
                                drug_embeds[smile] = embed_smiles(smile)
                            embeds.append(drug_embeds[smile])
                        pickle.dump(drug_embeds, open("drug_embeds.pkl", "wb"))
                        info["embed"] = embeds
                        print("Stored drug embeds")

                    if mode[-1] == "f":
                        for i, aaseq in enumerate(aaseqs[split]):
                            print(f"\r{mode}|{split} - {i}/{len(smiles[split])}", end="\r")
                            if aaseq not in prot_embeds:
                                prot_embeds[aaseq] = embed_aaseqs(aaseq)
                            embeds.append(prot_embeds[aaseq])
                        pickle.dump(prot_embeds, open("prot_embeds.pkl", "wb"))
                        info["embed"] = embeds
                        print("Stored prot embeds")
                    print()
                info["split"] = train_size

            if os.path.exists(f"{info['folder']}/split_{r}/results/valid_markdowntable.txt"):
                info["metric"].append(read_markdowntable(f"{info['folder']}/split_{r}/results/valid_markdowntable.txt"))
        if len(info["metric"]) == 0:
            info["metric"] = np.array((5, 50))
        else:
            info["metric"] = np.array(info["metric"])

    id_part = len(data["icse"]["embed"])
    transforms = drug_umap.fit_transform(np.array(data["icse"]["embed"] + data["ccse"]["embed"]))
    data["icse"]["embed"] = transforms[:id_part]
    data["ccse"]["embed"] = transforms[id_part:]

    id_part = len(data["ccse"]["embed"])
    transforms = prot_umap.fit_transform(np.array(data["icsf"]["embed"] + data["ccsf"]["embed"]))
    data["icsf"]["embed"] = transforms[:id_part]
    data["ccsf"]["embed"] = transforms[id_part:]
    return data


def plot_embeds(ax, data, mode, legend=None):
    ax.scatter(data["embed"][:data["split"], 0], data["embed"][:data["split"], 1], s=1, c="blue", label="train",
               alpha=0.5)
    ax.scatter(data["embed"][data["split"]:, 0], data["embed"][data["split"]:, 1], s=1, c="orange", label="test",
               alpha=0.5)
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


def get_bounds(values, axis=0):
    mean = np.mean(values, axis=axis)
    return mean - np.std(values, axis=axis), mean + np.std(values, axis=axis)


def plot_full(data):
    matplotlib.rc('font', **{'size': 16})
    axis = 0
    rand, icse, icsf, icd, ccse, ccsf, ccd = data["rand"], data["icse"], data["icsf"], data["icd"], data["ccse"], data[
        "ccsf"], data["ccd"]
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
        # lower, upper = get_bounds(tech["metric"], axis=axis)
        # ax_full.fill_between(x, *get_bounds(tech["metric"], axis=axis), alpha=0.5)
        ax_full.plot(np.median(tech["metric"], axis=axis), label=name)
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
    pkl_name = f"data_{'umap' if USE_UMAP else 'tsne'}.pkl"
    if not os.path.exists(pkl_name):
        data = read()
        with open(pkl_name, "wb") as out:
            pickle.dump(data, out)
    else:
        with open(pkl_name, "rb") as pickled_data:
            data = pickle.load(pickled_data)
    plot_full(data)


analyze()
