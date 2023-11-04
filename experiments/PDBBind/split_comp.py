import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import umap

from experiments.utils import embed_aaseqs


# Comparison of DataSAIL and GraphPart splits (Hi-Lo-Splitter didn't manage to split)

def plot_embeds(ax, data, mask, name):
    # ax.scatter(data[mask == 2, 0], data[mask == 2, 1], s=1, c="gray", label="drop", alpha=0.5)
    ax.scatter(data[mask == 0, 0], data[mask == 0, 1], s=1, c="blue", label="train", alpha=0.5)
    ax.scatter(data[mask == 1, 0], data[mask == 1, 1], s=1, c="orange", label="test", alpha=0.5)
    ax.set_xlabel(name + " 1")
    ax.set_ylabel(name + " 2")
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.set_yticks([], minor=True)
    ax.legend()


# ds_proteins = {}
# ds_ligands = {}
# gp_proteins = {}
lohi_ligands = {}

# df = pd.read_csv(Path("experiments") / "PDBBind" / "lppdbbind" / "C1e" / "split_0" / "pdbbind.csv")
# ds_ligands = dict(df[["ligands", "split"]].values.tolist())
path_pdbbind = Path("experiments") / "PDBBind"

df = pd.read_csv(path_pdbbind / "lppdbbind" / "C1f" / "split_0" / "pdbbind.csv")
df["cluster"] = df["split"].apply(lambda x: 0 if x == "train" else 1)
ds_proteins = dict(df[["ids", "cluster"]].values.tolist())

df = pd.read_csv("graphpart_result.csv")
gp_proteins = dict(df[["AC", "cluster"]].values.tolist())

prot_embeds = pickle.load(open(path_pdbbind / "prot_embeds.pkl", "rb"))
print(len(prot_embeds), "protein embedding loaded")

pdbbind = pd.read_csv("/home/rjo21/Downloads/LP_PDBBind.csv")
embeds = []
ds_prot_mask = []
gp_prot_mask = []
for idx, seq in pdbbind[['Unnamed: 0', 'seq']].values.tolist():
    seq = seq.replace(":", "G")[:1022]
    ds_prot_mask.append(ds_proteins.get(idx, 2))
    gp_prot_mask.append(gp_proteins.get(idx, 2))
    if seq not in prot_embeds:
        prot_embeds[seq] = embed_aaseqs(seq)
    embeds.append(prot_embeds[seq])
pickle.dump(prot_embeds, open(path_pdbbind / "prot_embeds.pkl", "wb"))

tsne = TSNE(n_components=2, learning_rate="auto", init="random", random_state=42)
tsne_embeds = tsne.fit_transform(np.array(embeds))
umap = umap.UMAP()
umap_embeds = umap.fit_transform(np.array(embeds))

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
plot_embeds(axs[0], tsne_embeds, np.array(ds_prot_mask, dtype=int), "tSNE")
plot_embeds(axs[1], tsne_embeds, np.array(gp_prot_mask, dtype=int), "tSNE")
# plot_embeds(axs[0, 1], umap_embeds, np.array(ds_prot_mask, dtype=int), "UMAP")
# plot_embeds(axs[1, 1], umap_embeds, np.array(gp_prot_mask, dtype=int), "UMAP")
axs[0].set_title("DataSAIL - tSNE")
axs[1].set_title("GraphPart - tSNE")
# axs[0, 1].set_title("DataSAIL - UMAP")
# axs[1, 1].set_title("GraphPart - UMAP")
plt.tight_layout()
plt.savefig("split_comp.png")
plt.show()
