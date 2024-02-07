import os.path
import pickle
from pathlib import Path
from typing import Literal, Dict, List, Union

import numpy as np
import pandas as pd
import umap
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from mpl_toolkits.axes_grid1 import make_axes_locatable

from datasail.reader import read
from datasail.settings import *
from experiments.utils import RUNS, USE_UMAP, embed_smiles, COLORS, set_subplot_label, embed_sequence, TECHNIQUES, \
    DRUG_TECHNIQUES, PROTEIN_TECHNIQUES

LINES = {
    "Random baseline": (COLORS["0d"], "solid"),
    "Random drug baseline (I1)": (COLORS["r1d"], "solid"),
    "Random protein baseline (I1)": (COLORS["r1d"], "dashed"),
    "DataSAIL drug-based (S1)": (COLORS["s1d"], "solid"),
    "DataSAIL protein-based (S1)": (COLORS["s1d"], "dashed"),
    "ID-based baseline (I2)": (COLORS["i2"], "solid"),
    "DataSAIL 2D split (S2)": (COLORS["c2"], "dashed"),
    "LoHi": (COLORS["lohi"], "solid"),
    "DeepChem Butina Splits": (COLORS["butina"], "solid"),
    "DeepChem Fingerprint Splits": (COLORS["fingerprint"], "solid"),
    "DeepChem MaxMin Splits": (COLORS["maxmin"], "solid"),
    "DeepChem Scaffold Splits": (COLORS["scaffold"], "solid"),
    "DeepChem Weight Splits": (COLORS["weight"], "solid"),
    "GraphPart": (COLORS["graphpart"], "solid"),
}


def read_lp_pdbbind():
    """
    Read the LP_PDBBind dataset.

    Returns:
        pd.DataFrame: The dataset
    """
    df = pd.read_csv(Path(__file__).parent / "lppdbbind" / "dataset" / "LP_PDBBind.csv")
    df.rename(columns={"Unnamed: 0": "ids"}, inplace=True)
    e_dataset, f_dataset, inter = read.read_data(**{
        KW_INTER: [(x[0], x[0]) for x in df[["ids"]].values.tolist()],
        KW_E_TYPE: "M",
        KW_E_DATA: df[["ids", "smiles"]].values.tolist(),
        KW_E_WEIGHTS: None,
        KW_E_STRAT: None,
        KW_E_SIM: "ecfp",
        KW_E_DIST: None,
        KW_E_ARGS: "",
        KW_F_TYPE: "P",
        KW_F_DATA: df[["ids", "seq"]].values.tolist(),
        KW_F_WEIGHTS: None,
        KW_F_STRAT: None,
        KW_F_SIM: "mmseqs",
        KW_F_DIST: None,
        KW_F_ARGS: "",
    })
    out = pd.DataFrame(
        [(e_dataset.data[e_dataset.id_map[idx]], f_dataset.data[f_dataset.id_map[idx]]) for idx, _ in inter if
         idx in e_dataset.id_map and idx in f_dataset.id_map], columns=["smiles", "seq"])
    return out


def load_embed(path: Path):
    """
    Load the embeddings.

    Args:
        path: Path to the embeddings

    Returns:
        dict: The embeddings
    """
    if os.path.exists(path):
        with open(path, "rb") as data:
            embeds = pickle.load(data)
    else:
        embeds = {}
    return embeds


def read_single_data(tech, path, encodings, data_path) -> dict:
    """
    Read the data for a single technique.

    Args:
        tech: The technique name
        path: Path to the technique folder
        encodings: The encodings dictionary
        data_path: Path to the folder holding the embeddings for proteins and drugs

    Returns:
        dict: The data
    """
    data = {"folder": path,
            "train_ids_drug": [], "test_ids_drug": [], "drop_ids_drug": [],
            "train_ids_prot": [], "test_ids_prot": [], "drop_ids_prot": []}
    full = read_lp_pdbbind()
    train = pd.read_csv(path / "split_0" / "train.csv")
    test = pd.read_csv(path / "split_0" / "test.csv")
    smiles = {"train": train["Ligand"].tolist(), "test": test["Ligand"].tolist()}
    aaseqs = {"train": train["Target"].tolist(), "test": test["Target"].tolist()}

    if tech in PROTEIN_TECHNIQUES + [TEC_C2]:
        prot_embeds = load_embed(data_path / "prot_embeds.pkl")

        def register_prot(x):
            if x not in encodings["p_map"]:
                x = x.replace(":", "G")[:1022]
                prot_embeds[x] = embed_sequence(x, prot_embeds)
                encodings["p_map"][x] = len(encodings["prots"])
                encodings["prots"].append(prot_embeds[x])
            return encodings["p_map"][x]

        for split in ["train", "test"]:
            for aa_seq in aaseqs[split]:
                data[f"{split}_ids_prot"].append(register_prot(aa_seq))
            data["drop_ids_prot"] = [register_prot(aa_seq) for aa_seq in
                                     set(full["seq"].values) - (set(aaseqs["train"] + aaseqs["test"]))]
        with open(data_path / "prot_embeds.pkl", "wb") as prots:
            pickle.dump(prot_embeds, prots)

    if tech in DRUG_TECHNIQUES + TEC_C2:
        drug_embeds = load_embed(data_path / "drug_embeds.pkl")

        def register_drug(x):
            if x not in encodings["d_map"]:
                drug_embeds[x] = embed_smiles(x, drug_embeds)
                encodings["d_map"][x] = len(encodings["drugs"])
                encodings["drugs"].append(drug_embeds[x])
            return encodings["d_map"][x]

        for split in ["train", "test"]:
            for smile in smiles[split]:
                data[f"{split}_ids_drug"].append(register_drug(smile))
            data["drop_ids_drug"] = [register_drug(smile) for smile in
                                     set(full["smiles"].values) - (set(smiles["train"] + smiles["test"]))]
        with open(data_path / "drug_embeds.pkl", "wb") as drugs:
            pickle.dump(drug_embeds, drugs)
    return data


def read_data(base_path: Path) -> dict:
    """
    Read the data for all techniques.

    Args:
        base_path: Path to the base directory

    Returns:
        dict: The data
    """
    encodings = {
        "drugs": [],
        "prots": [],
        "d_map": {},
        "p_map": {},
    }
    data = {tech: read_single_data(tech, base_path / tool / tech, encodings, base_path / "data")
            for tool, techniques in TECHNIQUES for tech in techniques}

    if USE_UMAP:
        prot_umap, drug_umap = umap.UMAP(), umap.UMAP()
    else:
        prot_umap = TSNE(n_components=2, learning_rate="auto", init="random", random_state=42)
        drug_umap = TSNE(n_components=2, learning_rate="auto", init="random", random_state=42)

    d_trans = drug_umap.fit_transform(np.array(encodings["drugs"]))
    p_trans = prot_umap.fit_transform(np.array(encodings["prots"]))

    for d, trans, techniques in [
        (MODE_E, d_trans, [TEC_I1 + MODE_E, TEC_C1 + MODE_E, TEC_C2, "Butina", "MaxMin", "Fingerprint", "Scaffold",
                           "Weight", "lohi"]),
        (MODE_F, p_trans, [TEC_I1 + MODE_F, TEC_C1 + MODE_F, TEC_C2, "graphpart"])
    ]:
        for tech in techniques:
            for split in ["train", "test", "drop"]:
                data[tech][f"{split}_coord_{'drug' if d == 'e' else 'prot'}"] = trans[
                    data[tech][f"{split}_ids_{'drug' if d == 'e' else 'prot'}"]]
    # data[TEC_C2]["train_coord_drug"] = d_trans[data[TEC_C2]["train_ids_drug"]]
    # data[TEC_C2]["test_coord_drug"] = d_trans[data[TEC_C2]["test_ids_drug"]]
    # data[TEC_C2]["drop_coord_drug"] = d_trans[data[TEC_C2]["drop_ids_drug"]]
    # data[TEC_C2]["train_coord_prot"] = p_trans[data[TEC_C2]["train_ids_prot"]]
    # data[TEC_C2]["test_coord_prot"] = p_trans[data[TEC_C2]["test_ids_prot"]]
    # data[TEC_C2]["drop_coord_prot"] = p_trans[data[TEC_C2]["drop_ids_prot"]]
    return data


def plot_embeds(
        ax: plt.Axes,
        data: Dict,
        postfix: Literal["prot", "drug"],
        title: str,
        drop: bool = True,
        legend: Optional[Union[str, int]] = None,
) -> None:
    """
    Plot the embeddings.

    Args:
        ax: The axis to plot on
        data: Dictionary with the data to plot
        postfix: Whether to plot the protein or drug embeddings
        title: Title of the plot
        drop: Whether to include the drop set
        legend: Location of the legend
    """
    n_train = len(data[f"train_ids_{postfix}"])
    n_test = len(data[f"test_ids_{postfix}"])
    n_drop = len(data[f"drop_ids_{postfix}"])

    if drop:
        p = np.concatenate(
            [data[f"train_coord_{postfix}"], data[f"test_coord_{postfix}"], data[f"drop_coord_{postfix}"]])
        c = np.array([COLORS["train"]] * n_train + [COLORS["test"]] * n_test + [COLORS["drop"]] * n_drop)
    else:
        p = np.concatenate([data[f"train_coord_{postfix}"], data[f"test_coord_{postfix}"]])
        c = np.array([COLORS["train"]] * n_train + [COLORS["test"]] * n_test)
    perm = np.random.permutation(len(p))
    ax.scatter(p[perm, 0], p[perm, 1], s=5, c=c[perm])
    ax.set_xlabel(f"{'UMAP' if USE_UMAP else 't-SNE'} 1")
    ax.set_ylabel(f"{'UMAP' if USE_UMAP else 't-SNE'} 2")
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        train_dot = Line2D([0], [0], marker='o', label="train", color=COLORS["train"], linestyle='None')
        test_dot = Line2D([0], [0], marker='o', label="test", color=COLORS["test"], linestyle='None')
        if drop:
            drop_dot = Line2D([0], [0], marker='o', label="drop", color=COLORS["drop"], linestyle='None')
            handles.extend([train_dot, test_dot, drop_dot])
        else:
            handles.extend([train_dot, test_dot])
        ax.legend(handles=handles, loc=legend, markerscale=2)
    ax.set_title(title)


def viz_sl_models(
        base_path: Path,
        ax: plt.Axes,
        fig: plt.Figure,
        techniques: List,
        ptype: Literal["htm", "bar"] = "htm",
        legend: Optional[Union[str, int]] = None,
        ncol: int = 1,
) -> None:
    """
    Visualize the statistical learning models.

    Args:
        base_path: Path to the base directory
        ax: The axis to plot on
        fig: The figure to plot on
        techniques: List of techniques to plot with additional information
        ptype: Type of plot to use
        legend: Location of the legend
        ncol: Number of columns in the legend
    """
    models = ["RF", "SVM", "XGB", "MLP", "DeepDTA"]
    values = [[] for _ in range(len(techniques))]

    for s, (tool, _, t) in enumerate(techniques):
        for model in models:
            try:
                df = pd.read_csv(base_path / f"{model.lower()}.csv")
                df = df[df["tool"] == tool]
                values[s].append(df[['perf', 'tech']].groupby("tech").mean().loc[t].values[0])
            except Exception as e:
                print(e)
                values[s].append(0)
    if ptype == "bar":
        df = pd.DataFrame(np.array(values).T, columns=[x[1] for x in techniques], index=models)
        c_map = {"I1f": "I1e", "C1f": "C1e", "R": "0d"}
        df.plot.bar(ax=ax, rot=0, ylabel="RMSE (↓)", color=[COLORS[c_map.get(x[2], x[1]).lower()] for x in techniques])
        if legend and ptype == "bar":
            ax.legend(loc=legend, ncol=ncol)
    elif ptype == "htm":
        values = np.array(values, dtype=float)
        cmap = LinearSegmentedColormap.from_list("Custom", [COLORS["train"], COLORS["test"]], N=256)
        cmap.set_bad(color="white")
        pic = ax.imshow(values, cmap=cmap, vmin=np.nanmin(values), vmax=np.nanmax(values))
        ax.set_yticks(range(len(techniques)), [t[1] for t in techniques])
        ax.set_xticks(range(len(models)), models)
        for b in range(len(models)):
            for a in range(len(techniques)):
                if np.isnan(values[a, b]):
                    continue
                label = f"{values[a, b]:.2f}"
                ax.text(b, a, label, ha='center', va='center')
        cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
        fig.colorbar(pic, cax=cax, orientation='vertical', label="RMSE (↓)")
    ax.set_xlabel("ML Models")
    ax.set_title("Performance comparison")


def plot_3x3(full_path: Path, data: Dict) -> None:
    """
    Plot the 3x3 grid of embeddings and models.

    Args:
        full_path: Path to the base directory
        data: The data to plot
    """
    matplotlib.rc('font', **{'size': 16})

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig)

    ax_rd = fig.add_subplot(gs[0, 0])
    ax_sd = fig.add_subplot(gs[1, 0])
    ax_cd = fig.add_subplot(gs[2, 0])
    ax_rp = fig.add_subplot(gs[0, 1])
    ax_sp = fig.add_subplot(gs[1, 1])
    ax_cp = fig.add_subplot(gs[2, 1])
    ax_c2d = fig.add_subplot(gs[0, 2])
    ax_c2p = fig.add_subplot(gs[1, 2])
    ax_c2 = fig.add_subplot(gs[2, 2])

    plot_embeds(ax_rd, data["I1e"], "drug", "Random drug baseline (I1)", drop=False)
    plot_embeds(ax_sd, data["C1e"], "drug", "DataSAIL drug-based (S1)", drop=False)
    plot_embeds(ax_rp, data["I1f"], "prot", "Random protein baseline (I1)", drop=False)
    plot_embeds(ax_sp, data["C1f"], "prot", "DataSAIL protein-based (S1)", drop=False)
    plot_embeds(ax_c2d, data["C2"], "drug", "DataSAIL 2D split (S2) - drugs", legend=4)
    plot_embeds(ax_c2p, data["C2"], "prot", "DataSAIL 2D split (S2) - proteins")

    viz_sl_models(full_path, ax_cd, fig, [
        ("datasail", "Random drug baseline (I1)", "I1e"),
        ("datasail", "DataSAIL drug-based (S1)", "C1e")
    ], legend="upper right", ptype="bar")
    viz_sl_models(full_path, ax_cp, fig, [
        ("datasail", "Random protein baseline (I1)", "I1f"),
        ("datasail", "DataSAIL protein-based (S1)", "C1f")
    ], legend="upper right", ptype="bar")
    viz_sl_models(full_path, ax_c2, fig, [
        ("datasail", "Random baseline", "R"),
        ("datasail", "ID-based baseline (I2)", "I2"),
        ("datasail", "DataSAIL 2D split (S2)", "C2")
    ], legend="lower right", ptype="bar")

    for ax, l in zip([ax_rd, ax_sd, ax_cd, ax_rp, ax_sp, ax_cp, ax_c2d, ax_c2p, ax_c2],
                     ["A", "B", "C", "D", "E", "F", "G", "H", "I"]):
        set_subplot_label(ax, fig, l)

    ax_cd.sharey(ax_c2)
    ax_cp.sharey(ax_c2)

    fig.tight_layout()
    plt.savefig(full_path / "plots" / f"PDBBind_{'umap' if USE_UMAP else 'tsne'}_3x3.png", transparent=True)
    plt.show()


def plot_cold_drug(full_path: Path, data: Dict) -> None:
    """
    Plot the cold drug embeddings.

    Args:
        full_path: Path to the base directory
        data: The data to plot
    """
    matplotlib.rc('font', **{'size': 16})

    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.25, 3])
    gs_upper = gs[0].subgridspec(1, 2)
    gs_lower = gs[1].subgridspec(1, 2, width_ratios=[1.33, 1], wspace=0.35)
    gs_comp = gs_lower[0].subgridspec(3, 2, hspace=0.3, wspace=0.15)

    i1e, c1e, lohi, butina, fingerprint, minmax, scaffold, weight = \
        data["I1e"], data["C1e"], data["lohi"], data["Butina"], data["Fingerprint"], data["MaxMin"], data["Scaffold"], \
        data["Weight"]

    ax_i1 = fig.add_subplot(gs_upper[0])
    ax_c1 = fig.add_subplot(gs_upper[1])
    ax_lh = fig.add_subplot(gs_comp[0, 0])
    ax_bu = fig.add_subplot(gs_comp[0, 1])
    ax_fi = fig.add_subplot(gs_comp[1, 0])
    ax_mm = fig.add_subplot(gs_comp[1, 1])
    ax_sc = fig.add_subplot(gs_comp[2, 0])
    ax_we = fig.add_subplot(gs_comp[2, 1])
    ax_full = fig.add_subplot(gs_lower[1])

    plot_embeds(ax_i1, i1e, "drug", "Random drug baseline (I1)", legend=4, drop=False)
    plot_embeds(ax_c1, c1e, "drug", "DataSAIL drug-based (S1)", drop=False)
    plot_embeds(ax_lh, lohi, "drug", "LoHi", drop=False)
    plot_embeds(ax_bu, butina, "drug", "DC - Butina Splits", drop=False)
    plot_embeds(ax_fi, fingerprint, "drug", "DC - Fingerprint Splits", drop=False)
    plot_embeds(ax_mm, minmax, "drug", "DC - MaxMin Splits", drop=False)
    plot_embeds(ax_sc, scaffold, "drug", "DC - Scaffold Splits", drop=False)
    plot_embeds(ax_we, weight, "drug", "DC - Weight Splits", drop=False)

    viz_sl_models(full_path, ax_full, fig, [
        ("datasail", "DataSAIL (S2)", "C2"),
        ("datasail", "DataSAIL (S1)", "C1e"),
        ("datasail", "Rd. basel. (I1)", "I1e"),
        ("lohi", "LoHi", "lohi"),
        ("deepchem", "DC - Butina", "Butina"),
        ("deepchem", "DC - Fingerprint", "Fingerprint"),
        ("deepchem", "DC - MaxMin", "MaxMin"),
        ("deepchem", "DC - Scaffold", "Scaffold"),
        ("deepchem", "DC - Weight", "Weight")
    ], ptype="htm")

    for ax, l in zip([ax_i1, ax_c1, ax_lh, ax_bu, ax_fi, ax_mm, ax_sc, ax_we, ax_full],
                     ["A", "B", "C", "D", "E", "F", "G", "H", "I"]):
        set_subplot_label(ax, fig, l)

    fig.tight_layout()
    plt.savefig(full_path / "plots" / f"PDBBind_CD_{'umap' if USE_UMAP else 'tsne'}.png")
    plt.show()


def plot_cold_prot(full_path: Path, data: Dict) -> None:
    """
    Plot the cold protein embeddings.

    Args:
        full_path: Path to the base directory
        data: The data to plot
    """
    matplotlib.rc('font', **{'size': 16})

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    i1f, c1f, graphpart = data["I1f"], data["C1f"], data["graphpart"]
    ax_i1 = fig.add_subplot(gs[0, 0])
    ax_c1 = fig.add_subplot(gs[0, 1])
    ax_gp = fig.add_subplot(gs[1, 0])
    ax_full = fig.add_subplot(gs[1, 1])

    plot_embeds(ax_i1, i1f, "prot", "Random protein baseline (I1)", legend=4, drop=False)
    plot_embeds(ax_c1, c1f, "prot", "DataSAIL protein-based (S1)", drop=False)
    plot_embeds(ax_gp, graphpart, "prot", "GraphPart", drop=False)

    viz_sl_models(full_path, ax_full, fig, [
        ("datasail", "DataSAIL (S2)", "C2"),
        ("datasail", "DataSAIL (S1)", "C1f"),
        ("datasail", "Baseline (I1)", "I1f"),
        ("graphpart", "GraphPart", "graphpart"),
    ], legend="lower right", ptype="bar", ncol=2)

    for ax, l in zip([ax_i1, ax_c1, ax_gp, ax_full], ["A", "B", "C", "D"]):
        set_subplot_label(ax, fig, l)

    fig.tight_layout()
    plt.savefig(full_path / "plots" / f"PDBBind_CT_{'umap' if USE_UMAP else 'tsne'}.png")
    plt.show()


def plot(full_path: Path):
    """
    Plot the embeddings and models for the LP_PDBBind dataset.

    Args:
        full_path: Path to the base directory
    """
    (full_path / "plots").mkdir(exist_ok=True)
    (full_path / "data").mkdir(exist_ok=True)

    pkl_name = full_path / "data" / f"{'umap' if USE_UMAP else 'tsne'}_embeds.pkl"
    if not os.path.exists(pkl_name):
        data = read_data(full_path)
        with open(pkl_name, "wb") as out:
            pickle.dump(data, out)
    else:
        with open(pkl_name, "rb") as pickled_data:
            data = pickle.load(pickled_data)

    print("Plot 3x3")
    plot_3x3(full_path, data)
    print("Plot cold drug")
    plot_cold_drug(full_path, data)
    print("Plot cold prot")
    plot_cold_prot(full_path, data)


if __name__ == '__main__':
    plot(Path(sys.argv[1]))
