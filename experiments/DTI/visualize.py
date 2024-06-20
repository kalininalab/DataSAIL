import os.path
import pickle
from pathlib import Path
from typing import Literal, Dict, List, Union, Optional

import numpy as np
import pandas as pd
import umap
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt, gridspec, cm, colors as mpl_colors
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1 import make_axes_locatable

from datasail.reader import read
from datasail.reader.utils import DataSet
from datasail.cluster.diamond import run_diamond
from datasail.cluster.ecfp import run_ecfp
from experiments.ablation import david
from datasail.settings import *
from experiments.utils import USE_UMAP, embed_smiles, COLORS, set_subplot_label, embed_sequence, TECHNIQUES, \
    DRUG_TECHNIQUES, PROTEIN_TECHNIQUES, plot_bars_2y, create_heatmap


def comp_il(base_path: Path):
    if (leak_path := (base_path / "data" / "leakage.pkl")).exists():
        with open(leak_path, "rb") as f:
            output = pickle.load(f)
    else:
        output = {}
    df = pd.read_csv(Path(__file__).parent / "lppdbbind" / "dataset" / "LP_PDBBind.csv")
    df.rename(columns={"Unnamed: 0": "ids"}, inplace=True)

    if (lig_path:= (base_path / "data" / "lig.pkl")).exists():
        with open(lig_path, "rb") as f:
            lig_dataset = pickle.load(f)
        print("Loaded pickled ligands")
    else:
        lig_dataset = DataSet(
            type="M",
            names=df["ids"].tolist(),
            data=dict(df[["ids", "smiles"]].values.tolist()),
            id_map={x: x for x in df["ids"].tolist()},
        )
        run_ecfp(lig_dataset)
        print("Computed liagnds:", lig_dataset.cluster_similarity.shape)
        with open(lig_path, "wb") as f:
            pickle.dump(lig_dataset, f)

    if (tar_path := (base_path / "data" / "tar.pkl")).exists():
        with open(tar_path, "rb") as f:
            tar_dataset = pickle.load(f)
        print("Loaded pickled targets")
    else:
        tar_dataset = DataSet(
            type="P",
            names=df["ids"].tolist(),
            data=dict(df[["ids", "seq"]].values.tolist()),
            id_map={x: x for x in df["ids"].tolist()},
        )
        run_diamond(tar_dataset)
        print("Computed targets:", tar_dataset.cluster_similarity.shape)
        with open(tar_path, "wb") as f:
            pickle.dump(tar_dataset, f)

    if "lig_sim" not in output:
        output["lig_sim"] = np.sum(lig_dataset.cluster_similarity)
    if "tar_sim" not in output:
        output["tar_sim"] = np.sum(tar_dataset.cluster_similarity)

    # root = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "DataSAIL" / "v10" / "DTI" / "datasail"
    for tech in [t for techs in TECHNIQUES.values() for t in techs]:
        base = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "DataSAIL" / "v10" / "DTI"
        if tech in ["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"]:
            root = base / "datasail"
        elif tech == "lohi":
            root = base / "lohi"
        elif tech == "graphpart":
            root = base / "graphpart"
        else:
            root = base / "deepchem"

        if tech in ["R", "I2", "C2"]:
            dss = [(lig_dataset, "_lig"), (tar_dataset, "_tar")]
        elif tech in ["I1e", "C1e", "lohi", "Butina", "Fingerprint", "MaxMin", "Scaffold", "Weight"]:
            dss = [(lig_dataset, "_lig")]
        elif tech in ["I1f", "C1f", "graphpart"]:
            dss = [(tar_dataset, "_tar")]
        else:
            print(f"Unknown technique: {tech}")
            continue

        for ds, n in dss:
            name = tech + n
            if name not in output:
                output[name] = []
            elif tech not in TECHNIQUES["datasail"]:
                continue
            for run in range(5):
                print(name, run, end="\t")
                base = root / tech / f"split_{run}"
                train_ids = pd.read_csv(base / "train.csv")["ids"]
                test_ids = pd.read_csv(base / "test.csv")["ids"]
                assi = np.array(
                    [1 if x in train_ids.values else -1 if x in test_ids.values else 0 for x in ds.cluster_names])
                il, total = david.eval(
                    assi.reshape(-1, 1),
                    ds.cluster_similarity,
                    [ds.cluster_weights[c] for c in ds.cluster_names],
                )
                print(il)
                output[name].append((il, total))
    with open(leak_path, "wb") as f:
        pickle.dump(output, f)


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
        KW_E_CLUSTERS: 50,
        KW_F_TYPE: "P",
        KW_F_DATA: df[["ids", "seq"]].values.tolist(),
        KW_F_WEIGHTS: None,
        KW_F_STRAT: None,
        KW_F_SIM: "mmseqs",
        KW_F_DIST: None,
        KW_F_ARGS: "",
        KW_F_CLUSTERS: 50,
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


def read_single_data(tech, path, encodings, data_path, full) -> dict:
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
    data = {
        "folder": path,
        "train_ids_drug": [], "test_ids_drug": [], "drop_ids_drug": [],
        "train_ids_prot": [], "test_ids_prot": [], "drop_ids_prot": [],
    }
    train = pd.read_csv(path / "split_0" / "train.csv")
    test = pd.read_csv(path / "split_0" / "test.csv")
    train.rename(columns={"Ligand": "smiles", "Target": "seq"}, inplace=True)
    test.rename(columns={"Ligand": "smiles", "Target": "seq"}, inplace=True)
    smiles = {"train": train["smiles"].tolist(), "test": test["smiles"].tolist()}
    aaseqs = {"train": train["seq"].tolist(), "test": test["seq"].tolist()}

    if tech in PROTEIN_TECHNIQUES + [TEC_C2]:
        prot_embeds = load_embed(data_path / "prot_embeds.pkl")

        def register_prot(x):
            if x not in encodings["p_map"]:
                x = x.replace(":", "G")[:1022]
                prot_embeds[x] = embed_sequence(x, prot_embeds)
                if prot_embeds[x] is not None:
                    encodings["p_map"][x] = len(encodings["prots"])
                    encodings["prots"].append(prot_embeds[x])
            return encodings["p_map"].get(x, None)

        for split in ["train", "test"]:
            for aa_seq in aaseqs[split]:
                data[f"{split}_ids_prot"].append(register_prot(aa_seq))
            data["drop_ids_prot"] = [register_prot(aa_seq) for aa_seq in
                                     set(full["seq"].values) - (set(aaseqs["train"] + aaseqs["test"]))]
        with open(data_path / "prot_embeds.pkl", "wb") as prots:
            pickle.dump(prot_embeds, prots)

    if tech in DRUG_TECHNIQUES + [TEC_C2]:
        drug_embeds = load_embed(data_path / "drug_embeds.pkl")

        def register_drug(x):
            if x not in encodings["d_map"]:
                drug_embeds[x] = embed_smiles(x, drug_embeds)
                if drug_embeds[x] is not None:
                    encodings["d_map"][x] = len(encodings["drugs"])
                    encodings["drugs"].append(drug_embeds[x])
            return encodings["d_map"].get(x, None)

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
    
    full = read_lp_pdbbind()
    data = {tech: read_single_data(tech, base_path / tool / tech, encodings, base_path / "data", full)
            for tool, techniques in TECHNIQUES.items() for tech in techniques}

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
                tmp = data[tech][f"{split}_ids_{'drug' if d == 'e' else 'prot'}"]
                tmp = np.array(list(filter(lambda v: v == v and v is not None, tmp)))
                if len(tmp) > 0:
                    data[tech][f"{split}_coord_{'drug' if d == 'e' else 'prot'}"] = trans[tmp]
    return data


def plot_embeds(
        ax: plt.Axes,
        fig: plt.Figure,
        data: Dict,
        postfix: Literal["prot", "drug"],
        title: str,
        drop: bool = True,
        legend: Optional[Union[str, int]] = None,
        label: Optional[str] = None,
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
    set_subplot_label(ax, fig, label)


def viz_sl_models(
        base_path: Path,
        # ax: plt.Axes,
        gs: gridspec.SubplotSpec,
        fig: plt.Figure,
        techniques: List,
        ptype: Literal["htm", "bar"] = "htm",
        legend: Optional[Union[str, int]] = None,
        ncol: int = 1,
        label: Optional[str] = None,
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
    leak_cmap = mpl_colors.LinearSegmentedColormap.from_list("cyan_magenta", ["cyan", "magenta"])
    with open(base_path / "data" / "leakage.pkl", "rb") as f:
        l = pickle.load(f)

    def leakage(tech):
        if tech in ["R", "I2", "C2"]:
            return [[(l[f"{tech}_lig"][i][j] + l[f"{tech}_tar"][i][j] / 2) for j in range(len(l[f"{tech}_lig"][i]))]
                    for i in range(5)]
        if tech in ["I1f", "C1f", "graphpart"]:
            return l[f"{tech}_tar"]
        return l[f"{tech}_lig"]

    for s, (tool, _, t) in enumerate(techniques):
        for model in models:
            try:
                df = pd.read_csv(base_path / f"{tool}.csv")
                values[s].append(df[(df["model"] == model.lower()) & (df["tech"] == t)][["perf"]].mean().values[0])
            except Exception as e:
                print(e)
                values[s].append(0)
    
    if ptype == "bar":
        ax = fig.add_subplot(gs)
        df = pd.DataFrame(np.array(values).T, columns=[x[1] for x in techniques], index=models)
        c_map = {"I1f": "I1e", "C1f": "C1e", "R": "0d"}
        df.loc["Splits"] = [np.average([x for x, _ in leakage(tech)]) for _, _, tech in techniques]
        il = plot_bars_2y(df.T, ax, color=[COLORS[c_map.get(x[2], x[2]).lower()] for x in techniques])
        ax.set_ylabel("RMSE (↓)")
        ax.set_xlabel("ML Models")
        if legend is not None:
            il.legend(loc=legend, ncol=ncol, framealpha=1)
        ax.set_title("Performance comparison")
        ax.set_xlabel("ML Models")
    elif ptype == "htm":
        gs_main = gs.subgridspec(1, 2, width_ratios=[15, 1], wspace=0.4)
        ax = fig.add_subplot(gs_main[1])
        values = np.array(values, dtype=float)
        leak = np.array([np.average([x for x, _ in leakage(tech)]) for _, _, tech in techniques]).reshape(-1, 1)
        cmap = LinearSegmentedColormap.from_list("Custom", [COLORS["train"], COLORS["test"]], N=256)
        cmap.set_bad(color="white")
        create_heatmap(values, leak, cmap, leak_cmap, fig, gs_main[0], "Performance", "RMSE (↓)", y_labels=True, mode="MMB", max_val=max(leak), label=label, yticklabels=[t[1] for t in techniques])
        plt.colorbar(cm.ScalarMappable(mpl_colors.Normalize(0, max(leak)), leak_cmap), cax=ax, label="$L(\pi)$ ↓")
    else:
        raise ValueError(f"Unknown plottype {ptype}")
    return ax


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
    ax_rp = fig.add_subplot(gs[0, 1])
    ax_sp = fig.add_subplot(gs[1, 1])
    ax_c2d = fig.add_subplot(gs[0, 2])
    ax_c2p = fig.add_subplot(gs[1, 2])

    plot_embeds(ax_rd, fig, data["I1e"], "drug", "Random drug baseline (I1)", drop=False, label="A")
    plot_embeds(ax_sd, fig, data["C1e"], "drug", "DataSAIL drug-based (S1)", drop=False, label="B")
    plot_embeds(ax_rp, fig, data["I1f"], "prot", "Random protein baseline (I1)", drop=False, label="D")
    plot_embeds(ax_sp, fig, data["C1f"], "prot", "DataSAIL protein-based (S1)", drop=False, label="E")
    plot_embeds(ax_c2d, fig, data["C2"], "drug", "DataSAIL 2D split (S2) - drugs", legend="lower right", label="G")
    plot_embeds(ax_c2p, fig, data["C2"], "prot", "DataSAIL 2D split (S2) - proteins", label="H")

    ax_cd = viz_sl_models(full_path, gs[2, 0], fig, [
        ("datasail", "DataSAIL drug-based (S1)", "C1e"),
        ("lohi", "LoHi", "lohi"),
        ("deepchem", "Fingerprint", "Fingerprint"),
        ("datasail", "Random drug baseline (I1)", "I1e"),
    ], legend="lower left", ptype="bar", label="C")
    ax_cp = viz_sl_models(full_path, gs[2, 1], fig, [
        ("datasail", "DataSAIL protein-based (S1)", "C1f"),
        ("graphpart", "GraphPart", "graphpart"),
        ("datasail", "Random protein baseline (I1)", "I1f")
    ], legend="lower left", ptype="bar", label="F")
    ax_c2 = viz_sl_models(full_path, gs[2, 2], fig, [
        ("datasail", "DataSAIL 2D split (S2)", "C2"),
        ("datasail", "ID-based baseline (I2)", "I2"),
        ("datasail", "Random baseline", "R")
    ], legend="lower left", ptype="bar", label="I")

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
    # ax_full = fig.add_subplot(gs_lower[1])

    plot_embeds(ax_i1, fig, i1e, "drug", "Random drug baseline (I1)", legend=4, drop=False, label="A")
    plot_embeds(ax_c1, fig, c1e, "drug", "DataSAIL drug-based (S1)", drop=False, label="B")
    plot_embeds(ax_lh, fig, lohi, "drug", "LoHi", drop=False, label="C")
    plot_embeds(ax_bu, fig, butina, "drug", "DC - Butina Splits", drop=False, label="D")
    plot_embeds(ax_fi, fig, fingerprint, "drug", "DC - Fingerprint Splits", drop=False, label="E")
    plot_embeds(ax_mm, fig, minmax, "drug", "DC - MaxMin Splits", drop=False, label="F")
    plot_embeds(ax_sc, fig, scaffold, "drug", "DC - Scaffold Splits", drop=False, label="G")
    plot_embeds(ax_we, fig, weight, "drug", "DC - Weight Splits", drop=False, label="H")

    viz_sl_models(full_path, gs_lower[1], fig, [
        ("datasail", "DataSAIL (S2)", "C2"),
        ("datasail", "DataSAIL (S1)", "C1e"),
        ("datasail", "Rd. basel. (I1)", "I1e"),
        ("lohi", "LoHi", "lohi"),
        ("deepchem", "DC - Butina", "Butina"),
        ("deepchem", "DC - Fingerprint", "Fingerprint"),
        ("deepchem", "DC - MaxMin", "MaxMin"),
        ("deepchem", "DC - Scaffold", "Scaffold"),
        ("deepchem", "DC - Weight", "Weight")
    ], ptype="htm", label="I")

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

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    i1f, c1f, graphpart = data["I1f"], data["C1f"], data["graphpart"]
    ax_i1 = fig.add_subplot(gs[0, 0])
    ax_c1 = fig.add_subplot(gs[0, 1])
    ax_gp = fig.add_subplot(gs[1, 0])

    plot_embeds(ax_i1, fig, i1f, "prot", "Random protein baseline (I1)", legend=4, drop=False, label="A")
    plot_embeds(ax_c1, fig, c1f, "prot", "DataSAIL protein-based (S1)", drop=False, label="B")
    plot_embeds(ax_gp, fig, graphpart, "prot", "GraphPart", drop=False, label="C")

    viz_sl_models(full_path, gs[1, 1], fig, [
        ("datasail", "DataSAIL (S2)", "C2"),
        ("datasail", "DataSAIL (S1)", "C1f"),
        ("datasail", "Baseline (I1)", "I1f"),
        ("graphpart", "GraphPart", "graphpart"),
    ], legend="lower left", ptype="bar", ncol=2, label="D")

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
    # comp_il(Path(sys.argv[1]))
