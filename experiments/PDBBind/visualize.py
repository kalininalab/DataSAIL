import os.path
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from matplotlib import gridspec, patches, legend_handler
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sympy.physics.control.control_plots import matplotlib, plt

from datasail.reader import read
from datasail.settings import *
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


def read_lp_pdbbind():
    df = pd.read_csv(Path("experiments") / "PDBBind" / "LP_PDBBind.csv")
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
        KW_F_SIM: "ecfp",
        KW_F_DIST: None,
        KW_F_ARGS: "",
    })
    out = pd.DataFrame([(e_dataset.data[e_dataset.id_map[idx]], f_dataset.data[f_dataset.id_map[idx]]) for idx, _ in inter if idx in e_dataset.id_map and idx in f_dataset.id_map], columns=["smiles", "seq"])
    return out


def read_single_data(name, path, encodings):
    data = {"folder": path, "train_ids_drug": [], "test_ids_drug": [], "drop_ids_drug": [], "train_ids_prot": [], "test_ids_prot": [], "drop_ids_prot": []}
    metric = []
    full = read_lp_pdbbind()
    for r in range(RUNS):
        if r == 0:  # and name[-1] not in "2R":
            df = pd.read_csv(path / f"split_{r}" / "pdbbind.csv")
            train = df[df["split"] == "train"]
            test = df[df["split"] == "test"]
            smiles = {"train": train["ligands"].tolist(), "test": test["ligands"].tolist()}
            aaseqs = {"train": train["proteins"].tolist(), "test": test["proteins"].tolist()}

            if name in {"I1f", "C1f", "C2"}:  # , "graphpart"}:
                print(name, "-", "protein")
                embed_path = Path("experiments") / "PDBBind" / "prot_embeds.pkl"
                if os.path.exists(embed_path):
                    prot_embeds = pickle.load(open(embed_path, "rb"))
                else:
                    prot_embeds = {}
                print(len(prot_embeds), "protein embedding loaded")

                def register_prot(x):
                    if x not in encodings["p_map"]:
                        x = x.replace(":", "G")[:1022]
                        if x not in prot_embeds:
                            prot_embeds[x] = embed_aaseqs(x)
                        encodings["p_map"][x] = len(encodings["prots"])
                        encodings["prots"].append(prot_embeds[x])
                    return encodings["p_map"][x]

                for split in ["train", "test"]:
                    for aa_seq in aaseqs[split]:
                        data[f"{split}_ids_prot"].append(register_prot(aa_seq))
                    data["drop_ids_prot"] = [register_prot(aa_seq) for aa_seq in set(full["seq"].values) - (set(aaseqs["train"] + aaseqs["test"]))]
                pickle.dump(prot_embeds, open(embed_path, "wb"))

            if name in {"I1e", "C1e", "C2"}:  # , "LoHi", "Butina", "Fingerprint", "MinMax", "Scaffold", "Weight"}:
                print(name, "-", "drug")
                embed_path = Path("experiments") / "PDBBind" / "drug_embeds.pkl"
                if os.path.exists(embed_path):
                    drug_embeds = pickle.load(open(embed_path, "rb"))
                else:
                    drug_embeds = {}
                print(len(drug_embeds), "drug embedding loaded")

                def register_drug(x):
                    if x not in encodings["d_map"]:
                        if x not in drug_embeds:
                            drug_embeds[x] = embed_smiles(x)
                        encodings["d_map"][x] = len(encodings["drugs"])
                        encodings["drugs"].append(drug_embeds[x])
                    return encodings["d_map"][x]

                for split in ["train", "test"]:
                    for smile in smiles[split]:
                        data[f"{split}_ids_drug"].append(register_drug(smile))
                    data["drop_ids_drug"] = [register_drug(smile) for smile in set(full["smiles"].values) - (set(smiles["train"] + smiles["test"]))]
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
            for split in ["train", "test", "drop"]:
                data[tech][f"{split}_coord_{'drug' if d == 'e' else 'prot'}"] = trans[data[tech][f"{split}_ids_{'drug' if d == 'e' else 'prot'}"]]
    data["C2"]["train_coord_drug"] = d_trans[data["C2"]["train_ids_drug"]]
    data["C2"]["test_coord_drug"] = d_trans[data["C2"]["test_ids_drug"]]
    data["C2"]["drop_coord_drug"] = d_trans[data["C2"]["drop_ids_drug"]]
    data["C2"]["train_coord_prot"] = p_trans[data["C2"]["train_ids_prot"]]
    data["C2"]["test_coord_prot"] = p_trans[data["C2"]["test_ids_prot"]]
    data["C2"]["drop_coord_prot"] = p_trans[data["C2"]["drop_ids_prot"]]
    return data


def plot_embeds(ax, data, postfix, title, drop=True, legend=None):
    n_train = len(data[f"train_ids_{postfix}"])
    n_test = len(data[f"test_ids_{postfix}"])
    n_drop = len(data[f"drop_ids_{postfix}"])

    if drop:
        p = np.concatenate([data[f"train_coord_{postfix}"], data[f"test_coord_{postfix}"], data[f"drop_coord_{postfix}"]])
        c = np.array([colors["train"]] * n_train + [colors["test"]] * n_test + [colors["drop"]] * n_drop)
    else:
        p = np.concatenate([data[f"train_coord_{postfix}"], data[f"test_coord_{postfix}"]])
        c = np.array([colors["train"]] * n_train + [colors["test"]] * n_test)
    perm = np.random.permutation(len(p))
    ax.scatter(p[perm, 0], p[perm, 1], s=5, c=c[perm])
    ax.set_xlabel(f"{'UMAP' if USE_UMAP else 'tSNE'} 1")
    ax.set_ylabel(f"{'UMAP' if USE_UMAP else 'tSNE'} 2")
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        train_dot = Line2D([0], [0], marker='o', label="train", color=colors["train"], linestyle='None')
        test_dot = Line2D([0], [0], marker='o', label="test", color=colors["test"], linestyle='None')
        if drop:
            drop_dot = Line2D([0], [0], marker='o', label="drop", color=colors["drop"], linestyle='None')
            handles.extend([train_dot, test_dot, drop_dot])
        else:
            handles.extend([train_dot, test_dot])
        ax.legend(handles=handles, loc=legend, markerscale=2)
    ax.set_title(title)


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
        df.plot.bar(ax=ax_full[i], rot=0, ylabel="MSE",
                    color=[colors[c_map.get(x[2], x[2]).lower()] for x in techniques])
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


def plot_3x3(data, perf_right=True):
    matplotlib.rc('font', **{'size': 16})

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig)

    if perf_right:
        ax_rd = fig.add_subplot(gs[0, 0])
        ax_sd = fig.add_subplot(gs[0, 1])
        ax_cd = fig.add_subplot(gs[0, 2])
        ax_rp = fig.add_subplot(gs[1, 0])
        ax_sp = fig.add_subplot(gs[1, 1])
        ax_cp = fig.add_subplot(gs[1, 2])
        ax_c2d = fig.add_subplot(gs[2, 0])
        ax_c2p = fig.add_subplot(gs[2, 1])
        ax_c2 = fig.add_subplot(gs[2, 2])
    else:
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
    set_subplot_label(ax_rd, fig, "A")
    plot_embeds(ax_sd, data["C1e"], "drug", "DataSAIL drug-based (S1)", drop=False)
    set_subplot_label(ax_sd, fig, "B")
    plot_embeds(ax_rp, data["I1f"], "prot", "Random protein baseline (I1)", drop=False)
    set_subplot_label(ax_rp, fig, "D")
    plot_embeds(ax_sp, data["C1f"], "prot", "DataSAIL protein-based (S1)", drop=False)
    set_subplot_label(ax_sp, fig, "E")
    plot_embeds(ax_c2d, data["C2"], "drug", "DataSAIL 2D split (S2) - drugs", legend=4)
    set_subplot_label(ax_c2d, fig, "G")
    plot_embeds(ax_c2p, data["C2"], "prot", "DataSAIL 2D split (S2) - proteins")
    set_subplot_label(ax_c2p, fig, "H")

    ax_full = [ax_cd, ax_cp, ax_c2]
    root = Path("..") / "DataSAIL" / "experiments" / "PDBBind" / "datasail"
    for i, techniques in enumerate([
        [(data["I1e"], "Random drug baseline (I1)", "I1e"), (data["C1e"], "DataSAIL drug-based (S1)", "C1e")],
        [(data["I1f"], "Random protein baseline (I1)", "I1f"), (data["C1f"], "DataSAIL protein-based (S1)", "C1f")],
        [(data["R"], "Random baseline", "R"), (data["I2"], "ID-based baseline (I2)", "I2"), (data["C2"], "DataSAIL 2D split (S2)", "C2")]
    ]):
        models = ["RF", "SVM", "XGB", "MLP", "DeepDTA"]
        values = [[] for _ in range(len(techniques))]

        for s, (tech, _, t) in enumerate(techniques):
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
        df.plot.bar(ax=ax_full[i], rot=0, ylabel="MSE",
                    color=[colors[c_map.get(x[2], x[2]).lower()] for x in techniques])
        if i == 2:
            ax_full[i].legend(loc="lower center")
        else:
            ax_full[i].legend()
        set_subplot_label(ax_full[i], fig, ["C", "F", "I"][i])
        ax_full[i].set_xlabel("ML Models")
        ax_full[i].set_title("Performance comparison")
    if not perf_right:
        print("ShareIT!")
        ax_full[0].sharey(ax_full[2])
        ax_full[1].sharey(ax_full[2])

    fig.tight_layout()
    plt.savefig(f"PDBBind_{'umap' if USE_UMAP else 'tsne'}_3x3{'_Olga' if perf_right else ''}.png", transparent=True)
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
    pkl_name = Path("..") / "DataSAIL" / "experiments" / "PDBBind" / f"{'umap' if USE_UMAP else 'tsne'}_embeds_2.pkl"
    if not os.path.exists(pkl_name):  # or True:
        data = read_data()
        with open(pkl_name, "wb") as out:
            pickle.dump(data, out)
    else:
        with open(pkl_name, "rb") as pickled_data:
            data = pickle.load(pickled_data)
    # plot_full(data)
    plot_3x3(data, perf_right=False)
    # plot_cold_drug(data)
    # plot_cold_prot(data)


if __name__ == '__main__':
    analyze()
    # viz_sl()
