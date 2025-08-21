import pickle
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np

from experiments.DTI.visualize import viz_sl_models
from experiments.MPP.visualize import comp_all_il, get_perf
from experiments.Strat.visualize import plot_perf
from experiments.ablation.david import visualize
from experiments.utils import COLORS, DATASETS, METRICS, TECHNIQUES, embed, plot_bars_2y, plot_embeds, set_subplot_label

FULL_PATH = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "DataSAIL" / "v10"
matplotlib.rc('font', **{'size': 16})


def results_plot():

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig)

    rand_tsne_tox21 = fig.add_subplot(gs[0, 0])
    datasail_tsne_tox21 = fig.add_subplot(gs[0, 1])
    performance_tox21 = fig.add_subplot(gs[0, 2])
    rand_tsne_srare = fig.add_subplot(gs[1, 0])
    datasail_tsne_srare = fig.add_subplot(gs[1, 1])
    performance_srare = fig.add_subplot(gs[1, 2])
    cd_lppdbbind = fig.add_subplot(gs[2, 0])
    ct_lppdbbind = fig.add_subplot(gs[2, 1])
    c2d_lppdbbind = fig.add_subplot(gs[2, 2])

    #######################################################################################
    # Plot Tox21 tSNEs and performances
    tox21_full_path = FULL_PATH / "MPP"
    perf = {"tox21": get_perf(tox21_full_path, "tox21")}
    with open(tox21_full_path / "data" / "leakage.pkl", "rb") as f:
        leakage = pickle.load(f)

    il = {tech: leakage["tox21"][tool][tech] for tool, techniques in [("datasail", ["I1e", "C1e"]), ("deepchem", TECHNIQUES["deepchem"]), ("lohi", TECHNIQUES["lohi"])] for tech in techniques}
    df = perf["tox21"]
    df["model"] = df["model"].apply(lambda x: x.upper())
    df = df.loc[df["tech"].isin(["I1e", "Fingerprint", "lohi", "C1e"]), ["tech", "model", "perf"]].groupby(["model", "tech"])["perf"] \
        .mean().reset_index().pivot(index="model", columns="tech", values="perf")
    df.loc["Split"] = [np.average(il[tech]) for tech in df.columns]
    df = df.loc[["RF", "SVM", "XGB", "MLP", "D-MPNN", "Split"], ["C1e", "lohi", "Fingerprint", "I1e"]]
    df.rename(columns={"I1e": "Random baseline (I1)", "C1e": "DataSAIL split (S1)", "lohi": "LoHi"}, inplace=True)
    plot_bars_2y(df.T, performance_tox21, color=[COLORS["r1d"], COLORS["lohi"], COLORS["fingerprint"], COLORS["s1d"]])
    performance_tox21.set_ylabel(METRICS[DATASETS["tox21"][2]])
    performance_tox21.legend(loc="lower left", framealpha=1)
    performance_tox21.set_title("Performance comparison")
    performance_tox21.set_xlabel("ML Models")

    i_tr, i_te, c_tr, c_te = embed(tox21_full_path, "tox21")
    plot_embeds(rand_tsne_tox21, i_tr, i_te, "Random baseline (I1)", legend=True)
    plot_embeds(datasail_tsne_tox21, c_tr, c_te, "DataSAIL split (S1)")
    performance_tox21.set_xlabel("")

    #######################################################################################
    # Plot Tox21 tSNEs SR-ARE performances
    strat_full_path = FULL_PATH / "Strat"
    dc_tr, dc_te, ds_tr, ds_te = embed(strat_full_path)
    plot_embeds(rand_tsne_srare, dc_tr, dc_te, "Stratified baseline", legend=True)
    plot_embeds(datasail_tsne_srare, ds_tr, ds_te, "DataSAIL split (S1 w/ classes)")
    plot_perf(strat_full_path,  performance_srare)
    performance_srare.set_xlabel("")

    #######################################################################################
    # Plot LP-PDBBind performances
    lppdbbind_full_path = FULL_PATH / "DTI"
    cd_lppdbbind, il_cd = viz_sl_models(lppdbbind_full_path, cd_lppdbbind, fig, [
        ("datasail", "DataSAIL drug-based (S1)", "C1e"),
        ("lohi", "LoHi", "lohi"),
        ("deepchem", "DC - Fingerprint", "Fingerprint"),
        ("datasail", "Random drug baseline (I1)", "I1e"),
    ], legend="lower left", ptype="bar")
    cd_lppdbbind.set_xlabel("")
    ct_lppdbbind, il_cp = viz_sl_models(lppdbbind_full_path, ct_lppdbbind, fig, [
        ("datasail", "DataSAIL protein-based (S1)", "C1f"),
        ("graphpart", "GraphPart", "graphpart"),
        ("datasail", "Random protein baseline (I1)", "I1f")
    ], legend="lower left", ptype="bar")
    ct_lppdbbind.set_xlabel("")
    c2d_lppdbbind, il_c2 = viz_sl_models(lppdbbind_full_path, c2d_lppdbbind, fig, [
        ("datasail", "DataSAIL 2D split (S2)", "C2"),
        ("datasail", "ID-based baseline (I2)", "I2"),
        ("datasail", "Random baseline", "R")
    ], legend="lower left", ptype="bar")
    c2d_lppdbbind.set_xlabel("")

    cd_lppdbbind.sharey(c2d_lppdbbind)
    ct_lppdbbind.sharey(c2d_lppdbbind)
    il_c2.sharey(il_cd)
    il_cp.sharey(il_cd)

    fig.tight_layout()
    plt.savefig("ISMB.png", transparent=False, dpi=300)


def solver_plot():
    fig = plt.figure(figsize=(10, 14))
    gs = gridspec.GridSpec(2, 1, figure=fig)
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
    visualize(FULL_PATH / "Clusters", list(range(10, 50, 5)) + list(range(50, 150, 10)) + list(range(150, 401, 50)), ["GUROBI", "MOSEK", "SCIP"], ax=(ax[0], ax[1]), fig=fig)
    ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[0].set_xlabel("")
    ax[0].set_title("")
    ax[1].set_title("")
    # ax[0].set_xticklabels([])
    ax[0].sharex(ax[1])
    plt.tight_layout()
    plt.savefig("ISMB_ablation.png", transparent=False, dpi=300)


if __name__ == "__main__":
    results_plot()
    # solver_plot()