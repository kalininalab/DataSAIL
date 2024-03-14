import copy
import pickle
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib
import cvxpy
import deepchem as dc
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from datasail.cluster.clustering import additional_clustering
from datasail.cluster.utils import read_molecule_encoding
from datasail.reader.read_molecules import read_molecule_data
from datasail.solver.utils import solve, compute_limits, cluster_y_constraints
from experiments.ablation.time import MARKERS
from experiments.utils import dc2pd, DATASETS, COLORS


def solve_ccs_blp(
        clusters,
        weights,
        similarities,
        epsilon,
        splits,
        names,
        max_sec,
        solver,
        log_file,
        threads,
):
    min_lim = compute_limits(epsilon, sum(weights), splits)

    x = cvxpy.Variable((len(splits), len(clusters)), boolean=True)
    y = [[cvxpy.Variable(1, boolean=True) for _ in range(e)] for e in range(len(clusters))]

    constraints = [cvxpy.sum(x, axis=0) == np.ones((len(clusters)))]

    for s, split in enumerate(splits):
        constraints.append(min_lim[s] <= cvxpy.sum(cvxpy.multiply(x[s], weights)))

    constraints += cluster_y_constraints(clusters, y, x, splits)

    tmp = [[similarities[e1, e2] * y[e1][e2] for e2 in range(e1)] for e1 in range(len(clusters))]
    loss = cvxpy.sum([t for tmp_list in tmp for t in tmp_list])

    start = time.time()
    problem = solve(loss, constraints, max_sec, solver, log_file, threads)
    ttime = time.time() - start

    assignment = {e: names[s] for s in range(len(splits)) for i, e in enumerate(clusters) if x[s, i].value > 0.1}
    return problem, assignment, ttime


def run_ecfp(dataset):
    invalid_mols = []
    molecules = {}
    for name in dataset.names:
        mol = read_molecule_encoding(dataset.data[name])
        if mol is None:
            invalid_mols.append(name)
            continue
        molecules[name] = mol

    for invalid_name in invalid_mols:  # obsolete code?
        dataset.names.remove(invalid_name)
        dataset.data.pop(invalid_name)
        poppable = []
        for key, value in dataset.id_map.items():
            if value == invalid_name:
                poppable.append(key)
        for pop in poppable:
            dataset.id_map.pop(pop)

    fps = []
    cluster_names = list(set(Chem.MolToSmiles(s) for s in list(molecules.values())))
    for scaffold in cluster_names:
        fps.append(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(scaffold), 2, nBits=1024))

    count = len(cluster_names)
    sim_matrix = np.zeros((count, count))
    for i in range(count):
        sim_matrix[i, i] = 1
        sim_matrix[i, :i] = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        sim_matrix[:i, i] = sim_matrix[i, :i]

    cluster_map = dict((name, Chem.MolToSmiles(molecules[name])) for name in dataset.names)

    cluster_weights = {}
    for key, value in cluster_map.items():
        if value not in cluster_weights:
            cluster_weights[value] = 0
        cluster_weights[value] += 1

    return cluster_names, cluster_map, sim_matrix, cluster_weights


def run_solver(full_path: Path, ds_name: str, clusters: List[int], solvers: List[str]):
    full_path.mkdir(parents=True, exist_ok=True)

    ds_path = full_path / "data.pkl"
    if not ds_path.exists():
        dataset = DATASETS[ds_name][0](featurizer=dc.feat.DummyFeaturizer(), splitter=None)[1][0]
        df = dc2pd(dataset, ds_name)
        dataset = read_molecule_data(dict(df[["ID", "SMILES"]].values.tolist()), sim="ecfp")
        dataset.cluster_names, dataset.cluster_map, dataset.cluster_similarity, dataset.cluster_weights = run_ecfp(
            dataset
        )
        with open(ds_path, "wb") as f:
            pickle.dump(dataset, f)
    else:
        with open(ds_path, "rb") as f:
            dataset = pickle.load(f)

    for nc, num_clusters in enumerate(clusters):
        if (full_path / "GUROBI" / f"data_{num_clusters}.pkl").exists():
            with open(full_path / "GUROBI" / f"data_{num_clusters}.pkl", "rb") as pkl:
                ds, _ = pickle.load(pkl)
        else:
            ds = copy.deepcopy(dataset)
            ds = additional_clustering(ds, n_clusters=num_clusters, linkage="average")

        for solver_name in solvers:
            solver_path = full_path / solver_name
            solver_path.mkdir(parents=True, exist_ok=True)

            try:
                problem, assignment, ttime = solve_ccs_blp(
                    clusters=ds.cluster_names,
                    weights=[ds.cluster_weights[name] for name in ds.cluster_names],
                    similarities=ds.cluster_similarity,
                    epsilon=0.05,
                    splits=[0.8, 0.2],
                    names=["train", "test"],
                    max_sec=7200,
                    solver=solver_name,
                    log_file=full_path / solver_name / f"solve_{num_clusters}.log",
                    threads=16,
                )
            except cvxpy.error.SolverError as e:
                print(e)
                with open(full_path / "log.txt", "a") as log:
                    print(f"{num_clusters} - SolverError", file=log)
                return

            with open(full_path / solver_name / f"data_{num_clusters}.pkl", "wb") as f:
                pickle.dump((ds, assignment), f)
            tmp = np.array([1 if assignment[n] == "train" else -1 for n in range(num_clusters)]).reshape(-1, 1)
            leakage = eval(tmp, ds.cluster_similarity)
            try:
                solver_time = problem.solver_stats.solve_time
            except:
                solver_time = 0
            with open(full_path / "log.txt", "a") as log:
                print(solver_name, num_clusters, leakage, solver_time, ttime, sep=" | ", file=log)


def time_overhead(full_path):
    times = {}
    with open(full_path / "log.txt", "r") as data:
        for line in data:
            parts = line.split(" | ")
            if parts[0] not in times:
                times[parts[0]] = []
            times[parts[0]].append((int(parts[1]), float(parts[4]), float("nan" if parts[3] == "None" else parts[3])))
    for key, value in times.items():
        times[key] = np.array(value)

    m_reg = LinearRegression()
    m_reg.fit(times["MOSEK"][:, :2], np.log(times["MOSEK"][:, 1] - times["MOSEK"][:, 2]))
    times["SCIP"][:, 2] = times["SCIP"][:, 1] - np.exp(m_reg.predict(times["SCIP"][:, :2]))
    for solver in ["GUROBI", "MOSEK", "SCIP"]:
        np.clip(times[solver][:, 1:], 0, 7200, out=times[solver][:, 1:])
    return times


def eval(assignments, similarity):
    mask = assignments @ assignments.T
    mask[mask == -1] = 0
    return 1 - np.sum(similarity * mask) / np.sum(similarity)


def visualize(full_path: Path, clusters: List[int], solvers, ax: Optional[Tuple] = None, fig=None):
    if show := ax is None:
        matplotlib.rc('font', **{'size': 16})
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(1, 2, figure=fig)
        ax_p, ax_t = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    else:
        ax_p, ax_t = ax

    times = time_overhead(full_path)

    perf_path = Path("performances.pkl")
    if perf_path.exists():
        with open(perf_path, "rb") as f:
            performances = pickle.load(f)
    else:
        performances = {"MOSEK": [], "SCIP": [], "GUROBI": []}
        with open(full_path / "data.pkl", "rb") as f:
            dataset = pickle.load(f)

        for solver in solvers:
            for num_clusters in clusters:
                print(f"\r{solver} - {num_clusters}", end="")
                try:
                    with open(full_path / solver / f"data_{num_clusters}.pkl", "rb") as f:
                        ds, assi = pickle.load(f)
                    tmp2 = np.array([1 if assi[ds.cluster_map[n]] == "train" else -1 for n in dataset.names]).reshape(
                        -1, 1)
                    s_value = eval(tmp2, dataset.cluster_similarity)
                except Exception:
                    s_value = 1
                performances[solver].append(s_value)
        with open(perf_path, "wb") as f:
            pickle.dump(performances, f)

    ax_p.set_xlabel("Number of clusters")
    ax_t.set_xlabel("Number of clusters")
    ax_t.set_ylabel("Time for solving [s] (↓)")
    ax_p.set_ylabel("$L(\pi)$ (↓)")

    ax_t.plot(times["GUROBI"][:, 0], times["GUROBI"][:, 1], label="GUROBI", color=COLORS["train"],
              marker=MARKERS["gurobi"], markersize=9)
    ax_t.plot(times["MOSEK"][:, 0], times["MOSEK"][:, 1], label="MOSEK", color=COLORS["test"],
              marker=MARKERS["mosek"], markersize=9)
    ax_t.plot(times["SCIP"][:, 0], times["SCIP"][:, 1], label="SCIP", color=COLORS["r1d"],
              marker=MARKERS["scip"], markersize=9)

    ax_p.plot(times["GUROBI"][:, 0], performances["GUROBI"], label="GUROBI", color=COLORS["train"],
              marker=MARKERS["gurobi"], markersize=9)
    ax_p.plot(times["MOSEK"][:, 0], performances["MOSEK"], label="MOSEK", color=COLORS["test"],
              marker=MARKERS["mosek"], markersize=9)
    ax_p.plot(times["SCIP"][:, 0], performances["SCIP"], label="SCIP", color=COLORS["r1d"],
              marker=MARKERS["scip"], markersize=9)

    ax_p.legend()
    ax_p.set_title("Leaked Information on Tox21")

    ax_t.set_yscale("log")
    ax_t.legend()
    ax_t.set_title("Runtime on Tox21")
    if show:
        ax_p.title.set_text("Unleaked information measured by sample similarity")
        fig.tight_layout()
        plt.savefig(full_path / "clusters.png")
        plt.show()


def run_cluster_ablation(full_path: Path):
    run_solver(full_path, "tox21", list(range(10, 50, 5)) + list(range(50, 150, 10)) + list(range(150, 401, 50)),
               ["GUROBI", "MOSEK", "SCIP"])
    visualize(full_path, list(range(10, 50, 5)) + list(range(50, 150, 10)) + list(range(150, 401, 50)),
              ["GUROBI", "MOSEK", "SCIP"])


if __name__ == '__main__':
    run_cluster_ablation(Path(sys.argv[1]))
