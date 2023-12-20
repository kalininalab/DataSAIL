import copy
import pickle
import time
from pathlib import Path
from typing import List, Dict

import cvxpy
import deepchem as dc
import numpy as np
from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from datasail.cluster.clustering import additional_clustering
from datasail.cluster.utils import read_molecule_encoding
from datasail.reader.read_molecules import read_molecule_data
from datasail.solver.utils import solve, compute_limits, cluster_y_constraints
from experiments.utils import dc2pd, telegram

blocker = rdBase.BlockLogs()

FOLDER = 'time'
SOLVERS = ["CBC", "GLPK_MI", "GUROBI", "MOSEK", "SCIP"]
CLUSTERS = list(range(10, 50, 5)) + list(range(50, 150, 10)) + [200]  # list(range(150, 501, 50))


def solve_ccs_blp(
        clusters,
        weights,
        similarities,
        epsilon,
        splits,
        names,
        max_sec,
        max_sol,
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


def run_solver():
    path = Path("experiments") / FOLDER
    path.mkdir(parents=True, exist_ok=True)

    ds_path = Path("experiments") / FOLDER / "data.pkl"
    if not ds_path.exists():
        dataset = dc.molnet.load_clintox(featurizer=dc.feat.DummyFeaturizer(), splitter=None)[1][0]
        df = dc2pd(dataset, "clintox")
        dataset = read_molecule_data(dict(df[["ID", "SMILES"]].values.tolist()), sim="ecfp")
        dataset.cluster_names, dataset.cluster_map, dataset.cluster_similarity, dataset.cluster_weights = run_ecfp(
            dataset)
        norm = np.sum(dataset.cluster_similarity)
        with open(path / "data.pkl", "wb") as f:
            pickle.dump(dataset, f)
    else:
        with open(ds_path, "rb") as f:
            dataset = pickle.load(f)
        norm = np.sum(dataset.cluster_similarity)

    with open(path / "log.txt", "w") as log:
        for num_clusters in CLUSTERS:
            # old_path = Path("experiments") / FOLDER / "MOSEK" / f"data_{num_clusters}.pkl"
            # if not old_path.exists():
            ds = copy.deepcopy(dataset)
            ds = additional_clustering(ds, n_clusters=num_clusters)
            for solver_name in SOLVERS:
                # if old_path.exists():
                #     with open(old_path, "rb") as f:
                #         ds, _ = pickle.load(f)
                (path / solver_name).mkdir(parents=True, exist_ok=True)
                try:
                    problem, assignment, ttime = solve_ccs_blp(
                        clusters=ds.cluster_names,
                        weights=[ds.cluster_weights[name] for name in ds.cluster_names],
                        similarities=ds.cluster_similarity,
                        epsilon=0.05,
                        splits=[0.8, 0.2],
                        names=["train", "test"],
                        max_sec=7200,
                        max_sol=-1,
                        solver=solver_name,
                        log_file=path / solver_name / f"solve_{num_clusters}.log",
                        threads=16,
                    )
                except cvxpy.error.SolverError as e:
                    print(e)
                    print(f"{num_clusters} - SolverError", file=log)
                    break
                with open(path / solver_name / f"data_{num_clusters}.pkl", "wb") as f:
                    pickle.dump((ds, assignment), f)
                tmp = np.array([1 if assignment[ds.cluster_map[n]] == "train" else -1 for n in ds.names]).reshape(-1, 1)
                mask = tmp @ tmp.T
                mask[mask == -1] = 0
                leakage = np.sum(dataset.cluster_similarity * mask) / norm
                print(solver_name, num_clusters, leakage, problem.solver_stats.solve_time, ttime, sep=" | ", file=log)
                telegram(f"[Timing] {solver_name} - {num_clusters} - {ttime:.1f}s - {leakage:.4f}")


def weighted_random(names: List[str], weights: Dict[str, int], splits: List[float], epsilon: float):
    min_lim = compute_limits(epsilon, sum(weights.values()), splits)
    sizes = np.zeros(len(splits))
    splits = [[] for _ in range(len(splits))]
    names = sorted(names, key=lambda x: weights[x], reverse=True)
    for name in names:
        ratios = [(sizes[s] + weights[name]) / min_lim[s] for s in range(len(splits))]
        mindex = np.argmin(ratios)
        sizes[mindex] += weights[name]
        splits[mindex].append(name)
    return splits


def random_baseline():
    base = Path("experiments") / FOLDER
    with open(base / "data.pkl", "rb") as f:
        dataset = pickle.load(f)

    # cluster-based baseline
    s_leakage, c_leakage = [], []
    for num_clusters in CLUSTERS:
        with open(base / "MOSEK" / f"data_{num_clusters}.pkl", "rb") as f:
            ds, assi = pickle.load(f)
        n_leakage = [], []
        for i in range(5):
            train_test = weighted_random(ds.cluster_names, ds.cluster_weights, [0.8, 0.2], 0.05)

            c_tmp = np.array([1 if n in train_test[0] else -1 for n in ds.cluster_names]).reshape(-1, 1)
            c_value = eval(c_tmp, ds.cluster_similarity)
            n_leakage[0].append(c_value)

            s_tmp = np.array([1 if ds.cluster_map[n] in train_test[0] else -1 for n in dataset.names]).reshape(-1, 1)
            s_value = eval(s_tmp, dataset.cluster_similarity)
            n_leakage[1].append(1 - s_value)
            ds.shuffle()
        c_leakage.append((np.mean(n_leakage[0]), np.min(n_leakage[0]), np.max(n_leakage[0]), np.std(n_leakage[0])))
        s_leakage.append((np.mean(n_leakage[1]), np.min(n_leakage[1]), np.max(n_leakage[1]), np.std(n_leakage[1])))

    return np.array(c_leakage), np.array(s_leakage)


def time_overhead():
    times = {}
    with open(Path("experiments") / FOLDER / "log.txt", "r") as data:
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
    times["SCIP"][-1, 2] = 7200
    return times


def viz_single(similarity, assignment):
    permutation = [i for i, s in assignment.items() if s == "train"] + [i for i, s in assignment.items() if s == "test"]
    similarity = similarity[permutation, :][:, permutation]
    plt.imshow(similarity)
    plt.show()


def blub():
    for num_cluster in list(range(10, 50, 5)):
        with open(Path("experiments") / FOLDER / "MOSEK" / f"data_{num_cluster}.pkl", "rb") as f:
            ds, assi = pickle.load(f)
        viz_single(ds.cluster_similarity, assi)


def eval(assignments, similarity):
    mask = assignments @ assignments.T
    mask[mask == -1] = 0
    return np.sum(similarity * mask) / np.sum(similarity)


def visualize():
    times = time_overhead()
    c_random, s_random = random_baseline()
    c_performances = {"MOSEK": [], "SCIP": [], "GUROBI": []}
    s_performances = {"MOSEK": [], "SCIP": [], "GUROBI": []}
    with open(Path("experiments") / FOLDER / "data.pkl", "rb") as f:
        dataset = pickle.load(f)

    for solver in ["GUROBI", "MOSEK", "SCIP"]:
        for num_cluster in CLUSTERS:
            print(f"\r{solver} - {num_cluster}", end="")
            try:
                with open(Path("experiments") / FOLDER / solver / f"data_{num_cluster}.pkl", "rb") as f:
                    ds, assi = pickle.load(f)
                tmp = np.array([1 if assi[n] == "train" else -1 for n in range(num_cluster)]).reshape(-1, 1)
                tmp2 = np.array([1 if assi[ds.cluster_map[n]] == "train" else -1 for n in dataset.names]).reshape(-1, 1)
                c_value = eval(tmp, ds.cluster_similarity)
                s_value = eval(tmp2, dataset.cluster_similarity)
            except Exception:
                c_value = 1
                s_value = 1
            c_performances[solver].append(c_value)
            s_performances[solver].append(s_value)

    fig, (axl, axr) = plt.subplots(1, 2)
    visualize2(axl, times, c_performances, c_random)
    axl.title.set_text("Unleaked information measured by cluster similarity")
    visualize2(axr, times, s_performances, s_random)
    axr.title.set_text("Unleaked information measured by sample similarity")

    fig.set_size_inches(20, 8)
    fig.tight_layout()
    plt.savefig(Path("experiments") / FOLDER / "time_overhead.pdf")
    plt.show()


def visualize2(ax1, times, performances, random):
    ax2 = ax1.twinx()
    ax1.set_xlabel("Number of clusters")
    ax1.set_ylabel("Time [s]")
    ax2.set_ylabel("Intra-cluster Similarity ratio (maximize)")

    ax1.plot(times["GUROBI"][:, 0], times["GUROBI"][:, 1], label="GUROBI", color="blue", linestyle="dashed")
    ax1.plot(times["MOSEK"][:, 0], times["MOSEK"][:, 1], label="MOSEK", color="orange", linestyle="dashed")
    ax1.plot(times["SCIP"][:, 0], times["SCIP"][:, 1], label="SCIP", color="green", linestyle="dashed")

    ax2.plot(times["GUROBI"][:, 0], performances["GUROBI"], label="GUROBI", color="blue")
    ax2.plot(times["MOSEK"][:, 0], performances["MOSEK"], label="MOSEK", color="orange")
    ax2.plot(times["SCIP"][:, 0], performances["SCIP"], label="SCIP", color="green")

    ax2.fill_between(times["GUROBI"][:, 0], random[:, 1], random[:, 2], color="black", alpha=0.2)
    ax2.plot(times["GUROBI"][:, 0], random[:, 0], color="black", label="Random")

    ax1.legend(loc="upper left")
    ax1.set_yscale("log")
    ax2.legend(loc="lower right")


if __name__ == '__main__':
    run_solver()
    # time_overhead()
    # random_baseline()
    # visualize()
