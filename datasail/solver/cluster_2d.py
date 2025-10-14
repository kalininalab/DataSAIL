from typing import Optional, Union
from pathlib import Path
import cvxpy
import numpy as np
from scipy.optimize import fsolve

from datasail.solver.utils import solve, compute_limits, stratification_constraints


def solve_c2(
        e_clusters: list[Union[str, int]],
        e_s_matrix: Optional[np.ndarray],
        e_similarities: Optional[np.ndarray],
        e_distances: Optional[np.ndarray],
        e_weights: Optional[np.ndarray],
        e_splits: list[float],
        e_names: list[str],
        f_clusters: list[Union[str, int]],
        f_s_matrix: Optional[np.ndarray],
        f_similarities: Optional[np.ndarray],
        f_distances: Optional[np.ndarray],
        f_weights: Optional[np.ndarray],
        f_splits: list[float],
        f_names: list[str],
        delta: float,
        epsilon: float,
        max_sec: int,
        solver: str,
        log_file: Path,
) -> Optional[tuple[dict[str, str], dict[str, str]]]:
    """
    Solve cluster-based double-cold splitting using disciplined quasi-convex programming and binary quadratic
    programming.

    Args:
        e_clusters: List of cluster names to split from the e-dataset
        e_s_matrix: Stratification for the e-dataset
        e_similarities: Pairwise similarity matrix of clusters in the order of their names
        e_distances: Pairwise distance matrix of clusters in the order of their names
        e_weights: Weights of the clusters in the order of their names in e_clusters
        e_splits: List of split sizes of the e-dataset
        e_names: List of names of the splits of the e-dataset in the order of the splits argument
        f_clusters: List of cluster names to split from the f-dataset
        f_s_matrix: Stratification for the f-dataset
        f_similarities: Pairwise similarity matrix of clusters in the order of their names
        f_distances: Pairwise distance matrix of clusters in the order of their names
        f_weights: Weights of the clusters in the order of their names in f_clusters
        f_splits: List of split sizes of the f-dataset
        f_names: List of names of the splits of the f-dataset in the order of the splits argument
        delta: Additive bound for stratification imbalance
        epsilon: Additive bound for exceeding the requested split size
        max_sec: Maximal number of seconds to take when optimizing the problem (not for finding an initial solution)
        solver: Solving algorithm to use to solve the formulated program
        log_file: File to store the detailed log from the solver to

    Returns:
        A list of interactions and their assignment to a split and two mappings from entities to splits, one for each
        dataset
    """
    e_splits = convert(e_splits)
    f_splits = convert(f_splits)
    min_lim_e = compute_limits(epsilon, sum(e_weights), e_splits)
    min_lim_f = compute_limits(epsilon, sum(f_weights), f_splits)

    x_e = cvxpy.Variable((len(e_splits), len(e_clusters)), boolean=True)
    x_f = cvxpy.Variable((len(f_splits), len(f_clusters)), boolean=True)

    # check if the cluster relations are uniform
    e_intra_weights = e_similarities if e_similarities is not None else 1 - e_distances
    f_intra_weights = f_similarities if f_similarities is not None else 1 - f_distances

    constraints = [
        cvxpy.sum(x_e, axis=0) == np.ones((len(e_clusters)), dtype=int),
        cvxpy.sum(x_f, axis=0) == np.ones((len(f_clusters)), dtype=int),
    ]
    for s, (lim_e, lim_f) in enumerate(zip(min_lim_e, min_lim_f)):
        constraints.append(lim_e <= cvxpy.sum(cvxpy.multiply(x_e[s], e_weights)))
        constraints.append(lim_f <= cvxpy.sum(cvxpy.multiply(x_f[s], f_weights)))

    if e_s_matrix is not None:
        constraints.append(stratification_constraints(e_s_matrix, e_splits, delta, x_e))
    if f_s_matrix is not None:
        constraints.append(stratification_constraints(f_s_matrix, f_splits, delta, x_f))

    e_tmp = [[e_weights[e1] * e_weights[e2] * e_intra_weights[e1, e2] * cvxpy.max(
        cvxpy.vstack([x_e[s, e1] - x_e[s, e2] for s in range(len(e_splits))])
    ) for e2 in range(e1 + 1, len(e_clusters))] for e1 in range(len(e_clusters))]
    f_tmp = [[f_weights[f1] * f_weights[f2] * f_intra_weights[f1, f2] * cvxpy.max(
        cvxpy.vstack([x_f[s, f1] - x_f[s, f2] for s in range(len(f_splits))])
    ) for f2 in range(f1 + 1, len(f_clusters))] for f1 in range(len(f_clusters))]
    e_loss = cvxpy.sum([e for e_tmp_list in e_tmp for e in e_tmp_list])
    f_loss = cvxpy.sum([f for f_tmp_list in f_tmp for f in f_tmp_list])

    problem = solve(e_loss + f_loss, constraints, max_sec, solver, log_file)

    if problem is None:
        return None

    # report the found solution
    return {e: e_names[s] for s in range(len(e_splits)) for i, e in enumerate(e_clusters) if x_e[s, i].value > 0.1}, \
           {f: f_names[s] for s in range(len(f_splits)) for j, f in enumerate(f_clusters) if x_f[s, j].value > 0.1}


def func(x, targets):
    denom = sum([a ** 2 for a in x])
    return [(x[i] ** 2 / denom - targets[i]) + (1e6 if x[i] < 0 else 0) for i in range(len(x))]


def convert(targets):
    targets = [t / sum(targets) for t in targets]
    sol = fsolve(
        lambda x: func(x, targets),
        [1 / len(targets) for _ in targets]
    )
    return [s / sum(sol) for s in sol]
