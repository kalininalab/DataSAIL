from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
import cvxpy
import numpy as np
from scipy.optimize import fsolve

from datasail.solver.utils import solve, interaction_contraints, collect_results_2d, leakage_loss, compute_limits, \
    stratification_constraints, collect_results_2d2


def solve_c2(
        e_clusters: List[Union[str, int]],
        e_s_matrix: Optional[np.ndarray],
        e_similarities: Optional[np.ndarray],
        e_distances: Optional[np.ndarray],
        e_weights: Optional[np.ndarray],
        f_clusters: List[Union[str, int]],
        f_s_matrix: Optional[np.ndarray],
        f_similarities: Optional[np.ndarray],
        f_distances: Optional[np.ndarray],
        f_weights: Optional[np.ndarray],
        inter: np.ndarray,
        delta: float,
        epsilon: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
        solver: str,
        log_file: Path,
) -> Optional[Tuple[Dict[Tuple[str, str], str], Dict[str, str], Dict[str, str]]]:
    """
    Solve cluster-based double-cold splitting using disciplined quasi-convex programming and binary quadratic
    programming.

    Args:
        e_clusters: List of cluster names to split from the e-dataset
        e_s_matrix: Stratification for the e-dataset
        e_similarities: Pairwise similarity matrix of clusters in the order of their names
        e_distances: Pairwise distance matrix of clusters in the order of their names
        e_weights: Weights of the clusters in the order of their names in e_clusters
        f_clusters: List of cluster names to split from the f-dataset
        f_s_matrix: Stratification for the f-dataset
        f_similarities: Pairwise similarity matrix of clusters in the order of their names
        f_distances: Pairwise distance matrix of clusters in the order of their names
        f_weights: Weights of the clusters in the order of their names in f_clusters
        inter: Matrix storing the amount of interactions between the entities in the e-clusters and f-clusters
        delta: Additive bound for stratification imbalance
        epsilon: Additive bound for exceeding the requested split size
        splits: List of split sizes
        names: List of names of the splits in the order of the splits argument
        max_sec: Maximal number of seconds to take when optimizing the problem (not for finding an initial solution)
        max_sol: Maximal number of solution to consider
        solver: Solving algorithm to use to solve the formulated program
        log_file: File to store the detailed log from the solver to

    Returns:
        A list of interactions and their assignment to a split and two mappings from entities to splits, one for each
        dataset
    """
    if splits[0] == 0.771:  # time_v0
        splits = [0.5650, 0.2175, 0.2175]
    elif splits[0] == 0.889:  # pl50_v0
        splits = [2/3, 1/6, 1/6]
    elif splits[0] == 0.776:  # ecod_v0
        splits = [0.5685, 0.2254, 0.2061]
    elif splits[0] == 0.898:  # pl50_v1
        splits = [0.6773, 0.1645, 0.1582]
    elif splits[0] == 0.8:  # full splits
        splits = [0.5858, 0.2071, 0.2071]
    else:
        splits = convert(splits)
    min_lim = compute_limits(epsilon, int(np.sum(inter)), splits)  # [s / 2 for s in splits])
    #np.set_printoptions(threshold=np.inf)
    #print(np.sum(inter))
    #print(e_weights)
    #print(np.sum(e_weights))
    #print(f_weights)
    #print(np.sum(f_weights))
    #print(min_lim)
    #np.set_printoptions(threshold=np.inf)
    #print(inter)
    x_e = cvxpy.Variable((len(splits), len(e_clusters)), boolean=True)
    x_f = cvxpy.Variable((len(splits), len(f_clusters)), boolean=True)
    #x_i = {(e, f): cvxpy.Variable(len(splits), boolean=True) for e in range(len(e_clusters)) for f in
    #       range(len(f_clusters)) if inter[e, f] != 0}

    # check if the cluster relations are uniform
    e_intra_weights = e_similarities if e_similarities is not None else 1 - e_distances
    f_intra_weights = f_similarities if f_similarities is not None else 1 - f_distances
    #e_uniform = e_intra_weights is None or np.allclose(e_intra_weights, np.ones_like(e_intra_weights)) or \
    #    np.allclose(e_intra_weights, np.zeros_like(e_intra_weights))
    #f_uniform = f_intra_weights is None or np.allclose(f_intra_weights, np.ones_like(f_intra_weights)) or \
    #    np.allclose(f_intra_weights, np.zeros_like(f_intra_weights))

    #def index(x, y):
    #    return (x, y) if (x, y) in x_i else None

    constraints = [
        cvxpy.sum(x_e, axis=0) == np.ones((len(e_clusters)), dtype=int),
        cvxpy.sum(x_f, axis=0) == np.ones((len(f_clusters)), dtype=int),
    ]
    for s, split in enumerate(splits):
        constraints.append(min_lim[s] <= cvxpy.sum(cvxpy.multiply(x_e[s], e_weights)))
        constraints.append(min_lim[s] <= cvxpy.sum(cvxpy.multiply(x_f[s], f_weights)))

    #print(len(constraints))
    #if e_s_matrix is not None:
    #    constraints.append(stratification_constraints(e_s_matrix, [s / 2 for s in splits], delta / 2, x_e))
    #if f_s_matrix is not None:
    #    constraints.append(stratification_constraints(f_s_matrix, [s / 2 for s in splits], delta / 2, x_f))
    #print(len(constraints))
    #exit(0)

    #interaction_contraints(e_clusters, f_clusters, x_i, constraints, splits, x_e, x_f, min_lim, lambda key: inter[key],
    #                       index)
    e_tmp = [[e_weights[e1] * e_weights[e2] * e_intra_weights[e1, e2] * cvxpy.max(
        cvxpy.vstack([x_e[s, e1] - x_e[s, e2] for s in range(len(splits))])
    ) for e2 in range(e1 + 1, len(e_clusters))] for e1 in range(len(e_clusters))]
    f_tmp = [[f_weights[f1] * f_weights[f2] * f_intra_weights[f1, f2] * cvxpy.max(
        cvxpy.vstack([x_f[s, f1] - x_f[s, f2] for s in range(len(splits))])
    ) for f2 in range(f1 + 1, len(f_clusters))] for f1 in range(len(f_clusters))]
    e_loss = cvxpy.sum([e for e_tmp_list in e_tmp for e in e_tmp_list])  # leakage_loss(e_uniform, e_intra_weights, x_e, e_clusters, e_weights, len(splits))
    f_loss = cvxpy.sum([f for f_tmp_list in f_tmp for f in f_tmp_list])  # leakage_loss(f_uniform, f_intra_weights, x_f, f_clusters, f_weights, len(splits))

    problem = solve(e_loss + f_loss, constraints, max_sec, solver, log_file)

    #return collect_results_2d(problem, names, splits, e_clusters, f_clusters, x_e, x_f, x_i, index)
    return collect_results_2d2(problem, names, splits, e_clusters, f_clusters, x_e, x_f, inter)


def func(x, targets):
    denom = sum([a ** 2 for a in x])
    return [(x[i] ** 2 / denom - targets[i]) for i in range(len(x))]


def convert(targets):
    targets = [t / sum(targets) for t in targets]
    sol = fsolve(
        lambda x: func(x, targets),
        [1 / len(targets) for _ in targets]
    )
    return [s / sum(sol) for s in sol]

