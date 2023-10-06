from typing import List, Tuple, Optional, Dict, Union

import cvxpy
import numpy as np
from cvxpy import conj

from datasail.settings import NOT_ASSIGNED
from datasail.solver.utils import solve, compute_limits, interaction_constraints, cluster_sim_dist_constraint, \
    cluster_sim_dist_objective


def solve_ccd_blp(
        e_clusters: List[Union[str, int]],
        e_weights: List[float],
        e_similarities: Optional[np.ndarray],
        e_distances: Optional[np.ndarray],
        e_threshold: float,
        f_clusters: List[Union[str, int]],
        f_weights: List[float],
        f_similarities: Optional[np.ndarray],
        f_distances: Optional[np.ndarray],
        f_threshold: float,
        inter: np.ndarray,
        epsilon: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
        solver: str,
        log_file: str,
) -> Optional[Tuple[Dict[Tuple[str, str], str], Dict[str, str], Dict[str, str]]]:
    """
    Solve cluster-based double-cold splitting using disciplined quasi-convex programming and binary quadratic
    programming.

    Args:
        e_clusters: List of cluster names to split from the e-dataset
        e_weights: List of weights of the clusters in order of the e_cluster argument
        e_similarities: Pairwise similarity matrix of clusters in the order of their names
        e_distances: Pairwise distance matrix of clusters in the order of their names
        e_threshold: Threshold to not undergo when optimizing
        f_clusters: List of cluster names to split from the f-dataset
        f_weights: List of weights of the clusters in order of the f_cluster argument
        f_similarities: Pairwise similarity matrix of clusters in the order of their names
        f_distances: Pairwise distance matrix of clusters in the order of their names
        f_threshold: Threshold to not undergo when optimizing
        inter: Matrix storing the amount of interactions between the entities in the e-clusters and f-clusters
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
    min_lim = [int((split - epsilon) * np.sum(inter)) for split in splits]

    x_e = [cvxpy.Variable((len(e_clusters), 1), boolean=True) for _ in range(len(splits))]
    x_f = [cvxpy.Variable((len(f_clusters), 1), boolean=True) for _ in range(len(splits))]
    x_i = {(e, f): cvxpy.Variable(len(splits), boolean=True) for e in range(len(e_clusters)) for f in
           range(len(f_clusters)) if inter[e, f] != 0}
    y_e = [[cvxpy.Variable(1, boolean=True) for _ in range(e)] for e in range(len(e_clusters))]
    y_f = [[cvxpy.Variable(1, boolean=True) for _ in range(f)] for f in range(len(f_clusters))]

    # check if the cluster relations are uniform
    e_intra_weights = e_similarities if e_similarities is not None else e_distances
    f_intra_weights = f_similarities if f_similarities is not None else f_distances
    e_uniform = e_intra_weights is not None and np.allclose(e_intra_weights, np.ones_like(e_intra_weights))
    f_uniform = f_intra_weights is not None and np.allclose(f_intra_weights, np.ones_like(f_intra_weights))

    # print("\n".join(" ".join(f"{int(v):2d}" for v in row) for row in inter))
    constraints = [
        cvxpy.sum([x[:, 0] for x in x_e]) == np.ones((len(e_clusters))),
        cvxpy.sum([x[:, 0] for x in x_f]) == np.ones((len(f_clusters))),
    ]  # + [cvxpy.sum(x_i[key]) <= 1 for key in x_i.keys()]  # unnecessary as guaranteed by other constraints

    for s, split in enumerate(splits):
        constraints.append(min_lim[s] <= cvxpy.sum([x_i[key][s] * inter[key] for key in x_i]))
        for i in range(len(e_clusters)):
            for j in range(len(f_clusters)):
                if (i, j) in x_i:
                    # constraints.append(x_i[i, j][s] >= cvxpy.maximum(x_e[s][i, 0] + x_f[s][j, 0] - 1, 0))
                    # constraints.append(x_i[i, j][s] <= 0.75 * (x_e[s][i, 0] + x_f[s][j, 0]))
                    constraints.append(x_i[i, j][s] >= x_e[s][i, 0] - x_f[s][j, 0])

    if not e_uniform:
        for e1 in range(len(e_clusters)):
            for e2 in range(e1):
                constraints.append(y_e[e1][e2] >= cvxpy.max(cvxpy.vstack([x_e[s][e1, 0] - x_e[s][e2, 0] for s in range(len(splits))])))
                # constraints.append(y_e[e1][e2] == 1 / 2 * cvxpy.sum([x_e[s][e1, 0] + x_e[s][e2, 0] for s in range(len(splits))]))
    if not f_uniform:
        for f1 in range(len(f_clusters)):
            for f2 in range(f1):
                constraints.append(y_f[f1][f2] >= cvxpy.max(cvxpy.vstack([x_f[s][f1, 0] - x_f[s][f2, 0] for s in range(len(splits))])))
                # constraints.append(y_f[f1][f2] == 1 / 2 * cvxpy.sum([x_f[s][f1, 0] + x_f[s][f2, 0] for s in range(len(splits))]))

    inter_loss = (np.sum(inter) - sum(cvxpy.sum(x) for x in x_i.values())) / np.sum(inter)
    if e_uniform:
        e_loss = 0
    else:
        e_tmp = [[e_intra_weights[e1, e2] * y_e[e1][e2] for e2 in range(e1)] for e1 in range(len(e_clusters))]
        e_loss = cvxpy.sum([t for tmp_list in e_tmp for t in tmp_list])
        if e_similarities is None:
            e_loss = -e_loss

    if f_uniform:
        f_loss = 0
    else:
        f_tmp = [[f_intra_weights[f1, f2] * y_f[f1][f2] for f2 in range(f1)] for f1 in range(len(f_clusters))]
        f_loss = cvxpy.sum([t for tmp_list in f_tmp for t in tmp_list])
        if f_similarities is None:
            f_loss = -f_loss

    problem = solve(inter_loss, constraints, max_sec, solver, log_file)

    if problem is None:
        return None

    # report the found solution
    output = (
        {},
        {e: names[s] for s in range(len(splits)) for i, e in enumerate(e_clusters) if x_e[s][i, 0].value > 0.1},
        {f: names[s] for s in range(len(splits)) for j, f in enumerate(f_clusters) if x_f[s][j, 0].value > 0.1},
    )
    for i, e in enumerate(e_clusters):
        for j, f in enumerate(f_clusters):
            if (i, j) in x_i:
                for s in range(len(splits)):
                    if x_i[i, j][s].value > 0:
                        output[0][(e_clusters[i], f_clusters[j])] = names[s]
                if sum(x_i[i, j][b].value for b in range(len(splits))) == 0:
                    output[0][(e_clusters[i], f_clusters[j])] = NOT_ASSIGNED
    return output
