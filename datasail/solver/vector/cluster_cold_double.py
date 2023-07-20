from typing import List, Tuple, Optional, Dict, Union

import cvxpy
import numpy as np

from datasail.solver.utils import solve
from datasail.solver.vector.utils import interaction_constraints, cluster_sim_dist_constraint, \
    cluster_sim_dist_objective


def solve_ccd_bqp(
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
    alpha = 0.1
    inter_count = np.sum(inter)
    e_ones = np.ones((1, len(e_clusters)))
    f_ones = np.ones((1, len(f_clusters)))
    inter_ones = np.ones_like(inter)
    e_t = np.full((len(e_clusters), len(e_clusters)), e_threshold)
    f_t = np.full((len(f_clusters), len(f_clusters)), f_threshold)
    min_lim = [int((split - epsilon) * inter_count) for split in splits]
    max_lim = [int((split + epsilon) * inter_count) for split in splits]

    x_e = [cvxpy.Variable((len(e_clusters), 1), boolean=True) for _ in range(len(splits))]
    x_f = [cvxpy.Variable((len(f_clusters), 1), boolean=True) for _ in range(len(splits))]
    x_i = [cvxpy.Variable((len(e_clusters), len(f_clusters)), boolean=True) for _ in range(len(splits))]

    constraints = [
        cvxpy.sum([x[:, 0] for x in x_e]) == np.ones((len(e_clusters))),
        cvxpy.sum([x[:, 0] for x in x_f]) == np.ones((len(f_clusters))),
        cvxpy.sum([x for x in x_i]) <= np.ones((len(e_clusters), len(f_clusters))),
    ]

    for s, split in enumerate(splits):
        constraints += [
            min_lim[s] <= cvxpy.sum(cvxpy.sum(cvxpy.multiply(inter, x_i[s]), axis=0), axis=0),
            cvxpy.sum(cvxpy.sum(cvxpy.multiply(inter, x_i[s]), axis=0), axis=0) <= max_lim[s],
        ] + interaction_constraints(len(e_clusters), len(f_clusters), x_e, x_f, x_i, s) + [
            cluster_sim_dist_constraint(e_similarities, e_distances, e_t, e_ones, x_e, s),
            cluster_sim_dist_constraint(f_similarities, f_distances, f_t, f_ones, x_f, s),
        ]

    inter_loss = cvxpy.sum(
        [cvxpy.sum(cvxpy.sum(cvxpy.multiply(inter_ones - x_i[s], inter), axis=0), axis=0) for s in range(len(splits))]
    )

    e_loss = cluster_sim_dist_objective(e_similarities, e_distances, e_ones, e_weights, x_e, splits)
    f_loss = cluster_sim_dist_objective(f_similarities, f_distances, f_ones, f_weights, x_f, splits)

    solve(alpha * inter_loss + e_loss + f_loss, constraints, max_sec, solver, log_file)

    # report the found solution
    output = ({}, dict(
        (e, names[s]) for s in range(len(splits)) for i, e in enumerate(e_clusters) if x_e[s][:, 0][i].value > 0.1
    ), dict(
        (f, names[s]) for s in range(len(splits)) for j, f in enumerate(f_clusters) if x_f[s][:, 0][j].value > 0.1
    ))
    for i, e in enumerate(e_clusters):
        for j, f in enumerate(f_clusters):
            for s in range(len(splits)):
                if x_i[s][i, j].value > 0:
                    output[0][(e_clusters[i], f_clusters[j])] = names[s]
            if sum(x_i[b][i, j].value for b in range(len(splits))) == 0:
                output[0][(e_clusters[i], f_clusters[j])] = "not selected"
    return output
