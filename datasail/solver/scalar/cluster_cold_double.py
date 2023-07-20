from typing import List, Optional, Tuple, Dict, Union

import numpy as np

from datasail.solver.scalar.utils import init_variables, init_inter_variables_cluster, cluster_sim_dist_constraint, \
    cluster_sim_dist_objective
from datasail.solver.utils import solve


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
        e_weights: List of weights of the clusters from e_clusters in the order of the e_cluster argument
        e_similarities: Pairwise similarity matrix of clusters in the order of their names
        e_distances: Pairwise distance matrix of clusters in the order of their names
        e_threshold: Threshold to not undergo when optimizing
        f_clusters: List of cluster names to split from the f-dataset
        f_weights: List of weights of the clusters from f_clusters in the order of the f_cluster argument
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

    x_e = init_variables(len(splits), len(e_clusters))
    x_f = init_variables(len(splits), len(f_clusters))
    x_i = init_inter_variables_cluster(len(splits), e_clusters, f_clusters)

    constraints = []

    for i in range(len(e_clusters)):
        constraints.append(sum(x_e[i, s] for s in range(len(splits))) == 1)
    for j in range(len(f_clusters)):
        constraints.append(sum(x_f[j, s] for s in range(len(splits))) == 1)
    for i in range(len(e_clusters)):
        for j in range(len(f_clusters)):
            constraints.append(sum(x_i[i, j, s] for s in range(len(splits))) <= 1)

    for s in range(len(splits)):
        var = sum(
            x_i[i, j, s] * inter[i, j] for i in range(len(e_clusters)) for j in range(len(f_clusters))
        )
        constraints += [
            (splits[s] - epsilon) * inter_count <= var,
            var <= (splits[s] + epsilon) * inter_count,
        ]
        for i in range(len(e_clusters)):
            for j in range(len(f_clusters)):
                constraints.append(x_i[i, j, s] >= (x_e[i, s] + x_f[j, s] - 1.5))
                constraints.append(x_i[i, j, s] <= (x_e[i, s] + x_f[j, s]) * 0.5)
                constraints.append(x_e[i, s] >= x_i[i, j, s])
                constraints.append(x_f[j, s] >= x_i[i, j, s])

        constraints += cluster_sim_dist_constraint(
            e_similarities, e_distances, e_threshold, len(e_clusters), x_e, s
        )
        constraints += cluster_sim_dist_constraint(
            f_similarities, f_distances, f_threshold, len(f_clusters), x_f, s
        )

    normalization = 2 / np.sum(inter)
    inter_loss = sum(
        (1 - x_i[i, j, b]) * inter[i, j]
        for i in range(len(e_clusters)) for j in range(len(f_clusters)) for b in range(len(splits))
    ) * normalization

    e_loss = cluster_sim_dist_objective(e_similarities, e_distances, len(e_clusters), e_weights, x_e, len(splits))
    f_loss = cluster_sim_dist_objective(f_similarities, f_distances, len(f_clusters), f_weights, x_f, len(splits))

    solve(alpha * inter_loss + e_loss + f_loss, constraints, max_sec, solver, log_file)

    # report the found solution
    output = ({}, dict(
        (e, names[s]) for s in range(len(splits)) for i, e in enumerate(e_clusters) if x_e[i, s].value > 0.1
    ), dict(
        (f, names[s]) for s in range(len(splits)) for j, f in enumerate(f_clusters) if x_f[j, s].value > 0.1
    ))
    for i in range(len(e_clusters)):
        for j in range(len(f_clusters)):
            for s in range(len(splits)):
                if x_i[i, j, s].value > 0:
                    output[0][(e_clusters[i], f_clusters[j])] = names[s]
            if sum(x_i[i, j, b].value for b in range(len(splits))) == 0:
                output[0][(e_clusters[i], f_clusters[j])] = "not selected"
    return output
