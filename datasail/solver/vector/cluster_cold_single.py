from typing import List, Union, Optional, Dict

import cvxpy
import numpy as np

from datasail.solver.utils import solve
from datasail.solver.vector.utils import cluster_sim_dist_constraint, cluster_sim_dist_objective


def solve_ccs_bqp(
        e_clusters: List[Union[str, int]],
        e_weights: List[float],
        e_similarities: Optional[np.ndarray],
        e_distances: Optional[np.ndarray],
        e_threshold: float,
        epsilon: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
        solver: str,
        log_file: str,
) -> Optional[Dict[str, str]]:
    """
    Solve cluster-based cold splitting using disciplined quasi-convex programming and binary quadratic programming.

    Args:
        e_clusters: List of cluster names to split
        e_weights: Weights of the clusters in the order of their names in e_clusters
        e_similarities: Pairwise similarity matrix of clusters in the order of their names
        e_distances: Pairwise distance matrix of clusters in the order of their names
        e_threshold: Threshold to not undergo when optimizing
        epsilon: Additive bound for exceeding the requested split size
        splits: List of split sizes
        names: List of names of the splits in the order of the splits argument
        max_sec: Maximal number of seconds to take when optimizing the problem (not for finding an initial solution)
        max_sol: Maximal number of solution to consider
        solver: Solving algorithm to use to solve the formulated program
        log_file: File to store the detailed log from the solver to

    Returns:
        Mapping from clusters to splits optimizing the objective function
    """
    ones = np.ones((1, len(e_clusters)))

    x_e = [cvxpy.Variable((len(e_clusters), 1), boolean=True) for _ in range(len(splits))]

    e_t = np.full((len(e_clusters), len(e_clusters)), e_threshold)
    min_lim = [int(split * epsilon * sum(e_weights)) for split in splits]
    max_lim = [int(split / epsilon * sum(e_weights)) for split in splits]
    
    constraints = [
        cvxpy.sum([a[:, 0] for a in x_e]) == np.ones((len(e_clusters))),
    ]
    for s, split in enumerate(splits):
        constraints += [
            min_lim[s] <= cvxpy.sum(cvxpy.multiply(x_e[s][:, 0], e_weights)),
            cvxpy.sum(cvxpy.multiply(x_e[s][:, 0], e_weights)) <= max_lim[s],
            cluster_sim_dist_constraint(e_similarities, e_distances, e_t, ones, x_e, s)
        ]

    normalization = 1 / (len(splits) * sum(e_weights) * epsilon)
    size_loss = cvxpy.sum([
        cvxpy.abs(cvxpy.sum(cvxpy.multiply(e_weights, x_e[s][:, 0])) - split * sum(e_weights))
        for s, split in enumerate(splits)]
    ) * normalization

    e_loss = cluster_sim_dist_objective(e_similarities, e_distances, ones, e_weights, x_e, splits)

    alpha = 0.5
    problem = solve(alpha * size_loss + e_loss, constraints, max_sec, solver, log_file)

    if problem is None:
        return {}

    return {e: names[s] for s in range(len(splits)) for i, e in enumerate(e_clusters) if x_e[s][i, 0].value > 0.1}

