from typing import List, Union, Optional, Dict

import cvxpy
import numpy as np

from datasail.solver.utils import solve, compute_limits, cluster_sim_dist_constraint, cluster_sim_dist_objective


def solve_ccs_bqp(
        clusters: List[Union[str, int]],
        weights: List[float],
        similarities: Optional[np.ndarray],
        distances: Optional[np.ndarray],
        threshold: float,
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
        clusters: List of cluster names to split
        weights: Weights of the clusters in the order of their names in e_clusters
        similarities: Pairwise similarity matrix of clusters in the order of their names
        distances: Pairwise distance matrix of clusters in the order of their names
        threshold: Threshold to not undergo when optimizing
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
    ones = np.ones((1, len(clusters)))

    x_e = [cvxpy.Variable((len(clusters), 1), boolean=True) for _ in range(len(splits))]

    e_t = np.full((len(clusters), len(clusters)), threshold)
    max_lim, min_lim = compute_limits(epsilon, sum(weights), splits)
    
    constraints = [
        cvxpy.sum([a[:, 0] for a in x_e]) == np.ones((len(clusters))),
    ]
    for s, split in enumerate(splits):
        constraints += [
            min_lim[s] <= cvxpy.sum(cvxpy.multiply(x_e[s][:, 0], weights)),
            cvxpy.sum(cvxpy.multiply(x_e[s][:, 0], weights)) <= max_lim[s],
            cluster_sim_dist_constraint(similarities, distances, e_t, ones, x_e, s)
        ]

    normalization = 1 / (len(splits) * sum(weights) * epsilon)
    size_loss = cvxpy.sum([
        cvxpy.abs(cvxpy.sum(cvxpy.multiply(weights, x_e[s][:, 0])) - split * sum(weights))
        for s, split in enumerate(splits)]
    ) * normalization

    e_loss = cluster_sim_dist_objective(similarities, distances, ones, weights, x_e, splits)

    alpha = 0.5
    problem = solve(alpha * size_loss + e_loss, constraints, max_sec, solver, log_file)

    if problem is None:
        return None

    return {e: names[s] for s in range(len(splits)) for i, e in enumerate(clusters) if x_e[s][i, 0].value > 0.1}

