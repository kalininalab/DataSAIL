from typing import List, Union, Optional, Dict
from pathlib import Path

import cvxpy
import numpy as np

from datasail.solver.utils import solve, cluster_y_constraints, compute_limits, stratification_constraints


def solve_c1(
        clusters: List[Union[str, int]],
        weights: List[float],
        s_matrix: Optional[np.ndarray],
        similarities: Optional[np.ndarray],
        distances: Optional[np.ndarray],
        delta: float,
        epsilon: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
        solver: str,
        log_file: Path,
) -> Optional[Dict[str, str]]:
    """
    Solve cluster-based cold splitting using disciplined quasi-convex programming and binary quadratic programming.

    Args:
        clusters: List of cluster names to split
        weights: Weights of the clusters in the order of their names in e_clusters
        s_matrix: Stratification for the clusters
        similarities: Pairwise similarity matrix of clusters in the order of their names
        distances: Pairwise distance matrix of clusters in the order of their names.
        delta: Additive bound for stratification imbalance
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
    min_lim = compute_limits(epsilon, sum(weights), splits)

    x = cvxpy.Variable((len(splits), len(clusters)), boolean=True)
    y = [[cvxpy.Variable(1, boolean=True) for _ in range(e)] for e in range(len(clusters))]
    
    constraints = [cvxpy.sum(x, axis=0) == np.ones((len(clusters)))]

    for s, split in enumerate(splits):
        constraints.append(min_lim[s] <= cvxpy.sum(cvxpy.multiply(x[s], weights)))

    if s_matrix is not None:
        constraints.append(stratification_constraints(s_matrix, splits, delta, x))

    constraints += cluster_y_constraints(False, clusters, y, x, splits)

    intra_weights = similarities if similarities is not None else distances
    tmp = [[intra_weights[e1, e2] * y[e1][e2] for e2 in range(e1)] for e1 in range(len(clusters))]
    loss = cvxpy.sum([t for tmp_list in tmp for t in tmp_list])
    if distances is not None:
        loss = -loss
    problem = solve(loss, constraints, max_sec, solver, log_file)

    return None if problem is None else {
        e: names[s] for s in range(len(splits)) for i, e in enumerate(clusters) if x[s, i].value > 0.1
    }

