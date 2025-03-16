from typing import List, Union, Optional, Dict
from pathlib import Path

import cvxpy
import numpy as np

from datasail.solver.utils import solve, compute_limits, stratification_constraints


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

    x = cvxpy.Variable((len(splits), len(clusters)), boolean=True)  # 19

    constraints = [cvxpy.sum(x, axis=0) == np.ones((len(clusters)))]  # 16

    for s, lim in enumerate(min_lim):
        constraints.append(lim <= cvxpy.sum(cvxpy.multiply(x[s], weights)))  # 17

    if s_matrix is not None:
        constraints.append(stratification_constraints(s_matrix, splits, delta, x))

    intra_weights = similarities if similarities is not None else np.max(distances) - distances

    # Because of different weights tmp != len(clusters) * (len(clusters) - 1) / 2
    tmp = [[weights[e1] * weights[e2] * intra_weights[e1, e2] * cvxpy.max(
        cvxpy.vstack([x[s, e1] - x[s, e2] for s in range(len(splits))])
    ) for e2 in range(e1 + 1, len(clusters))] for e1 in range(len(clusters))]  # 15

    loss = cvxpy.sum([t for tmp_list in tmp for t in tmp_list])
    problem = solve(loss, constraints, max_sec, solver, log_file)
    print(problem)

    return None if problem is None else {
        e: names[s] for s in range(len(splits)) for i, e in enumerate(clusters) if x[s, i].value > 0.1
    }
