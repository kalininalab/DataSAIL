from typing import List, Union, Optional, Dict
from pathlib import Path

import time
import cvxpy
import numpy as np

from datasail.solver.utils import solve, cluster_y_constraints, compute_limits, stratification_constraints
# from experiments.ablation import david


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
    # y = [[cvxpy.Variable(1, boolean=True) for _ in range(e)] for e in range(len(clusters))]  # 20
    
    constraints = [cvxpy.sum(x, axis=0) == np.ones((len(clusters)))]  # 16

    for s, split in enumerate(splits):
        constraints.append(min_lim[s] <= cvxpy.sum(cvxpy.multiply(x[s], weights)))  # 17

    if s_matrix is not None:
        constraints.append(stratification_constraints(s_matrix, splits, delta, x))

    # constraints += cluster_y_constraints(clusters, y, x, splits)  # 18

    intra_weights = similarities if similarities is not None else np.max(distances) - distances
    # tmp = [[intra_weights[e1, e2] * y[e1][e2] for e2 in range(e1)] for e1 in range(len(clusters))]  # 15

    # Because of different weights tmp != len(clusters) * (len(clusters) - 1) / 2
    tmp = [[weights[e1] * weights[e2] * intra_weights[e1, e2] * cvxpy.max(cvxpy.vstack([x[s, e1] - x[s, e2] for s in range(len(splits))])) for e2 in range(e1 + 1, len(clusters))] for e1 in range(len(clusters))]  # 15

    loss = cvxpy.sum([t for tmp_list in tmp for t in tmp_list])
    # if distances is not None:
    #     loss = -loss
    # loss += cvxpy.sum([cvxpy.sum([y[e1][e2] for e2 in range(e1)]) for e1 in range(len(clusters))])  # 14
    problem = solve(loss, constraints, max_sec, solver, log_file)
    # print("============= Evaluation =============")
    # y_mat = np.full((len(clusters), len(clusters)), 0)
    # w_mat = np.full((len(clusters), len(clusters)), 0)
    # for e1 in range(len(clusters)):
    #     w_mat[e1, e1] = weights[e1] ** 2
    #     for e2 in range(e1):
    #         y_mat[e1, e2] = np.max([x[s, e1].value - x[s, e2].value for s in range(len(splits))])
    #         w_mat[e1, e2] = weights[e1] * weights[e2]
    #         y_mat[e2, e1] = y_mat[e1, e2]
    #         w_mat[e2, e1] = w_mat[e1, e2]
    # print(problem.value)
    # weights = np.array(weights).reshape(-1, 1)
    # print(david.eval(np.array([
    #     [1 if x[0, i].value > 0.1 else -1] for i in range(len(clusters))
    # ]), similarities, weights @ weights.T))  # , y_mat=y_mat, w_mat=w_mat))
    # print("======================================")

    return None if problem is None else {
        e: names[s] for s in range(len(splits)) for i, e in enumerate(clusters) if x[s, i].value > 0.1
    }

