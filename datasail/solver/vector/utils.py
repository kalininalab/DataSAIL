from typing import List, Optional, Union

import cvxpy
import numpy as np
from cvxpy import Variable, Expression
from cvxpy.constraints.constraint import Constraint


def interaction_constraints(
        len_e_data: int, len_f_data: int, x_e: List[Variable], x_f: List[Variable], x_i: List[Variable], s: int
) -> List[Constraint]:
    """
    Define the constraints that two clusters are in the same split iff their interaction (if exists) is in that split.

    Args:
        len_e_data: Number of datapoints in the e-dataset
        len_f_data: Number of datapoints in the f-dataset
        x_e: List of variables for the e-dataset
        x_f: List of variables for the f-dataset
        x_i: List of variables for the interactions
        s: Current split to consider

    Returns:
        A list of cvxpy constraints
    """
    constraints = []
    for i in range(len_e_data):
        for j in range(len_f_data):
            constraints.append(x_i[s][i, j] >= (x_e[s][:, 0][i] + x_f[s][:, 0][j] - 1.5))
            constraints.append(x_i[s][i, j] <= (x_e[s][:, 0][i] + x_f[s][:, 0][j]) * 0.5)
            constraints.append(x_e[s][:, 0][i] >= x_i[s][i, j])
            constraints.append(x_f[s][:, 0][j] >= x_i[s][i, j])
    return constraints


def cluster_sim_dist_constraint(
        similarities: Optional[np.ndarray],
        distances: Optional[np.ndarray],
        threshold: np.ndarray,
        ones: np.ndarray,
        x: List[Variable],
        s: int
) -> List[Constraint]:
    """

    Args:
        similarities: Similarity matrix of the data
        distances: Distance matrix of the data
        threshold: Threshold to apply
        ones: Vector to help in the computations
        x: List of variables for the dataset
        s: Split to consider

    Returns:
        A list of cvxpy constraints
    """
    if distances is not None:
        return cvxpy.multiply(
            cvxpy.maximum((x[s] @ ones) + cvxpy.transpose(x[s] @ ones) - (ones.T @ ones), 0), distances
        ) <= threshold
    return cvxpy.multiply(((x[s] @ ones) - cvxpy.transpose(x[s] @ ones)) ** 2, similarities) <= threshold


def cluster_sim_dist_objective(
        similarities: Optional[np.ndarray],
        distances: Optional[np.ndarray],
        ones: np.ndarray,
        weights: Union[np.ndarray, List[float]],
        x: List[Variable],
        splits: List[float]
) -> Expression:
    """
    Construct an objective function of the variables based on a similarity or distance matrix.

    Args:
        similarities: Similarity matrix of the dataset
        distances: Distance matrix of the dataset
        ones: Vector to help in the computations
        weights: weights of the entities
        x: Dictionary of indices and variables for the e-dataset
        splits: Splits as list of their relative size

    Returns:
        An objective function to minimize
    """
    # n = len(distances) if distances is not None else len(similarities)
    # normalization = 1 / (2 * cvxpy.sum([cvxpy.sum(x[s]) * n - cvxpy.sum(x[s]) ** 2 for s in range(len(splits))]))
    # if distances is not None:
    #     return cvxpy.sum([cvxpy.sum(cvxpy.multiply(
    #         cvxpy.maximum((x[s] @ ones) + cvxpy.transpose(x[s] @ ones) - (ones.T @ ones), 0), distances
    #     )) for s in range(len(splits))]) * normalization
    # return cvxpy.sum([cvxpy.sum(cvxpy.multiply(
    #     ((x[s] @ ones) - cvxpy.transpose(x[s] @ ones)) ** 2, similarities
    # )) for s in range(len(splits))]) * normalization
    if isinstance(weights, List):
        weights = np.array(weights)

    weight_matrix = weights.T @ weights

    if distances is not None:
        hit_matrix = cvxpy.sum([cvxpy.maximum((x[s] @ ones) + cvxpy.transpose(x[s] @ ones) - (ones.T @ ones), 0) for s in range(len(splits))])
        leak_matrix = cvxpy.multiply(hit_matrix, distances)
    else:
        hit_matrix = cvxpy.sum([((x[s] @ ones) - cvxpy.transpose(x[s] @ ones)) for s in range(len(splits))])
        leak_matrix = cvxpy.multiply(hit_matrix, similarities)

    leak_matrix = cvxpy.multiply(leak_matrix, weight_matrix)
    leakage = cvxpy.sum(leak_matrix) / cvxpy.sum(cvxpy.multiply(hit_matrix, weight_matrix))
    return cvxpy.sum(leak_matrix)  # leakage
