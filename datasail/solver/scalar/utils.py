from typing import Dict, Tuple, Sized, List, Optional, Set

import cvxpy
import numpy as np
from cvxpy import Expression
from cvxpy.constraints.constraint import Constraint

DataVars = Dict[Tuple[int, int], cvxpy.Variable]
InterVars = Dict[Tuple[int, int, int], cvxpy.Variable]


def init_variables(num_splits: int, len_data: int) -> DataVars:
    """
    Initialize a dictionary of variables based on the number of splits and entities.

    Args:
        num_splits: Number of splits
        len_data: Number of entities

    Returns:
        A dictionary of variables
    """
    x = {}
    for s in range(num_splits):
        for i in range(len_data):
            x[i, s] = cvxpy.Variable(boolean=True)
    return x


def init_inter_variables_cluster(num_splits: int, e_clusters: Sized, f_clusters: Sized) -> InterVars:
    """
    Initialize a dictionary of variables to assign interactions of clusters to splits.

    Args:
        num_splits: Number of splits
        e_clusters: Number of entities in e-dataset
        f_clusters: Number of entities in e-dataset

    Returns:
        A dictionary of variables
    """
    x = {}
    for s in range(num_splits):
        for i in range(len(e_clusters)):
            for j in range(len(f_clusters)):
                x[i, j, s] = cvxpy.Variable(boolean=True)
    return x


def sum_constraint(x: DataVars, len_data: int, num_splits: int) -> List[Constraint]:
    """
    Define the summation constraints, i.e. every entity $i$ has to be assigned to exactly one cluster.

    Args:
        x: Dictionary of indices and variables
        len_data: Number of entities in the dataset
        num_splits: Number of splits to assign to

    Returns:
        A list of cvxpy constraints
    """
    return [sum(x[i, s] for s in range(num_splits)) == 1 for i in range(len_data)]


def cluster_sim_dist_constraint(
        similarities: Optional[np.ndarray],
        distances: Optional[np.ndarray],
        threshold: float,
        num_clusters: int,
        x: DataVars,
        s: int
) -> List[Constraint]:
    """
    Construct constraints on similarity or distance of entities within/across splits.

    Args:
        similarities: Similarity matrix of the data
        distances: Distance matrix of the data
        threshold: Threshold to apply
        num_clusters: Number of clusters
        x: Dictionary of indices and variables for the dataset
        s: Split to consider

    Returns:
        A list of cvxpy constraints
    """
    constraints = []
    if similarities is not None:
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                constraints.append((x[i, s] - x[j, s]) ** 2 * similarities[i][j] <= threshold)
    else:
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                constraints.append(cvxpy.maximum((x[i, s] + x[j, s]) - 1, 0) * distances[i][j] <= threshold)
    return constraints


def cluster_sim_dist_objective(
        similarities: Optional[np.ndarray],
        distances: Optional[np.ndarray],
        num_clusters: int,
        x: DataVars,
        num_splits: int
) -> Expression:
    """
    Construct an objective function of the variables based on a similarity or distance matrix.

    Args:
        similarities: Similarity matrix of the dataset
        distances: Distance matrix of the dataset
        num_clusters: Number of clusters
        x: Dictionary of indices and variables for the e-dataset
        num_splits: Number of splits

    Returns:
        An objective function to minimize
    """
    if similarities is not None:
        return sum(
            (x[i, s] - x[j, s]) ** 2 * similarities[i][j]
            for i in range(num_clusters) for j in range(i + 1, num_clusters) for s in range(num_splits)
        )
    else:
        return sum(
            cvxpy.maximum((x[i, s] + x[j, s]) - 1, 0) * distances[i][j]
            for i in range(num_clusters) for j in range(i + 1, num_clusters) for s in range(num_splits)
        )
