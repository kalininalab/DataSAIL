from typing import Optional, Union
from pathlib import Path
from typing import Optional, Union

import cvxpy
import numpy as np
from scipy.optimize import fsolve

from datasail.dataset import DataSet
from datasail.reader.utils import DimTechnique
from datasail.solver.utils import solve, compute_limits, stratification_constraints, collect_results_2d2


def solve_multi_d(
        dim_techs: list[DimTechnique],
        datasets: list[DataSet],
        splits: list[list[float]],
        names: list[list[str]],
        delta: float,
        epsilon: float,
        max_sec: int,
        solver: str,
        log_file: Path,
) -> Optional[tuple[dict[str, str], dict[str, str]]]:
    """
    Solve cluster-based double-cold splitting using disciplined quasi-convex programming and binary quadratic
    programming.

    Args:
        e_clusters: List of cluster names to split from the e-dataset
        e_s_matrix: Stratification for the e-dataset
        e_similarities: Pairwise similarity matrix of clusters in the order of their names
        e_distances: Pairwise distance matrix of clusters in the order of their names
        e_weights: Weights of the clusters in the order of their names in e_clusters
        e_splits: List of split sizes of the e-dataset
        e_names: List of names of the splits of the e-dataset in the order of the splits argument
        f_clusters: List of cluster names to split from the f-dataset
        f_s_matrix: Stratification for the f-dataset
        f_similarities: Pairwise similarity matrix of clusters in the order of their names
        f_distances: Pairwise distance matrix of clusters in the order of their names
        f_weights: Weights of the clusters in the order of their names in f_clusters
        f_splits: List of split sizes of the f-dataset
        f_names: List of names of the splits of the f-dataset in the order of the splits argument
        delta: Additive bound for stratification imbalance
        epsilon: Additive bound for exceeding the requested split size
        max_sec: Maximal number of seconds to take when optimizing the problem (not for finding an initial solution)
        solver: Solving algorithm to use to solve the formulated program
        log_file: File to store the detailed log from the solver to

    Returns:
        A list of interactions and their assignment to a split and two mappings from entities to splits, one for each
        dataset
    """
    min_limits = [compute_limits(epsilon, sum(dataset.cluster_weights), splits[i]) for i, dataset in enumerate(datasets)]
    xs = [cvxpy.Variable((len(splits[i]), len(dataset.cluster_weights)), boolean=True) for i, dataset in enumerate(datasets)]

    # check if the cluster relations are uniform
    e_intra_weights = e_similarities if e_similarities is not None else 1 - e_distances
    f_intra_weights = f_similarities if f_similarities is not None else 1 - f_distances

    constraints = [
        cvxpy.sum(x_e, axis=0) == np.ones((len(e_clusters)), dtype=int),
        cvxpy.sum(x_f, axis=0) == np.ones((len(f_clusters)), dtype=int),
    ]
    for s, (lim_e, lim_f) in enumerate(zip(min_lim_e, min_lim_f)):
        constraints.append(lim_e <= cvxpy.sum(cvxpy.multiply(x_e[s], e_weights)))
        constraints.append(lim_f <= cvxpy.sum(cvxpy.multiply(x_f[s], f_weights)))

    if e_s_matrix is not None:
        constraints.append(stratification_constraints(e_s_matrix, e_splits, delta, x_e))
    if f_s_matrix is not None:
        constraints.append(stratification_constraints(f_s_matrix, f_splits, delta, x_f))
