from typing import Optional
from pathlib import Path

import cvxpy
import numpy as np

from datasail.solver.cluster_2d import convert
from datasail.solver.utils import solve, compute_limits, stratification_constraints


def solve_i2(
        e_entities: list[str],
        e_stratification: Optional[np.ndarray],
        e_weights: np.ndarray,
        e_splits: list[float],
        e_names: list[str],
        f_entities: list[str],
        f_stratification: Optional[np.ndarray],
        f_weights: np.ndarray,
        f_splits: list[float],
        f_names: list[str],
        delta: float,
        epsilon: float,
        max_sec: int,
        solver: str,
        log_file: Path,
) -> Optional[tuple[dict[object, str], dict[object, str]]]:
    """
    Solve identity-based double-cold splitting using disciplined quasi-convex programming and binary quadratic
    programming.

    Args:
        e_entities: List of entity names to split in e-dataset
        e_weights: Weights of the entities in the e-dataset
        e_stratification: Stratification for the e-dataset
        e_splits: List of split sizes for the e-dataset
        e_names: List of names of the splits for the e-dataset in the order of the splits argument
        f_entities: List of entity names to split in f-dataset
        f_weights: Weights of the entities in the f-dataset
        f_stratification: Stratification for the f-dataset
        f_splits: List of split sizes for the f-dataset
        f_names: List of names of the splits for the f-dataset in the order of the splits argument
        inter: List of interactions
        delta: Additive bound for stratification imbalance
        epsilon: Additive bound for exceeding the requested split size
        max_sec: Maximal number of seconds to take when optimizing the problem (not for finding an initial solution)
        solver: Solving algorithm to use to solve the formulated program
        log_file: File to store the detailed log from the solver to

    Returns:
        A list of interactions and their assignment to a split and two mappings from entities to splits, one for each
        dataset
    """
    e_splits = convert(e_splits)
    f_splits = convert(f_splits)
    min_lim_e = compute_limits(epsilon, sum(e_weights), e_splits)
    min_lim_f = compute_limits(epsilon, sum(f_weights), f_splits)

    x_e = cvxpy.Variable((len(e_splits), len(e_entities)), boolean=True)
    x_f = cvxpy.Variable((len(f_splits), len(f_entities)), boolean=True)

    constraints = [cvxpy.sum(x_e, axis=0) == np.ones((len(e_entities))),
                   cvxpy.sum(x_f, axis=0) == np.ones((len(f_entities)))]
    
    for s, (lim_e, lim_f) in enumerate(zip(min_lim_e, min_lim_f)):
        constraints.append(lim_e <= cvxpy.sum(cvxpy.multiply(x_e[s], e_weights)))
        constraints.append(lim_f <= cvxpy.sum(cvxpy.multiply(x_f[s], f_weights)))

    if e_stratification is not None:
        stratification_constraints(e_stratification, e_splits, delta, x_e)
    if f_stratification is not None:
        stratification_constraints(f_stratification, f_splits, delta, x_f)

    problem = solve(1, constraints, max_sec, solver, log_file)

    if problem is None:
        return None

    # report the found solution
    return {e: e_names[s] for s in range(len(e_splits)) for i, e in enumerate(e_entities) if x_e[s, i].value > 0.1}, \
           {f: f_names[s] for s in range(len(f_splits)) for j, f in enumerate(f_entities) if x_f[s, j].value > 0.1}
