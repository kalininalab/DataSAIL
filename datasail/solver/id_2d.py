from typing import Optional, Tuple, List, Set, Dict
from pathlib import Path

import cvxpy
import numpy as np

from datasail.solver.utils import solve, interaction_contraints, collect_results_2d, compute_limits, \
    stratification_constraints


def solve_i2(
        e_entities: List[str],
        e_stratification: Optional[np.ndarray],
        f_entities: List[str],
        f_stratification: Optional[np.ndarray],
        inter: Set[Tuple[str, str]],
        delta: float,
        epsilon: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
        solver: str,
        log_file: Path,
) -> Optional[Tuple[Dict[Tuple[str, str], str], Dict[object, str], Dict[object, str]]]:
    """
    Solve identity-based double-cold splitting using disciplined quasi-convex programming and binary quadratic
    programming.

    Args:
        e_entities: List of entity names to split in e-dataset
        e_stratification: Stratification for the e-dataset
        f_entities: List of entity names to split in f-dataset
        f_stratification: Stratification for the f-dataset
        inter: List of interactions
        delta: Additive bound for stratification imbalance
        epsilon: Additive bound for exceeding the requested split siz
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
    inter_count = len(inter)
    min_lim = compute_limits(epsilon / 2, len(inter), [s / 2 for s in splits])

    x_e = cvxpy.Variable((len(splits), len(e_entities)), boolean=True)
    x_f = cvxpy.Variable((len(splits), len(f_entities)), boolean=True)
    x_i = {(e, f): cvxpy.Variable(len(splits), boolean=True) for (e, f) in inter}

    def index(x, y):
        return (e_entities[x], f_entities[y]) if (e_entities[x], f_entities[y]) in x_i else None

    constraints = [cvxpy.sum(x_e, axis=0) == np.ones((len(e_entities))),
                   cvxpy.sum(x_f, axis=0) == np.ones((len(f_entities)))]

    if 0 not in e_stratification.shape:
        stratification_constraints(e_stratification, splits, delta, x_e)
    if 0 not in f_stratification.shape:
        stratification_constraints(f_stratification, splits, delta, x_f)

    interaction_contraints(e_entities, f_entities, x_i, constraints, splits, x_e, x_f, min_lim, lambda key: 1, index)

    inter_loss = (inter_count - sum(cvxpy.sum(x) for x in x_i.values())) / inter_count
    problem = solve(inter_loss, constraints, max_sec, solver, log_file)

    return collect_results_2d(problem, names, splits, e_entities, f_entities, x_e, x_f, x_i, index)
