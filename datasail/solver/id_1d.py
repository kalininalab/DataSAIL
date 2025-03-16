from pathlib import Path
from typing import List, Dict, Optional

import cvxpy
import numpy as np

from datasail.solver.utils import solve, compute_limits, stratification_constraints


def solve_i1(
        entities: List[str],
        weights: Optional[List[float]],
        stratification: Optional[np.ndarray],
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
    Solve identity-based cold splitting using disciplined quasi-convex programming and binary linear programming.

    Args:
        entities: List of entity names to split
        weights: Weights of the entities in order of their names in entities
        stratification: Stratification of the entities in order of their names in entities
        delta: Additive bound for stratification imbalance
        epsilon: Additive bound for exceeding the requested split size
        splits: List of split sizes
        names: List of names of the splits in the order of the splits argument
        max_sec: Maximal number of seconds to take when optimizing the problem (not for finding an initial solution)
        max_sol: Maximal number of solution to consider
        solver: Solving algorithm to use to solve the formulated program
        log_file: File to store the detailed log from the solver to

    Returns:
        Mapping from entities to splits optimizing the objective function
    """
    min_lim = compute_limits(epsilon, sum(weights), splits)
    x = cvxpy.Variable((len(splits), len(entities)), boolean=True)

    constraints = [cvxpy.sum(x, axis=0) == np.ones((len(entities)))]

    for s, lim in enumerate(min_lim):
        constraints.append(lim <= cvxpy.sum(cvxpy.multiply(x[s], weights)))

    if stratification is not None:
        constraints.append(stratification_constraints(stratification, splits, delta, x))

    problem = solve(1, constraints, max_sec, solver, log_file)

    return None if problem is None else dict(
        (e, names[s]) for s in range(len(splits)) for i, e in enumerate(entities) if x[s, i].value > 0.1
    )
