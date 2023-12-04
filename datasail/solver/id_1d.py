from pathlib import Path
from typing import List, Dict, Optional

import cvxpy
import numpy as np

from datasail.solver.utils import solve


def solve_i1(
        entities: List[str],
        weights: List[float],
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
        weights: Weights of the entities in order of their names in e_entities
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
    x = cvxpy.Variable((len(entities), len(splits)), boolean=True)

    o = [split * sum(weights) for split in splits]
    w = np.stack([weights] * len(splits))
    normalization = 1 / (len(splits) * sum(weights) * epsilon)

    constraints = [cvxpy.sum(x, axis=1) == np.ones((len(entities)))]
    loss = cvxpy.sum(cvxpy.abs(cvxpy.sum(cvxpy.multiply(w.T, x), axis=0) - o)) * normalization
    problem = solve(loss, constraints, max_sec, solver, log_file)

    return None if problem is None else dict(
        (e, names[s]) for s in range(len(splits)) for i, e in enumerate(entities) if x[i, s].value > 0.1
    )
