from typing import List, Dict, Optional

import cvxpy
import numpy as np

from datasail.solver.utils import solve


def solve_ics_bqp(
        e_entities: List[str],
        e_weights: List[float],
        epsilon: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
) -> Optional[Dict[str, str]]:
    """
    Solve identity-based cold splitting using disciplined quasi-convex programming and binary quadratic programming.

    Args:
        e_entities: List of entity names to split
        e_weights: Weights of the entities in order of their names in e_entities
        epsilon: Additive bound for exceeding the requested split size
        splits: List of split sizes
        names: List of names of the splits in the order of the splits argument
        max_sec: Maximal number of seconds to take when optimizing the problem (not for finding an initial solution)
        max_sol: Maximal number of solution to consider

    Returns:
        Mapping from entities to splits optimizing the objective function
    """
    m = np.ones((len(e_entities)))
    o = [split * sum(e_weights) for split in splits]
    w = np.stack([e_weights] * len(splits))
    min_lim = [int(split * sum(e_weights) * (1 - epsilon)) for split in splits]
    max_lim = [int(split * sum(e_weights) * (1 + epsilon)) for split in splits]

    x_e = cvxpy.Variable((len(e_entities), len(splits)), boolean=True)
    constraints = [
        cvxpy.sum(x_e, axis=1) == m,
        min_lim <= cvxpy.sum(cvxpy.multiply(w.T, x_e), axis=0),
        cvxpy.sum(cvxpy.multiply(w.T, x_e), axis=0) <= max_lim,
    ]

    loss = cvxpy.sum_squares(cvxpy.sum(cvxpy.multiply(w.T, x_e), axis=0) - o)
    solve(loss, constraints, max_sec, 1)

    return dict(
        (e, names[s]) for s in range(len(splits)) for i, e in enumerate(e_entities) if x_e[i, s].value > 0.1
    )
