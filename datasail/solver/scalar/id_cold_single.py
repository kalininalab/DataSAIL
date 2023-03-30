import logging
from typing import List, Dict, Optional

from datasail.solver.scalar.utils import init_variables, sum_constraint
from datasail.solver.utils import solve


def solve_ics_bqp(
        e_entities: List[str],
        e_weights: List[float],
        epsilon: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
        solver: str,
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
        solver: Solving algorithm to use to solve the formulated program

    Returns:
        Mapping from entities to splits optimizing the objective function
    """
    x_e = init_variables(len(splits), len(e_entities))

    constraints = sum_constraint(x_e, len(e_entities), len(splits))

    for b in range(len(splits)):
        var = sum(x_e[i, b] * e_weights[i] for i in range(len(e_entities)))
        constraints += [
            int((splits[b] - epsilon) * sum(e_weights)) <= var,
            var <= int((splits[b] + epsilon) * sum(e_weights))
        ]

    dist_loss = sum(
        (sum(x_e[i, b] * e_weights[i] for i in range(len(e_entities))) - splits[b] * sum(e_weights)) ** 2
        for b in range(len(splits))
    )

    solve(dist_loss, constraints, max_sec, len(x_e), solver)

    return dict(
        (e, names[b]) for b in range(len(splits)) for i, e in enumerate(e_entities) if x_e[i, b].value > 0.1
    )
