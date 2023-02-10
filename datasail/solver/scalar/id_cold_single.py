import logging
from typing import List, Dict, Optional

from datasail.solver.scalar.utils import init_variables, sum_constraint
from datasail.solver.utils import solve


def solve_ics_bqp(
        e_entities: List[str],
        e_weights: List[float],
        limit: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
) -> Optional[Dict[str, str]]:
    logging.info("Define optimization problem")

    x_e = init_variables(len(splits), len(e_entities))

    constraints = sum_constraint(e_entities, x_e, splits)

    for b in range(len(splits)):
        var = sum(x_e[i, b] * e_weights[i] for i in range(len(e_entities)))
        constraints += [
            int(splits[b] * sum(e_weights) * (1 - limit)) <= var,
            var <= int(splits[b] * sum(e_weights) * (1 + limit))
        ]

    dist_loss = sum(
        (sum(x_e[i, b] * e_weights[i] for i in range(len(e_entities))) - splits[b] * sum(e_weights)) ** 2
        for b in range(len(splits))
    )

    solve(dist_loss, constraints, max_sec, len(x_e))

    return dict(
        (e, names[b]) for b in range(len(splits)) for i, e in enumerate(e_entities) if x_e[i, b].value > 0.1
    )
