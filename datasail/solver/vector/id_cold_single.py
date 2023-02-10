import logging
from typing import List, Dict, Optional

import cvxpy
import numpy as np

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

    m = np.ones((len(e_entities)))
    o = [split * sum(e_weights) for split in splits]
    w = np.stack([e_weights] * len(splits))
    min_lim = [int(split * sum(e_weights) * (1 - limit)) for split in splits]
    max_lim = [int(split * sum(e_weights) * (1 + limit)) for split in splits]

    x_e = cvxpy.Variable((len(e_entities), len(splits)), boolean=True)
    constraints = [
        cvxpy.sum(x_e, axis=1) == m,
        min_lim <= cvxpy.sum(cvxpy.multiply(w.T, x_e), axis=0),
        cvxpy.sum(cvxpy.multiply(w.T, x_e), axis=0) <= max_lim,
    ]

    logging.info("Start solving with SCIP")

    loss = cvxpy.sum_squares(cvxpy.sum(cvxpy.multiply(w.T, x_e), axis=0) - o)
    solve(loss, constraints, max_sec, 1)

    return dict(
        (e, names[s]) for s in range(len(splits)) for i, e in enumerate(e_entities) if x_e[i, s].value > 0.1
    )
