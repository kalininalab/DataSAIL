import logging
from typing import List, Dict, Optional

import cvxpy
import numpy as np


def solve_ics_bqp_matrix(
        molecules: List[str],
        weights: List[float],
        limit: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
) -> Optional[Dict[str, str]]:
    logging.info("Define optimization problem")

    m = np.ones((len(molecules)))
    o = [split * sum(weights) for split in splits]
    w = np.stack([weights] * len(splits))
    min_lim = [int(split * sum(weights) * (1 - limit)) for split in splits]
    max_lim = [int(split * sum(weights) * (1 + limit)) for split in splits]

    x = cvxpy.Variable((len(molecules), len(splits)), boolean=True)
    constraints = [
        cvxpy.sum(x, axis=1) == m,
        min_lim <= cvxpy.sum(cvxpy.multiply(w.T, x), axis=0),
        cvxpy.sum(cvxpy.multiply(w.T, x), axis=0) <= max_lim,
    ]

    logging.info("Start solving with SCIP")

    objective = cvxpy.Minimize(cvxpy.sum_squares(cvxpy.sum(cvxpy.multiply(w.T, x), axis=0) - o))
    problem = cvxpy.Problem(objective, constraints)
    problem.solve(
        solver=cvxpy.SCIP,
        qcp=True,
        scip_params={
            "limits/time": max_sec,
        },
    )

    logging.info(f"SCIP status: {problem.status}")
    logging.info(f"Solution's score: {problem.value}")

    if "optimal" not in problem.status:
        logging.warning(
            'SCIP cannot solve the problem. Please consider relaxing split restrictions, '
            'e.g., less splits, or a higher tolerance level for exceeding cluster limits.'
        )
        return None

    output = {}
    for i in range(len(molecules)):
        for b in range(len(splits)):
            if x[i, b].value > 0.1:
                output[molecules[i]] = names[b]

    return output


def main():
    logging.basicConfig(level=logging.INFO)
    solve_ics_bqp_matrix(
        ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"],
        [6, 6, 6, 6, 6, 6, 4, 4, 4, 4],
        0.2,
        [0.7, 0.3],
        ["train", "test"],
        10,
        0,
    )


if __name__ == '__main__':
    main()
