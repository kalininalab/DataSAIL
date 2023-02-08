import logging
from typing import List, Dict, Optional

import cvxpy


def solve_ics_bqp(
        molecules: List[str],
        weights: List[float],
        limit: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
) -> Optional[Dict[str, str]]:
    logging.info("Define optimization problem")

    x = {}
    for i in range(len(molecules)):
        for b in range(len(splits)):
            x[i, b] = cvxpy.Variable(boolean=True)

    constraints = []

    for i in range(len(molecules)):
        constraints.append(sum(x[i, b] for b in range(len(splits))) == 1)

    for b in range(len(splits)):
        var = sum(x[i, b] * weights[i] for i in range(len(molecules)))
        constraints.append(int(splits[b] * sum(weights) * (1 - limit)) <= var)
        constraints.append(var <= int(splits[b] * sum(weights) * (1 + limit)))

    dist_loss = sum(
        (sum(x[i, b] * weights[i] for i in range(len(molecules))) - splits[b] * sum(weights)) ** 2
        for b in range(len(splits))
    )

    logging.info("Start solving with SCIP")

    objective = cvxpy.Minimize(dist_loss)
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
    solve_ics_bqp(
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
