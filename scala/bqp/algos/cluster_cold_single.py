import logging
from typing import List, Optional, Dict

import cvxpy


def solve_ccx_iqp(
        clusters: List[str],
        weights: List[float],
        similarities: List[List[float]],
        threshold: float,
        limit: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
) -> Optional[Dict[str, str]]:

    alpha = 0.5

    x = {}
    for i in range(len(clusters)):
        for b in range(len(splits)):
            x[i, b] = cvxpy.Variable(boolean=True)

    constraints = []

    for i in range(len(clusters)):
        constraints.append(sum(x[i, b] for b in range(len(splits))) == 1)

    for b in range(len(splits)):
        var = sum(x[i, b] * weights[i] for i in range(len(clusters)))
        constraints.append(int(splits[b] * sum(weights) * (1 - limit)) <= var)
        constraints.append(var <= int(splits[b] * sum(weights) * (1 + limit)))

    for b in range(len(splits)):
        for i in range(len(clusters)):
            for j in range(len(clusters)):
                constraints.append((x[i, b] - x[j, b]) ** 2 * similarities[i][j] <= threshold)

    cmb = sum(
        # minimize distance to target size of clusters
        alpha * (sum(x[i, b] * weights[i] for i in range(len(clusters))) - splits[b] * sum(weights)) ** 2 +
        # minimize similarities between elements of clusters
        sum((x[i, b] - x[j, b]) ** 2 * similarities[i][j] for i in range(len(clusters)) for j in range(len(clusters)))
        for b in range(len(splits))
    )

    # solve
    objective = cvxpy.Minimize(cmb)
    problem = cvxpy.Problem(objective, constraints)
    problem.solve(
        solver=cvxpy.MOSEK,
        qcp=True,
        # mosek_params={
        #     mosek.dparam.optimizer_max_time: max_sec * 1_000,
        # },
        # verbose=True,
    )

    logging.info(f"MOSEK status: {problem.status}")
    logging.info(f"Solution's score: {problem.value}")

    if problem.status != "optimal":
        logging.warning(
            'MOSEK cannot solve the problem. Please consider relaxing split restrictions, '
            'e.g., less splits, or a higher tolerance level for exceeding cluster limits.'
        )
        return None

    output = {}
    for i in range(len(clusters)):
        for b in range(len(splits)):
            if x[i, b].value > 0.1:
                output[clusters[i]] = names[b]

    return output
