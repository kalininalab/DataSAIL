import logging
from typing import List, Optional, Dict

import cvxpy
import numpy as np


def solve_ccx_iqp(
        clusters: List[str],
        weights: List[float],
        similarities: np.ndarray,
        threshold: float,
        limit: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
) -> Optional[Dict[str, str]]:
    print("Defining the optimization problem")
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
    logging.info("Start solving with MOSEK")
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

    print(f"MOSEK status: {problem.status}")
    print(f"Solution's score: {problem.value}")

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


if __name__ == '__main__':
    print("5 clusters")
    print(
        solve_ccx_iqp(
            ["1", "2", "3", "4", "5"],
            [3, 3, 3, 2, 2],
            np.asarray([
                [0, 0, 0, 4, 4],
                [0, 0, 0, 4, 4],
                [0, 0, 0, 4, 4],
                [4, 4, 4, 0, 0],
                [4, 4, 4, 0, 0],
            ]),
            1,
            0.2,
            [0.7, 0.3],
            ["train", "test"],
            0,
            0,
        )
    )
    exit(0)

    print("10 clusters")
    print(
        solve_ccx_iqp(
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            [6, 6, 6, 6, 6, 6, 4, 4, 4, 4],
            np.asarray([
                [0, 0, 0, 0, 0, 0, 4, 4, 4, 4],
                [0, 0, 0, 0, 0, 0, 4, 4, 4, 4],
                [0, 0, 0, 0, 0, 0, 4, 4, 4, 4],
                [0, 0, 0, 0, 0, 0, 4, 4, 4, 4],
                [0, 0, 0, 0, 0, 0, 4, 4, 4, 4],
                [0, 0, 0, 0, 0, 0, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 0, 0, 0, 0],
                [4, 4, 4, 4, 4, 4, 0, 0, 0, 0],
                [4, 4, 4, 4, 4, 4, 0, 0, 0, 0],
                [4, 4, 4, 4, 4, 4, 0, 0, 0, 0],
            ]),
            1,
            0.2,
            [0.7, 0.3],
            ["train", "test"],
            0,
            0,
        )
    )
