import logging
from typing import List, Optional, Dict, Union

import cvxpy
import mosek
import numpy as np


def solve_ccs_bqp(
        clusters: List[Union[str, int]],
        weights: List[float],
        similarities: Optional[np.ndarray],
        distances: Optional[np.ndarray],
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

    if distances is not None:
        for b in range(len(splits)):
            for i in range(len(clusters)):
                for j in range(len(clusters)):
                    constraints.append(cvxpy.maximum((x[i, b] + x[j, b]) - 1, 0) * distances[i, j] <= threshold)
    else:
        for b in range(len(splits)):
            for i in range(len(clusters)):
                for j in range(len(clusters)):
                    constraints.append((x[i, b] - x[j, b]) ** 2 * similarities[i, j] <= threshold)

    size_loss = sum(
        (sum(x[i, b] * weights[i] for i in range(len(clusters))) - splits[b] * sum(weights)) ** 2
        for b in range(len(splits))
    )
    if distances is not None:
        cluster_loss = sum(
            cvxpy.maximum((x[i, b] + x[j, b]) - 1, 0) * distances[i, j]
            for i in range(len(clusters)) for j in range(len(clusters)) for b in range(len(splits))
        )
    else:
        cluster_loss = sum(
            (x[i, b] - x[j, b]) ** 2 * similarities[i, j]
            for i in range(len(clusters)) for j in range(len(clusters)) for b in range(len(splits))
        )

    # solve
    logging.info("Start solving with MOSEK")
    objective = cvxpy.Minimize(alpha * size_loss + cluster_loss)
    problem = cvxpy.Problem(objective, constraints)
    problem.solve(
        solver=cvxpy.SCIP,
        qcp=True,
        # mosek_params={
        #     mosek.dparam.optimizer_max_time: max_sec,
        # },
        verbose=True,
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
    print("5 clusters - distance")
    print(
        solve_ccs_bqp(
            clusters=["1", "2", "3", "4", "5"],
            weights=[3, 3, 3, 2, 2],
            similarities=None,
            distances=np.asarray([
                [0, 0, 0, 4, 4],
                [0, 0, 0, 4, 4],
                [0, 0, 0, 4, 4],
                [4, 4, 4, 0, 0],
                [4, 4, 4, 0, 0],
            ]),
            threshold=1,
            limit=0.2,
            splits=[0.7, 0.3],
            names=["train", "test"],
            max_sec=10,
            max_sol=0,
        )
    )
    exit(0)

    print("5 clusters - similarity")
    print(
        solve_ccs_bqp(
            clusters=["1", "2", "3", "4", "5"],
            weights=[3, 3, 3, 2, 2],
            similarities=np.asarray([
                [5, 5, 5, 0, 0],
                [5, 5, 5, 0, 0],
                [5, 5, 5, 0, 0],
                [0, 0, 0, 5, 5],
                [0, 0, 0, 5, 5],
            ]),
            distances=None,
            threshold=1,
            limit=0.2,
            splits=[0.7, 0.3],
            names=["train", "test"],
            max_sec=10,
            max_sol=0,
        )
    )
