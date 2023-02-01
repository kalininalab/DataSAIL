import logging
from typing import List, Union, Optional, Dict

import cvxpy
import numpy as np


def solve_ccs_bqp_matrix(
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

    alpha = 0.5

    m = np.ones((len(clusters)))
    ones = np.ones((1, len(clusters)))

    x = [cvxpy.Variable((len(clusters), 1), boolean=True) for _ in range(len(splits))]

    t = np.full((len(clusters), len(clusters)), threshold)
    min_lim = [int(split * sum(weights) * (1 - limit)) for split in splits]
    max_lim = [int(split * sum(weights) * (1 + limit)) for split in splits]

    constraints = [
        cvxpy.sum([a[:, 0] for a in x]) == m,
    ]
    for s, split in enumerate(splits):
        print()
        constraints += [
            min_lim[s] <= cvxpy.sum(cvxpy.multiply(x[s][:, 0], weights)),
            cvxpy.sum(cvxpy.multiply(x[s][:, 0], weights)) <= max_lim[s],
        ]
        if distances is not None:
            constraints += [
                cvxpy.multiply(cvxpy.maximum((x[s] @ ones) + cvxpy.transpose(x[s] @ ones) - (ones @ ones.T), 0), distances) <= t,
            ]
        else:
            constraints += [
                cvxpy.multiply(((x[s] @ ones) - cvxpy.transpose(x[s] @ ones)) ** 2, similarities) <= t
            ]

    size_loss = sum(
        (cvxpy.sum(cvxpy.multiply(weights, x[s][:, 0])) - s) ** 2
        for s in range(len(splits))
    )

    if distances is not None:
        cluster_loss = cvxpy.sum(
            [cvxpy.sum(cvxpy.multiply(cvxpy.maximum((x[s] @ ones) + cvxpy.transpose(x[s] @ ones) - (ones @ ones.T), 0), distances)) for s in range(len(splits))]
        )
    else:
        cluster_loss = cvxpy.sum(
            [cvxpy.sum(cvxpy.multiply(((x[s] @ ones) - cvxpy.transpose(x[s] @ ones)) ** 2, similarities)) for s in range(len(splits))]
        )

    logging.info("Start solving with SCIP")
    logging.info(f"The problem has {len(x)} variables and {len(constraints)} constraints.")

    objective = cvxpy.Minimize(alpha * size_loss + cluster_loss)
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
    for i in range(len(clusters)):
        for b in range(len(splits)):
            if x[b][i, 0].value > 0.1:
                output[clusters[i]] = names[b]
    return output

if __name__ == '__main__':
    if True:
        print("5 clusters - distance")
        print(
            solve_ccs_bqp_matrix(
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
    else:
        print("5 clusters - similarity")
        print(
            solve_ccs_bqp_matrix(
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
