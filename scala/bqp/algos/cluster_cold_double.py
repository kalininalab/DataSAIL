import logging
from typing import List, Optional, Tuple, Dict, Union

import cvxpy
import numpy as np


def solve_ccd_bqp(
        drug_clusters: List[Union[str, int]],
        drug_similarities: Optional[np.ndarray],
        drug_distances: Optional[np.ndarray],
        drug_threshold: float,
        prot_clusters: List[Union[str, int]],
        prot_similarities: Optional[np.ndarray],
        prot_distances: Optional[np.ndarray],
        prot_threshold: float,
        inter: np.ndarray,
        limit: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
) -> Optional[Tuple[List[Tuple[str, str, str]], Dict[str, str], Dict[str, str]]]:

    alpha = 0.1
    inter_count = np.sum(inter)

    x_d = {}
    for b in range(len(splits)):
        for i in range(len(drug_clusters)):
            x_d[i, b] = cvxpy.Variable(boolean=True)
    x_p = {}
    for b in range(len(splits)):
        for j in range(len(prot_clusters)):
            x_p[j, b] = cvxpy.Variable(boolean=True)
    x_e = {}
    for b in range(len(splits)):
        for i, drug in enumerate(drug_clusters):
            for j, protein in enumerate(prot_clusters):
                x_e[i, j, b] = cvxpy.Variable(boolean=True)

    constraints = []

    for i in range(len(drug_clusters)):
        constraints.append(sum(x_d[i, b] for b in range(len(splits))) == 1)
    for j in range(len(prot_clusters)):
        constraints.append(sum(x_p[j, b] for b in range(len(splits))) == 1)
    for i in range(len(drug_clusters)):
        for j in range(len(prot_clusters)):
            constraints.append(sum(x_e[i, j, b] for b in range(len(splits))) <= 1)

    # all_inter = sum(x_e[i, j, b] for i in range(len(drug_clusters)) for j in range(len(prot_clusters)) for b in range(len(splits)))
    for b in range(len(splits)):
        var = sum(
            x_e[i, j, b] * inter[i, j] for i in range(len(drug_clusters)) for j in range(len(prot_clusters))
        )
        constraints.append(splits[b] * inter_count * (1 - limit) <= var)
        constraints.append(var <= splits[b] * inter_count * (1 + limit))

        for i in range(len(drug_clusters)):
            for j in range(len(prot_clusters)):
                constraints.append(x_e[i, j, b] >= (x_d[i, b] + x_p[j, b] - 1.5))
                constraints.append(x_e[i, j, b] <= (x_d[i, b] + x_p[j, b]) * 0.5)
                constraints.append(x_d[i, b] >= x_e[i, j, b])
                constraints.append(x_p[j, b] >= x_e[i, j, b])

        if drug_similarities is not None:
            for i in range(len(drug_clusters)):
                for j in range(i + 1, len(drug_clusters)):
                    constraints.append((x_d[i, b] - x_d[j, b]) ** 2 * drug_similarities[i][j] <= drug_threshold)
        else:
            for i in range(len(drug_clusters)):
                for j in range(i + 1, len(drug_clusters)):
                    constraints.append(cvxpy.maximum((x_d[i, b] + x_d[j, b]) - 1, 0) * drug_distances[i][j] <= drug_threshold)

        if prot_similarities is not None:
            for i in range(len(prot_clusters)):
                for j in range(i + 1, len(prot_clusters)):
                    constraints.append((x_p[i, b] - x_p[j, b]) ** 2 * prot_similarities[i][j] <= prot_threshold)
        else:
            for i in range(len(prot_clusters)):
                for j in range(i + 1, len(prot_clusters)):
                    constraints.append(cvxpy.maximum((x_p[i, b] + x_p[j, b]) - 1, 0) * prot_distances[i][j] <= prot_threshold)

    inter_loss = sum(
        (1 - x_e[i, j, b]) * inter[i, j]
        for i in range(len(drug_clusters)) for j in range(len(prot_clusters)) for b in range(len(splits))
    )

    if drug_similarities is not None:
        drug_loss = sum(
            (x_d[i, b] - x_d[j, b]) ** 2 * drug_similarities[i][j]
            for i in range(len(drug_clusters)) for j in range(i + 1, len(drug_clusters)) for b in range(len(splits))
        )
    else:
        drug_loss = sum(
            cvxpy.maximum((x_d[i, b] + x_d[j, b]) - 1, 0) * drug_distances[i][j]
            for i in range(len(drug_clusters)) for j in range(i + 1, len(drug_clusters)) for b in range(len(splits))
        )

    if prot_similarities is not None:
        prot_loss = sum(
            (x_p[i, b] - x_p[j, b]) ** 2 * prot_similarities[i][j]
            for i in range(len(prot_clusters)) for j in range(i + 1, len(prot_clusters)) for b in range(len(splits))
        )
    else:
        prot_loss = sum(
            cvxpy.maximum((x_p[i, b] + x_p[j, b]) - 1, 0) * prot_distances[i][j]
            for i in range(len(prot_clusters)) for j in range(i + 1, len(prot_clusters)) for b in range(len(splits))
        )

    objective = cvxpy.Minimize(alpha * inter_loss + drug_loss + prot_loss)
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

    # report the found solution
    output = ([], {}, {})
    for i, drug in enumerate(drug_clusters):
        for b in range(len(splits)):
            if x_d[i, b].value > 0:
                output[1][drug_clusters[i]] = names[b]
    for j, protein in enumerate(prot_clusters):
        for b in range(len(splits)):
            if x_p[j, b].value > 0:
                output[2][prot_clusters[j]] = names[b]
    for i, drug in enumerate(drug_clusters):
        for j, protein in enumerate(prot_clusters):
            for b in range(len(splits)):
                if x_e[i, j, b].value > 0:
                    output[0].append((drug_clusters[i], prot_clusters[j], names[b]))
            if sum(x_e[i, j, b].value for b in range(len(splits))) == 0:
                output[0].append((drug_clusters[i], prot_clusters[j], "not selected"))
    return output


def main():
    print(
        solve_ccd_bqp(
            ["D1", "D2", "D3"],
            np.asarray([
                [5, 5, 0],
                [5, 5, 0],
                [0, 0, 5],
            ]),
            None,
            4,
            ["P1", "P2", "P3"],
            np.asarray([
                [5, 5, 0],
                [5, 5, 0],
                [0, 0, 5],
            ]),
            None,
            4,
            [
                [9, 9, 0],
                [9, 9, 0],
                [0, 0, 9],
            ],
            0.2,
            [0.8, 0.2],
            ["train", "test"],
            0,
            0,
        )
    )


if __name__ == '__main__':
    main()
