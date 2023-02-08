import logging
from typing import List, Set, Tuple, Optional, Dict, Union

import cvxpy
import numpy as np


def solve_ccd_bqp_matrix(
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
    logging.info("Define optimization problem")

    alpha = 0.1
    inter_count = np.sum(inter)
    drug_ones = np.ones((1, len(drug_clusters)))
    prot_ones = np.ones((1, len(prot_clusters)))
    inter_ones = np.ones_like(inter)
    drug_t = np.full((len(drug_clusters), len(drug_clusters)), drug_threshold)
    prot_t = np.full((len(prot_clusters), len(prot_clusters)), prot_threshold)
    min_lim = [int(split * inter_count * (1 - limit)) for split in splits]
    max_lim = [int(split * inter_count * (1 + limit)) for split in splits]

    x_d = [cvxpy.Variable((len(drug_clusters), 1), boolean=True) for _ in range(len(splits))]
    x_p = [cvxpy.Variable((len(prot_clusters), 1), boolean=True) for _ in range(len(splits))]
    x_e = [cvxpy.Variable((len(drug_clusters), len(prot_clusters)), boolean=True) for _ in range(len(splits))]

    constraints = [
        cvxpy.sum([x[:, 0] for x in x_d]) == np.ones((len(drug_clusters))),
        cvxpy.sum([x[:, 0] for x in x_p]) == np.ones((len(prot_clusters))),
    ]
    constraints += [
        cvxpy.sum([x for x in x_e]) <= np.ones((len(drug_clusters), len(prot_clusters))),
    ]
    for s, split in enumerate(splits):
        constraints += [
            min_lim[s] <= cvxpy.sum(cvxpy.sum(cvxpy.multiply(inter, x_e[s]), axis=0), axis=0),
            cvxpy.sum(cvxpy.sum(cvxpy.multiply(inter, x_e[s]), axis=0), axis=0) <= max_lim[s],
        ]

        for i in range(len(drug_clusters)):
            for j in range(len(prot_clusters)):
                constraints.append(x_e[s][i, j] >= (x_d[s][:, 0][i] + x_p[s][:, 0][j] - 1.5))
                constraints.append(x_e[s][i, j] <= (x_d[s][:, 0][i] + x_p[s][:, 0][j]) * 0.5)
                constraints.append(x_d[s][:, 0][i] >= x_e[s][i, j])
                constraints.append(x_p[s][:, 0][j] >= x_e[s][i, j])

        if drug_distances is not None:
            constraints += [
                cvxpy.multiply(cvxpy.maximum(
                    (x_d[s] @ drug_ones) + cvxpy.transpose(x_d[s] @ drug_ones) - (drug_ones.T @ drug_ones), 0),
                    drug_distances) <= drug_t,
            ]
        else:
            constraints += [
                cvxpy.multiply(((x_d[s] @ drug_ones) - cvxpy.transpose(x_d[s] @ drug_ones)) ** 2,
                               drug_similarities) <= drug_t
            ]

        if prot_distances is not None:
            constraints += [
                cvxpy.multiply(cvxpy.maximum(
                    (x_p[s] @ prot_ones) + cvxpy.transpose(x_p[s] @ prot_ones) - (prot_ones.T @ prot_ones), 0),
                    prot_distances) <= prot_t,
            ]
        else:
            constraints.append(
                cvxpy.multiply(((x_p[s] @ prot_ones) - cvxpy.transpose(x_p[s] @ prot_ones)) ** 2,
                               prot_similarities) <= prot_t
            )

    inter_loss = cvxpy.sum(
        [cvxpy.sum(cvxpy.sum(cvxpy.multiply(inter_ones - x_e[s], inter), axis=0), axis=0) for s in range(len(splits))]
    )

    if drug_distances is not None:
        drug_loss = cvxpy.sum(
            [cvxpy.sum(cvxpy.multiply(
                cvxpy.maximum(
                    (x_d[s] @ drug_ones) + cvxpy.transpose(x_d[s] @ drug_ones) - (drug_ones.T @ drug_ones), 0
                ), drug_distances
            )) for s in range(len(splits))]
        )
    else:
        drug_loss = cvxpy.sum(
            [cvxpy.sum(
                cvxpy.multiply(((x_d[s] @ drug_ones) - cvxpy.transpose(x_d[s] @ drug_ones)) ** 2, drug_similarities)
            ) for s in range(len(splits))]
        )

    if prot_distances is not None:
        prot_loss = cvxpy.sum(
            [cvxpy.sum(cvxpy.multiply(
                cvxpy.maximum(
                    (x_p[s] @ prot_ones) + cvxpy.transpose(x_p[s] @ prot_ones) - (prot_ones.T @ prot_ones), 0
                ), prot_distances
            )) for s in range(len(splits))]
        )
    else:
        prot_loss = cvxpy.sum(
            [cvxpy.sum(
                cvxpy.multiply(((x_p[s] @ prot_ones) - cvxpy.transpose(x_p[s] @ prot_ones)) ** 2, prot_similarities)
            ) for s in range(len(splits))]
        )

    logging.info("Start solving with SCIP")
    logging.info(f"The problem has {len(x_d) + len(x_p) + len(x_e)} variables and {len(constraints)} constraints.")

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
            if x_d[b][:, 0][i].value > 0:
                output[1][drug_clusters[i]] = names[b]
    for j, protein in enumerate(prot_clusters):
        for b in range(len(splits)):
            if x_p[b][:, 0][j].value > 0:
                output[2][prot_clusters[j]] = names[b]
    for i, drug in enumerate(drug_clusters):
        for j, protein in enumerate(prot_clusters):
            for b in range(len(splits)):
                if x_e[b][i, j].value > 0:
                    output[0].append((drug_clusters[i], prot_clusters[j], names[b]))
            if sum(x_e[b][i, j].value for b in range(len(splits))) == 0:
                output[0].append((drug_clusters[i], prot_clusters[j], "not selected"))
    return output


def main():
    logging.basicConfig(level=logging.INFO)
    solve_ccd_bqp_matrix(
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
        np.asarray([
            [9, 9, 0],
            [9, 9, 0],
            [0, 0, 9],
        ]),
        0.2,
        [0.8, 0.2],
        ["train", "test"],
        10,
        0,
    )


if __name__ == '__main__':
    main()
