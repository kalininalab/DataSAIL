import logging
from typing import Optional, Tuple, List, Set, Dict

import cvxpy
import numpy as np


def inter_mask(drugs, prots, inter):
    return inter_mask_dense(drugs, prots, inter) if len(inter) / (len(drugs) + len(prots)) \
        else inter_mask_sparse(drugs, prots, inter)


def inter_mask_dense(drugs, prots, inter):
    output = np.zeros((len(drugs), len(prots)))
    for i, d in enumerate(drugs):
        for j, p in enumerate(prots):
            if (d, p) in inter:
                output[i, j] = 1
    return output


def inter_mask_sparse(drugs, prots, inter):
    output = np.zeros((len(drugs), len(prots)))
    d_map = dict((d, i) for i, d in enumerate(drugs))
    p_map = dict((p, i) for i, p in enumerate(prots))
    for d, p in inter:
        output[d_map[d], p_map[p]] = 1
    return output


def solve_icd_bqp_matrix(
        drugs: List[object],
        proteins: List[object],
        inter: Set[Tuple[str, str]],
        limit: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
) -> Optional[Tuple[List[Tuple[str, str, str]], Dict[str, str], Dict[str, str]]]:
    logging.info("Define optimization problem")

    inter_count = len(inter)
    inter_ones = inter_mask(drugs, proteins, inter)
    min_lim = [int(split * inter_count * (1 - limit)) for split in splits]
    max_lim = [int(split * inter_count * (1 + limit)) for split in splits]

    x_d = [cvxpy.Variable((len(drugs), 1), boolean=True) for _ in range(len(splits))]
    x_p = [cvxpy.Variable((len(proteins), 1), boolean=True) for _ in range(len(splits))]
    x_e = [cvxpy.Variable((len(drugs), len(proteins)), boolean=True) for _ in range(len(splits))]

    constraints = [
        cvxpy.sum([x[:, 0] for x in x_d]) == np.ones((len(drugs))),
        cvxpy.sum([x[:, 0] for x in x_p]) == np.ones((len(proteins))),
    ]
    constraints += [
        cvxpy.sum([x for x in x_e]) <= inter_ones,
    ]

    for s, split in enumerate(splits):
        constraints += [
            min_lim[s] <= cvxpy.sum(cvxpy.sum(cvxpy.multiply(inter_ones, x_e[s]), axis=0), axis=0),
            cvxpy.sum(cvxpy.sum(cvxpy.multiply(inter_ones, x_e[s]), axis=0), axis=0) <= max_lim[s],
        ]

        for i in range(len(drugs)):
            for j in range(len(proteins)):
                constraints.append(x_e[s][i, j] >= (x_d[s][:, 0][i] + x_p[s][:, 0][j] - 1.5))
                constraints.append(x_e[s][i, j] <= (x_d[s][:, 0][i] + x_p[s][:, 0][j]) * 0.5)
                constraints.append(x_d[s][:, 0][i] >= x_e[s][i, j])
                constraints.append(x_p[s][:, 0][j] >= x_e[s][i, j])

    inter_loss = cvxpy.sum(cvxpy.sum(inter_ones - cvxpy.sum([x for x in x_e]), axis=0), axis=0)

    logging.info("Start solving with SCIP")

    objective = cvxpy.Minimize(inter_loss)
    problem = cvxpy.Problem(objective, constraints)
    problem.solve(
        solver=cvxpy.SCIP,
        qcp=True,
        scip_params={
            "limits/time": max_sec,
        }
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
    for i, drug in enumerate(drugs):
        for b in range(len(splits)):
            if x_d[b][:, 0][i].value > 0:
                output[1][drug] = names[b]
    for j, protein in enumerate(proteins):
        for b in range(len(splits)):
            if x_p[b][:, 0][j].value > 0:
                output[2][protein] = names[b]
    for i, drug in enumerate(drugs):
        for j, protein in enumerate(proteins):
            if (drug, protein) in inter:
                for b in range(len(splits)):
                    if x_e[b][i, j].value > 0:
                        output[0].append((drug, protein, names[b]))
                if sum(x_e[b][i, j].value for b in range(len(splits))) == 0:
                    output[0].append((drug, protein, "not selected"))
    print(output)
    return output


def main():
    logging.basicConfig(level=logging.INFO)
    solve_icd_bqp_matrix(
        ["D1", "D2", "D3", "D4", "D5"],
        ["P1", "P2", "P3", "P4", "P5"],
        {
            ("D1", "P1"), ("D1", "P2"), ("D1", "P3"),
            ("D2", "P1"), ("D2", "P2"), ("D2", "P3"),
            ("D3", "P1"), ("D3", "P2"), ("D3", "P3"),
            ("D4", "P4"), ("D4", "P5"),
            ("D5", "P4"), ("D5", "P5"),
        },
        0.2,
        [0.7, 0.3],
        ["train", "test"],
        10,
        0,
    )


if __name__ == '__main__':
    main()
