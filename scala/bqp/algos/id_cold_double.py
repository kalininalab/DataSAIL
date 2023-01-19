import logging
from typing import Optional, Tuple, List, Set, Dict

import cvxpy
import mosek


def solve_ic_iqp(
        drugs: List[object],
        proteins: List[object],
        inter: Set[Tuple[str, str]],
        limit: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
) -> Optional[Tuple[List[Tuple[str, str, str]], Dict[str, str], Dict[str, str]]]:
    x_d = {}
    for b in range(len(splits)):
        for i in range(len(drugs)):
            x_d[i, b] = cvxpy.Variable(boolean=True)
    x_p = {}
    for b in range(len(splits)):
        for j in range(len(proteins)):
            x_p[j, b] = cvxpy.Variable(boolean=True)
    x_e = {}
    for b in range(len(splits)):
        for i, drug in enumerate(drugs):
            for j, protein in enumerate(proteins):
                if (drug, protein) in inter:
                    x_e[i, j, b] = cvxpy.Variable(boolean=True)

    constraints = []
    print("Constraint defining")

    for i in range(len(drugs)):
        constraints.append(sum(x_d[i, b] for b in range(len(splits))) == 1)
    for j in range(len(proteins)):
        constraints.append(sum(x_p[j, b] for b in range(len(splits))) == 1)
    for i, drug in enumerate(drugs):
        for j, protein in enumerate(proteins):
            if (drug, protein) in inter:
                constraints.append(sum(x_e[i, j, b] for b in range(len(splits))) <= 1)
    print("D, P, and I constraints defined")

    all_inter = sum(
        x_e[i, j, b] for b in range(len(splits)) for i, drug in enumerate(drugs) for j, protein in enumerate(proteins)
        if (drug, protein) in inter
    )

    print("Define size constraints")
    for b in range(len(splits)):
        var = sum(
            x_e[i, j, b] for i, drug in enumerate(drugs) for j, protein in enumerate(proteins) if
            (drug, protein) in inter
        )
        # constraints.append(splits[b] * all_inter * (1 - limit) <= var)
        # constraints.append(var <= splits[b] * all_inter * (1 + limit))

        for i, drug in enumerate(drugs):
            for j, protein in enumerate(proteins):
                if (drug, protein) in inter:
                    constraints.append(x_e[i, j, b] >= (x_d[i, b] + x_p[j, b] - 1.5))
                    constraints.append(x_e[i, j, b] <= (x_d[i, b] + x_p[j, b]) * 0.5)
                    constraints.append(x_d[i, b] >= x_e[i, j, b])
                    constraints.append(x_p[j, b] >= x_e[i, j, b])

    print("Define objective")

    inter_loss = sum(
        (1 - sum(x_e[i, j, b] for b in range(len(splits)))) for i, drug in enumerate(drugs)
        for j, protein in enumerate(proteins) if (drug, protein) in inter
    )

    print("Start solving the problem")

    objective = cvxpy.Minimize(inter_loss)
    problem = cvxpy.Problem(objective, constraints)
    problem.solve(solver=cvxpy.MOSEK, qcp=True, mosek_params={
            mosek.dparam.optimizer_max_time: max_sec * 1_000,
            # mosek.iparam.max_iterations: max_sol,
        }
    )

    logging.info(f"MOSEK status: {problem.status}")
    logging.info(f"Solution's score: {problem.value}")
    print(f"MOSEK status: {problem.status}")
    print(f"Solution's score: {problem.value}")

    if problem.status != "optimal":
        logging.warning(
            'MOSEK cannot solve the problem. Please consider relaxing split restrictions, '
            'e.g., less splits, or a higher tolerance level for exceeding cluster limits.'
        )
        print(
            'MOSEK cannot solve the problem. Please consider relaxing split restrictions, '
            'e.g., less splits, or a higher tolerance level for exceeding cluster limits.'
        )
        return None

    # report the found solution
    output = ([], {}, {})
    for i, drug in enumerate(drugs):
        for b in range(len(splits)):
            if x_d[i, b].value > 0:
                output[1][drug] = names[b]
    for j, protein in enumerate(proteins):
        for b in range(len(splits)):
            if x_p[j, b].value > 0:
                output[2][protein] = names[b]
    for i, drug in enumerate(drugs):
        for j, protein in enumerate(proteins):
            if (drug, protein) in inter:
                for b in range(len(splits)):
                    if x_e[i, j, b].value > 0:
                        output[0].append((drug, protein, names[b]))
                if sum(x_e[i, j, b].value for b in range(len(splits))) == 0:
                    output[0].append((drug, protein, "not selected"))
    return output
