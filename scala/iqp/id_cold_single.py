from typing import List

import cvxpy


def solve_icx_qip(
        molecules: List[str],
        weights: List[float],
        limit: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
):
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
    objective = cvxpy.Minimize(dist_loss)
    problem = cvxpy.Problem(objective, constraints)

    problem.solve(solver=cvxpy.MOSEK, qcp=True)

    output = {}
    for i in range(len(molecules)):
        for b in range(len(splits)):
            if x[i, b].value > 0.1:
                print(i, b, x[i, b].value, names[b])
                output[molecules[i]] = names[b]

    return output
