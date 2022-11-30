from typing import List

import numpy as np
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
    problem, x = solve_icx_iqp_cvxpy(molecules, weights, limit, splits)
    problem.solve(solver=cvxpy.MOSEK, qcp=True)

    print(problem.status)
    print(problem.value)

    for i in range(len(molecules)):
        for b in range(len(splits)):
            if x[i, b].value > 0.1:
                print(i, b, x[i, b].value, names[b])

    exit(0)


def solve_icx_iqp_cvxpy(
        molecules: List[str],
        weights: List[float],
        limit: float,
        splits: List[float],
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

    dist_loss = sum((sum(x[i, b] * weights[i] for i in range(len(molecules))) - splits[b] * sum(weights)) ** 2 for b in range(len(splits))) ** 0.5
    objective = cvxpy.Minimize(dist_loss)
    problem = cvxpy.Problem(objective, constraints)

    return problem, x


def solve_icx_iqp_cvxpy_pro(
        molecules: List[str],
        weights: List[float],
        limit: float,
        splits: List[float],
):
    splits = np.array(splits)
    n_mol = len(molecules)
    n_spl = len(splits)
    shape = (n_mol, n_spl)

    x = cvxpy.Variable(shape, boolean=True)

    bs = np.ones(n_spl)
    sizes = np.ones(n_mol) @ cvxpy.multiply(x, np.repeat(weights, n_spl).reshape(shape))
    target = splits * sum(weights)

    constrains = [
        x @ bs == np.ones(n_mol),
        target * (1 - limit) <= sizes,
        sizes <= target * (1 + limit),
    ]

    # objective = cvxpy.Minimize(cvxpy.quad_form(sizes - target, np.ones((n_spl, n_spl))))
    objective = cvxpy.Minimize(((sizes - target) @ (sizes - target).T) ** 0.5)

    problem = cvxpy.Problem(objective, constrains)

    return problem, x
