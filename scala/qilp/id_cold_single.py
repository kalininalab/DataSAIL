from typing import List

import numpy as np
from cvxopt.modeling import variable, op
import cvxpy
from cvxpy import CVXOPT, GLPK, GLPK_MI, GLOP, CBC, ECOS, ECOS_BB, SCS, DIFFCP, GUROBI, OSQP, CPLEX, MOSEK, XPRESS, NAG, PDLP, SCIP, SCIPY


NON_MIP = {
    "CVXOPT": CVXOPT,
    "GLPK": GLPK,
    "GLOP": GLOP,
    "ECOS": ECOS,
    "SCS": SCS,
    "OSQP": OSQP,
    "PDLP": PDLP,
    "SCIPY": SCIPY,
}


NOT_INSTALLED = {
    "CBC": CBC,
    "DIFFCP": DIFFCP,
    "GUROBI": GUROBI,
    "CPLEX": CPLEX,
    "MOSEK": MOSEK,
    "XPRESS": XPRESS,
    "NAG": NAG,
    "SCIP": SCIP,
}


SOLVERS = {
    "GLPK_MI": GLPK_MI,
    "ECOS_BB": ECOS_BB,
}
SOLVERS.update(NON_MIP)
SOLVERS.update(NOT_INSTALLED)


def solve_icx_qilp(
        molecules: List[str],
        weights: List[float],
        limit: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
):
    return solve_icx_qilp_cvxpy_pro(
        molecules,
        weights,
        limit,
        splits,
        names,
        max_sec,
        max_sol,
    )


def solve_icx_qilp_cvxopt(
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
            x[i, b] = variable(1, f"x_{i}_{b}")

    c = []
    for i in range(len(molecules)):
        for b in range(len(splits)):
            c.append(x[i, b] in [0, 1])

    for i in range(len(molecules)):
        c.append(sum(x[i, b] for b in range(len(splits))) == 1)

    for b in range(len(splits)):
        var = sum(x[i, b] * weights[i] for i in range(len(molecules)))
        c.append(int(splits[b] * sum(weights) * (1 - limit)) <= var)
        c.append(var <= int(splits[b] * sum(weights) * (1 - limit)))

    model = op(sum((sum(x[i, b] * weights[i] for i in range(len(molecules))) - int(splits[b] * sum(weights) * limit)) ** 2 for b in range(len(splits))) ** 0.5, c)
    model.solve()
    print(model.status)

    return model


def solve_icx_qilp_cvxpy(
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
            x[i, b] = cvxpy.Variable(integer=True)

    c = []
    for i in range(len(molecules)):
        for b in range(len(splits)):
            c.append(0 <= x[i, b])
            c.append(x[i, b] <= 1)

    for i in range(len(molecules)):
        c.append(sum(x[i, b] for b in range(len(splits))) == 1)

    for b in range(len(splits)):
        print(int(splits[b] * sum(weights) * (1 - limit)), int(splits[b] * sum(weights) * (1 + limit)))
        var = sum(x[i, b] * weights[i] for i in range(len(molecules)))
        c.append(int(splits[b] * sum(weights) * (1 - limit)) <= var)
        c.append(var <= int(splits[b] * sum(weights) * (1 + limit)))

    objective = cvxpy.Minimize(sum((sum(x[i, b] * weights[i] for i in range(len(molecules))) - int(splits[b] * sum(weights) * limit)) ** 2 for b in range(len(splits))) ** 0.5)
    # objective = cvxpy.Minimize(1)
    solver = cvxpy.Problem(objective, c)

    solver.solve(solver=GLPK_MI, qcp=True)

    """for algo in SOLVERS:
        print(algo)
        try:
            solver.solve(solver=SOLVERS[algo], qcp=True)
        except Exception as e:
            print(f"\t{e}")
            continue
        print("\tWorked")
        print(f"\t{solver.status}")
        print(f"\t{solver.value}")
    exit(0)
    """

    print(solver.status)
    print(solver.value)

    # exit(0)

    for i in range(len(molecules)):
        for b in range(len(splits)):
            if x[i, b].value > 0.1:
                print(i, b, x[i, b].value, names[b])

    exit(0)
    return solver


def solve_icx_qilp_cvxpy_pro(
        molecules: List[str],
        weights: List[float],
        limit: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
):
    splits = np.array(splits)
    n_mol = len(molecules)
    n_spl = len(splits)
    shape = (n_mol, n_spl)

    x = cvxpy.Variable(shape, integer=True)

    lower = np.zeros(shape)
    upper = np.ones(shape)
    bs = np.ones(n_spl)
    sizes = np.ones(n_mol) @ cvxpy.multiply(x, np.repeat(weights, n_spl).reshape(shape))
    target = splits * sum(weights)

    constrains = [
        lower <= x,
        x <= upper,
        x @ bs == np.ones(n_mol),
        target * (1 - limit) <= sizes,
        sizes <= target * (1 + limit),
    ]

    problem = cvxpy.Problem(cvxpy.Minimize(cvxpy.quad_form(sizes - target, np.ones((n_spl, n_spl)))), constrains)

    for algo in SOLVERS:
        print(algo)
        try:
            problem.solve(solver=SOLVERS[algo], qcp=True)
        except Exception as e:
            print(f"\t{e}")
            continue
        print("\tWorked")
        print(f"\t{problem.status}")
        print(f"\t{problem.value}")
    exit(0)

    problem.solve()
    print(problem.status)
    print(x.value)
    exit(0)
