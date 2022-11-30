from typing import List
import cvxpy
from cvxpy import CVXOPT, GLOP, GLPK, GLPK_MI, CBC, CPLEX, ECOS, ECOS_BB, SCS, SCIP, SCIPY, DIFFCP, GUROBI, OSQP, \
    MOSEK, XPRESS, NAG, PDLP
from cvxpy.transforms import scalarize

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
    "XPRESS": XPRESS,
    "NAG": NAG,
    "SCIP": SCIP,
}


SOLVERS = {
    "GLPK_MI": GLPK_MI,
    "ECOS_BB": ECOS_BB,
    "MOSEK": MOSEK,
}
SOLVERS.update(NON_MIP)
SOLVERS.update(NOT_INSTALLED)


def solve_ccx_iqp_cvxpy(
        clusters: List[object],
        weights: List[float],
        similarities: List[List[float]],
        threshold: float,
        limit: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
):
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
    """
    # minimize the distance between the size of the actual splits and the target relations
    dist_loss = sum((sum(x[i, b] * weights[i] for i in range(len(clusters))) - splits[b] * sum(weights)) ** 2
                    for b in range(len(splits))) ** 0.5

    # maximize the similarity within the splits
    sim_loss = sum(x[i, b] * x[j, b] * similarities[i][j] for i in range(len(clusters)) for j in range(len(clusters))
                   for b in range(len(splits)))
    diff_loss = sum((x[i, b] - x[j, b]) ** 2 * similarities[i][j] for i in range(len(clusters))
                    for j in range(len(clusters)) for b in range(len(splits)))

    # combine
    weighted_sum = scalarize.weighted_sum([dist_loss, diff_loss], [1, 1])"""
    cmb = sum(
        # minimize distance to target size of clusters
        alpha * (sum(x[i, b] * weights[i] for i in range(len(clusters))) - splits[b] * sum(weights)) ** 2 +
        # minimize similarities between elements of clusters
        sum((x[i, b] - x[j, b]) ** 2 * similarities[i][j] for i in range(len(clusters)) for j in range(len(clusters)))
        for b in range(len(splits))
    )

    # solve
    objective = cvxpy.Minimize(cmb)
    problem = cvxpy.Problem(objective, constraints)

    """
    for algo in SOLVERS:
        print(algo)
        try:
            problem.solve(solver=SOLVERS[algo], qcp=True)
        except Exception as e:
            print(e)
            continue
        print("\tWorked")
        print(f"\t{problem.status}")
        print(f"\t{problem.value}")
    exit(0)
    """

    problem.solve(solver=cvxpy.MOSEK, qcp=True)
    print(problem.status)
    print(problem.value)

    for i in range(len(clusters)):
        for b in range(len(splits)):
            if x[i, b].value > 0.1:
                print(i, b, x[i, b].value, names[b])


if __name__ == '__main__':
    print("5 clusters")
    solve_ccx_iqp_cvxpy(
        [1, 2, 3, 4, 5],
        [3, 3, 3, 2, 2],
        [
            [5, 5, 5, 0, 0],
            [5, 5, 5, 0, 0],
            [5, 5, 5, 0, 0],
            [0, 0, 0, 5, 5],
            [0, 0, 0, 5, 5],
        ],
        1,
        0.2,
        [0.7, 0.3],
        ["train", "test"],
        0,
        0,
    )
    print("10 clusters")
    solve_ccx_iqp_cvxpy(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [6, 6, 6, 6, 6, 6, 4, 4, 4, 4],
        [
            [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
            [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
            [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
            [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
            [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
            [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 5, 5, 5],
            [0, 0, 0, 0, 0, 0, 5, 5, 5, 5],
            [0, 0, 0, 0, 0, 0, 5, 5, 5, 5],
            [0, 0, 0, 0, 0, 0, 5, 5, 5, 5],
        ],
        1,
        0.2,
        [0.7, 0.3],
        ["train", "test"],
        0,
        0,
    )
