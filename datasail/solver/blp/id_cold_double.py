from typing import Optional, Tuple, List, Set, Dict

import cvxpy
import numpy as np
from datasail.settings import NOT_ASSIGNED

from datasail.solver.utils import solve, inter_mask, compute_limits, interaction_constraints


def solve_icd_blp(
        e_entities: List[str],
        f_entities: List[str],
        inter: Set[Tuple[str, str]],
        epsilon: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
        solver: str,
        log_file: str,
) -> Optional[Tuple[Dict[Tuple[str, str], str], Dict[object, str], Dict[object, str]]]:
    """
    Solve identity-based double-cold splitting using disciplined quasi-convex programming and binary quadratic
    programming.

    Args:
        e_entities: List of entity names to split in e-dataset
        f_entities: List of entity names to split in f-dataset
        inter: List of interactions
        epsilon: Additive bound for exceeding the requested split size
        splits: List of split sizes
        names: List of names of the splits in the order of the splits argument
        max_sec: Maximal number of seconds to take when optimizing the problem (not for finding an initial solution)
        max_sol: Maximal number of solution to consider
        solver: Solving algorithm to use to solve the formulated program
        log_file: File to store the detailed log from the solver to

    Returns:
        A list of interactions and their assignment to a split and two mappings from entities to splits, one for each
        dataset
    """
    inter_count = len(inter)
    max_lim, min_lim = compute_limits(epsilon, inter_count, splits)

    x_e = [cvxpy.Variable((len(e_entities), 1), boolean=True) for _ in range(len(splits))]
    x_f = [cvxpy.Variable((len(f_entities), 1), boolean=True) for _ in range(len(splits))]
    x_i = {(e, f): cvxpy.Variable(len(splits), boolean=True) for (e, f) in inter}

    constraints = [
        cvxpy.sum([x[:, 0] for x in x_e]) == np.ones((len(e_entities))),
        cvxpy.sum([x[:, 0] for x in x_f]) == np.ones((len(f_entities))),
    ]

    for s, split in enumerate(splits):
        constraints.append(min_lim[s] <= cvxpy.sum([x_i[key][s] for key in x_i]))
        for i, e1 in enumerate(e_entities):
            for j, e2 in enumerate(f_entities):
                if (e1, e2) in x_i:
                    # constraints.append(x_i[e1, e2][s] >= cvxpy.maximum(x_e[s][:, 0][i] + x_f[s][:, 0][j] - 1, 0))
                    # constraints.append(x_i[e1, e2][s] <= 0.75 * (x_e[s][:, 0][i] + x_f[s][:, 0][j]))
                    constraints.append(x_i[e1, e2][s] >= x_e[s][:, 0][i] - x_f[s][:, 0][j])

    inter_loss = (inter_count - sum(cvxpy.sum(x) for x in x_i.values())) / inter_count
    problem = solve(inter_loss, constraints, max_sec, solver, log_file)

    if problem is None:
        return None

    # report the found solution
    output = (
        {},
        {e: names[s] for s in range(len(splits)) for i, e in enumerate(e_entities) if x_e[s][:, 0][i].value > 0.1},
        {f: names[s] for s in range(len(splits)) for j, f in enumerate(f_entities) if x_f[s][:, 0][j].value > 0.1},
    )
    for i, e in enumerate(e_entities):
        for j, f in enumerate(f_entities):
            if (e, f) in inter:
                for b in range(len(splits)):
                    if x_i[e, f][b].value > 0:
                        output[0][(e, f)] = names[b]
                if sum(x_i[e, f][b].value for b in range(len(splits))) == 0:
                    output[0][(e, f)] = NOT_ASSIGNED

    return output
