from typing import Optional, Tuple, List, Set, Dict

import cvxpy
import numpy as np

from datasail.solver.utils import solve, inter_mask, estimate_surviving_interactions
from datasail.solver.vector.utils import interaction_constraints


def solve_icd_bqp(
        e_entities: List[object],
        f_entities: List[object],
        inter: Set[Tuple[str, str]],
        epsilon: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
        solver: str,
) -> Optional[Tuple[List[Tuple[str, str, str]], Dict[object, str], Dict[object, str]]]:
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

    Returns:
        A list of interactions and their assignment to a split and two mappings from entities to splits, one for each
        dataset
    """
    inter_count = len(inter)
    background = estimate_surviving_interactions(inter_count, len(e_entities), len(f_entities), splits)
    inter_ones = inter_mask(e_entities, f_entities, inter)
    min_lim = [int((split - epsilon) * inter_count) for split in splits]
    max_lim = [int((split + epsilon) * inter_count) for split in splits]

    x_e = [cvxpy.Variable((len(e_entities), 1), boolean=True) for _ in range(len(splits))]
    x_f = [cvxpy.Variable((len(f_entities), 1), boolean=True) for _ in range(len(splits))]
    x_i = [cvxpy.Variable((len(e_entities), len(f_entities)), boolean=True) for _ in range(len(splits))]

    constraints = [
        cvxpy.sum([x[:, 0] for x in x_e]) == np.ones((len(e_entities))),
        cvxpy.sum([x[:, 0] for x in x_f]) == np.ones((len(f_entities))),
    ]
    constraints += [
        cvxpy.sum([x for x in x_i]) <= inter_ones,
    ]

    for s, split in enumerate(splits):
        constraints += [
            min_lim[s] <= cvxpy.sum(cvxpy.sum(cvxpy.multiply(inter_ones, x_i[s]), axis=0), axis=0),
            cvxpy.sum(cvxpy.sum(cvxpy.multiply(inter_ones, x_i[s]), axis=0), axis=0) <= max_lim[s],
        ]

        interaction_constraints(len(e_entities), len(f_entities), x_e, x_f, x_i, s)

    inter_loss = cvxpy.sum(cvxpy.sum(inter_ones - cvxpy.sum([x for x in x_i]), axis=0), axis=0) / background

    problem = solve(inter_loss, constraints, max_sec, len(x_e) + len(x_f) + len(x_i), solver)

    print(inter_loss.value)

    # report the found solution
    output = ([], dict(
        (e, names[s]) for s in range(len(splits)) for i, e in enumerate(e_entities) if x_e[s][:, 0][i].value > 0.1
    ), dict(
        (f, names[s]) for s in range(len(splits)) for j, f in enumerate(f_entities) if x_f[s][:, 0][j].value > 0.1
    ))
    for i, e in enumerate(e_entities):
        for j, f in enumerate(f_entities):
            if (e, f) in inter:
                for b in range(len(splits)):
                    if x_i[b][i, j].value > 0:
                        output[0].append((e, f, names[b]))
                if sum(x_i[b][i, j].value for b in range(len(splits))) == 0:
                    output[0].append((e, f, "not selected"))

    return output
