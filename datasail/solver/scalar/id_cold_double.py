from typing import Optional, Tuple, List, Set, Dict

import cvxpy

from datasail.solver.scalar.utils import init_variables, sum_constraint
from datasail.solver.utils import solve, estimate_number_target_interactions


def solve_icd_bqp(
        e_entities: List[str],
        f_entities: List[str],
        inter: Set[Tuple[str, str]],
        epsilon: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
) -> Optional[Tuple[List[Tuple[str, str, str]], Dict[str, str], Dict[str, str]]]:
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

    Returns:
        A list of interactions and their assignment to a split and two mappings from entities to splits, one for each
        dataset
    """
    all_inter = estimate_number_target_interactions(inter, len(e_entities), len(f_entities), splits)

    x_e = init_variables(len(splits), len(e_entities))
    x_f = init_variables(len(splits), len(f_entities))
    x_i = {}
    for s in range(len(splits)):
        for i, e in enumerate(e_entities):
            for j, f in enumerate(f_entities):
                if (e, f) in inter:
                    x_i[i, j, s] = cvxpy.Variable(boolean=True)

    constraints = sum_constraint(x_e, len(e_entities), len(splits)) + sum_constraint(x_f, len(f_entities), len(splits))

    for i, e in enumerate(e_entities):
        for j, f in enumerate(f_entities):
            if (e, f) in inter:
                constraints.append(sum(x_i[i, j, s] for s in range(len(splits))) <= 1)

    for s in range(len(splits)):
        var = sum(
            x_i[i, j, s] for i, e in enumerate(e_entities) for j, f in enumerate(f_entities) if (e, f) in inter
        )
        constraints += [
            splits[s] * all_inter * (1 - epsilon) <= var,
            var <= splits[s] * all_inter * (1 + epsilon),
        ]
        for i, e1 in enumerate(e_entities):
            for j, e2 in enumerate(f_entities):
                if (e1, e2) in inter:
                    constraints.append(x_i[i, j, s] >= (x_e[i, s] + x_f[j, s] - 1.5))
                    constraints.append(x_i[i, j, s] <= (x_e[i, s] + x_f[j, s]) * 0.5)
                    constraints.append(x_e[i, s] >= x_i[i, j, s])
                    constraints.append(x_f[j, s] >= x_i[i, j, s])

    inter_loss = sum(
        (1 - sum(x_i[i, j, b] for b in range(len(splits)))) for i, e in enumerate(e_entities)
        for j, f in enumerate(f_entities) if (e, f) in inter
    )

    solve(inter_loss, constraints, max_sec, len(x_e) + len(x_f) + len(x_i))

    # report the found solution
    output = ([], dict(
        (e, names[s]) for s in range(len(splits)) for i, e in enumerate(e_entities) if x_e[i, s].value > 0.1
    ), dict(
        (f, names[s]) for s in range(len(splits)) for j, f in enumerate(f_entities) if x_f[j, s].value > 0.1
    ))
    for i, e in enumerate(e_entities):
        for j, f in enumerate(f_entities):
            if (e, f) in inter:
                for s in range(len(splits)):
                    if x_i[i, j, s].value > 0:
                        output[0].append((e, f, names[s]))
                if sum(x_i[i, j, b].value for b in range(len(splits))) == 0:
                    output[0].append((e, f, "not selected"))

    return output
