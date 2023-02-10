import logging
from typing import Optional, Tuple, List, Set, Dict

from datasail.solver.scalar.utils import init_variables, sum_constraint, interaction_constraints
from datasail.solver.utils import solve, init_inter_variables_id_scalar, estimate_number_target_interactions


def solve_icd_bqp(
        e_entities: List[object],
        f_entities: List[object],
        inter: Set[Tuple[str, str]],
        limit: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
) -> Optional[Tuple[List[Tuple[str, str, str]], Dict[str, str], Dict[str, str]]]:
    logging.info("Define optimization problem")

    all_inter = estimate_number_target_interactions(inter, len(e_entities), len(f_entities), splits)

    x_e = init_variables(len(splits), len(e_entities))
    x_f = init_variables(len(splits), len(f_entities))
    x_i = init_inter_variables_id_scalar(len(splits), e_entities, f_entities, inter)

    constraints = sum_constraint(e_entities, x_e, splits) + sum_constraint(f_entities, x_f, splits)

    for i, e in enumerate(e_entities):
        for j, f in enumerate(f_entities):
            if (e, f) in inter:
                constraints.append(sum(x_i[i, j, s] for s in range(len(splits))) <= 1)

    for s in range(len(splits)):
        var = sum(
            x_i[i, j, s] for i, drug in enumerate(e_entities) for j, protein in enumerate(f_entities) if
            (drug, protein) in inter
        )
        constraints += [
            splits[s] * all_inter * (1 - limit) <= var,
            var <= splits[s] * all_inter * (1 + limit),
        ] + interaction_constraints(e_entities, f_entities, inter, x_e, x_f, x_i, s)

    inter_loss = sum(
        (1 - sum(x_i[i, j, b] for b in range(len(splits)))) for i, drug in enumerate(e_entities)
        for j, protein in enumerate(f_entities) if (drug, protein) in inter
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
