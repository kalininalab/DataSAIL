import logging
from typing import List, Tuple

import cvxpy
import numpy as np


def estimate_number_target_interactions(inter, num_e_data, num_f_data, splits):
    if isinstance(inter, np.ndarray):
        return np.sum(inter)
    else:
        return len(inter)
    # return estimate_number_target_interactions(inter)


def estimate_surviving_interactions(num_inter: int, num_e: int, num_f: int, splits: List[float]) -> int:
    sparsity = num_inter / (num_e * num_f)
    dense_survivors = sum(s ** 2 for s in splits) * num_e * num_f
    return int(dense_survivors * sparsity + 0.5)


def inter_mask(e_entities, f_entities, inter):
    return inter_mask_dense(e_entities, f_entities, inter) if len(inter) / (len(e_entities) + len(f_entities)) \
        else inter_mask_sparse(e_entities, f_entities, inter)


def inter_mask_dense(e_entities, f_entities, inter):
    output = np.zeros((len(e_entities), len(f_entities)))
    for i, e in enumerate(e_entities):
        for j, f in enumerate(f_entities):
            output[i, j] = (e, f) in inter
    return output


def inter_mask_sparse(e_entities, f_entities, inter):
    output = np.zeros((len(e_entities), len(f_entities)))
    d_map = dict((e, i) for i, e in enumerate(e_entities))
    p_map = dict((f, i) for i, f in enumerate(f_entities))
    for e, f in inter:
        output[d_map[e], p_map[f]] = 1
    return output


def init_inter_variables_id_scalar(num_splits, e_entities, f_entities, inter):
    x = {}
    for s in range(num_splits):
        for i, e in enumerate(e_entities):
            for j, f in enumerate(f_entities):
                if (e, f) in inter:
                    x[i, j, s] = cvxpy.Variable(boolean=True)
    return x


def solve(loss, constraints, max_sec, num_vars):
    logging.info("Start solving with SCIP")
    logging.info(f"The problem has {num_vars} variables and {len(constraints)} constraints.")

    problem = cvxpy.Problem(cvxpy.Minimize(loss), constraints)
    problem.solve(
        solver=cvxpy.SCIP,
        qcp=True,
        scip_params={
            "limits/time": max_sec,
        },
    )

    logging.info(f"SCIP status: {problem.status}")
    logging.info(f"Solution's score: {problem.value}")

    if "optimal" not in problem.status:
        logging.warning(
            'SCIP cannot solve the problem. Please consider relaxing split restrictions, '
            'e.g., less splits, or a higher tolerance level for exceeding cluster limits.'
        )
        return None


def sample_categorical(
        data: List[Tuple[str, str]],
        splits: List[float],
        names: List[str],
):
    np.random.shuffle(data)

    def gen():
        for index in range(len(splits) - 1):
            yield data[int(sum(splits[:index]) * len(data)):int(sum(splits[:(index + 1)]) * len(data))]
        yield data[int(sum(splits[:-1]) * len(data)):]

    output = []
    for i, split in enumerate(gen()):
        output += [(d, p, names[i]) for d, p in split]
    return output