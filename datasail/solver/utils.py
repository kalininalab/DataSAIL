import logging
from typing import List, Tuple, Union

import cvxpy
import mosek
import numpy as np


def estimate_number_target_interactions(
        inter: Union[np.ndarray, List[Tuple[str, str]]],
        num_e_data: int,
        num_f_data: int,
        splits: List[float]
) -> float:
    """
    Estimate number of interactions in the final dataset to use them to compute size-bounds for splits.

    Args:
        inter: Interactions as list of pairwise interactions or as numpy array counting interactions of clusters
        num_e_data: Number of entities in e-dataset
        num_f_data: Number of entities in f-dataset
        splits: List of splits given by their relative size

    Returns:
        Estimated number of interactions in the final dataset
    """
    if isinstance(inter, np.ndarray):
        return int(np.sum(inter))
    else:
        return len(inter)
    # return estimate_number_target_interactions(inter)


def estimate_surviving_interactions(num_inter: int, num_e: int, num_f: int, splits: List[float]) -> int:
    """
    Estimate number of interactions in the final dataset based on how many interactions would be lost on a
    fully-connected dataset and how sparse the current dataset is.

    Notes:
        Not working for perfectly separable dataset (Too many interactions are estimated to be lost)

    Args:
        num_inter: Number of interactions
        num_e: number of entities in e-dataset
        num_f: number of entities in f-dataset
        splits: List of splits given by their relative size

    Returns:
        Estimated number of interactions in the final dataset
    """
    sparsity = num_inter / (num_e * num_f)
    dense_survivors = sum(s ** 2 for s in splits) * num_e * num_f
    return int(dense_survivors * sparsity + 0.5)


def inter_mask(e_entities: List[str], f_entities: List[str], inter: List[Tuple[str, str]]) -> np.ndarray:
    """
    Compute an interaction mask, i.e. an adjacency matrix from the list of interactions.

    Args:
        e_entities: Entities in e-dataset
        f_entities: Entities in f-dataset
        inter: List of interactions between entities in e-dataset and entities in f-dataset

    Returns:
        Adjacency matrix based on the list of interactions
    """
    # TODO: Figure out which of these methods is faster and for which degree of density
    return inter_mask_dense(e_entities, f_entities, inter) if len(inter) / (len(e_entities) + len(f_entities)) \
        else inter_mask_sparse(e_entities, f_entities, inter)


def inter_mask_dense(e_entities, f_entities, inter):
    """
    Compute adjacency matrix by setting every single value to 1 if there is an interaction accordingly.

    Notes:
        Supposedly fast for sparse matrices, but slow for dense ones

    Args:
        e_entities: Entities in e-dataset
        f_entities: Entities in f-dataset
        inter: List of interactions between entities in e-dataset and entities in f-dataset

    Returns:
        Adjacency matrix based on the list of interactions
    """
    output = np.zeros((len(e_entities), len(f_entities)))
    for i, e in enumerate(e_entities):
        for j, f in enumerate(f_entities):
            output[i, j] = (e, f) in inter
    return output


def inter_mask_sparse(e_entities, f_entities, inter):
    """
    Compute adjacency matrix by first compute mappings from entity names to their index and then setting the
    individual interactions to 1.

    Notes:
        Supposedly fast for dense matrices, but slow for sparse ones

    Args:
        e_entities: Entities in e-dataset
        f_entities: Entities in f-dataset
        inter: List of interactions between entities in e-dataset and entities in f-dataset

    Returns:
        Adjacency matrix based on the list of interactions
    """
    output = np.zeros((len(e_entities), len(f_entities)))
    d_map = dict((e, i) for i, e in enumerate(e_entities))
    p_map = dict((f, i) for i, f in enumerate(f_entities))
    for e, f in inter:
        output[d_map[e], p_map[f]] = 1
    return output


def solve(loss, constraints, max_sec, num_vars):
    """
    Minimize the loss function based on the constraints with the timelimit specified by max_sec.

    Args:
        loss: Loss function to minimize
        constraints: Constraints that have to hold
        max_sec: Maximal number of seconds to optimize the initial solution
        num_vars: Number of variables for statistics

    Returns:

    """
    logging.info("Start solving with SCIP")
    logging.info(f"The problem has {num_vars} variables and {len(constraints)} constraints.")

    problem = cvxpy.Problem(cvxpy.Minimize(loss), constraints)
    problem.solve(
        solver=cvxpy.MOSEK,
        qcp=True,
        # scip_params={
        #     "limits/time": max_sec,
        # },
        mosek_params={
            mosek.dparam.optimizer_max_time: max_sec
        },
        # verbose=True,
    )

    logging.info(f"SCIP status: {problem.status}")
    logging.info(f"Solution's score: {problem.value}")

    if "optimal" not in problem.status:
        logging.warning(
            'SCIP cannot solve the problem. Please consider relaxing split restrictions, '
            'e.g., less splits, or a higher tolerance level for exceeding cluster limits.'
        )
        return None
    return problem


def sample_categorical(
        inter: List[Tuple[str, str]],
        splits: List[float],
        names: List[str],
):
    """
    Sample interactions randomly into splits. This is the random split. It relies on the idea of categorical sampling.

    Args:
        inter: List of interactions to split
        splits: List of splits given by their relative size
        names: List of names given by their relative size

    Yields:
        Chunks of interactions in order of the splits
    """
    np.random.shuffle(inter)

    def gen():
        for index in range(len(splits) - 1):
            yield inter[int(sum(splits[:index]) * len(inter)):int(sum(splits[:(index + 1)]) * len(inter))]
        yield inter[int(sum(splits[:-1]) * len(inter)):]

    output = []
    for i, split in enumerate(gen()):
        output += [(d, p, names[i]) for d, p in split]
    return output
