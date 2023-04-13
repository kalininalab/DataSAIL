import logging
import math
import sys
from typing import List, Tuple, Collection

import cvxpy
import numpy as np

from datasail.settings import LOGGER


def inter_mask(
        e_entities: List[str],
        f_entities: List[str],
        inter: Collection[Tuple[str, str]],
) -> np.ndarray:
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


def inter_mask_dense(e_entities: List[str], f_entities: List[str], inter: Collection[Tuple[str, str]]):
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


def inter_mask_sparse(e_entities: List[str], f_entities: List[str], inter: Collection[Tuple[str, str]]):
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


class LoggerRedirect:
    def __init__(self, logfile_name):
        self.file_handler = logging.FileHandler(logfile_name)
        self.old_stdout = sys.stdout
        self.disabled = {}

    def __enter__(self):
        for name, logger in logging.root.manager.loggerDict.items():
            if isinstance(logger, logging.Logger) and len(logger.handlers) > 0:
                for handler in logger.handlers:
                    if handler.stream.name == "<stdout>":
                        if name not in self.disabled:
                            self.disabled[name] = []
                        self.disabled[name].append(handler)
                for handler in self.disabled[name]:
                    logger.removeHandler(handler)
                logger.addHandler(self.file_handler)
        sys.stdout = self.file_handler.stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, handlers in self.disabled.items():
            logger = logging.root.manager.loggerDict[name]
            logger.removeHandler(self.file_handler)
            for handler in handlers:
                logger.addHandler(handler)
        sys.stdout = self.old_stdout


def solve(loss, constraints: List, max_sec: int, solver: str, log_file: str):
    """
    Minimize the loss function based on the constraints with the timelimit specified by max_sec.

    Args:
        loss: Loss function to minimize
        constraints: Constraints that have to hold
        max_sec: Maximal number of seconds to optimize the initial solution
        solver: Solving algorithm to use to solve the formulated program
        log_file: File to store the detailed log from the solver to

    Returns:

    """
    problem = cvxpy.Problem(cvxpy.Minimize(loss), constraints)
    LOGGER.info(f"Start solving with {solver}")
    LOGGER.info(f"The problem has {sum([math.prod(v.shape) for v in problem.variables()])} variables "
                f"and {sum([math.prod(c.shape) for c in problem.constraints])} constraints.")

    if solver == "MOSEK":
        solve_algo = cvxpy.MOSEK
        kwargs = {"mosek_params": {"MSK_DPAR_OPTIMIZER_MAX_TIME": max_sec}}
    else:
        solve_algo = cvxpy.SCIP
        kwargs = {"scip_params": {"limits/time": max_sec}}
    with LoggerRedirect(log_file):
        print("std-out captured?")
        LOGGER.info("INFO captured?")
        problem.solve(
            solver=solve_algo,
            qcp=True,
            verbose=True,
            **kwargs,
        )

    LOGGER.info(f"{solver} status: {problem.status}")
    LOGGER.info(f"Solution's score: {problem.value}")

    if "optimal" not in problem.status:
        LOGGER.warning(
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
