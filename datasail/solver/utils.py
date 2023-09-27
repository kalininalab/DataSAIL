import functools
import logging
import operator
import sys

from typing import List, Optional, Union, Tuple, Set, Collection, Dict

import cvxpy
from cvxpy import Variable, Expression
from cvxpy.constraints.constraint import Constraint
import numpy as np

from datasail.settings import LOGGER, SOLVER_CPLEX, SOLVER_XPRESS, SOLVER_SCIP, SOLVER_MOSEK, \
    SOLVER_GUROBI, SOLVERS


def compute_limits(epsilon: float, total: int, splits: List[float]) -> Tuple[List[float], List[float]]:
    """
    Compute the lower and upper limits for the splits based on the total number of interactions and the epsilon.

    Args:
        epsilon: epsilon to use
        total: total number of interactions
        splits: list of splits

    Returns:
        lower and upper limits for the splits
    """
    return [int(split * (1 + epsilon) * total) for split in splits], \
        [int(split * (1 - epsilon) * total) for split in splits]


def inter_mask(
        e_entities: List[str],
        f_entities: List[str],
        inter: Collection[Tuple[str, str]],
) -> np.ndarray:
    """
    Compute an interaction mask, i.e. an adjacency matrix from the list of interactions.
    Compute adjacency matrix by first compute mappings from entity names to their index and then setting the
    individual interactions to 1.

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
        """
        Initialize this redirection module to be used to pipe messages to stdout to some file.
        Args:
            logfile_name: Filename to write stdout logs to instead of the console
        """
        if logfile_name is None:
            self.silent = True
            return
        self.file_handler = logging.FileHandler(logfile_name)
        self.old_stdout = sys.stdout
        self.disabled = {}
        self.silent = False

    def __enter__(self):
        """
        Remove the stream from all loggers that print to stdout.
        """
        if self.silent:
            return
        for name, logger in logging.root.manager.loggerDict.items():
            if isinstance(logger, logging.Logger) and len(logger.handlers) > 0:
                for handler in logger.handlers:
                    if not hasattr(handler, "stream"):
                        continue
                    if handler.stream.name == "<stdout>":
                        if name not in self.disabled:
                            self.disabled[name] = []
                        self.disabled[name].append(handler)
                if name in self.disabled:
                    for handler in self.disabled[name]:
                        logger.removeHandler(handler)
                logger.addHandler(self.file_handler)
        sys.stdout = self.file_handler.stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Re-instantiate all loggers with their streams.

        Args:
            exc_type: ignored
            exc_val: ignored
            exc_tb: ignored
        """
        if self.silent:
            return
        for name, handlers in self.disabled.items():
            logger = logging.root.manager.loggerDict[name]
            logger.removeHandler(self.file_handler)
            for handler in handlers:
                logger.addHandler(handler)
        sys.stdout = self.old_stdout


def solve(loss, constraints: List, max_sec: int, solver: str, log_file: str) -> Optional[cvxpy.Problem]:
    """
    Minimize the loss function based on the constraints with the timelimit specified by max_sec.

    Args:
        loss: Loss function to minimize
        constraints: Constraints that have to hold
        max_sec: Maximal number of seconds to optimize the initial solution
        solver: Solving algorithm to use to solve the formulated program
        log_file: File to store the detailed log from the solver to

    Returns:
        The problem object after solving. None if the problem could not be solved.
    """
    problem = cvxpy.Problem(cvxpy.Minimize(loss), constraints)
    LOGGER.info(f"Start solving with {solver}")
    LOGGER.info(
        f"The problem has {sum([functools.reduce(operator.mul, v.shape, 1) for v in problem.variables()])} variables "
        f"and {sum([functools.reduce(operator.mul, c.shape, 1) for c in problem.constraints])} constraints.")

    # if solver == SOLVER_GLPK:
    #     kwargs = {"glpk_mi_params": {"tm_lim": max_sec}}
    if solver == SOLVER_SCIP:
        kwargs = {"scip_params": {"limits/time": max_sec}}
    elif solver == SOLVER_CPLEX:
        kwargs = {"cplex_params": {}}
    elif solver == SOLVER_GUROBI:
        kwargs = {"gurobi_params": {}}
    elif solver == SOLVER_MOSEK:
        kwargs = {"mosek_params": {
            "MSK_DPAR_OPTIMIZER_MAX_TIME": max_sec,
            "MSK_IPAR_NUM_THREADS": 14,
        }}
    elif solver == SOLVER_XPRESS:
        kwargs = {"xpress_params": {}}
    else:
        raise ValueError("Unknown solver error")
    with LoggerRedirect(log_file):
        try:
            problem.solve(
                solver=SOLVERS[solver],
                qcp=True,
                verbose=True,
                **kwargs,
            )

            LOGGER.info(f"{solver} status: {problem.status}")
            LOGGER.info(f"Solution's score: {problem.value}")

            if "optimal" not in problem.status:
                LOGGER.warning(
                    f'{solver} cannot solve the problem. Please consider relaxing split restrictions, '
                    'e.g., less splits, or a higher tolerance level for exceeding cluster limits.'
                )
                return None
            return problem
        except KeyError:
            LOGGER.warning(f"Solving failed for {''}. Please use try another solver or update your python version.")
            return None


def sample_categorical(
        inter: List[Tuple[str, str]],
        splits: List[float],
        names: List[str],
) -> Dict[Tuple[str, str], str]:
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

    output = {}
    for i, split in enumerate(gen()):
        output.update({(d, p): names[i] for d, p in split})
    return output


def interaction_constraints(
        e_data: List[str],
        f_data: List[str],
        inter: Union[Set[Tuple[str, str]], np.ndarray],
        x_e: List[Variable],
        x_f: List[Variable],
        x_i: List[Variable],
        s: int
) -> List[Constraint]:
    """
    Define the constraints that two clusters are in the same split iff their interaction (if exists) is in that split.

    Args:
        e_data: Names of datapoints in the e-dataset
        f_data: Names of datapoints in the f-dataset
        inter: a set of interactions between pairs of entities
        x_e: List of variables for the e-dataset
        x_f: List of variables for the f-dataset
        x_i: List of variables for the interactions
        s: Current split to consider

    Returns:
        A list of cvxpy constraints
    """
    constraints = []
    for i, e1 in enumerate(e_data):
        for j, e2 in enumerate(f_data):
            if isinstance(inter, np.ndarray) or (e1, e2) in inter:
                # constraints.append(x_i[s][i, j] >= (x_e[s][:, 0][i] + x_f[s][:, 0][j] - 1.5))
                # constraints.append(x_i[s][i, j] <= (x_e[s][:, 0][i] + x_f[s][:, 0][j]) * 0.5)
                # constraints.append(x_e[s][:, 0][i] >= x_i[s][i, j])
                # constraints.append(x_f[s][:, 0][j] >= x_i[s][i, j])
                constraints.append(x_i[s][i, j] >= cvxpy.maximum(x_e[s][:, 0][i] + x_f[s][:, 0][j] - 1, 0))
                constraints.append(x_i[s][i, j] <= 0.75 * (x_e[s][:, 0][i] + x_f[s][:, 0][j]))
    return constraints


def cluster_sim_dist_constraint(
        similarities: Optional[np.ndarray],
        distances: Optional[np.ndarray],
        threshold: np.ndarray,
        ones: np.ndarray,
        x: List[Variable],
        s: int
) -> List[Constraint]:
    """
    Define the constraints on similarities between samples in difference splits or distances of samples in the same
    split.

    Args:
        similarities: Similarity matrix of the data
        distances: Distance matrix of the data
        threshold: Threshold to apply
        ones: Vector to help in the computations
        x: List of variables for the dataset
        s: Split to consider

    Returns:
        A list of cvxpy constraints
    """
    if distances is not None:
        return cvxpy.multiply(
            cvxpy.maximum((x[s] @ ones) + cvxpy.transpose(x[s] @ ones) - (ones.T @ ones), 0), distances
        ) <= threshold
    return cvxpy.multiply(((x[s] @ ones) - cvxpy.transpose(x[s] @ ones)) ** 2, similarities) <= threshold


def generate_baseline(
        splits: List[float],
        weights: Union[np.ndarray, List[float]],
        similarities: Optional[np.ndarray],
        distances: Optional[np.ndarray],
):
    indices = sorted(list(range(len(weights))), key=lambda i: -weights[i])
    max_sizes = np.array(splits) * sum(weights)
    sizes = [0] * len(splits)
    assignments = [-1] * len(weights)
    oh_val, oh_idx = float("inf"), -1
    for idx in indices:
        for s in range(len(splits)):
            if sizes[s] + weights[idx] <= max_sizes[s]:
                assignments[idx] = s
                sizes[s] += weights[idx]
                break
            elif (sizes[s] + weights[idx]) / max_sizes[s] < oh_val:
                oh_val = (sizes[s] + weights[idx]) / max_sizes[s]
                oh_idx = s
        if assignments[idx] == -1:
            assignments[idx] = oh_idx
            sizes[oh_idx] += weights[idx]
    x = np.zeros((len(assignments), max(assignments) + 1))
    x[np.arange(len(assignments)), assignments] = 1
    ones = np.ones((1, len(weights)))

    if distances is not None:
        hit_matrix = np.sum([np.maximum(
            (np.expand_dims(x[:, s], axis=1) @ ones) + (np.expand_dims(x[:, s], axis=1) @ ones).T - (ones.T @ ones), 0)
                             for s in range(len(splits))], axis=0)
        leak_matrix = np.multiply(hit_matrix, distances)
    else:
        hit_matrix = np.sum(
            [((np.expand_dims(x[:, s], axis=1) @ ones) - (np.expand_dims(x[:, s], axis=1) @ ones).T) ** 2 for s in
             range(len(splits))], axis=0) / (len(splits) - 1)
        leak_matrix = np.multiply(hit_matrix, similarities)

    return np.sum(leak_matrix)


def cluster_sim_dist_objective(
        similarities: Optional[np.ndarray],
        distances: Optional[np.ndarray],
        ones: np.ndarray,
        weights: Union[np.ndarray, List[float]],
        x: List[Variable],
        splits: List[float]
) -> Expression:
    """
    Construct an objective function of the variables based on a similarity or distance matrix.

    Args:
        similarities: Similarity matrix of the dataset
        distances: Distance matrix of the dataset
        ones: Vector to help in the computations
        weights: weights of the entities
        x: Dictionary of indices and variables for the e-dataset
        splits: Splits as list of their relative size

    Returns:
        An objective function to minimize
    """
    if isinstance(weights, List):
        weights = np.array(weights)

    baseline = generate_baseline(splits, weights, similarities, distances)

    weight_matrix = weights.T @ weights

    if distances is not None:
        hit_matrix = cvxpy.sum(
            [cvxpy.maximum((x[s] @ ones) + cvxpy.transpose(x[s] @ ones) - (ones.T @ ones), 0) for s in
             range(len(splits))])
        leak_matrix = cvxpy.multiply(hit_matrix, distances)
    else:
        hit_matrix = cvxpy.sum([((x[s] @ ones) - cvxpy.transpose(x[s] @ ones)) ** 2 for s in range(len(splits))]) / (
                    len(splits) - 1)
        leak_matrix = cvxpy.multiply(hit_matrix, similarities)

    leak_matrix = cvxpy.multiply(leak_matrix, weight_matrix)
    # leakage = cvxpy.sum(leak_matrix) / cvxpy.sum(cvxpy.multiply(hit_matrix, weight_matrix))  # accurate computation
    return cvxpy.sum(leak_matrix) / baseline
