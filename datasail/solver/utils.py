import functools
import logging
import operator
import sys

from typing import List, Optional, Union, Tuple, Collection, Dict, Callable

import cvxpy
from cvxpy import Variable
from cvxpy.constraints.constraint import Constraint
import numpy as np

from datasail.settings import LOGGER, SOLVER_CPLEX, SOLVER_GLPK, SOLVER_XPRESS, SOLVER_SCIP, SOLVER_MOSEK, \
    SOLVER_GUROBI, SOLVERS, NOT_ASSIGNED


def compute_limits(epsilon: float, total: int, splits: List[float]) -> List[float]:
    """
    Compute the lower and upper limits for the splits based on the total number of interactions and the epsilon.

    Args:
        epsilon: epsilon to use
        total: total number of interactions
        splits: list of splits

    Returns:
        lower and upper limits for the splits
    """
    return [int((split - epsilon) * total) for split in splits]


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


def generate_baseline(
        splits: List[float],
        weights: Union[np.ndarray, List[float]],
        similarities: Optional[np.ndarray],
        distances: Optional[np.ndarray],
) -> float:
    """
    Generate a baseline solution for the double-cold splitting problem.

    Args:
        splits: List of relative sizes of the splits
        weights: List of weights of the entities
        similarities: Pairwise similarity matrix of entities in the order of their names
        distances: Pairwise distance matrix of entities in the order of their names

    Returns:
        The amount of information leakage in a random double-cold splitting
    """
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

    return float(np.sum(leak_matrix))


def interaction_contraints(
        e_entities: List[str],
        f_entities: List[str],
        x_i: Dict[Tuple[str, str], Variable],
        constraints: List,
        splits: List[float],
        x_e: Variable,
        x_f: Variable,
        min_lim: List[float],
        weighting: Callable,
        is_valid: Callable
) -> None:
    """
    Generate constraints for the interactions in the cluster-based double-cold splitting.

    Args:
        e_entities: List of names of the entities in the e-dataset
        f_entities: List of names of the entities in the f-dataset
        x_i: Optimization variables for the interactions
        constraints: List of constraints
        splits: List of splits
        x_e: Optimization variables for the e-dataset
        x_f: Optimization variables for the f-dataset
        min_lim: List of lower limits for the split sizes
        weighting: Function to compute the weight of an interaction
        is_valid: Function to check if an interaction is valid
    """
    for s, split in enumerate(splits):
        constraints.append(min_lim[s] <= cvxpy.sum([x_i[key][s] * weighting(key) for key in x_i]))
        for i in range(len(e_entities)):
            for j in range(len(f_entities)):
                index = is_valid(i, j)
                if index is not None:
                    constraints.append(x_i[index][s] >= cvxpy.maximum(x_e[s][i] + x_f[s][j] - 1, 0))
                    constraints.append(x_i[index][s] <= 0.75 * (x_e[s][i] + x_f[s][j]))


def cluster_y_constraints(
        uniform: bool,
        clusters: List[str],
        y: List[List[Variable]],
        x: Variable,
        splits: List[float],
) -> List[Constraint]:
    """
    Generate constraints for the helper variables y in the cluster-based double-cold splitting.

    Args:
        uniform: Boolean flag if the cluster metric is uniform
        clusters: List of cluster names
        y: List of helper variables
        x: Optimization variables
        splits: List of splits

    Returns:
        List of constraints for the helper variables y
    """
    if uniform:
        return []
    return [y[c1][c2] >= cvxpy.max(cvxpy.vstack([x[s, c1] - x[s, c2] for s in range(len(splits))]))
            for c1 in range(len(clusters)) for c2 in range(c1)]


def collect_results_2d(
        problem: cvxpy.Problem,
        names: List[str],
        splits: List[float],
        e_entities: List[str],
        f_entities: List[str],
        x_e: Variable,
        x_f: Variable,
        x_i: Dict[Tuple[str, str], Variable],
        is_valid: Callable,
) -> Optional[Tuple[Dict[Tuple[str, str], str], Dict[object, str], Dict[object, str]]]:
    """
    Report the found solution for two-dimensional splits.

    Args:
        problem: Problem object after solving.
        names: List of names of the splits.
        splits: List of the relative sizes of the splits.
        e_entities: List of names of entities in the e-dataset.
        f_entities: List of names of entities in the f-dataset.
        x_e: Optimization variables for the e-dataset.
        x_f: Optimization variables for the f-dataset.
        x_i: Optimization variables for the interactions.
        is_valid: Function to check if an interaction is valid.

    Returns:
        A list of interactions and their assignment to a split and two mappings from entities to splits, one for each
    """
    if problem is None:
        return None

    # report the found solution
    output = (
        {},
        {e: names[s] for s in range(len(splits)) for i, e in enumerate(e_entities) if x_e[s, i].value > 0.1},
        {f: names[s] for s in range(len(splits)) for j, f in enumerate(f_entities) if x_f[s, j].value > 0.1},
    )
    for i, e in enumerate(e_entities):
        for j, f in enumerate(f_entities):
            index = is_valid(i, j)
            if index is not None:
                for b in range(len(splits)):
                    if x_i[index][b].value > 0:
                        output[0][e, f] = names[b]
                if sum(x_i[index][b].value for b in range(len(splits))) == 0:
                    output[0][e, f] = NOT_ASSIGNED

    return output


def leakage_loss(
        uniform: bool,
        intra_weights,
        y,
        clusters,
        similarities
):
    """
    Compute the leakage loss for the cluster-based double-cold splitting.

    Args:
        uniform: Boolean flag if the cluster metric is uniform
        intra_weights: Weights of the intra-cluster edges
        y: Helper variables
        clusters: List of cluster names
        similarities: Pairwise similarity matrix of clusters in the order of their names

    Returns:
        Loss describing the leakage between clusters
    """
    if uniform:
        return 0
    else:
        tmp = [intra_weights[c1, c2] * y[c1][c2] for c1 in range(len(clusters)) for c2 in range(c1)]
        e_loss = cvxpy.sum(tmp)
        if similarities is None:
            return -e_loss
