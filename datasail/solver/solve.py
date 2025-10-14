from pathlib import Path

import numpy as np
from cvxpy import SolverError

from datasail.cluster.clustering import reverse_clustering
from datasail.reader.utils import DataSet, DictMap
from datasail.settings import LOGGER, MODE_F, TEC_I1, TEC_C1, TEC_I2, TEC_C2, MMSEQS, CDHIT, MMSEQS2, MODE_E, MODE_F
from datasail.solver.id_1d import solve_i1
from datasail.solver.id_2d import solve_i2
from datasail.solver.cluster_1d import solve_c1
from datasail.solver.cluster_2d import solve_c2
from datasail.solver.utils import sample_categorical


def insert(dictionary: dict, key: str, value) -> None:
    """
    Append a value into a dictionary with the given key. If key is not in dictionary, create an empty list and append
    the value.

    Args:
        dictionary: Dict to insert in
        key: Key to insert at
        value: Value to insert
    """
    if key not in dictionary:
        dictionary[key] = []
    dictionary[key].append(value)


def random_inter_split(runs, inter, splits, split_names):
    return [sample_categorical(inter=inter, splits=splits, names=split_names) for _ in range(runs)]


def run_solver(
        techniques: list[str],
        e_dataset: DataSet,
        f_dataset: DataSet,
        delta: float,
        epsilon: float,
        runs: int,
        split_ratios: dict[str, list[float]],
        split_names: dict[str, list[str]],
        max_sec: int,
        solver: str,
        log_dir: Path,
) -> tuple[DictMap, DictMap, DictMap, DictMap]:
    """
    Run a solver based on the selected technique.

    Args:
        techniques: List of techniques to use to split the dataset
        e_dataset: First dataset
        f_dataset: Second dataset
        inter: Interactions of elements or clusters of the two datasets
        delta: Additive bound for stratification imbalance
        epsilon: Additive bound for exceeding the requested split size
        runs: Number of runs to perform
        split_ratios: List of split sizes
        split_names: List of names of the splits in the order of the splits argument
        max_sec: Maximal number of seconds to take when optimizing the problem (not for finding an initial solution)
        solver: Solving algorithm to use to solve the formulated program
        log_dir: path to folder to store log files in

    Returns:
        A list of interactions and their assignment to a split and two mappings from entities to splits, one for each
        dataset
    """
    output_e_entities, output_f_entities, output_e_clusters, output_f_clusters = {}, {}, {}, {}

    LOGGER.info("Define optimization problem")

    for run in range(runs):
        if run > 0:
            e_dataset.shuffle()
            f_dataset.shuffle()
        for technique in techniques:
            try:
                LOGGER.info(f"Splitting technique {technique} of run {run + 1} of {runs}")
                mode = technique[-1]
                dataset = f_dataset if mode == MODE_F else e_dataset
                log_file = None if log_dir is None else log_dir / f"{dataset.get_name()}_{technique}.log"
                
                if technique.startswith(TEC_I1) or \
                        (technique.startswith(TEC_C1) and isinstance(dataset.similarity, str) and
                         dataset.similarity.lower() in [CDHIT, MMSEQS, MMSEQS2]):
                    if technique.startswith(TEC_C1) and (isinstance(dataset.similarity, str) and
                                                         dataset.similarity.lower() in [CDHIT, MMSEQS, MMSEQS2]):
                        names = dataset.cluster_names
                        weights = [dataset.cluster_weights.get(x, 0) for x in dataset.cluster_names]
                        if dataset.stratification is None or len(dataset.classes) <= 1:
                            stratification = None
                        else:
                            stratification = np.stack([dataset.cluster_stratification.get(c, np.zeros(len(dataset.classes))) for c in dataset.cluster_names])
                    else:
                        names = dataset.names
                        weights = [dataset.weights.get(x, 0) for x in dataset.names]
                        if dataset.stratification is None or len(dataset.classes) <= 1:
                            stratification = None
                        else:
                            stratification = np.stack([dataset.stratification[n] for n in dataset.names])

                    solution = solve_i1(
                        entities=names,
                        weights=weights,
                        stratification=stratification,
                        delta=delta,
                        epsilon=epsilon,
                        splits=split_ratios[technique],
                        names=split_names[technique],
                        max_sec=max_sec,
                        solver=solver,
                        log_file=log_file,
                    )

                    if solution is not None:
                        if technique.startswith(TEC_C1) and \
                                isinstance(dataset.similarity, str) and \
                                dataset.similarity.lower() in [CDHIT, MMSEQS, MMSEQS2]:
                            if mode == MODE_F:
                                insert(output_f_clusters, technique, solution)
                                insert(output_f_entities, technique,
                                       reverse_clustering(solution, f_dataset.cluster_map))
                            else:
                                insert(output_e_clusters, technique, solution)
                                insert(output_e_entities, technique,
                                       reverse_clustering(solution, e_dataset.cluster_map))
                        else:
                            if mode == MODE_F:
                                insert(output_f_entities, technique, solution)
                            else:
                                insert(output_e_entities, technique, solution)
                elif technique.startswith(TEC_I2):
                    solution = solve_i2(
                        e_entities=e_dataset.names,
                        e_stratification=np.stack([
                            e_dataset.stratification.get(n, np.zeros(len(dataset.classes))) for n in e_dataset.names
                        ]) if e_dataset.stratification is not None and len(e_dataset.classes) > 1 else None,
                        e_weights=[e_dataset.weights.get(c, 0) for c in e_dataset.names],
                        e_splits=split_ratios[TEC_I1 + MODE_E],
                        e_names=split_names[TEC_I1 + MODE_E],
                        f_entities=f_dataset.names,
                        f_stratification=np.stack([
                            f_dataset.stratification.get(n, np.zeros(len(dataset.classes))) for n in f_dataset.names
                        ]) if f_dataset.stratification is not None and len(f_dataset.classes) > 1 else None,
                        f_weights=[f_dataset.weights.get(c, 0) for c in f_dataset.names],
                        f_splits=split_ratios[TEC_I1 + MODE_F],
                        f_names=split_names[TEC_I1 + MODE_F],
                        delta=delta,
                        epsilon=epsilon,
                        max_sec=max_sec,
                        solver=solver,
                        log_file=log_file,
                    )
                    if solution is not None:
                        insert(output_e_entities, technique, solution[0])
                        insert(output_f_entities, technique, solution[1])
                elif technique.startswith(TEC_C1):
                    cluster_split = solve_c1(
                        clusters=dataset.cluster_names,
                        weights=[dataset.cluster_weights.get(c, 0) for c in dataset.cluster_names],
                        s_matrix=np.stack([
                            dataset.cluster_stratification.get(c, np.zeros(len(dataset.classes)))
                            for c in dataset.cluster_names
                        ]) if dataset.cluster_stratification is not None and len(dataset.classes) > 1 else None,
                        similarities=dataset.cluster_similarity,
                        distances=dataset.cluster_distance,
                        delta=delta,
                        epsilon=epsilon,
                        splits=split_ratios[technique],
                        names=split_names[technique],
                        max_sec=max_sec,
                        solver=solver,
                        log_file=log_file,
                    )
                    if cluster_split is not None:
                        if mode == MODE_F:
                            insert(output_f_clusters, technique, cluster_split)
                            insert(output_f_entities, technique,
                                   reverse_clustering(cluster_split, f_dataset.cluster_map))
                        else:
                            insert(output_e_clusters, technique, cluster_split)
                            insert(output_e_entities, technique,
                                   reverse_clustering(cluster_split, e_dataset.cluster_map))
                elif technique.startswith(TEC_C2):
                    cluster_split = solve_c2(
                        e_clusters=e_dataset.cluster_names,
                        e_s_matrix=np.stack([
                            e_dataset.cluster_stratification.get(c, np.zeros(len(dataset.classes)))
                            for c in e_dataset.cluster_names
                        ]) if e_dataset.cluster_stratification is not None and len(e_dataset.classes) > 1 else None,
                        e_similarities=e_dataset.cluster_similarity,
                        e_distances=e_dataset.cluster_distance,
                        e_weights=np.array([e_dataset.cluster_weights.get(c, 1) for c in e_dataset.cluster_names]),
                        e_splits=split_ratios[TEC_C1 + MODE_E],
                        e_names=split_names[TEC_C1 + MODE_E],
                        f_clusters=f_dataset.cluster_names,
                        f_s_matrix=np.stack([
                            f_dataset.cluster_stratification.get(c, np.zeros(len(dataset.classes)))
                            for c in f_dataset.cluster_names
                        ]) if f_dataset.cluster_stratification is not None and len(f_dataset.classes) > 1 else None,
                        f_similarities=f_dataset.cluster_similarity,
                        f_distances=f_dataset.cluster_distance,
                        f_weights=np.array([f_dataset.cluster_weights.get(c, 1) for c in f_dataset.cluster_names]),
                        f_splits=split_ratios[TEC_C1 + MODE_F],
                        f_names=split_names[TEC_C1 + MODE_F],
                        delta=delta,
                        epsilon=epsilon,
                        max_sec=max_sec,
                        solver=solver,
                        log_file=log_file,
                    )

                    if cluster_split is not None:
                        insert(output_e_clusters, technique, cluster_split[0])
                        insert(output_f_clusters, technique, cluster_split[1])
                        insert(output_e_entities, technique,
                               reverse_clustering(cluster_split[0], e_dataset.cluster_map))
                        insert(output_f_entities, technique,
                               reverse_clustering(cluster_split[1], f_dataset.cluster_map))
            except SolverError:
                LOGGER.error(f"Splitting failed for {technique}, try to increase the timelimit or the epsilon value.")

    return output_e_entities, output_f_entities, output_e_clusters, output_f_clusters
