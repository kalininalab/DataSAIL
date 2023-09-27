import os
from typing import Tuple, Optional, List, Dict, Union

import numpy as np
from cvxpy import SolverError

from datasail.cluster.clustering import reverse_clustering, cluster_interactions, reverse_interaction_clustering
from datasail.reader.utils import DataSet, DictMap
from datasail.settings import LOGGER, MODE_F, TEC_R, TEC_ICS, TEC_CCS, TEC_ICD, TEC_CCD, MMSEQS, CDHIT, MMSEQS2
from datasail.solver.blp.id_cold_single import solve_ics_blp as solve_ics_bqp
from datasail.solver.blp.id_cold_double import solve_icd_blp as solve_icd_bqp
from datasail.solver.blp.cluster_cold_single import solve_ccs_blp as solve_ccs_bqp
from datasail.solver.blp.cluster_cold_double import solve_ccd_blp as solve_ccd_bqp
from datasail.solver.utils import sample_categorical


def insert(dictionary: dict, key: str, value):
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


def run_solver(
        techniques: List[str],
        e_dataset: DataSet,
        f_dataset: DataSet,
        inter: Optional[Union[np.ndarray, List[Tuple[str, str]]]],
        epsilon: float,
        runs: int,
        splits: List[float],
        split_names: List[str],
        max_sec: int,
        max_sol: int,
        solver: str,
        log_dir: str,
) -> Tuple[Dict[str, List[Dict[Tuple[str, str], str]]], DictMap, DictMap, DictMap, DictMap]:
    """
    Run a solver based on the selected technique.

    Args:
        techniques: List of techniques to use to split the dataset
        e_dataset: First dataset
        f_dataset: Second dataset
        inter: Interactions of elements or clusters of the two datasets
        epsilon: Additive bound for exceeding the requested split size
        runs:
        splits: List of split sizes
        split_names: List of names of the splits in the order of the splits argument
        max_sec: Maximal number of seconds to take when optimizing the problem (not for finding an initial solution)
        max_sol: Maximal number of solution to consider
        solver: Solving algorithm to use to solve the formulated program
        log_dir: path to folder to store log files in

    Returns:
        A list of interactions and their assignment to a split and two mappings from entities to splits, one for each
        dataset
    """
    output_inter, output_e_entities, output_f_entities, output_e_clusters, output_f_clusters = \
        dict(), dict(), dict(), dict(), dict()

    LOGGER.info("Define optimization problem")

    for run in range(runs):
        if run > 0:
            e_dataset.shuffle()
            f_dataset.shuffle()
        for technique in techniques:
            try:
                LOGGER.info(technique)
                mode = technique[-1]
                dataset = f_dataset if mode == MODE_F else e_dataset
                log_file = None if log_dir is None else os.path.join(log_dir, f"{dataset.get_name()}_{technique}.log")

                if technique == TEC_R:
                    solution = sample_categorical(
                        inter=inter,
                        splits=splits,
                        names=split_names,
                    )
                    insert(output_inter, technique, solution)
                elif technique[:3] == TEC_ICS or (technique[:3] == TEC_CCS and isinstance(dataset.similarity, str) and
                                                  dataset.similarity.lower() in [CDHIT, MMSEQS, MMSEQS2]):
                    if technique[:3] == TEC_CCS and (isinstance(dataset.similarity, str) and
                                                     dataset.similarity.lower() in [CDHIT, MMSEQS, MMSEQS2]):
                        names = dataset.cluster_names
                        weights = [dataset.cluster_weights.get(x, 0) for x in dataset.cluster_names]
                    else:
                        names = dataset.names
                        weights = [dataset.weights.get(x, 0) for x in dataset.names]

                    solution = solve_ics_bqp(
                        entities=names,
                        weights=weights,
                        epsilon=epsilon,
                        splits=splits,
                        names=split_names,
                        max_sec=max_sec,
                        max_sol=max_sol,
                        solver=solver,
                        log_file=log_file,
                    )

                    if solution is not None:
                        if technique[:3] == TEC_CCS \
                                and isinstance(dataset.similarity, str) \
                                and dataset.similarity.lower() in [CDHIT, MMSEQS, MMSEQS2]:
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
                elif technique[:3] == TEC_ICD:
                    solution = solve_icd_bqp(
                        e_entities=e_dataset.names,
                        f_entities=f_dataset.names,
                        inter=set(inter),
                        epsilon=epsilon,
                        splits=splits,
                        names=split_names,
                        max_sec=max_sec,
                        max_sol=max_sol,
                        solver=solver,
                        log_file=log_file,
                    )
                    if solution is not None:
                        insert(output_inter, technique, solution[0])
                        insert(output_e_entities, technique, solution[1])
                        insert(output_f_entities, technique, solution[2])
                        # output_inter[technique], output_e_entities[technique], output_f_entities[technique] = solution
                elif technique[:3] == TEC_CCS:
                    cluster_split = solve_ccs_bqp(
                        clusters=dataset.cluster_names,
                        weights=[dataset.cluster_weights.get(c, 0) for c in dataset.cluster_names],
                        similarities=dataset.cluster_similarity,
                        distances=dataset.cluster_distance,
                        threshold=dataset.threshold,
                        epsilon=epsilon,
                        splits=splits,
                        names=split_names,
                        max_sec=max_sec,
                        max_sol=max_sol,
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
                elif technique[:3] == TEC_CCD:
                    cluster_inter = cluster_interactions(
                        inter,
                        e_dataset.cluster_map,
                        e_dataset.cluster_names,
                        f_dataset.cluster_map,
                        f_dataset.cluster_names,
                    )
                    cluster_split = solve_ccd_bqp(
                        e_clusters=e_dataset.cluster_names,
                        e_weights=[e_dataset.cluster_weights.get(c, 0) for c in e_dataset.cluster_names],
                        e_similarities=e_dataset.cluster_similarity,
                        e_distances=e_dataset.cluster_distance,
                        e_threshold=e_dataset.threshold,
                        f_clusters=f_dataset.cluster_names,
                        f_weights=[f_dataset.cluster_weights.get(c, 0) for c in f_dataset.cluster_names],
                        f_similarities=f_dataset.cluster_similarity,
                        f_distances=f_dataset.cluster_distance,
                        f_threshold=f_dataset.threshold,
                        inter=cluster_inter,
                        epsilon=epsilon,
                        splits=splits,
                        names=split_names,
                        max_sec=max_sec,
                        max_sol=max_sol,
                        solver=solver,
                        log_file=log_file,
                    )

                    if cluster_split is not None:
                        insert(output_e_clusters, technique, cluster_split[1])
                        insert(output_f_clusters, technique, cluster_split[2])
                        # output_inter[technique], output_e_clusters[technique], output_f_clusters[technique] = cluster_split
                        insert(output_inter, technique, reverse_interaction_clustering(
                            cluster_split[0],
                            e_dataset.cluster_map,
                            f_dataset.cluster_map,
                            inter,
                        ))
                        insert(output_e_entities, technique,
                               reverse_clustering(cluster_split[1], e_dataset.cluster_map))
                        insert(output_f_entities, technique,
                               reverse_clustering(cluster_split[2], f_dataset.cluster_map))
            except SolverError:
                LOGGER.error(f"Splitting failed for {technique}, try to increase the timelimit or the epsilon value.")

    return output_inter, output_e_entities, output_f_entities, output_e_clusters, output_f_clusters
