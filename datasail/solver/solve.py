import logging
import os
from typing import Tuple, Optional, List, Dict, Union, Set

import numpy as np
from cvxpy import SolverError

from datasail.cluster.clustering import reverse_clustering, cluster_interactions
from datasail.reader.utils import DataSet
from datasail.solver.scalar.id_cold_single import solve_ics_bqp as solve_ics_bqp_scalar
from datasail.solver.vector.id_cold_single import solve_ics_bqp as solve_ics_bqp_vector
from datasail.solver.scalar.id_cold_double import solve_icd_bqp as solve_icd_bqp_scalar
from datasail.solver.vector.id_cold_double import solve_icd_bqp as solve_icd_bqp_vector
from datasail.solver.scalar.cluster_cold_single import solve_ccs_bqp as solve_ccs_bqp_scalar
from datasail.solver.vector.cluster_cold_single import solve_ccs_bqp as solve_ccs_bqp_vector
from datasail.solver.scalar.cluster_cold_double import solve_ccd_bqp as solve_ccd_bqp_scalar
from datasail.solver.vector.cluster_cold_double import solve_ccd_bqp as solve_ccd_bqp_vector
from datasail.solver.utils import sample_categorical


DictMap = Dict[str, Dict[str, str]]


def run_solver(
        techniques: List[str],
        e_dataset: DataSet,
        f_dataset: DataSet,
        inter: Optional[Union[np.ndarray, List[Tuple[str, str]]]],
        vectorized: bool,
        epsilon: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
        solver: str,
        log_dir: str,
) -> Tuple[Dict[str, List[Tuple[str, str, str]]], DictMap, DictMap, DictMap, DictMap]:
    """
    Run a solver based on the selected technique.

    Args:
        techniques: List of techniques to use to split the dataset
        e_dataset: First dataset
        f_dataset: Second dataset
        inter: Interactions of elements or clusters of the two datasets
        vectorized: Boolean flag indicating to run it in vectorized form
        epsilon: Additive bound for exceeding the requested split size
        splits: List of split sizes
        names: List of names of the splits in the order of the splits argument
        max_sec: Maximal number of seconds to take when optimizing the problem (not for finding an initial solution)
        max_sol: Maximal number of solution to consider
        solver: Solving algorithm to use to solve the formulated program
        log_dir: path to folder to store log files in

    Returns:
        A list of interactions and their assignment to a split and two mappings from entities to splits, one for each
        dataset
    """
    output_inter, output_e_entities, output_f_entities, output_e_clusters, output_f_clusters = dict(), dict(), dict(), dict(), dict()

    logging.info("Define optimization problem")

    for technique in techniques:
        try:
            logging.info(technique)
            technique, mode = technique[:3], technique[-1]
            dataset = f_dataset if mode == "f" else e_dataset
            log_file = os.path.join(log_dir, f"{dataset.get_name()}_{technique}{mode}.log")

            if technique == "R":
                solution = sample_categorical(
                    inter=inter,
                    splits=splits,
                    names=names,
                )
                output_inter[technique] = solution
            elif technique == "ICS" or (technique == "CCS" and isinstance(dataset.similarity, str) and dataset.similarity.lower() in ["cdhit", "mmseqs"]):
                if vectorized:
                    fun = solve_ics_bqp_vector
                else:
                    fun = solve_ics_bqp_scalar

                solution = fun(
                    e_entities=dataset.names,
                    e_weights=[dataset.weights.get(x, 0) for x in dataset.names],
                    epsilon=epsilon,
                    splits=splits,
                    names=names,
                    max_sec=max_sec,
                    max_sol=max_sol,
                    solver=solver,
                    log_file=log_file,
                )

                if solution is not None:
                    if mode == "f":
                        output_f_entities[technique] = solution
                    else:
                        output_e_entities[technique] = solution
            elif technique == "ICD":
                if vectorized:
                    fun = solve_icd_bqp_vector
                else:
                    fun = solve_icd_bqp_scalar
                solution = fun(
                    e_entities=e_dataset.names,
                    f_entities=f_dataset.names,
                    inter=set(inter),
                    epsilon=epsilon,
                    splits=splits,
                    names=names,
                    max_sec=max_sec,
                    max_sol=max_sol,
                    solver=solver,
                    log_file=log_file,
                )
                if solution is not None:
                    output_inter[technique], output_e_entities[technique], output_f_entities[technique] = solution
            elif technique == "CCS":
                fun = solve_ccs_bqp_vector if vectorized else solve_ccs_bqp_scalar

                cluster_split = fun(
                    e_clusters=dataset.cluster_names,
                    e_weights=[dataset.cluster_weights.get(c, 0) for c in dataset.cluster_names],
                    e_similarities=dataset.cluster_similarity,
                    e_distances=dataset.cluster_distance,
                    e_threshold=dataset.threshold,
                    epsilon=epsilon,
                    splits=splits,
                    names=names,
                    max_sec=max_sec,
                    max_sol=max_sol,
                    solver=solver,
                    log_file=log_file,
                )
                if cluster_split is not None:
                    if mode == "f":
                        output_f_clusters[technique] = cluster_split
                        output_f_entities[technique] = reverse_clustering(cluster_split, f_dataset.cluster_map)
                    else:
                        output_e_clusters[technique] = cluster_split
                        output_e_entities[technique] = reverse_clustering(cluster_split, e_dataset.cluster_map)
            elif technique == "CCD":
                cluster_inter = cluster_interactions(
                    inter,
                    e_dataset.cluster_map,
                    e_dataset.cluster_names,
                    f_dataset.cluster_map,
                    f_dataset.cluster_names,
                )
                fun = solve_ccd_bqp_vector if vectorized else solve_ccd_bqp_scalar
                cluster_split = fun(
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
                    names=names,
                    max_sec=max_sec,
                    max_sol=max_sol,
                    solver=solver,
                    log_file=log_file,
                )

                if cluster_split is not None:
                    output_inter[technique], output_e_clusters[technique], output_f_clusters[technique] = cluster_split
                    output_e_entities[technique] = reverse_clustering(cluster_split[1], e_dataset.cluster_map)
                    output_f_entities[technique] = reverse_clustering(cluster_split[2], f_dataset.cluster_map)
        except SolverError:
            logging.error(f"Splitting failed for {technique}, try to increase the timelimit or the epsilon value.")

    return output_inter, output_e_entities, output_f_entities, output_e_clusters, output_f_clusters
