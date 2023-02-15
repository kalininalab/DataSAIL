import logging
from typing import Tuple, Optional, List, Dict, Union

import numpy as np

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


def run_solver(
        technique: str,
        e_dataset: DataSet,
        f_dataset: DataSet,
        inter: Optional[Union[np.ndarray, List[Tuple[str, str]]]],
        vectorized: bool,
        epsilon: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
) -> Tuple[Optional[List[Tuple[str, str, str]]], Optional[Dict[str, str]], Optional[Dict[str, str]]]:
    """
    Run a solver based on the selected technique.

    Args:
        technique: Technique to use to split the dataset
        e_dataset: First dataset
        f_dataset: Second dataset
        inter: Interactions of elements or clusters of the two datasets
        vectorized: Boolean flag indicating to run it in vectorized form
        epsilon: Additive bound for exceeding the requested split size
        splits: List of split sizes
        names: List of names of the splits in the order of the splits argument
        max_sec: Maximal number of seconds to take when optimizing the problem (not for finding an initial solution)
        max_sol: Maximal number of solution to consider

    Returns:
        A list of interactions and their assignment to a split and two mappings from entities to splits, one for each
        dataset
    """
    output_inter, output_e_entities, output_f_entities = None, None, None

    logging.info("Define optimization problem")

    if technique == "R":
        output_inter = sample_categorical(
            inter=inter,
            splits=splits,
            names=names,
        )
    elif technique == "ICS":
        if vectorized:
            fun = solve_ics_bqp_vector
        else:
            fun = solve_ics_bqp_scalar
        solution = fun(
            e_entities=e_dataset.names,
            e_weights=[e_dataset.weights[e] for e in e_dataset.names],
            epsilon=epsilon,
            splits=splits,
            names=names,
            max_sec=max_sec,
            max_sol=max_sol,
        )
        if solution is not None:
            output_e_entities = solution
    if technique == "ICD":
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
        )
        if solution is not None:
            output_inter, output_e_entities, output_f_entities = solution
    if technique == "CCS":
        if vectorized:
            fun = solve_ccs_bqp_vector
        else:
            fun = solve_ccs_bqp_scalar
        cluster_split = fun(
            e_clusters=e_dataset.cluster_names,
            e_weights=[e_dataset.cluster_weights[c] for c in e_dataset.cluster_names],
            e_similarities=e_dataset.cluster_similarity,
            e_distances=e_dataset.cluster_distance,
            e_threshold=e_dataset.threshold,
            epsilon=epsilon,
            splits=splits,
            names=names,
            max_sec=max_sec,
            max_sol=max_sol,
        )
        if cluster_split is not None:
            output_e_entities = reverse_clustering(cluster_split, e_dataset.cluster_map)
    if technique == "CCD":
        cluster_inter = cluster_interactions(
            inter,
            e_dataset.cluster_map,
            e_dataset.cluster_names,
            f_dataset.cluster_map,
            f_dataset.cluster_names,
        )
        if vectorized:
            fun = solve_ccd_bqp_vector
        else:
            fun = solve_ccd_bqp_scalar
        cluster_split = fun(
            e_clusters=e_dataset.cluster_names,
            e_similarities=e_dataset.cluster_similarity,
            e_distances=e_dataset.cluster_distance,
            e_threshold=e_dataset.threshold,
            f_clusters=f_dataset.cluster_names,
            f_similarities=f_dataset.cluster_similarity,
            f_distances=f_dataset.cluster_distance,
            f_threshold=f_dataset.threshold,
            inter=cluster_inter,
            epsilon=epsilon,
            splits=splits,
            names=names,
            max_sec=max_sec,
            max_sol=max_sol,
        )

        if cluster_split is not None:
            output_inter, output_e_entities, output_f_entities = cluster_split

    return output_inter, output_e_entities, output_f_entities
