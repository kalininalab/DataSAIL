import copy
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
from cvxpy import SolverError

from datasail.cluster.clustering import reverse_clustering, cluster_interactions, reverse_interaction_clustering
from datasail.dataset import DataSet
from datasail.constants import LOGGER, MODE_F, TEC_R, TEC_I1, TEC_C1, TEC_I2, TEC_C2, MMSEQS, CDHIT, MMSEQS2, DictMap
from datasail.reader.utils import Technique
from datasail.solver.id_1d import solve_i1
from datasail.solver.id_2d import solve_i2
from datasail.solver.cluster_1d import solve_c1
from datasail.solver.cluster_2d import convert, solve_c2
from datasail.solver.overflow import check_dataset
from datasail.solver.utils import sample_categorical


def insert(dictionary: dict, key: str, dim: int, value) -> None:
    """
    Append a value into a dictionary with the given key. If key is not in dictionary, create an empty list and append
    the value.

    Args:
        dictionary: Dict to insert in
        key: Key to insert at
        value: Value to insert
    """
    if key not in dictionary:
        dictionary[key] = {}
    if dim not in dictionary[key]:
        dictionary[key][dim] = []
    dictionary[key][dim].append(value)


def random_inter_split(runs, inter, splits, split_names):
    return [sample_categorical(inter=inter, splits=splits, names=split_names) for _ in range(runs)]


def run_solver(
        technique: Technique,
        datasets: list[DataSet],
        delta: float,
        epsilon: float,
        runs: int,
        splits: list[float],
        names: list[str],
        overflow: Literal["break", "assign"],
        linkage: Literal["average", "single", "complete"],
        max_sec: int,
        solver: str,
        log_dir: Path,
) -> tuple[DictMap, DictMap]:
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
    tmp_datasets, pre_name_split_maps, pre_cluster_split_maps, split_ratios, split_names = [], [], [], [], []
    output_entities, output_clusters = {}, {}

    ds_ids = [dim.dim - 1 for dim in technique]
    splits = convert(splits, dimensions=len(ds_ids))

    for idx in ds_ids:
        dataset, pre_tmp_name_split_map, pre_tmp_cluster_split_map, ratio, names = check_dataset(
            copy.deepcopy(datasets[idx]),
            splits,
            names,
            overflow,
            linkage,
            technique.is_clustered(),  # [idx].clustering,
        )
        tmp_datasets.append(dataset)
        pre_name_split_maps.append(pre_tmp_name_split_map)
        pre_cluster_split_maps.append(pre_tmp_cluster_split_map)
        split_ratios.append(ratio)
        split_names.append(names)

    LOGGER.info("Define optimization problem")

    for run in range(runs):
        if run > 0:
            for dataset in datasets:
                dataset.shuffle()
        try:
            LOGGER.info(f"Splitting technique {technique} of run {run + 1} of {runs}")
            log_file = None if log_dir is None else log_dir / f"{dataset.get_name()}_{technique}.log"
            
            if technique.is_oned():
                tech = technique.dims[0]
                dataset = tmp_datasets[0]
                unidistant = isinstance(dataset.similarity, str) and dataset.similarity.lower() in [CDHIT, MMSEQS, MMSEQS2]
                if not tech.clustering or unidistant:
                    if tech.clustering and unidistant:
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
                        splits=split_ratios[0],
                        names=split_names[0],
                        max_sec=max_sec,
                        solver=solver,
                        log_file=log_file,
                    )

                    if solution is not None:
                        if tech.clustering and unidistant:
                            insert(output_clusters, technique, technique[0].dim, solution)
                            insert(output_entities, technique, technique[0].dim, reverse_clustering(solution, tmp_datasets[0].cluster_map))
                        else:
                            insert(output_entities, technique, technique[0].dim, solution)
                else:
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
                        splits=split_ratios[0],
                        names=split_names[0],
                        max_sec=max_sec,
                        solver=solver,
                        log_file=log_file,
                    )
                    if cluster_split is not None:
                        insert(output_clusters, technique, technique[0].dim, cluster_split)
                        insert(output_entities, technique, technique[0].dim, reverse_clustering(cluster_split, tmp_datasets[0].cluster_map))
            elif len(technique) == 2 and not technique.is_clustered():
                e_dataset = tmp_datasets[0]
                f_dataset = tmp_datasets[1]
                solution = solve_i2(
                    e_entities=e_dataset.names,
                    e_stratification=np.stack([
                        e_dataset.stratification.get(n, np.zeros(len(dataset.classes))) for n in e_dataset.names
                    ]) if e_dataset.stratification is not None and len(e_dataset.classes) > 1 else None,
                    e_weights=[e_dataset.weights.get(c, 0) for c in e_dataset.names],
                    e_splits=split_ratios[0],
                    e_names=split_names[0],
                    f_entities=f_dataset.names,
                    f_stratification=np.stack([
                        f_dataset.stratification.get(n, np.zeros(len(dataset.classes))) for n in f_dataset.names
                    ]) if f_dataset.stratification is not None and len(f_dataset.classes) > 1 else None,
                    f_weights=[f_dataset.weights.get(c, 0) for c in f_dataset.names],
                    f_splits=split_ratios[1],
                    f_names=split_names[1],
                    delta=delta,
                    epsilon=epsilon,
                    max_sec=max_sec,
                    solver=solver,
                    log_file=log_file,
                )
                if solution is not None:
                    insert(output_entities, technique, technique[0].dim, solution[0])
                    insert(output_entities, technique, technique[1].dim, solution[1])
            elif len(technique) == 2 and technique.is_clustered():
                e_dataset = tmp_datasets[0]
                f_dataset = tmp_datasets[1]
                cluster_split = solve_c2(
                    e_clusters=e_dataset.cluster_names,
                    e_s_matrix=np.stack([
                        e_dataset.cluster_stratification.get(c, np.zeros(len(dataset.classes)))
                        for c in e_dataset.cluster_names
                    ]) if e_dataset.cluster_stratification is not None and len(e_dataset.classes) > 1 else None,
                    e_similarities=e_dataset.cluster_similarity,
                    e_distances=e_dataset.cluster_distance,
                    e_weights=np.array([e_dataset.cluster_weights.get(c, 1) for c in e_dataset.cluster_names]),
                    e_splits=split_ratios[0],
                    e_names=split_names[0],
                    f_clusters=f_dataset.cluster_names,
                    f_s_matrix=np.stack([
                        f_dataset.cluster_stratification.get(c, np.zeros(len(dataset.classes)))
                        for c in f_dataset.cluster_names
                    ]) if f_dataset.cluster_stratification is not None and len(f_dataset.classes) > 1 else None,
                    f_similarities=f_dataset.cluster_similarity,
                    f_distances=f_dataset.cluster_distance,
                    f_weights=np.array([f_dataset.cluster_weights.get(c, 1) for c in f_dataset.cluster_names]),
                    f_splits=split_ratios[1],
                    f_names=split_names[1],
                    delta=delta,
                    epsilon=epsilon,
                    max_sec=max_sec,
                    solver=solver,
                    log_file=log_file,
                )

                if cluster_split is not None:
                    insert(output_clusters, technique, technique[0].dim, cluster_split[0])
                    insert(output_clusters, technique, technique[1].dim, cluster_split[1])
                    insert(output_entities, technique, technique[0].dim, reverse_clustering(cluster_split[0], e_dataset.cluster_map))
                    insert(output_entities, technique, technique[1].dim, reverse_clustering(cluster_split[1], f_dataset.cluster_map))
        except SolverError:
            LOGGER.error(f"Splitting failed for {technique}, try to increase the timelimit or the epsilon value.")

    for i, name_split_map in enumerate(pre_name_split_maps):
        ds_id = ds_ids[i] + 1
        for dim_map in output_entities.values():
            for run_idx in range(len(dim_map[ds_id])):
                dim_map[ds_id][run_idx].update(name_split_map)
    for i, cluster_split_map in enumerate(pre_cluster_split_maps):
        ds_id = ds_ids[i] + 1
        for dim_map in output_clusters.values():
            for run_idx in range(len(dim_map[ds_id])):
                dim_map[ds_id][run_idx].update(cluster_split_map)

    return output_entities, output_clusters
