from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from datasail.reader.read import read_data_type
from datasail.cluster.clustering import cluster
from datasail.settings import KW_THREADS, KW_LOGDIR, KW_LINKAGE

SPLIT_ASSIGNMENT_TYPE = Union[dict[str, Any], str, Path]


def eval_splits(datatype, data: Optional[Union[dict[str, Any], str, Path]], weights: Optional[Union[dict[str, float], str, Path]], similarity, distance, split_assignments: Union[SPLIT_ASSIGNMENT_TYPE, list[SPLIT_ASSIGNMENT_TYPE], tuple[SPLIT_ASSIGNMENT_TYPE], dict[Any, SPLIT_ASSIGNMENT_TYPE]]):
    """
    Evaluate the leakage of a split assignment on a dataset.

    Args:
        datatype: The type of data (e.g., "text", "image", etc.)
        data: The dataset to evaluate, can be a dictionary, string (path), or Path object.
        weights: Optional weights for the dataset, can be a dictionary, string (path), or Path object.
        similarity: Optional similarity matrix, can be a string (path) or Path object.
        distance: Optional distance matrix, can be a string (path) or Path object.
        split_assignments: A single split assignment, a list or tuple of split assignments, or a dictionary mapping split names to split assignments.
    
    Returns: 
        A tuple containing the leakage ratio, leakage value, and total metric value for each split assignment
    """
    if isinstance(split_assignments, (list, tuple)):
        results = []
        for split_assignment in split_assignments:
            result = eval_single_split(datatype, data, weights, similarity, distance, split_assignment)
            results.append(result)
        return results
    elif isinstance(split_assignments, dict):
        results = {}
        for split_name, split_assignment in split_assignments.items():
            result = eval_single_split(datatype, data, weights, similarity, distance, split_assignment)
            results[split_name] = result
        return results
    else:
        return eval_single_split(datatype, data, weights, similarity, distance, split_assignments)


def eval_single_split(datatype, data: Optional[Union[dict[str, Any], str, Path]], weights: Optional[Union[dict[str, float], str, Path]], similarity, distance, split_assignment: Union[dict[str, Any], str, Path]):
    if isinstance(data, str):
        data = Path(data)
    if isinstance(weights, str):
        weights = Path(weights)
    if isinstance(similarity, str) and Path(similarity).exists():
        similarity = Path(similarity)
    if isinstance(distance, str) and Path(distance).exists():
        distance = Path(distance)
    if isinstance(split_assignment, str):
        split_assignment = Path(split_assignment)
    
    dataset = read_data_type(datatype)(data=data, weights=weights, sim=similarity, dist=distance, num_clusters=np.inf)
    dataset = cluster(dataset, **{KW_THREADS: 1, KW_LOGDIR: None, KW_LINKAGE: "average"})
    in_split_mask = np.zeros((len(dataset.cluster_names), len(dataset.cluster_names)))
    for split in set(split_assignment.values()):
        if split == "not assigned":
            continue
        split_array = np.array([split_assignment[name] == split for name in dataset.cluster_names], dtype=int).reshape(-1, 1)
        in_split_mask += split_array @ split_array.T
    
    metric, mode = dataset.cluster_similarity, "sim"
    if metric is None:
        metric, mode = dataset.cluster_distance, "dist"
    
    weight_array = np.array([dataset.cluster_weights[name] for name in dataset.cluster_names]).reshape(-1, 1)
    weight_matrix = weight_array @ weight_array.T
    metric *= weight_matrix

    metric_total = np.sum(metric)
    leakage = np.sum(in_split_mask * metric)
    if mode == "dist":
        leakage = metric_total - leakage
    return 1 - (leakage / metric_total), leakage, metric_total
