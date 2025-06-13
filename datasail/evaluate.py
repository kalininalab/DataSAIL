from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from datasail.reader.read import read_data_type


def eval_split(datatype, names: Optional[Union[list[str], str, Path]], data: Optional[Union[dict[str, Any], str, Path]], weights: Optional[Union[dict[str, float], str, Path]], similarity, distance, split_assignment: Union[dict[str, Any], str, Path]):
    if isinstance(data, str):
        data = Path(data)
    if isinstance(weights, str):
        weights = Path(weights)
    if isinstance(similarity, str):
        similarity = Path(similarity)
    if isinstance(distance, str):
        distance = Path(distance)
    if isinstance(split_assignment, str):
        split_assignment = Path(split_assignment)
    
    dataset = read_data_type(datatype)(data=data, weights=weights, sim=similarity, dist=distance)
    in_split_mask = np.zeros((len(dataset.names), len(dataset.names)))
    for split in set(split_assignment.values()):
        split_array = np.array([split_assignment[name] == split for name in dataset.names], dtype=int).reshape(-1, 1)
        in_split_mask += split_array @ split_array.T
    
    metric, mode = dataset.similarity, "sim"
    if metric is None:
        metric, mode = dataset.distance, "dist"
    
    weight_array = np.array([dataset.weights[name] for name in dataset.names]).reshape(-1, 1)
    weight_matrix = weight_array @ weight_array.T
    metric *= weight_matrix

    metric_total = np.sum(metric)
    leakage = np.sum(in_split_mask * metric)
    if mode == "dist":
        leakage = metric_total - leakage
    return leakage / metric_total, leakage, metric_total


def eval_cluster_split():
    pass
