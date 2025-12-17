from pathlib import Path
from typing import Any, Callable, Optional, Union
import copy
import numpy as np

from datasail.reader.read import read_data_type
from datasail.cluster.clustering import cluster
from datasail.reader.utils import MATRIX_INPUT
from datasail.settings import DIST_OPTIONS, KW_OUTDIR, KW_THREADS, KW_LOGDIR, KW_LINKAGE

SPLIT_ASSIGNMENT_TYPE = Union[dict[str, Any], str, Path]


def eval_split(
        datatype, 
        data: Optional[Union[dict[str, Any], str, Path]], 
        weights: Optional[Union[dict[str, float], str, Path]], 
        similarity: MATRIX_INPUT, 
        distance: MATRIX_INPUT, 
        dist_conv: Optional[Union[int, float, Callable]], 
        split_assignment: Union[dict[str, Any], str, Path]
    ) -> tuple[float, float, float]:
    """
    Evaluate the leakage of a single split assignment on a dataset. The inputs are mostly the same as for a normal DataSAIL run.

    Either a similarity or distance matrix must be provided. If a distance matrix is provided, a distance conversion function, string, or 
    a maximum distance value must also be provided to convert distances to similarities. In case of a function, it has to match the signature
    `func(distance_matrix: np.ndarray, len_fp: int = 1) -> np.ndarray`, where `len_fp` is the length of the fingerprints (or 1 if not applicable). 
    The len_fp parameter can be ignored if not needed.
    
    Args:
        datatype: The type of data, options are "M", "P", "G", "O"
        data: The dataset to evaluate, can be a dictionary, string (path), or Path object.
        weights: Optional weights for the dataset, can be a dictionary, string (path), or Path object.
        similarity: Optional similarity matrix, can be a string (path) or Path object.
        distance: Optional distance matrix, can be a string (path) or Path object.
        dist_conv: Optional distance conversion function or maximum distance value.
        split_assignment: A single split assignment, can be a dictionary, string (path), or Path object.

    Returns:
        A tuple containing 
            - the leakage ratio (lower is better), 
            - the absolute leakage value, and 
            - the total metric value for the split assignment (maximal leakage possible).
    """
    if distance is not None:
        if dist_conv is None:
            if not isinstance(distance, str):
                raise ValueError("If a distance matrix is provided, dist_conv must either be an int/float or a callable function.")
            if distance not in DIST_OPTIONS:
                raise ValueError("The provided distance matrix name is not recognized. Please check the documentation for supported distance metrics.")
            dist_conv = lambda M, _=1: 1 - M
        # Now dist_conv is either a float, a string, representing the max distance value, or a callable
        if isinstance(dist_conv, (int, float)):
            if 0 < dist_conv < np.inf:
                _dist_conv = lambda M, _=1: 1 - M / dist_conv
            else:
                _dist_conv = lambda M, _=1: np.log(M)
        elif isinstance(dist_conv, Callable):
            _dist_conv = dist_conv
        else:
            raise ValueError("dist_conv must be either a float, a string, or a callable function.")
    
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
    
    dataset = read_data_type(datatype)(data=data, weights=weights, sim=similarity, dist=distance, num_clusters=np.inf, detect_duplicates=False)
    dataset = cluster(dataset, **{KW_THREADS: 1, KW_LOGDIR: None, KW_LINKAGE: "average", KW_OUTDIR: None})
    in_split_mask = np.zeros((len(dataset.cluster_names), len(dataset.cluster_names)))
    for split in set(split_assignment.values()):
        if split == "not assigned":
            continue
        split_array = np.array([split_assignment[name] == split for name in dataset.cluster_names], dtype=int).reshape(-1, 1)
        in_split_mask += split_array @ split_array.T
    
    metric, mode = dataset.cluster_similarity, "sim"
    if metric is None:
        metric, mode = _dist_conv(dataset.cluster_distance, len(dataset.data[dataset.names[0]])), "dist"
    elif similarity == "mcconnaughey":
        metric = (metric + 1) / 2  # Convert McConnaughey to [0, 1] range
    
    weight_array = np.array([dataset.cluster_weights[name] for name in dataset.cluster_names]).reshape(-1, 1)
    weight_matrix = weight_array @ weight_array.T
    metric *= weight_matrix

    if mode == "sim":
        total = np.sum(metric * weight_matrix)
        leakage = np.sum((1 - in_split_mask) * weight_matrix * metric)
    if mode == "dist":
        total = np.sum((1 - metric) * weight_matrix)
        leakage = np.sum((1 - in_split_mask) * weight_matrix * (1 - metric))
    return leakage / total, leakage, total
