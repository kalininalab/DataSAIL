from typing import Any, Literal, Optional
import copy
import numpy as np
from collections import defaultdict

from datasail.reader.utils import DataSet, DictMap
from datasail.settings import LOGGER
from datasail.solver.cluster_2d import convert


def check_dataset(
    dataset,
    split_ratios,
    split_names,
    strategy: Literal["break", "assign"],
    linkage: Literal["average", "single", "complete"],
    id_tec: Optional[str],
    id_2d_tec: bool,
    cluster_tec: Optional[str],
    cluster_2d_tec: bool,
) -> tuple[DataSet, DictMap, DictMap,  list[float], list[str]]:
    name_split_map, cluster_split_map = {}, {}
    id_split_names, id_split_ratios = copy.deepcopy(split_names), copy.deepcopy(split_ratios)
    tec_split_names, tec_split_ratios = {}, {}
    if id_tec is not None:
        if id_2d_tec:
            id_split_ratios = convert(id_split_ratios)
        name_split_map[id_tec] = {}
        cluster_split_map[id_tec] = {}
        i = 0
        while (fixes := check_points(dataset, id_split_ratios, id_split_names, i)) is not None:
            i += 1
            dataset, tmp_name_split_map, tmp_cluster_split_map, id_split_ratios, id_split_names = fixes
            name_split_map[id_tec].update(tmp_name_split_map)
            cluster_split_map[id_tec].update(tmp_cluster_split_map)
        tec_split_names[id_tec] = id_split_names
        tec_split_ratios[id_tec] = id_split_ratios
    if cluster_tec is not None:
        if cluster_2d_tec:
            split_ratios = convert(split_ratios)
        name_split_map[cluster_tec] = {}
        cluster_split_map[cluster_tec] = {}
        i = 0
        while (fixes := check_clusters(dataset, split_ratios, split_names, strategy, linkage, i)) is not None:
            i += 1
            dataset, tmp_name_split_map, tmp_cluster_split_map, split_ratios, split_names = fixes
            name_split_map[cluster_tec].update(tmp_name_split_map)
            cluster_split_map[cluster_tec].update(tmp_cluster_split_map)
        tec_split_names[cluster_tec] = split_names
        tec_split_ratios[cluster_tec] = split_ratios
    return dataset, name_split_map, cluster_split_map, tec_split_ratios, tec_split_names


def check_points(dataset, split_ratios, split_names, i: int):
    sorted_points = sorted([(name, dataset.weights[name]) for name in dataset.names], key=lambda x: x[1], reverse=True)
    total_weight = sum(x[1] for x in sorted_points[i:])
    if [x[1] / total_weight for x in sorted_points[i:len(split_ratios)]] <= sorted(split_ratios, reverse=True):
        return None
    LOGGER.info("")
    overflows = [(pn, s) if ps > s else (None, None) for (pn, ps), s in zip(sorted_points, sorted(split_ratios, reverse=True))]
    overflow_point = next((i, pn, s) for i, (pn, s) in enumerate(overflows) if pn is not None)
    dataset, name_split_map, cluster_split_map, split_ratios, split_names = assign_cluster(dataset, overflow_point[1], split_ratios, split_names, overflow_point[0], clusters=False)
    return dataset, name_split_map, cluster_split_map, split_ratios, split_names


def check_clusters(dataset, split_ratios, split_names, strategy: Literal["break", "assign"], linkage: Literal["average", "single", "complete"], i: int):
    sorted_clusters = sorted([(name, dataset.cluster_weights[name]) for name in dataset.cluster_names], key=lambda x: x[1], reverse=True)
    total_weight = sum(x[1] for x in sorted_clusters[i:])
    if [x[1] / total_weight for x in sorted_clusters[i:len(split_ratios)]] <= sorted(split_ratios, reverse=True):
        return None
    LOGGER.info("")
    overflows = [(cn, s) if cs > s else (None, None) for (cn, cs), s in zip(sorted_clusters, sorted(split_ratios, reverse=True))]
    overflow_cluster = next((i, cn, s) for i, (cn, s) in enumerate(overflows) if cn is not None)
    if strategy == "break":
        dataset = break_cluster(dataset, overflow_cluster[1], overflow_cluster[2], linkage)
        name_split_map, cluster_split_map = {}, {}
    elif strategy == "assign":
        dataset, name_split_map, cluster_split_map, split_ratios, split_names = assign_cluster(dataset, overflow_cluster[1], split_ratios, split_names, overflow_cluster[0])
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'break' or 'assign'.")
    return dataset, name_split_map, cluster_split_map, split_ratios, split_names


def assign_cluster(dataset: DataSet, cluster_name: Any, split_ratios, split_names, split_index, clusters: bool = True) -> DataSet:
    split_name = split_names[split_index]
    split_ratios = split_ratios[:split_index] + split_ratios[split_index + 1:]
    split_names = split_names[:split_index] + split_names[split_index + 1:]

    if clusters:
        cluster_index = dataset.cluster_names.index(cluster_name)
        name_split_map = {}
        cluster_split_map = {cluster_name: split_name}
        for n in dataset.names:
            if dataset.cluster_map[n] == cluster_name:
                name_split_map[n] = split_name
        dataset.cluster_names = dataset.cluster_names[:cluster_index] + dataset.cluster_names[cluster_index + 1:]
        if dataset.cluster_similarity is not None:
            dataset.cluster_similarity = np.delete(dataset.cluster_similarity, cluster_index, axis=0)
            dataset.cluster_similarity = np.delete(dataset.cluster_similarity, cluster_index, axis=1)
        if dataset.cluster_distance is not None:
            dataset.cluster_distance = np.delete(dataset.cluster_distance, cluster_index, axis=0)
            dataset.cluster_distance = np.delete(dataset.cluster_distance, cluster_index, axis=1)
    else:
        name_split_map = {cluster_name: split_name}
        cluster_split_map = {}
        name_index = dataset.names.index(cluster_name)
        dataset.names =  dataset.names[:name_index] + dataset.names[name_index + 1:]
        if dataset.similarity is not None:
            dataset.similarity = np.delete(dataset.similarity, name_index, axis=0)
            dataset.similarity = np.delete(dataset.similarity, name_index, axis=1)
        if dataset.distance is not None:
            dataset.distance = np.delete(dataset.distance, name_index, axis=0)
            dataset.distance = np.delete(dataset.distance, name_index, axis=1)
    
    norm = sum(split_ratios)
    split_ratios = [r / norm for r in split_ratios]

    return dataset, name_split_map, cluster_split_map, split_ratios, split_names


def break_cluster(dataset: DataSet, cluster_name: Any, split_ratio: float, linkage: Literal["average", "single", "complete"]) -> DataSet:
    # index of the cluster to be broken
    cluster_index = dataset.cluster_names.index(cluster_name)

    # list of the cluster-names per data point in order of dataset.names
    labels, gap_map = [], {}

    # mapping of dataset.cluster_names to new cluster names
    unique_labels = {}
    for name in dataset.names:
        if (c := dataset.cluster_map[name]) != cluster_name:
            if c not in unique_labels:
                unique_labels[c] = len(unique_labels)
            labels.append(c)
        else:
            gap_map[name] = len(labels)
            labels.append("?")
    
    # reorder the to-be-reassigned datapoints (descending by weight) for assignment to new clusters
    re_assigns = sorted(gap_map.keys(), key=lambda x: -dataset.weights[x])

    # How many new clusters do we need to create? And what is the target (max) size for each?
    num_new = int(dataset.cluster_weights[cluster_name] / (sum(dataset.cluster_weights.values()) * split_ratio) + 1)
    target_size = dataset.cluster_weights[cluster_name] / num_new

    # counter with the current size of each new cluster
    sizes = [0] * num_new

    # define the names for the new clusters and update the unique_labels mapping
    new_names = [(cluster_name, n) for n in range(len(unique_labels), len(unique_labels) + num_new)]
    unique_labels.update({n: n[1] for n in new_names})

    # assign the datapoints of the to-be-broken cluster to one of the new clusters
    for name in re_assigns:
        assigned = False
        for i in range(num_new):
            if sizes[i] + dataset.weights[name] <= target_size:
                labels[gap_map[name]] = new_names[i]
                sizes[i] += dataset.weights[name]
                assigned = True
                break
        if not assigned:
            LOGGER.warning(f"Could not fit {name} into any new cluster, an assignment will be enforced.")
            mindex = np.argmin(sizes)
            labels[gap_map[name]] = new_names[mindex]
            sizes[mindex] += dataset.weights[name]

    cluster_map, cluster_weights = {}, defaultdict(float)
    for i, name in enumerate(dataset.names):
        cluster_map[name] = unique_labels[labels[i]]
        cluster_weights[unique_labels[labels[i]]] += dataset.weights[name]
    cluster_names = list(sorted(unique_labels.values()))

    if dataset.stratification is not None and len(dataset.classes) > 1:
        cluster_stratification = defaultdict(lambda: np.zeros(len(dataset.classes)))
        for name in dataset.names:  # key, value in dataset.cluster_map.items():
            cluster_stratification[dataset.cluster_map[name]] += dataset.stratification[name]
    else:
        cluster_stratification = None
    
    cluster_matrix = dataset.cluster_similarity if dataset.cluster_similarity is not None else dataset.cluster_distance
    row = np.delete(cluster_matrix[cluster_index], cluster_index)
    col = np.delete(cluster_matrix[:, cluster_index], cluster_index)
    cluster_matrix = np.delete(cluster_matrix, cluster_index, axis=0)
    cluster_matrix = np.delete(cluster_matrix, cluster_index, axis=1)
    # This assumes inner-cluster similarities to be 1, this is not necessarily true
    new_cluster_matrix = np.ones((len(cluster_names), len(cluster_names)))
    new_cluster_matrix[:cluster_matrix.shape[0], :cluster_matrix.shape[1]] = cluster_matrix
    new_cluster_matrix[cluster_matrix.shape[0]:, :-num_new] = np.tile(row, (num_new, 1))
    new_cluster_matrix[:-num_new, cluster_matrix.shape[1]:] = np.tile(col, (num_new, 1)).T

    dataset.cluster_names = cluster_names
    dataset.cluster_map = cluster_map
    dataset.cluster_weights = cluster_weights
    dataset.cluster_stratification = cluster_stratification
    dataset.cluster_similarity = new_cluster_matrix if dataset.cluster_similarity is not None else None

    return dataset
