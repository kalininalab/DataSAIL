import logging
import os
from typing import Dict, Tuple, List, Union, Optional

import numpy as np
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering

from datasail.cluster.caching import load_from_cache, store_to_cache
from datasail.cluster.cdhit import run_cdhit
from datasail.cluster.ecfp import run_ecfp
from datasail.cluster.foldseek import run_foldseek
from datasail.cluster.mash import run_mash
from datasail.cluster.mmseqs2 import run_mmseqs
from datasail.cluster.utils import heatmap
from datasail.cluster.wlk import run_wlk
from datasail.reader.utils import DataSet
from datasail.report import whatever


def cluster(dataset: DataSet, **kwargs) -> DataSet:
    """
    Cluster molecules based on a similarity or distance metric.

    Args:
        dataset: Dataset to cluster

    Returns:
        A dataset with modified properties according to clustering the data
    """
    cache = load_from_cache(dataset, **kwargs)
    if cache is not None:
        logging.info("Loaded clustering from cache")
        return cache

    if isinstance(dataset.similarity, str):  # compute the similarity
        dataset.cluster_names, dataset.cluster_map, dataset.cluster_similarity, dataset.cluster_weights = \
            similarity_clustering(dataset, kwargs["logdir"])

    elif isinstance(dataset.distance, str):  # compute the distance
        dataset.cluster_names, dataset.cluster_map, dataset.cluster_distance, dataset.cluster_weights = \
            distance_clustering(dataset)

    # if the similarity/distance is already given, store it
    elif dataset.similarity is not None or dataset.distance is not None:
        dataset.cluster_names = dataset.names
        dataset.cluster_map = dict([(d, d) for d in dataset.names])
        dataset.cluster_similarity = dataset.similarity
        dataset.cluster_distance = dataset.distance
        dataset.cluster_weights = dataset.weights

    if dataset.cluster_names is None:
        return dataset

    # if there are too many clusters, reduce their number based on some cluster algorithms.
    if any(isinstance(m, np.ndarray) for m in
           [dataset.similarity, dataset.cluster_similarity, dataset.cluster_distance]):
        num_old_cluster = len(dataset.cluster_names) + 1
        while 100 < len(dataset.cluster_names) < num_old_cluster:
            num_old_cluster = len(dataset.cluster_names)
            dataset = additional_clustering(dataset)

        if isinstance(dataset.similarity, np.ndarray) or isinstance(dataset.distance, np.ndarray):
            whatever(dataset.names, dataset.cluster_map, dataset.distance, dataset.similarity)
            metric = dataset.similarity if dataset.similarity is not None else dataset.distance
            form = "similarity" if dataset.similarity is not None else "distance"
            if kwargs["output"] is not None:
                heatmap(metric, os.path.join(kwargs["output"], dataset.get_name() + f"_{form}.png"))

    store_to_cache(dataset, **kwargs)

    return dataset


def similarity_clustering(dataset: DataSet, log_dir: Optional[str]) -> Tuple[
    List[str], Dict[str, str], np.ndarray, Dict[str, float],
]:
    """
    Compute the similarity based cluster based on a cluster method.

    Args:
        dataset: Mapping from molecule names to molecule description (fasta, PDB, SMILES, ...)
        log_dir: Absolute path to the directory to store all the logs in

    Returns:
        A tuple consisting of
          - The names of the current clusters
          - The mapping from cluster names to cluster representatives
          - Symmetric matrix of pairwise similarities between the current clusters
          - Symmetric matrix of pairwise similarities between the current clusters
          - Mapping from current clusters to their weights
    """
    if dataset.similarity.lower() == "wlk":
        cluster_names, cluster_map, cluster_sim = run_wlk(dataset)
    elif dataset.similarity.lower() == "mmseqs":
        cluster_names, cluster_map, cluster_sim = run_mmseqs(dataset, log_dir)
    elif dataset.similarity.lower() == "foldseek":
        cluster_names, cluster_map, cluster_sim = run_foldseek(dataset, log_dir)
    elif dataset.similarity.lower() == "cdhit":
        cluster_names, cluster_map, cluster_sim = run_cdhit(dataset, log_dir)
    elif dataset.similarity.lower() == "ecfp":
        cluster_names, cluster_map, cluster_sim = run_ecfp(dataset)
    else:
        raise ValueError(f"Unknown cluster method: {dataset.similarity}")

    # compute the weights for the clusters
    cluster_weights = {}
    for key, value in cluster_map.items():
        if value not in cluster_weights:
            cluster_weights[value] = 0
        cluster_weights[value] += 1

    # cluster_map maps members to their cluster names
    return cluster_names, cluster_map, cluster_sim, cluster_weights


def distance_clustering(dataset: DataSet) -> Tuple[
    List[str], Dict[str, str], np.ndarray, Dict[str, float],
]:
    """
    Compute the distance based cluster based on a cluster method or a file to extract pairwise distance
    from.

    Args:
        dataset: DataSet with all information what and how to cluster

    Returns:
        A tuple consisting of
          - The names of the current clusters
          - The mapping from cluster names to cluster representatives
          - Symmetric matrix of pairwise similarities between the current clusters
          - Symmetric matrix of pairwise similarities between the current clusters
          - Mapping from current clusters to their weights
    """
    if dataset.distance.lower() == "mash":
        cluster_names, cluster_map, cluster_dist = run_mash(dataset)
    else:
        raise ValueError(f"Unknown cluster method: {dataset.distance}")

    # compute the weights for the clusters
    cluster_weights = {}
    for key, value in cluster_map.items():
        if value not in cluster_weights:
            cluster_weights[key] = 0
        cluster_weights[key] += 1

    # cluster_map maps members to their cluster names
    return cluster_names, cluster_map, cluster_dist, cluster_weights


def additional_clustering(dataset: DataSet) -> DataSet:
    """
    Perform additional cluster based on a distance or similarity matrix. This is done to reduce the number of
    clusters and to speed up the further splitting steps.

    Args:
        dataset: DataSet to perform additional clustering on

    Returns:
        The dataset with updated clusters
    """
    logging.info(f"Cluster {len(dataset.cluster_names)} items based on "
                 f"{'similarities' if dataset.cluster_similarity is not None else 'distances'}")
    # set up the cluster algorithm for similarity or distance based cluster w/o specifying the number of clusters
    if dataset.cluster_similarity is not None:
        cluster_matrix = np.array(dataset.cluster_similarity, dtype=float)
        ca = AffinityPropagation(affinity='precomputed', random_state=42)
    else:
        cluster_matrix = np.array(dataset.cluster_distance, dtype=float)
        ca = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='average',
            distance_threshold=np.average(dataset.cluster_distance) * 0.9,
            # connectivity=np.asarray(cluster_matrix < np.average(cluster_distance) * 0.9, dtype=int),
        )
        logging.info(
            f"Clustering based on distances. "
            f"Distances above {np.average(dataset.cluster_distance) * 0.9} cannot end up in same cluster."
        )

    # cluster the clusters into new, fewer, and bigger clusters
    labels = ca.fit_predict(cluster_matrix)

    # extract the names of the new clusters and compute a mapping from the element names to the clusters
    old_cluster_map = dict((y, x) for x, y in enumerate(dataset.cluster_names))
    new_cluster_names = list(np.unique(labels))
    new_cluster_map = dict((n, labels[old_cluster_map[c]]) for n, c in dataset.cluster_map.items())

    # compute the distance or similarity matrix for the new clusters as the average sim/dist between their members
    new_cluster_matrix = np.zeros((len(new_cluster_names), len(new_cluster_names)))
    cluster_count = np.zeros((len(new_cluster_names), len(new_cluster_names)))
    for i in range(len(dataset.cluster_names)):
        for j in range(i + 1, len(dataset.cluster_names)):
            if labels[i] != labels[j]:
                new_cluster_matrix[labels[i], labels[j]] += cluster_matrix[i, j]
                cluster_count[labels[i], labels[j]] += 1

                new_cluster_matrix[labels[j], labels[i]] += cluster_matrix[i, j]
                cluster_count[labels[j], labels[i]] += 1
    new_cluster_matrix /= (cluster_count + np.eye(max(labels) + 1))

    # compute the mapping of new clusters to their weights as the sum of their members weights
    new_cluster_weights = {}
    for name in list(dataset.cluster_map.keys()):
        new_cluster = new_cluster_map[name]
        if new_cluster not in new_cluster_weights:
            new_cluster_weights[new_cluster] = 0
        new_cluster_weights[new_cluster] += dataset.cluster_weights[dataset.cluster_map[name]]

    logging.info(f"Reduced number of clusters to {len(new_cluster_names)}.")

    dataset.cluster_names = new_cluster_names
    dataset.cluster_map = new_cluster_map
    dataset.cluster_weights = new_cluster_weights

    # store the matrix at the correct field and set the main diagonal to either 1 or 0 depending on dist or sim
    if dataset.cluster_similarity is not None:
        dataset.cluster_similarity = np.maximum(new_cluster_matrix, np.eye(len(new_cluster_matrix)))
    else:
        dataset.cluster_distance = np.minimum(new_cluster_matrix, 1 - np.eye(len(new_cluster_matrix)))

    return dataset


def cluster_interactions(
        inter: List[Tuple[str, str]],
        e_cluster_map: Dict[str, Union[str, int]],
        e_cluster_names: List[Union[str, int]],
        f_cluster_map: Dict[str, Union[str, int]],
        f_cluster_names: List[Union[str, int]],
) -> np.ndarray:
    """
    Based on cluster information, count interactions in an interaction matrix between the individual clusters.

    Args:
        inter: List of pairs representing interactions
        e_cluster_map: Mapping from entity names to their cluster names
        e_cluster_names: List of custer names
        f_cluster_map: Mapping from entity names to their cluster names
        f_cluster_names: List of custer names

    Returns:
        Numpy array of matrix of interactions between two clusters
    """
    e_mapping = dict((y, x) for x, y in enumerate(e_cluster_names))
    f_mapping = dict((y, x) for x, y in enumerate(f_cluster_names))

    output = np.zeros((len(e_cluster_names), len(f_cluster_names)))
    for e, f in inter:
        output[e_mapping[e_cluster_map[e]]][f_mapping[f_cluster_map[f]]] += 1

    return output


def reverse_clustering(cluster_split: Dict[str, str], name_cluster: Dict[str, str]) -> Dict[str, str]:
    """
    Reverse clustering to uncover which entity is assigned to which split.

    Args:
        cluster_split: Assignment of clusters to splits
        name_cluster: Assignment of names to clusters

    Returns:
        Assignment of names to splits
    """
    output = {}
    for n, c in name_cluster.items():
        output[n] = cluster_split[c]
    return output
