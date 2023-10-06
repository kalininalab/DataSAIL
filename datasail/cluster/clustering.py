import copy
import os
from typing import Dict, Tuple, List, Union, Optional

import numpy as np
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, SpectralClustering

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
from datasail.settings import LOGGER, KW_THREADS, KW_LOGDIR, KW_OUTDIR, MAX_CLUSTERS, N_CLUSTERS


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
        LOGGER.info("Loaded clustering from cache")
        return cache

    if isinstance(dataset.similarity, str):  # compute the similarity
        dataset.cluster_names, dataset.cluster_map, dataset.cluster_similarity, dataset.cluster_weights = \
            similarity_clustering(dataset, kwargs[KW_THREADS], kwargs[KW_LOGDIR])

    elif isinstance(dataset.distance, str):  # compute the distance
        dataset.cluster_names, dataset.cluster_map, dataset.cluster_distance, dataset.cluster_weights = \
            distance_clustering(dataset, kwargs[KW_THREADS], kwargs[KW_LOGDIR])

    # if the similarity/distance is already given, store it
    elif isinstance(dataset.similarity, np.ndarray) or isinstance(dataset.distance, np.ndarray):
        dataset.cluster_names = dataset.names
        dataset.cluster_map = dict([(d, d) for d in dataset.names])
        dataset.cluster_similarity = dataset.similarity
        dataset.cluster_distance = dataset.distance
        dataset.cluster_weights = dataset.weights

    if dataset.cluster_names is None:  # No clustering to do?!
        return dataset

    # if there are too many clusters, reduce their number based on some cluster algorithms.
    if any(isinstance(m, np.ndarray) for m in
           [dataset.similarity, dataset.cluster_similarity, dataset.cluster_distance]):
        num_old_cluster = len(dataset.cluster_names) + 1
        while MAX_CLUSTERS < len(dataset.cluster_names) < num_old_cluster:
            num_old_cluster = len(dataset.cluster_names)
            dataset = stable_additional_clustering(dataset)

        if isinstance(dataset.similarity, np.ndarray) or isinstance(dataset.distance, np.ndarray):
            whatever(dataset.names, dataset.cluster_map, dataset.distance, dataset.similarity)
            metric = dataset.similarity if dataset.similarity is not None else dataset.distance
            form = "similarity" if dataset.similarity is not None else "distance"
            if kwargs[KW_OUTDIR] is not None:
                heatmap(metric, os.path.join(kwargs[KW_OUTDIR], dataset.get_name() + f"_{form}.png"))

    if len(dataset.cluster_names) > MAX_CLUSTERS:
        dataset = force_clustering(dataset)

    store_to_cache(dataset, **kwargs)

    return dataset


def similarity_clustering(
        dataset: DataSet,
        threads: int = 1,
        log_dir: Optional[str] = None
) -> Tuple[List[str], Dict[str, str], np.ndarray, Dict[str, float]]:
    """
    Compute the similarity based cluster based on a cluster method.

    Args:
        dataset: Mapping from molecule names to molecule description (fasta, PDB, SMILES, ...)
        threads: number of threads to use for one CD-HIT run
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
        cluster_names, cluster_map, cluster_sim = run_mmseqs(dataset, threads, log_dir)
    elif dataset.similarity.lower() == "foldseek":
        cluster_names, cluster_map, cluster_sim = run_foldseek(dataset, threads, log_dir)
    elif dataset.similarity.lower() == "cdhit":
        cluster_names, cluster_map, cluster_sim = run_cdhit(dataset, threads, log_dir)
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


def distance_clustering(
        dataset: DataSet,
        threads: int = 1,
        log_dir: Optional[str] = None
) -> Tuple[List[str], Dict[str, str], np.ndarray, Dict[str, float]]:
    """
    Compute the distance based cluster based on a cluster method or a file to extract pairwise distance from.

    Args:
        dataset: DataSet with all information what and how to cluster
        threads: number of threads to use for one CD-HIT run
        log_dir: Absolute path to the directory to store all the logs in

    Returns:
        A tuple consisting of
          - The names of the current clusters
          - The mapping from cluster names to cluster representatives
          - Symmetric matrix of pairwise similarities between the current clusters
          - Symmetric matrix of pairwise similarities between the current clusters
          - Mapping from current clusters to their weights
    """
    if dataset.distance.lower() == "mash":
        cluster_names, cluster_map, cluster_dist = run_mash(dataset, threads, log_dir)
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


def stable_additional_clustering(dataset: DataSet, min_num_clusters: int = 10) -> DataSet:
    """
    Wrapper method around additional clustering to stabilize results. This is necessary for Affinity Propagation
    as this might not converge and for agglomerative clustering as it might lead to too few clusters.

    Args:
        dataset: DataSet to perform additional clustering on
        min_num_clusters: minimal number of clusters

    Returns:
        The dataset with updated clusters
    """
    # make a deep copy of the data as we might need to cluster this dataset multiple times as it is modified in add_c.
    ds = copy.deepcopy(dataset)

    if dataset.cluster_similarity is not None:  # stabilize affinity propagation
        # define lower, current, and upper value for damping value
        min_d, curr_d, max_d = 0.5, 0.5, 0.95
        ds, conv = additional_clustering(ds, damping=min_d)
        if conv:
            return ds

        # increase damping factor until the algorithm converges
        while not conv:
            curr_d = (min_d + max_d) / 2
            min_d = curr_d
            ds = copy.deepcopy(dataset)
            ds, conv = additional_clustering(ds, damping=curr_d)
        return ds
    else:
        min_f, curr_f, max_f = 0, 0.9, 1
        ds, _ = additional_clustering(ds, dist_factor=curr_f)
        while len(ds.cluster_names) < min_num_clusters or MAX_CLUSTERS < len(ds.cluster_names):
            if len(ds.cluster_names) < min_num_clusters:
                max_f = curr_f
            else:
                min_f = curr_f
            curr_f = (min_f + max_f) / 2
            ds = copy.deepcopy(dataset)
            ds, _ = additional_clustering(ds, dist_factor=min_f)
        return ds


def additional_clustering(
        dataset: DataSet,
        damping: float = 0.5,
        max_iter: int = 1000,
        dist_factor: float = 0.9,
        n_clusters: int = MAX_CLUSTERS,
) -> Tuple[DataSet, bool]:
    """
    Perform additional cluster based on a distance or similarity matrix. This is done to reduce the number of
    clusters and to speed up the further splitting steps.

    Args:
        dataset: DataSet to perform additional clustering on
        damping: damping factor for affinity propagation
        max_iter: maximal number of iterations for affinity propagation
        dist_factor: factor to multiply the average distance with in agglomerate clustering
        n_clusters:

    Returns:
        The dataset with updated clusters and a bool flag indicating convergence of the used clustering algorithm
    """
    LOGGER.info(f"Cluster {len(dataset.cluster_names)} items based on "
                f"{'similarities' if dataset.cluster_similarity is not None else 'distances'}")
    # set up the cluster algorithm for similarity or distance based cluster w/o specifying the number of clusters
    if dataset.cluster_similarity is not None:
        cluster_matrix = np.array(dataset.cluster_similarity, dtype=float)
        # ca = AffinityPropagation(
        #     affinity='precomputed',
        #     random_state=42,
        #     verbose=True,
        #     damping=damping,
        #     max_iter=max_iter,
        # )
        ca = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=42,
        )
    else:
        cluster_matrix = np.array(dataset.cluster_distance, dtype=float)
        kwargs = {
            "n_clusters": None,
            "metric": 'precomputed',
            "linkage": 'average',
            "distance_threshold": np.average(dataset.cluster_distance) * dist_factor,
            # "verbose": True,
            # "connectivity": np.asarray(cluster_matrix < np.average(cluster_distance) * 0.9, dtype=int),
        }
        ca = AgglomerativeClustering(**kwargs)
        LOGGER.info(
            f"Clustering based on distances. "
            f"Distances above {np.average(dataset.cluster_distance) * 0.9} cannot end up in same cluster."
        )
    # cluster the clusters into new, fewer, and bigger clusters
    labels = ca.fit_predict(cluster_matrix)
    converged = not hasattr(ca, "n_iter_") or ca.n_iter_ < max_iter
    return labels2clusters(labels, dataset, cluster_matrix, converged)


def labels2clusters(
        labels: Union[List, np.ndarray],
        dataset: DataSet,
        cluster_matrix: np.ndarray,
        converged: bool
) -> Tuple[DataSet, bool]:
    """
    Convert a list of labels to a clustering and insert it into the dataset. This also updates cluster_weights and
    distance or similarity metrics.

    Args:
        labels: List of labels
        dataset: The dataset that is clustered
        cluster_matrix: Matrix storing distance or similarity values
        converged: a boolean to forward whether the clustering converged

    Returns:
        The updated dataset and the converged-flag
    """
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
    for i, name in enumerate(dataset.cluster_names):
        new_cluster = labels[i]
        if new_cluster not in new_cluster_weights:
            new_cluster_weights[new_cluster] = 0
        new_cluster_weights[new_cluster] += dataset.cluster_weights[name]

    LOGGER.info(f"Reduced number of clusters to {len(new_cluster_names)}.")

    dataset.cluster_names = new_cluster_names
    dataset.cluster_map = new_cluster_map
    dataset.cluster_weights = new_cluster_weights

    # store the matrix at the correct field and set the main diagonal to either 1 or 0 depending on dist or sim
    if dataset.cluster_similarity is not None:
        dataset.cluster_similarity = np.maximum(new_cluster_matrix, np.eye(len(new_cluster_matrix)))
    else:
        dataset.cluster_distance = np.minimum(new_cluster_matrix, 1 - np.eye(len(new_cluster_matrix)))

    return dataset, converged


def force_clustering(dataset: DataSet) -> DataSet:
    """
    Enforce a clustering to reduce the number of clusters to a reasonable amount. This is only done if the other
    clustering algorithms did not detect any reasonable similarity or distance in the dataset. The cluster assignment
    is fully random and distributes the samples equally into the samples (as far as possible given already detected
    clusters).

    Args:
        dataset: The dataset to be clustered

    Returns:
        The clustered dataset
    """
    LOGGER.info(f"Enforce clustering from {len(dataset.cluster_names)} clusters to {MAX_CLUSTERS} clusters")

    # define the list of clusters, the current sizes of the individual clusters, and how big they shall become
    labels = []
    sizes = np.zeros(MAX_CLUSTERS)
    fraction = sum(dataset.cluster_weights.values()) / MAX_CLUSTERS

    for name, weight in sorted(dataset.cluster_weights.items(), key=lambda x: x[1], reverse=True):
        assigned = False
        overlap = np.zeros(MAX_CLUSTERS)
        for i in range(MAX_CLUSTERS):
            # if the entity can be assigned to cluster i without exceeding the target size, assign it there, ...
            if sizes[i] + weight < fraction:
                labels.append(i)
                sizes[i] += weight
                assigned = True
                break
            # ... otherwise store the absolute overhead
            else:
                overlap[i] += weight + sizes[i] - fraction

        # if the entity hasn't been assigned, assign to the cluster where it causes the smallest overhead
        if not assigned:
            idx = overlap.argmin()
            labels.append(idx)
            sizes[idx] += weight

    # cluster the dataset based on the list of new clusters and return
    matrix = dataset.cluster_similarity if dataset.cluster_similarity is not None else dataset.cluster_distance
    return labels2clusters(labels, dataset, matrix, True)[0]


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
    return {n: cluster_split[c] for n, c in name_cluster.items()}


def reverse_interaction_clustering(
        inter_split: Dict[Tuple[str, str], str],
        e_name_cluster_map: Dict[str, str],
        f_name_cluster_map: Dict[str, str],
        inter: List[Tuple[str, str]]
) -> Dict[str, str]:
    """
    Revert the clustering of interactions.

    Args:
        inter_split: The assignment of each cell of an interaction matrix to a split based on the cluster names.
        e_name_cluster_map: Mapping from sample names to cluster names for the e-dataset.
        f_name_cluster_map: Mapping from sample names to cluster names for the f-dataset.
        inter: List of interactions as pairs of entity names from each dataset.

    Returns:

    """
    return {
        (e_name, f_name): inter_split[e_name_cluster_map[e_name], f_name_cluster_map[f_name]]
        for e_name, f_name in inter
    }
