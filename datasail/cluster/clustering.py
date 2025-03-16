from collections import defaultdict
from typing import Dict, Tuple, List, Union, Optional, Literal

import numpy as np
import sklearn
from sklearn.cluster import AgglomerativeClustering, SpectralClustering

from datasail.cluster.caching import load_from_cache, store_to_cache
from datasail.cluster.cdhit import run_cdhit
from datasail.cluster.cdhit_est import run_cdhit_est
from datasail.cluster.diamond import run_diamond
from datasail.cluster.ecfp import run_ecfp
from datasail.cluster.foldseek import run_foldseek
from datasail.cluster.mash import run_mash
from datasail.cluster.mmseqs2 import run_mmseqs
from datasail.cluster.mmseqspp import run_mmseqspp
from datasail.cluster.vectors import run_vector
from datasail.cluster.utils import heatmap
from datasail.cluster.wlk import run_wlk
from datasail.reader.utils import DataSet
from datasail.report import whatever
from datasail.settings import LOGGER, KW_THREADS, KW_LOGDIR, KW_OUTDIR, WLK, MMSEQS, MMSEQS2, MMSEQSPP, \
    FOLDSEEK, CDHIT, CDHIT_EST, ECFP, DIAMOND,TANIMOTO, KW_LINKAGE


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
        similarity_clustering(dataset, kwargs[KW_THREADS], kwargs[KW_LOGDIR])

    elif isinstance(dataset.distance, str):  # compute the distance
        distance_clustering(dataset, kwargs[KW_THREADS], kwargs[KW_LOGDIR])

    # if the similarity/distance is already given, store it
    elif isinstance(dataset.similarity, np.ndarray) or isinstance(dataset.distance, np.ndarray):
        dataset.cluster_names = dataset.names
        dataset.cluster_map = dict([(d, d) for d in dataset.names])
        dataset.cluster_similarity = dataset.similarity
        dataset.cluster_distance = dataset.distance
        dataset.cluster_weights = dataset.weights
        dataset.cluster_stratification = {name: dataset.strat2oh(name) for name in dataset.cluster_names}

    if dataset.cluster_names is None:  # No clustering to do?!
        return dataset

    # if there are too many clusters, reduce their number based on some cluster algorithms.
    if any(isinstance(m, np.ndarray) for m in
           [dataset.similarity, dataset.distance, dataset.cluster_similarity, dataset.cluster_distance]):
        num_old_cluster = len(dataset.cluster_names) + 1
        while dataset.num_clusters < len(dataset.cluster_names) < num_old_cluster:
            num_old_cluster = len(dataset.cluster_names)
            dataset = additional_clustering(dataset, dataset.num_clusters, kwargs[KW_LINKAGE])

        if isinstance(dataset.similarity, np.ndarray) or isinstance(dataset.distance, np.ndarray):
            whatever(dataset.names, dataset.cluster_map, dataset.distance, dataset.similarity)
            metric = dataset.similarity if dataset.similarity is not None else dataset.distance
            form = "similarity" if dataset.similarity is not None else "distance"
            if kwargs[KW_OUTDIR] is not None:
                heatmap(metric, kwargs[KW_OUTDIR] / (dataset.get_name() + f"_{form}.png"))

    if len(dataset.cluster_names) > dataset.num_clusters:
        dataset = force_clustering(dataset, kwargs[KW_LINKAGE])

    # store_to_cache(dataset, **kwargs)

    return dataset


def similarity_clustering(dataset: DataSet, threads: int = 1, log_dir: Optional[str] = None) -> None:
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
    if dataset.similarity.lower() == WLK:
        run_wlk(dataset)
    elif dataset.similarity.lower() == CDHIT:
        run_cdhit(dataset, threads, log_dir)
    elif dataset.similarity.lower() == CDHIT_EST:
        run_cdhit_est(dataset, threads, log_dir)
    elif dataset.similarity.lower() == DIAMOND:
        run_diamond(dataset, threads, log_dir)
    elif dataset.similarity.lower() == ECFP:
        run_ecfp(dataset)
    elif dataset.similarity.lower() == FOLDSEEK:
        run_foldseek(dataset, threads, log_dir)
    elif dataset.similarity.lower() in [MMSEQS, MMSEQS2]:
        run_mmseqs(dataset, threads, log_dir)
    elif dataset.similarity.lower() == MMSEQSPP:
        run_mmseqspp(dataset, threads, log_dir)
    elif dataset.similarity.lower() == TANIMOTO:
        run_vector(dataset)
    else:
        raise ValueError(f"Unknown cluster method: {dataset.similarity}")

    finish_clustering(dataset)


def distance_clustering(dataset: DataSet, threads: int = 1, log_dir: Optional[str] = None) -> None:
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
        run_mash(dataset, threads, log_dir)
    else:
        raise ValueError(f"Unknown cluster method: {dataset.distance}")

    finish_clustering(dataset)


def finish_clustering(dataset: DataSet) -> None:
    """
    Finish clustering by computing the weights of the clusters and the stratification of the clusters.

    Args:
        dataset: The dataset to finish clustering on
    """
    # compute the weights and the stratification for the clusters
    dataset.cluster_weights = defaultdict(float)
    for key, value in dataset.cluster_map.items():
        dataset.cluster_weights[value] += dataset.weights[key]

    if dataset.stratification is not None and len(dataset.classes) > 1:
        dataset.cluster_stratification = defaultdict(lambda: np.zeros(len(dataset.classes)))
        for key, value in dataset.cluster_map.items():
            dataset.cluster_stratification[value] += dataset.strat2oh(name=key)
    else:
        dataset.cluster_stratification = None


def additional_clustering(
        dataset: DataSet,
        n_clusters: int,
        linkage: Literal["average", "single", "complete"],
) -> DataSet:
    """
    Perform additional cluster based on a distance or similarity matrix. This is done to reduce the number of
    clusters and to speed up the further splitting steps.

    Args:
        dataset: DataSet to perform additional clustering on
        n_clusters: Number of clusters to reduce to
        linkage: List of linkage methods to use for the agglomerative clustering

    Returns:
        The dataset with updated clusters and a bool flag indicating convergence of the used clustering algorithm
    """
    LOGGER.info(f"Cluster {len(dataset.cluster_names)} items based on "
                f"{'similarities' if dataset.cluster_similarity is not None else 'distances'}")
    # set up the cluster algorithm for similarity or distance based cluster w/o specifying the number of clusters
    if dataset.cluster_similarity is not None:
        cluster_matrix = np.array(dataset.cluster_similarity, dtype=float)
        ca = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=42,
        )
    else:
        cluster_matrix = np.array(dataset.cluster_distance, dtype=float)
        if sklearn.__version__ < '1.2':
            ca = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                linkage=linkage,
            )
        else:
            ca = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="precomputed",
                linkage=linkage,
            )
        LOGGER.info(
            f"Clustering based on distances. "
            f"Distances above {np.average(dataset.cluster_distance) * 0.9} cannot end up in same cluster."
        )
    # cluster the clusters into new, fewer, and bigger clusters
    labels = ca.fit_predict(cluster_matrix)
    LOGGER.info("Clustering finished")
    return labels2clusters(labels, dataset, cluster_matrix, linkage)


def labels2clusters(
        labels: Union[List, np.ndarray],
        dataset: DataSet,
        cluster_matrix: np.ndarray,
        linkage: Literal["average", "single", "complete"],
) -> DataSet:
    """
    Convert a list of labels to a clustering and insert it into the dataset. This also updates cluster_weights and
    distance or similarity metrics.

    Args:
        labels: List of labels
        dataset: The dataset that is clustered
        cluster_matrix: Matrix storing distance or similarity values
        linkage: List of linkage methods to use for the agglomerative clustering

    Returns:
        The updated dataset and the converged-flag
    """
    # extract the names of the new clusters and compute a mapping from the element names to the clusters
    old_cluster_map = dict((y, x) for x, y in enumerate(dataset.cluster_names))
    new_cluster_names = list(np.unique(labels))
    new_cluster_map = dict((n, labels[old_cluster_map[c]]) for n, c in dataset.cluster_map.items())

    # compute the distance or similarity matrix for the new clusters as the average sim/dist between their members
    # new_cluster_matrix = np.zeros((len(new_cluster_names), len(new_cluster_names)))
    new_cluster_matrix = [[[] for _ in new_cluster_names] for _ in new_cluster_names]
    for i in range(len(dataset.cluster_names)):
        new_cluster_matrix[labels[i]][labels[i]].append(cluster_matrix[i, i])
        for j in range(i + 1, len(dataset.cluster_names)):
            if labels[i] != labels[j]:
                new_cluster_matrix[labels[i]][labels[j]].append(cluster_matrix[i, j])
                new_cluster_matrix[labels[j]][labels[i]].append(cluster_matrix[j, i])

    links = {"average": np.mean, "single": np.min, "complete": np.max}
    for i in range(len(new_cluster_matrix)):
        for j in range(len(new_cluster_matrix[i])):
            if len(new_cluster_matrix[i][j]) > 0:
                new_cluster_matrix[i][j] = links[linkage](new_cluster_matrix[i][j])
            else:
                new_cluster_matrix[i][j] = 0
    new_cluster_matrix = np.array(new_cluster_matrix)

    # compute the mapping of new clusters to their weights as the sum of their members weights
    new_cluster_weights = {}
    for i, name in enumerate(dataset.cluster_names):
        new_cluster = labels[i]
        if new_cluster not in new_cluster_weights:
            new_cluster_weights[new_cluster] = 0
        new_cluster_weights[new_cluster] += dataset.cluster_weights[name]
    if dataset.cluster_stratification is not None:
        new_cluster_stratification = defaultdict(lambda: np.zeros(len(dataset.classes)))
        for i, name in enumerate(dataset.cluster_names):
            new_cluster = labels[i]
            new_cluster_stratification[new_cluster] += dataset.cluster_stratification[name]
    else:
        new_cluster_stratification = None

    LOGGER.info(f"Reduced number of clusters to {len(new_cluster_names)}.")

    dataset.cluster_names = new_cluster_names
    dataset.cluster_map = new_cluster_map
    dataset.cluster_weights = new_cluster_weights
    dataset.cluster_stratification = new_cluster_stratification

    # store the matrix at the correct field and set the main diagonal to either 1 or 0 depending on dist or sim
    if dataset.cluster_similarity is not None:
        dataset.cluster_similarity = np.maximum(new_cluster_matrix, np.eye(len(new_cluster_matrix)))
    else:
        dataset.cluster_distance = np.minimum(new_cluster_matrix, 1 - np.eye(len(new_cluster_matrix)))

    return dataset


def force_clustering(dataset: DataSet, linkage: Literal["average", "single", "complete"] = "average") -> DataSet:
    """
    Enforce a clustering to reduce the number of clusters to a reasonable amount. This is only done if the other
    clustering algorithms did not detect any reasonable similarity or distance in the dataset. The cluster assignment
    is fully random and distributes the samples equally into the samples (as far as possible given already detected
    clusters).

    Args:
        dataset: The dataset to be clustered
        linkage: List of linkage methods to use for the agglomerative clustering

    Returns:
        The clustered dataset
    """
    LOGGER.info(f"Enforce clustering from {len(dataset.cluster_names)} clusters to {dataset.num_clusters} clusters")

    # define the list of clusters, the current sizes of the individual clusters, and how big they shall become
    labels = []
    sizes = np.zeros(dataset.num_clusters, dtype=float)
    fraction = sum(dataset.cluster_weights.values()) / dataset.num_clusters

    for name, weight in sorted(dataset.cluster_weights.items(), key=lambda x: x[1], reverse=True):
        assigned = False
        overlap = np.zeros(dataset.num_clusters)
        for i in range(dataset.num_clusters):
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
    return labels2clusters(labels, dataset, matrix, linkage)


def cluster_interactions(
        inter: List[Tuple[str, str]],
        e_dataset: DataSet,
        f_dataset: DataSet,
) -> np.ndarray:
    """
    Based on cluster information, count interactions in an interaction matrix between the individual clusters.

    Args:
        inter: List of pairs representing interactions
        e_dataset: Dataset of the e-dataset
        f_dataset: Dataset of the f-dataset

    Returns:
        Numpy array of matrix of interactions between two clusters
    """
    e_mapping = dict((y, x) for x, y in enumerate(e_dataset.cluster_names))
    f_mapping = dict((y, x) for x, y in enumerate(f_dataset.cluster_names))

    output = np.zeros((len(e_dataset.cluster_names), len(f_dataset.cluster_names)))
    for e, f in inter:
        e_key = e_mapping[e_dataset.cluster_map[e_dataset.id_map[e]]]
        f_key = f_mapping[f_dataset.cluster_map[f_dataset.id_map[f]]]
        output[e_key][f_key] += 1

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
