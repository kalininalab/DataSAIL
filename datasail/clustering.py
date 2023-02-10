import logging
from typing import Dict, Tuple, List, Optional, Union

import numpy as np
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering


def cluster_interactions(
        inter: List[Tuple[str, str]],
        drug_cluster_map: Dict[str, Union[str, int]],
        drug_cluster_names: List[Union[str, int]],
        prot_cluster_map: Dict[str, Union[str, int]],
        prot_cluster_names: List[Union[str, int]],
) -> np.ndarray:
    """
    Based on cluster information, count interactions in an interaction matrix between the individual clusters.

    Args:
        inter: List of pairs representing interactions
        drug_cluster_map: Mapping from drug names to their cluster names
        drug_cluster_names: List of custer names
        prot_cluster_map: Mapping from protein names to their cluster names
        prot_cluster_names: List of custer names

    Returns:

    """
    drug_mapping = dict((y, x) for x, y in enumerate(drug_cluster_names))
    prot_mapping = dict((y, x) for x, y in enumerate(prot_cluster_names))

    output = np.zeros((len(drug_cluster_names), len(prot_cluster_names)))
    for drug, protein in inter:
        output[drug_mapping[drug_cluster_map[drug]]][prot_mapping[prot_cluster_map[protein]]] += 1
        # output[prot_mapping[prot_cluster_map[protein]]][drug_mapping[drug_cluster_map[drug]]] += 1

    return output


def cluster(
        similarity: Optional[Union[np.ndarray, str]],
        distance: Optional[Union[np.ndarray, str]],
        molecules: Optional[Dict[str, str]],
        weights: Dict[str, float],
        **kwargs
) -> Tuple[
    Optional[List[Union[str, int]]],
    Optional[Dict[str, str]],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[Dict[str, float]]
]:
    """
    Cluster molecules based on a similarity or distance metric.

    Args:
        similarity: string of method to compute the similarity (mutually exclusive with similarity)
        distance: string of the method to compute the distance (mutually exclusive with similarity)
        molecules: mapping from molecule names to molecule description (fasta, PDB, SMILES, ...)
        weights: mapping from molecule names to weighting
        **kwargs: arguments given to the program in general

    Returns:
        A tuple consisting of
          - The names of the current clusters
          - The mapping from cluster names to cluster representatives
          - Symmetric matrix of pairwise similarities between the current clusters
          - Symmetric matrix of pairwise similarities between the current clusters
          - Mapping from current clusters to their weights
    """
    cluster_similarity, cluster_distance = None, None
    if isinstance(similarity, str):  # compute the similarity
        cluster_names, cluster_map, cluster_similarity, cluster_weights = \
            similarity_clustering(molecules, similarity, **kwargs)
    if isinstance(distance, str):  # compute the distance
        cluster_names, cluster_map, cluster_distance, cluster_weights = \
            distance_clustering(molecules, distance, **kwargs)
    elif similarity is not None or distance is not None:  # if the similarity/distance is already given, store it
        cluster_names = list(molecules.keys())
        cluster_map = dict([(d, d) for d, _ in molecules.items()])
        cluster_similarity = similarity
        cluster_distance = distance
        cluster_weights = weights
    else:
        cluster_names, cluster_map, cluster_weights = None, None, None

    if cluster_names is None:
        return cluster_names, cluster_map, cluster_similarity, cluster_distance, cluster_weights

    # if there are too many clusters, reduce their number based on some cluster algorithms.
    num_old_cluster = len(cluster_names) + 1
    while 100 < len(cluster_names) < num_old_cluster:
        num_old_cluster = len(cluster_names)
        cluster_names, cluster_map, cluster_similarity, cluster_distance, cluster_weights = additional_clustering(
            cluster_names, cluster_map, cluster_similarity, cluster_distance, cluster_weights
        )

    return cluster_names, cluster_map, cluster_similarity, cluster_distance, cluster_weights


def similarity_clustering(mols: Optional, cluster_method: str, **kwargs) -> Tuple[
    List[str], Dict[str, str], np.ndarray, Dict[str, float],
]:
    """
    Compute the similarity based cluster based on a cluster method.

    Args:
        mols: mapping from molecule names to molecule description (fasta, PDB, SMILES, ...)
        cluster_method: method to use for cluster

    Returns:
        A tuple consisting of
          - The names of the current clusters
          - The mapping from cluster names to cluster representatives
          - Symmetric matrix of pairwise similarities between the current clusters
          - Symmetric matrix of pairwise similarities between the current clusters
          - Mapping from current clusters to their weights
    """
    if cluster_method.lower() == "wlk":  # run Weisfeiler-Lehman kernel (only for graph data)
        cluster_names, cluster_map, cluster_sim = run_wlk(mols)
    elif cluster_method.lower() == "mmseqs":  # run mmseqs2 on the protein sequences
        cluster_names, cluster_map, cluster_sim = run_mmseqs(**kwargs)
    else:
        raise ValueError("Unknown cluster method.")

    # compute the weights for the clusters
    cluster_weights = {}
    for key, value in cluster_map.items():
        if value not in cluster_weights:
            cluster_weights[key] = 0
        cluster_weights[key] += 1

    # cluster_map maps members to their cluster names
    return cluster_names, cluster_map, cluster_sim, cluster_weights


def distance_clustering(mols: Optional, cluster_method: str, **kwargs) -> Tuple[
    List[str], Dict[str, str], np.ndarray, Dict[str, float],
]:
    """
    Compute the distance based cluster based on a cluster method or a file to extract pairwise distance
    from.

    Args:
        mols: mapping from molecule names to molecule description (fasta, PDB, SMILES, ...)
        cluster_method: method to use for cluster

    Returns:
        A tuple consisting of
          - The names of the current clusters
          - The mapping from cluster names to cluster representatives
          - Symmetric matrix of pairwise similarities between the current clusters
          - Symmetric matrix of pairwise similarities between the current clusters
          - Mapping from current clusters to their weights
    """
    return [], {}, np.array(1), {}


def additional_clustering(
        cluster_names: List[Union[str, int]],
        cluster_map: Dict[str, str],
        cluster_similarity: Optional[np.ndarray],
        cluster_distance: Optional[np.ndarray],
        cluster_weights: Dict[str, float],
) -> Tuple[List[Union[str, int]], Dict[str, str], Optional[np.ndarray], Optional[np.ndarray], Dict[str, float]]:
    """
    Perform additional cluster based on a distance or similarity matrix. This is done to reduce the number of
    clusters and to speed up the further splitting steps.

    Args:
        cluster_names: The names of the current clusters
        cluster_map: The mapping from cluster names to cluster representatives
        cluster_similarity: Symmetric matrix of pairwise similarities between the current clusters
        cluster_distance: Symmetric matrix of pairwise similarities between the current clusters
        cluster_weights: Mapping from current clusters to their weights

    Returns:
        A tuple consisting of
          - The names of the current clusters
          - The mapping from cluster names to cluster representatives
          - Symmetric matrix of pairwise similarities between the current clusters
          - Symmetric matrix of pairwise similarities between the current clusters
          - Mapping from current clusters to their weights
    """
    logging.info(f"Cluster {len(cluster_names)} items based on {'similarities' if cluster_similarity is not None else 'distances'}")
    # set up the cluster algorithm for similarity or distance based cluster w/o specifying the number of clusters
    if cluster_similarity is not None:
        cluster_matrix = np.array(cluster_similarity, dtype=float)
        ca = AffinityPropagation(affinity='precomputed', random_state=42)
    else:
        cluster_matrix = np.array(cluster_distance, dtype=float)
        ca = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='average',
            distance_threshold=np.average(cluster_distance) * 0.9,
            # connectivity=np.asarray(cluster_matrix < np.average(cluster_distance) * 0.9, dtype=int),
        )
        logging.info(
            f"Clustering based on distances. "
            f"Distances above {np.average(cluster_distance) * 0.9} cannot end up in same cluster."
        )

    # cluster the clusters into new, fewer, and bigger clusters
    labels = ca.fit_predict(cluster_matrix)

    # extract the names of the new clusters and compute a mapping from the element names to the clusters
    old_cluster_map = dict((y, x) for x, y in enumerate(cluster_names))
    new_cluster_names = list(np.unique(labels))
    new_cluster_map = dict((n, labels[old_cluster_map[c]]) for n, c in cluster_map.items())

    # compute the distance or similarity matrix for the new clusters as the average sim/dist between their members
    new_cluster_matrix = np.zeros((len(new_cluster_names), len(new_cluster_names)))
    cluster_count = np.zeros((len(new_cluster_names), len(new_cluster_names)))
    for i in range(len(cluster_names)):
        for j in range(i + 1, len(cluster_names)):
            if labels[i] != labels[j]:
                new_cluster_matrix[labels[i], labels[j]] += cluster_matrix[i, j]
                cluster_count[labels[i], labels[j]] += 1

                new_cluster_matrix[labels[j], labels[i]] += cluster_matrix[i, j]
                cluster_count[labels[j], labels[i]] += 1
    new_cluster_matrix /= (cluster_count + np.eye(max(labels) + 1))

    # compute the mapping of new clusters to their weights as the sum of their members weights
    new_cluster_weights = {}
    for name in list(cluster_map.keys()):
        new_cluster = new_cluster_map[name]
        if new_cluster not in new_cluster_weights:
            new_cluster_weights[new_cluster] = 0
        new_cluster_weights[new_cluster] += cluster_weights[cluster_map[name]]

    logging.info(f"Reduced number of clusters to {len(new_cluster_names)}.")

    if cluster_similarity is not None:
        return new_cluster_names, new_cluster_map, new_cluster_matrix, None, new_cluster_weights
    return new_cluster_names, new_cluster_map, None, new_cluster_matrix, new_cluster_weights


def reverse_clustering(cluster_split, name_cluster):
    output = {}
    for n, c in name_cluster.items():
        output[n] = cluster_split[c]
    return output






