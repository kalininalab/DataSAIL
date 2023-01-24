import os
import shutil
from typing import Dict, Tuple, List, Optional, Union

import numpy as np
from rdkit.Chem import MolFromSmiles
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering

from scala.utils.protein import mol_to_grakel, pdb_to_grakel
from scala.utils.wlk import run_wl_kernel


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
        A symmetric matrix counting interactions between the clusters of proteins and drugs
    """
    drug_mapping = dict((y, x) for x, y in enumerate(drug_cluster_names))
    prot_mapping = dict((y, x) for x, y in enumerate(prot_cluster_names))

    output = np.zeros((len(drug_cluster_names), len(prot_cluster_names)))
    for drug, protein in inter:
        output[drug_mapping[drug_cluster_map[drug]]][prot_mapping[prot_cluster_map[protein]]] += 1
        output[prot_mapping[prot_cluster_map[protein]]][drug_mapping[drug_cluster_map[drug]]] += 1

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

    # if there are too many clusters, reduce their number based on some clustering algorithms.
    num_old_cluster = len(cluster_names) + 1
    while 50 < len(cluster_names) < num_old_cluster:
        num_old_cluster = len(cluster_names)
        cluster_names, cluster_map, cluster_similarity, cluster_distance, cluster_weights = additional_clustering(
            cluster_names, cluster_map, cluster_similarity, cluster_distance, cluster_weights
        )

    return cluster_names, cluster_map, cluster_similarity, cluster_distance, cluster_weights


def similarity_clustering(mols: Optional, cluster_method: str, **kwargs) -> Tuple[
    List[str], Dict[str, str], np.ndarray, Dict[str, float],
]:
    """
    Compute the similarity based clustering based on a clustering method.

    Args:
        mols: mapping from molecule names to molecule description (fasta, PDB, SMILES, ...)
        cluster_method: method to use for clustering

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
        raise ValueError("Unknown clustering method.")

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
    Compute the distance based clustering based on a clustering method or a file to extract pairwise distance
    from.

    Args:
        mols: mapping from molecule names to molecule description (fasta, PDB, SMILES, ...)
        cluster_method: method to use for clustering

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
    Perform additional clustering based on a distance or similarity matrix. This is done to reduce the number of
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
    # setup the clustering algorithm for similarity or distance based clustering w/o specifying the number of clusters
    if cluster_similarity is not None:
        ca = AffinityPropagation(affinity='precomputed', random_state=42)
        cluster_matrix = np.array(cluster_similarity, dtype=float)
    else:
        ca = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='average',
            distance_threshold=np.average(cluster_distance) * 0.9
        )
        cluster_matrix = np.array(cluster_distance, dtype=float)

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
        new_cluster_weights[new_cluster] += cluster_weights[name]

    if cluster_similarity is not None:
        return new_cluster_names, new_cluster_map, new_cluster_matrix, None, new_cluster_weights
    return new_cluster_names, new_cluster_map, None, new_cluster_matrix, new_cluster_weights


def run_wlk(molecules: Dict) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    """
    Run Weisfeiler-Lehman kernel-based clustering on the input. As a result, every molecule will form its own cluster

    Args:
        molecules: A map from molecule identifies to either protein files or SMILES/SMARTS strings

    Returns:
        A tuple containing
          - the names of the clusters (cluster representatives)
          - the mapping from cluster members to the cluster names (cluster representatives)
          - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
    """
    if os.path.isfile(list(molecules.values())[1]):  # read PDB files into grakel graph objects
        cluster_names, graphs = list(zip(*((name, pdb_to_grakel(pdb_path)) for name, pdb_path in molecules.items())))
    else:  # read molecules from SMILES to grakel graph objects
        cluster_names, graphs = list(zip(*((name, mol_to_grakel(MolFromSmiles(mol))) for name, mol in molecules.items())))

    # compute similarity metric and the mapping from element names to cluster names
    cluster_sim = run_wl_kernel(graphs)
    cluster_map = dict((name, name) for name, _ in molecules.items())

    return cluster_names, cluster_map, cluster_sim


def run_mmseqs(**kwargs) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    """
    Run mmseqs in the commandline and read in the results into clusters.

    Args:
        **kwargs: General kwargs to the program

    Returns:
        A tuple containing
          - the names of the clusters (cluster representatives)
          - the mapping from cluster members to the cluster names (cluster representatives)
          - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
    """
    cmd = f"cd mmseqs_results && " \
          f"mmseqs " \
          f"easy-linclust " \
          f"../{kwargs['input']} " \
          f"mmseqs_out " \
          f"mmseqs_tmp " \
          f"--similarity-type 2 " \
          f"--cov-mode 0 " \
          f"-c 1.0 " \
          f"--min-seq-id 0.0"

    if not os.path.exists("mmseqs_results"):
        cmd = "mkdir mmseqs_results && " + cmd

    print(cmd)
    os.system(cmd)

    cluster_map = get_mmseqs_map("mmseqs_results/mmseqs_out_cluster.tsv")
    cluster_sim = np.ones((len(cluster_map), len(cluster_map)))
    cluster_names = list(set(cluster_map.values()))
    shutil.rmtree("mmseqs_results")

    return cluster_names, cluster_map, cluster_sim


def get_mmseqs_map(cluster_file: str) -> Dict[str, str]:
    """
    Read clusters from mmseqs output into map from cluster members to cluster representatives (cluster names).

    Args:
        cluster_file (str): Filepath of file containing the mapping information

    Returns:
        Map from cluster--members to cluster-representatives (cluster-names)
    """
    mapping = {}
    with open(cluster_file, 'r') as f:
        for line in f.readlines():
            words = line.strip().replace('β', 'beta').split('\t')
            if len(words) != 2:
                continue
            cluster_head, cluster_member = words
            mapping[cluster_member] = cluster_member
    return mapping
