import os
import shutil
from typing import Dict, Tuple, List

import numpy as np


def run_mmseqs(filepath: str) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    """
    Run mmseqs in the commandline and read in the results into clusters.

    Args:
        filepath: Filepath to the FASTA file storing the sequences to cluster

    Returns:
        A tuple containing
          - the names of the clusters (cluster representatives)
          - the mapping from cluster members to the cluster names (cluster representatives)
          - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
    """
    cmd = f"cd mmseqs_results && " \
          f"mmseqs " \
          f"easy-linclust " \
          f"{os.path.join('..', filepath)} " \
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
