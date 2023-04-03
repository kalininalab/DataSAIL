import logging
import os
import shutil
from typing import Dict, Tuple, List

import numpy as np

from datasail.cluster.utils import cluster_param_binary_search
from datasail.parsers import parse_mmseqs_args
from datasail.reader.utils import DataSet


def run_mmseqs(dataset: DataSet, log_dir: str) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    """
    Run mmseqs in the commandline and read in the results into clusters.

    Args:
        dataset: DataSet holding all information on the dta to be clustered
        log_dir: Absolute path to the directory to store all the logs in

    Returns:
        A tuple containing
          - the names of the clusters (cluster representatives)
          - the mapping from cluster members to the cluster names (cluster representatives)
          - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
    """
    args = parse_mmseqs_args(dataset.args)
    vals = (args["seq_id"],)
    logging.info("Starting MMseqs clustering")
    return cluster_param_binary_search(
        dataset,
        vals,
        (0,),
        (1,),
        mmseqs_trial,
        lambda x: f"--min-seq-id {x[0]}",
        lambda x, y: ((x[0] + y[0]) / 2,),
        log_dir,
    )


def mmseqs_trial(dataset, add_args):
    cmd = f"mkdir mmseqs_results && " \
          f"cd mmseqs_results && " \
          f"mmseqs " \
          f"easy-cluster " \
          f"{os.path.join('..', dataset.location)} " \
          f"mmseqs_out " \
          f"mmseqs_tmp " \
          f"--similarity-type 2 " \
          f"--cov-mode 0 " \
          f"-c 0.8 " \
          f"{add_args}"

    if logging.root.level != logging.DEBUG:
        cmd += " >/dev/null 2>&1"

    if os.path.exists("mmseqs_results"):
        cmd = "rm -rf mmseqs_results && " + cmd

    logging.info(cmd)
    os.system(cmd)

    cluster_map = get_mmseqs_map("mmseqs_results/mmseqs_out_cluster.tsv")
    cluster_names = list(set(cluster_map.values()))
    cluster_sim = np.ones((len(cluster_names), len(cluster_names)))
    logging.info(f"MMseqs2 clustered {len(cluster_map)} sequences into {len(cluster_names)} clusters")

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
            mapping[cluster_member] = cluster_head
    return mapping
