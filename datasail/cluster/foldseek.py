import logging
import os
import shutil
from typing import Tuple, List, Dict

import numpy as np

from datasail.reader.utils import DataSet


def run_foldseek(dataset: DataSet, log_dir: str) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    """
    Run FoldSeek to cluster the proteins based on their structure.

    Args:
        dataset: DataSet holding all information on the dta to be clustered
        log_dir: Absolute path to the directory to store all the logs in

    Returns:
        A tuple containing
          - the names of the clusters (cluster representatives)
          - the mapping from cluster members to the cluster names (cluster representatives)
          - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
    """
    log_name = os.path.join(log_dir, f"{dataset.get_name()}_foldseek.log")
    cmd = f"mkdir fs && " \
          f"cd fs && " \
          f"foldseek easy-search {os.path.join('..', dataset.location)} {os.path.join('..', dataset.location)} " \
          f"aln.m8 tmp --format-output 'query,target,fident' " \
          f">{log_name}"

    if os.path.exists("fs"):
        cmd = "rm -rf fs && " + cmd

    logging.info("Start FoldSeek clustering")

    logging.info(cmd)
    os.system(cmd)

    namap = dict((n, i) for i, n in enumerate(dataset.names))
    cluster_sim = np.zeros((len(dataset.names), len(dataset.names)))
    with open("fs/aln.m8", "r") as data:
        for line in data.readlines():
            q1, q2, sim = line.strip().split("\t")[:3]
            if "_" in q1 and "." in q1 and q1.rindex("_") > q1.index("."):
                q1 = "_".join(q1.split("_")[:-1])
            if "_" in q2 and "." in q2 and q2.rindex("_") > q2.index("."):
                q2 = "_".join(q2.split("_")[:-1])
            q1 = q1.replace(".pdb", "")
            q2 = q2.replace(".pdb", "")
            cluster_sim[namap[q1], namap[q2]] = sim
            cluster_sim[namap[q2], namap[q1]] = sim

    shutil.rmtree("fs")

    return dataset.names, dict((n, n) for n in dataset.names), cluster_sim
