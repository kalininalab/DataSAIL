import logging
import os
import shutil
from typing import Dict, Tuple, List

import numpy as np

from datasail.reader.utils import DataSet


def run_tmalign(dataset: DataSet) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    """
    Run TM-align in the commandline and read in the results into clusters.

    Args:
        dataset: DataSet holding all information on the dta to be clustered

    Returns:
        A tuple containing
          - the names of the clusters (cluster representatives)
          - the mapping from cluster members to the cluster names (cluster representatives)
          - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
    """
    cmd = f"mkdir tmalign && " \
          f"cd tmalign"

    if os.path.exists("tmalign"):
        cmd = "rm -rf tmalign && " + cmd

    count, total = 0, len(dataset.names) * (len(dataset.names) - 1) / 2
    for i, name1 in enumerate(dataset.names):
        for name2 in dataset.names[i + 1:]:
            count += 1
            cmd += f" && TMalign {dataset.data[name1]} {dataset.data[name2]} > out_{name1}_{name2}.txt"
            if count % 100 == 0:
                cmd += f" && echo {count} / {total}"

    logging.info("Start TMalign clustering")

    logging.info(cmd[:200])
    os.system(cmd)

    cluster_names, cluster_map, cluster_sim = dataset.names, dict((n, n) for n in dataset.names), read_tmalign_folder(dataset, "tmalign")
    shutil.rmtree("tmalign")

    return cluster_names, cluster_map, cluster_sim


def read_tmalign_folder(dataset: DataSet, tmalign_folder: str) -> np.ndarray:
    """
    Read clusters from TM-align output into map from cluster members to cluster representatives (cluster names).

    Args:
        dataset: Dataset with the data to cluster
        tmalign_folder (str): Folderpath of file containing the mapping information

    Returns:
        Map from cluster-members to cluster-representatives (cluster-names)
    """
    sims = np.ones((len(dataset.names), len(dataset.names)))
    for i, name1 in enumerate(dataset.names):
        for j, name2 in enumerate(dataset.names[i + 1:]):
            sims[i, i + j + 1] = read_tmalign_file(os.path.join(tmalign_folder, f"out_{name1}_{name2}.txt"))
            sims[i, i + j + 1] = sims[i + j + 1, i]
    return sims


def read_tmalign_file(filepath) -> float:
    with open(filepath, "r") as data:
        return sum(map(lambda x: float(x.split(" ")[1]), data.readlines()[17:19])) / 2
