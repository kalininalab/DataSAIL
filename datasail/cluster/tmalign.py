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
          f"cd tmalign && " \
          f"TMalign "

    if os.path.exists("mmseqs_results"):
        cmd = "rm -rf mmseqs_results && " + cmd

    os.system(cmd)

    cluster_names, cluster_map, cluster_sim = [], dict(), np.ndarray((1, 1))
    shutil.rmtree("mmseqs_results")

    return cluster_names, cluster_map, cluster_sim


def read_tmalign_file(tmalign_file: str) -> Dict[str, str]:
    """
    Read clusters from TM-align output into map from cluster members to cluster representatives (cluster names).

    Args:
        tmalign_file (str): Filepath of file containing the mapping information

    Returns:
        Map from cluster--members to cluster-representatives (cluster-names)
    """
    mapping = {}
    # TODO: Read output
    return mapping
