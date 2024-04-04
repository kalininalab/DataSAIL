import os
import shutil
from pathlib import Path

import numpy as np

from datasail.reader.utils import DataSet
from datasail.settings import LOGGER, INSTALLED, TMALIGN


def run_tmalign(dataset: DataSet) -> None:
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
    if not INSTALLED[TMALIGN]:
        raise ValueError("TM-align is not installed.")

    results_folder = Path("tmalign_results")

    cmd = f"mkdir {results_folder} && " \
          f"cd {results_folder}"

    if results_folder.exists():
        cmd = f"rm -rf {results_folder} && " + cmd

    count, total = 0, len(dataset.names) * (len(dataset.names) - 1) / 2
    for i, name1 in enumerate(dataset.names):
        for name2 in dataset.names[i + 1:]:
            count += 1
            cmd += f" && TMalign {dataset.data[name1]} {dataset.data[name2]} > out_{name1}_{name2}.txt"
            if count % 100 == 0:
                cmd += f" && echo {count} / {total}"

    LOGGER.info("Start TMalign clustering")
    LOGGER.info(cmd[:200] + ("..." if len(cmd) > 200 else ""))
    os.system(cmd)

    cluster_names, cluster_map, cluster_sim = dataset.names, dict((n, n) for n in dataset.names), \
        read_tmalign_folder(dataset, results_folder)
    shutil.rmtree(results_folder, ignore_errors=True)

    dataset.cluster_names = cluster_names
    dataset.cluster_map = cluster_map
    dataset.cluster_similarity = cluster_sim


def read_tmalign_folder(dataset: DataSet, tmalign_folder: Path) -> np.ndarray:
    """
    Read clusters from TM-align output into map from cluster members to cluster representatives (cluster names).

    Args:
        dataset: Dataset with the data to cluster
        tmalign_folder: Path to the folder of file containing the mapping information

    Returns:
        Map from cluster-members to cluster-representatives (cluster-names)
    """
    sims = np.ones((len(dataset.names), len(dataset.names)))
    for i, name1 in enumerate(dataset.names):
        for j, name2 in enumerate(dataset.names[i + 1:]):
            sims[i, i + j + 1] = read_tmalign_file(tmalign_folder / f"out_{name1}_{name2}.txt")
            sims[i, i + j + 1] = sims[i + j + 1, i]
    return sims


def read_tmalign_file(filepath: Path) -> float:
    """
    Read one TM-align file holding the output of one tmalign run.

    Args:
        filepath: path to the file to read from

    Returns:
        The average tm-score of both directions of that pairwise alignment
    """
    with open(filepath, "r") as data:
        return sum(map(lambda x: float(x.split(" ")[1]), data.readlines()[17:19])) / 2
