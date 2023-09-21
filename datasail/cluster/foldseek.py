import os
import shutil
from typing import Tuple, List, Dict, Optional

import numpy as np

from datasail.parsers import MultiYAMLParser
from datasail.reader.utils import DataSet
from datasail.settings import LOGGER, FOLDSEEK, INSTALLED


def run_foldseek(
        dataset: DataSet,
        threads: int = 1,
        log_dir: Optional[str] = None,
) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    """
    Run FoldSeek to cluster the proteins based on their structure.

    Args:
        dataset: DataSet holding all information on the dta to be clustered
        threads: number of threads to use for one CD-HIT run
        log_dir: Absolute path to the directory to store all the logs in

    Returns:
        A tuple containing
          - the names of the clusters (cluster representatives)
          - the mapping from cluster members to the cluster names (cluster representatives)
          - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
    """
    if not INSTALLED[FOLDSEEK]:
        raise ValueError("Foldseek is not installed.")
    user_args = MultiYAMLParser(FOLDSEEK).get_user_arguments(dataset.args, [])

    results_folder = "fs_results"

    cmd = f"mkdir {results_folder} && " \
          f"cd {results_folder} && " \
          f"foldseek " \
          f"easy-search " \
          f"{os.path.join('..', dataset.location)} " \
          f"{os.path.join('..', dataset.location)} " \
          f"aln.m8 tmp " \
          f"--format-output 'query,target,fident' " \
          f"-e inf " \
          f"--threads {threads} " \
          f"{user_args}"

    if log_dir is None:
        cmd += "> /dev/null 2>&1"
    else:
        cmd += f"> {os.path.join(log_dir, f'{dataset.get_name()}_foldseek.log')}"

    if os.path.exists(results_folder):
        cmd = f"rm -rf {results_folder} && " + cmd

    LOGGER.info("Start FoldSeek clustering")
    LOGGER.info(cmd)
    os.system(cmd)

    if not os.path.isfile(f"{results_folder}/aln.m8"):
        raise ValueError("Something went wrong with foldseek. The output file does not exist.")

    namap = dict((n, i) for i, n in enumerate(dataset.names))
    cluster_sim = np.zeros((len(dataset.names), len(dataset.names)))
    with open(f"{results_folder}/aln.m8", "r") as data:
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

    shutil.rmtree(results_folder)

    return dataset.names, dict((n, n) for n in dataset.names), cluster_sim
