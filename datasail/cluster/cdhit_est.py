import os
import shutil
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np

from datasail.cluster.utils import cluster_param_binary_search
from datasail.parsers import MultiYAMLParser
from datasail.reader.utils import DataSet
from datasail.settings import LOGGER, INSTALLED, CDHIT_EST


def run_cdhit_est(dataset: DataSet, threads: int = 1, log_dir: Optional[Path] = None) -> None:
    """
    Run the CD-HIT-EST tool for DNA or RNA input.

    Args:
        dataset: DataSet holding all information on the dta to be clustered
        log_dir: Absolute path to the directory to store all the logs in
        threads: number of threads to use for one CD-HIT-EST run
    """
    if not INSTALLED[CDHIT_EST]:
        raise ValueError("CD-HIT-EST is not installed.")

    user_args = MultiYAMLParser(CDHIT_EST).get_user_arguments(dataset.args, ["c", "n"])
    vals = (dataset.args.c, dataset.args.n)
    # extract_fasta(dataset)

    dataset.cluster_names, dataset.cluster_map, dataset.cluster_similarity =  cluster_param_binary_search(
        dataset,
        vals,
        (0.8, 5),
        (1, 10),
        user_args,
        threads,
        cdhit_est_trial,
        lambda x: f"-c {x[0]} -n {x[1]} -l {x[1] - 1}",
        lambda x, y: ((x[0] + y[0]) / 2, c2n((x[0] + y[0]) / 2)),
        log_dir,
    )


def cdhit_est_trial(
        dataset: DataSet,
        tune_args: Tuple,
        user_args: str,
        threads: int = 1,
        log_file: Optional[Path] = None
) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    """
    Run CD-HIT on the dataset with the given sequence similarity defined by add_args.

    Args:
        dataset: Dataset to run the clustering for
        tune_args: Tune-able arguments that are set by DataSAIL while finding the optimal clustering.
        user_args: Additional arguments specifying the sequence similarity parameter, those can be set by the user
        threads: number of threads to use for one CD-HIT run
        log_file: Filepath to log the output to

    Returns:
        A tuple containing
          - the names of the clusters (cluster representatives)
          - the mapping from cluster members to the cluster names (cluster representatives)
          - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
    """
    results_folder = Path("cdhit_est_results")

    with open("cdhitest.fasta", "w") as out:
        for name, seq in dataset.data.items():
            out.write(f">{name}\n{seq}\n")

    cmd = f"mkdir {results_folder} && " \
          f"cd {results_folder} && " \
          f"cd-hit-est " \
          f"-i ../cdhitest.fasta " \
          f"-o clusters " \
          f"-d 0 " \
          f"-T {threads} " \
          f"{tune_args} " \
          f"{user_args} && " \
          f"rm ../cdhitest.fasta"

    if log_file is None:
        cmd += "> /dev/null 2>&1"
    else:
        cmd += f"> {log_file.resolve()}"

    if results_folder.exists():
        cmd = f"rm -rf {results_folder} && " + cmd

    LOGGER.info(cmd)
    os.system(cmd)

    if not (results_folder / "clusters.clstr").exists():
        raise ValueError("Something went wrong with cd-hit-est. The output file does not exist.")

    cluster_map = get_cdhit_map(results_folder / "clusters.clstr")
    cluster_names = list(set(cluster_map.values()))
    cluster_sim = np.ones((len(cluster_names), len(cluster_names)))

    shutil.rmtree(results_folder, ignore_errors=True)

    return cluster_names, cluster_map, cluster_sim


def get_cdhit_map(cluster_file: Path) -> Dict[str, str]:
    """
    Read the cluster assignment from the output of CD-HIT.

    Args:
        cluster_file: filepath of the file that stores the cluster assignment.

    Returns:
        A mapping from entities to cluster representatives
    """
    mapping = {}
    rep = ""
    members = []
    with open(cluster_file, "r") as data:
        for line in data.readlines():
            line = line.strip()
            if line[0] == ">":
                if rep != "":
                    mapping[rep] = rep
                    for name in members:
                        mapping[name] = rep
                rep = ""
                members = []
            elif line[-1] == "*":
                rep = line.split(">")[1].split("...")[0]
            else:
                members.append(line.split(">")[1].split("...")[0])
    mapping[rep] = rep
    for name in members:
        mapping[name] = rep
    return mapping


def c2n(c: float):
    """
    For an input value for the C-parameter to CD-HIT, return an appropriate value for the parameter n.

    Args:
        c: c parameter to CD-HIT

    Returns:
        An according value for n based on c
    """
    if 0.8 <= c < 0.85:
        return 5
    elif 0.85 <= c < 0.88:
        return 6
    elif 0.88 <= c < 0.90:
        return 7
    elif 0.9 <= c < 0.93:
        return 8
    elif 0.93 <= c < 0.97:
        return 9
    return 10
