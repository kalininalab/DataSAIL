import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import shutil

import numpy as np

from datasail.cluster.utils import cluster_param_binary_search
from datasail.parsers import MultiYAMLParser
from datasail.reader.utils import DataSet
from datasail.settings import LOGGER, MMSEQS2, INSTALLED


def run_mmseqs(dataset: DataSet, threads: int, log_dir: Optional[Path]) -> None:
    """
    Run mmseqs in the commandline and read in the results into clusters.

    Args:
        dataset: DataSet holding all information on the dta to be clustered
        threads: number of threads to use for one CD-HIT run
        log_dir: Absolute path to the directory to store all the logs in
    """
    if not INSTALLED[MMSEQS2]:
        raise ValueError("MMseqs is not installed.")

    user_args = MultiYAMLParser(MMSEQS2).get_user_arguments(dataset.args, ["c"])
    optim_vals = (dataset.args.c,)  # values to be optimized

    dataset.cluster_names, dataset.cluster_map, dataset.cluster_similarity = cluster_param_binary_search(
        dataset,
        optim_vals,
        (0.1,),
        (1,),
        user_args,
        threads,
        mmseqs_trial,
        lambda x: f"-c {x[0]}",
        lambda x, y: ((x[0] + y[0]) / 2,),
        log_dir,
    )


def mmseqs_trial(
        dataset: DataSet,
        tune_args: Tuple,
        user_args: str,
        threads: int = 1,
        log_file: Optional[Path] = None
) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    """
    Run MMseqs2 on the dataset with the given sequence similarity defined by add_args.

    Args:
        dataset: Dataset to run the clustering for
        tune_args:
        user_args: Additional arguments specifying the sequence similarity parameter
        threads: number of threads to use for one CD-HIT run
        log_file: Filepath to log the output to

    Returns:
        A tuple containing
          - the names of the clusters (cluster representatives)
          - the mapping from cluster members to the cluster names (cluster representatives)
          - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
    """

    results_folder = Path("mmseqs_results")

    with open("mmseqs.fasta", "w") as out:
        for name, seq in dataset.data.items():
            out.write(f">{name}\n{seq}\n")

    cmd = f"mkdir {results_folder} && " \
          f"cd {results_folder} && " \
          f"mmseqs " \
          f"easy-cluster " \
          f"../mmseqs.fasta " \
          f"mmseqs_out " \
          f"mmseqs_tmp " \
          f"--threads {threads} " \
          f"{tune_args} " \
          f"{user_args} && " \
          f"rm ../mmseqs.fasta"

    if log_file is None:
        cmd += "> /dev/null 2>&1"
    else:
        cmd += f"> {log_file.resolve()}"

    if results_folder.exists():
        cmd = f"rm -rf {results_folder} && " + cmd

    LOGGER.info(cmd)
    os.system(cmd)

    if not (results_folder / "mmseqs_out_cluster.tsv").is_file():
        raise ValueError("Something went wrong with mmseqs. The output file does not exist.")

    cluster_map = get_mmseqs_map(results_folder / "mmseqs_out_cluster.tsv")
    cluster_names = list(set(cluster_map.values()))
    cluster_sim = np.ones((len(cluster_names), len(cluster_names)))

    shutil.rmtree(results_folder, ignore_errors=True)

    return cluster_names, cluster_map, cluster_sim


def get_mmseqs_map(cluster_file: Path) -> Dict[str, str]:
    """
    Read clusters from mmseqs output into map from cluster members to cluster representatives (cluster names).

    Args:
        cluster_file (str): Filepath of file containing the mapping information

    Returns:
        Map from cluster--members to cluster-representatives (cluster-names)
    """
    mapping = {}
    with open(cluster_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0 and "\t" not in line:
                return get_mmseqs_map_old(cluster_file)

            words = line.strip().split('\t')
            if len(words) != 2:
                continue
            cluster_head, cluster_member = words
            mapping[cluster_member] = cluster_head
    return mapping


def get_mmseqs_map_old(cluster_file: Path) -> Dict[str, str]:
    """
    This is a helper method for get_mmseqs_map that is necessary when DataSAIL is run on Windows and in a Python3.8
    build. In this case, MMseqs struggles with different linebreaks of Linux and Windows.

    Args:
        cluster_file (str): Filepath of file containing the mapping information

    Returns:
        Map from cluster--members to cluster-representatives (cluster-names)
    """
    mapping = {}
    rep = ""
    # The file basically contains \n\t-separated values
    with open(cluster_file, "r") as f:
        for line in f.readlines():
            if rep == "":
                rep = line.strip()
            else:
                mapping[rep] = line.strip()
                rep = ""
    return mapping
