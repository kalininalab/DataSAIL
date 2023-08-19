import os
import shutil
from typing import Tuple, List, Dict, Optional

import numpy as np

from datasail.cluster.utils import cluster_param_binary_search
from datasail.parsers import parse_cdhit_args
from datasail.reader.utils import DataSet
from datasail.settings import LOGGER, UNK_LOCATION


def run_cdhit(
        dataset: DataSet,
        threads: int = 1,
        log_dir: Optional[str] = None
) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    """
    Run the CD-HIT tool for protein input.

    Args:
        dataset: DataSet holding all information on the dta to be clustered
        log_dir: Absolute path to the directory to store all the logs in
        threads: number of threads to use for one CD-HIT run

    Returns:
        A tuple containing
          - the names of the clusters (cluster representatives)
          - the mapping from cluster members to the cluster names (cluster representatives)
          - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
    """
    args = parse_cdhit_args(dataset.args)
    vals = (args["c"], args["n"])

    if not os.path.exists(dataset.location):
        with open(dataset.location + ".fasta" if dataset.location.endswith(UNK_LOCATION) else "", "w") as out:
            for idx, seq in dataset.data.items():
                print(">" + idx, file=out)
                print(seq, file=out)
        dataset.location = dataset.location + ".fasta" if dataset.location.endswith(UNK_LOCATION) else ""

    return cluster_param_binary_search(
        dataset,
        vals,
        threads,
        (0.4, 2),
        (1, 5),
        cdhit_trial,
        lambda x: f"-c {x[0]} -n {x[1]} -l {x[1] - 1}",
        lambda x, y: ((x[0] + y[0]) / 2, c2n((x[0] + y[0]) / 2)),
        log_dir,
    )


def cdhit_trial(
        dataset: DataSet,
        add_args: Tuple,
        threads: int = 1,
        log_file: Optional[str] = None
) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    """
    Run CD-HIT on the dataset with the given sequence similarity defined by add_args.

    Args:
        dataset: Dataset to run the clustering for
        add_args: Additional arguments specifying the sequence similarity parameter
        threads: number of threads to use for one CD-HIT run
        log_file: Filepath to log the output to

    Returns:
        A tuple containing
          - the names of the clusters (cluster representatives)
          - the mapping from cluster members to the cluster names (cluster representatives)
          - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
    """
    cmd = f"mkdir cdhit && " \
          f"cd cdhit && " \
          f"cd-hit -i {os.path.join('..', dataset.location)} -o clusters -g 1 {add_args} -d 0 -T {threads} "

    if log_file is None:
        cmd += "> /dev/null 2>&1"
    else:
        cmd += f"> {log_file}"

    if os.path.exists("cdhit"):
        cmd = "rm -rf cdhit && " + cmd

    LOGGER.info(cmd)
    os.system(cmd)

    cluster_map = get_cdhit_map("cdhit/clusters.clstr")
    cluster_names = list(set(cluster_map.values()))
    cluster_sim = np.ones((len(cluster_names), len(cluster_names)))

    shutil.rmtree("cdhit")

    return cluster_names, cluster_map, cluster_sim


def get_cdhit_map(cluster_file: str) -> Dict[str, str]:
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
    if 0.4 <= c < 0.5:
        return 2
    elif 0.5 <= c < 0.6:
        return 3
    elif 0.6 <= c < 0.7:
        return 4
    else:
        return 5
