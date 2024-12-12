import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from datasail.parsers import MultiYAMLParser
from datasail.reader.utils import DataSet
from datasail.settings import LOGGER, FOLDSEEK, INSTALLED


def run_foldseek(dataset: DataSet, threads: int = 1, log_dir: Optional[Path] = None) -> None:
    """
    Run FoldSeek to cluster the proteins based on their structure.

    Args:
        dataset: DataSet holding all information on the dta to be clustered
        threads: number of threads to use for one CD-HIT run
        log_dir: Absolute path to the directory to store all the logs in
    """
    if not INSTALLED[FOLDSEEK]:
        raise ValueError("Foldseek is not installed.")
    user_args = MultiYAMLParser(FOLDSEEK).get_user_arguments(dataset.args, [])

    results_folder = Path("fs_results")

    tmp = Path("tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    for name, filepath in dataset.data.items():
        shutil.copy(filepath, tmp)

    cmd = f"mkdir {results_folder} && " \
          f"cd {results_folder} && " \
          f"foldseek " \
          f"easy-search " \
          f"../tmp " \
          f"../tmp " \
          f"aln.m8 " \
          f"tmp " \
          f"--format-output 'query,target,fident' " \
          f"-e inf " \
          f"--threads {threads} " \
          f"{user_args}"  # && " \
          # f"rm -rf ../tmp"

    if log_dir is None:
        cmd += "> /dev/null 2>&1"
    else:
        cmd += f"> {(log_dir / f'{dataset.get_name()}_foldseek.log').resolve()}"

    if results_folder.exists():
        cmd = f"rm -rf {results_folder} && " + cmd

    LOGGER.info("Start FoldSeek clustering")
    LOGGER.info(cmd)
    os.system(cmd)

    if not (results_folder / "aln.m8").exists():
        raise ValueError("Something went wrong with foldseek. The output file does not exist.")

    namap = dict((n, i) for i, n in enumerate(dataset.names))
    cluster_sim = np.zeros((len(dataset.names), len(dataset.names)), dtype=int)
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

    shutil.rmtree(results_folder, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)

    dataset.cluster_names = dataset.names
    dataset.cluster_map = dict((n, n) for n in dataset.names)
    dataset.cluster_similarity = cluster_sim
