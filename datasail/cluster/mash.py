import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from datasail.parsers import MultiYAMLParser
from datasail.reader.utils import DataSet
from datasail.settings import LOGGER, INSTALLED, MASH


def run_mash(dataset: DataSet, threads: int = 1, log_dir: Optional[Path] = None) -> None:
    """
    Run MASH on the provided dataset.

    Args:
        dataset: Dataset to run MASH for
        threads: number of threads to use for one CD-HIT run
        log_dir: Filepath to store the output of MASH to
    """
    if not INSTALLED[MASH]:
        raise ValueError("MASH is not installed.")
    parser = MultiYAMLParser(MASH)
    sketch_args = parser.get_user_arguments(dataset.args, [], 0)
    dist_args = parser.get_user_arguments(dataset.args, [], 1)

    results_folder = Path("mash_results")

    tmp = Path("tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    for name, filepath in dataset.data.items():
        shutil.copy(filepath, tmp)

    cmd = f"mkdir {results_folder} && " \
          f"cd mash_results && " \
          f"mash sketch -p {threads} -o ./cluster ../tmp/*.fna {sketch_args} && " \
          f"mash dist -p {threads} {dist_args} -t cluster.msh cluster.msh > cluster.tsv && " \
          f"rm -rf ../tmp"

    if log_dir is None:
        cmd += "> /dev/null 2>&1"
    else:
        cmd += f"> {(log_dir / f'{dataset.get_name()}_mash.log').resolve()}"

    if results_folder.exists():
        cmd = f"rm -rf {results_folder} && " + cmd

    LOGGER.info("Start MASH clustering")
    LOGGER.info(cmd)
    os.system(cmd)

    if not (results_folder / "cluster.tsv").exists():
        raise ValueError("Something went wrong with MASH. The output file does not exist.")

    dataset.cluster_map = dict((n, n) for n in dataset.names)
    dataset.cluster_distance = read_mash_tsv(results_folder / "cluster.tsv", len(dataset.names))
    dataset.cluster_names = dataset.names

    shutil.rmtree(results_folder, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)


def read_mash_tsv(filename: Path, num_entities: int) -> np.ndarray:
    """
    Read in the TSV file with pairwise distances produces by MASH.

    Args:
        filename: Filename of the file to read from
        num_entities: Number of entities in the set

    Returns:
        Symmetric 2D-numpy array storing pairwise distances
    """
    output = np.zeros((num_entities, num_entities))
    with open(filename, "r") as data:
        for i, line in enumerate(data.readlines()[1:]):
            for j, val in enumerate(line.strip().split("\t")[1:]):
                output[i, j] = float(val)
    return output
