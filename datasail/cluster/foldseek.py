import os
import shutil
from pathlib import Path
from typing import Optional
import pickle

import numpy as np
from pyarrow import compute, csv
from collections import defaultdict
from tqdm import tqdm

from datasail.parsers import MultiYAMLParser
from datasail.reader.utils import DataSet
from datasail.settings import LOGGER, FOLDSEEK, INSTALLED


def run_foldseek(dataset: DataSet, threads: int = 1, log_dir: Optional[Path] = None, tmp_dir: Optional[Path] = None) -> None:
    """
    Run FoldSeek to cluster the proteins based on their structure.

    Args:
        dataset: DataSet holding all information on the dta to be clustered
        threads: number of threads to use for one CD-HIT run
        log_dir: Absolute path to the directory to store all the logs in
        tmp_dir: Temporary directory to store intermediate files.
    """
    if not INSTALLED[FOLDSEEK]:
        raise ValueError("Foldseek is not installed.")
    user_args = MultiYAMLParser(FOLDSEEK).get_user_arguments(dataset.args, [])

    if tmp_dir is None:
        tmp_dir = Path(".")
    
    results_folder = tmp_dir / Path("fs_results")

    tmp = tmp_dir / Path("fs_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    for name in dataset.names:
        shutil.copy(dataset.data[name], tmp)

    cmd = f"mkdir {str(results_folder)} && " \
          f"cd {str(results_folder)} && " \
          f"foldseek " \
          f"easy-search " \
          f"{str(tmp.resolve())} " \
          f"{str(tmp.resolve())} " \
          f"aln.m8 " \
          f"tmp " \
          f"--format-output 'query,target,fident,qlen,lddt' " \
          f"-e inf " \
          f"--threads {threads} " \
          f"{user_args} " \
          f"--exhaustive-search 1 &&" \
          f"rm -rf ../fs_tmp"

    if log_dir is None:
        cmd += "> /dev/null 2>&1"
    else:
        cmd += f"> {(log_dir / f'{dataset.get_name()}_foldseek.log').resolve()}"

    if results_folder.exists():
        cmd = f"rm -rf {str(results_folder)} && " + cmd

    LOGGER.info("Start FoldSeek clustering")
    LOGGER.info(cmd)
    os.system(cmd)

    if not (results_folder / "aln.m8").exists():
        raise ValueError("Something went wrong with foldseek. The output file does not exist.")

    ds = read_with_pyarrow(results_folder / "aln.m8", dataset.names)
    namap = dict((n, i) for i, n in enumerate(dataset.names))
    filename_ids = {name for name in dataset.names if Path(dataset.data[name]).name == name}
    cluster_sim = np.zeros((len(dataset.names), len(dataset.names)))
    with open(results_folder / "aln.m8", "r") as data:
        for line in data.readlines():
            q1, q2, sim = line.strip().split("\t")[:3]
            # Exact source filenames already identify entities, including underscores.
            if q1 not in filename_ids:
                q1, _ = extract(q1.split("_"), namap)
            if q2 not in filename_ids:
                q2, _ = extract(q2.split("_"), namap)
            cluster_sim[namap[q1], namap[q2]] = sim
            cluster_sim[namap[q2], namap[q1]] = sim
    for i, name1 in enumerate(dataset.names):
        cluster_sim[i, i] = 1
        for j, name2 in enumerate(dataset.names[i + 1:], start=i + 1):
            if name2 in ds[name1]:
                cluster_sim[i, j] = ds[name1][name2][2] / ds[name1][name2][3]
            if name1 in ds[name2]:
                cluster_sim[j, i] = ds[name2][name1][2] / ds[name2][name1][3]
    cluster_sim = (cluster_sim + cluster_sim.T) / 2
    
    shutil.rmtree(results_folder, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)

    dataset.cluster_names = dataset.names
    dataset.cluster_map = dict((n, n) for n in dataset.names)
    dataset.cluster_similarity = cluster_sim


def extract(tmp, names=()):
    """Extract the filename stem and optional chain from a Foldseek PDB label."""
    label = "_".join(tmp)
    # Foldseek 10 omits the .pdb suffix for single-chain structures.
    if label in names and not (".pdb_" in label and label.rsplit(".pdb_", 1)[0] in names):
        return label, "?"
    if label.endswith(".pdb"):
        return label[:-4], "?"
    if ".pdb_" in label:
        return tuple(label.rsplit(".pdb_", 1))
    if len(tmp) == 1:
        return tmp[0], "?"
    else:
        return "_".join(tmp[:-1]), tmp[-1]


def inner_list():
    return ["", "", 0, 0]


def outer_dict():
    return defaultdict(inner_list)


def read_with_pyarrow(file_path, names=()):
    table = csv.read_csv(
        file_path,
        read_options=csv.ReadOptions(use_threads=True, column_names=["qid_chainid", "tid_chainid", "fident", "qlen", "lddt"]),
        parse_options=csv.ParseOptions(delimiter="\t"),
    )

    indices = compute.sort_indices(table, [("lddt", "descending"), ("fident", "descending")])
    ds = defaultdict(outer_dict)
    for idx in tqdm(indices):
        q_id, q_chain = extract(table["qid_chainid"][idx.as_py()].as_py().split("_"), names)
        t_id, t_chain = extract(table["tid_chainid"][idx.as_py()].as_py().split("_"), names)
        record = ds[q_id][t_id]
        if q_chain in record[0] or t_chain in record[1]:
            continue
        fident = table["fident"][idx.as_py()].as_py()
        q_len = table["qlen"][idx.as_py()].as_py()
        record[0] += q_chain
        record[1] += t_chain
        record[2] += fident * q_len
        record[3] += q_len
    return ds

