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

    tmp = Path("fs_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    for name in dataset.names:
        shutil.copy(dataset.data[name], tmp)

    cmd = f"mkdir {results_folder} && " \
          f"cd {results_folder} && " \
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
          f"rm -rf ../tmp"

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

    ds = read_with_pyarrow(f"{results_folder}/aln.m8")
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
    for i, name1 in enumerate(dataset.names):
        cluster_sim[i, i] = 1
        for j, name2 in enumerate(dataset.names[i + 1:]):
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


def extract(tmp):
    if len(tmp) == 1:
        return tmp[0], "?"
    else:
        return "_".join(tmp[:-1]), tmp[-1]


def inner_list():
    return ["", "", 0, 0]


def outer_dict():
    return defaultdict(inner_list)


def read_with_pyarrow(file_path):
    table = csv.read_csv(
        file_path,
        read_options=csv.ReadOptions(use_threads=True, column_names=["qid_chainid", "tid_chainid", "fident", "qlen", "lddt"]),
        parse_options=csv.ParseOptions(delimiter="\t"),
    )

    indices = compute.sort_indices(table, [("lddt", "descending"), ("fident", "descending")])
    ds = defaultdict(outer_dict)
    for idx in tqdm(indices):
        q_id, q_chain = extract(table["qid_chainid"][idx.as_py()].as_py().split("_"))
        t_id, t_chain = extract(table["tid_chainid"][idx.as_py()].as_py().split("_"))
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

