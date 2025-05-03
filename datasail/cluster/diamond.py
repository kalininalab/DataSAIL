import os
import re
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

from datasail.parsers import MultiYAMLParser
from datasail.reader.utils import DataSet
from datasail.settings import INSTALLED, MMSEQS2, DIAMOND, LOGGER


def run_diamond(dataset: DataSet, threads: int = 1, log_dir: Optional[Path] = None) -> None:
    """
    Run Diamond on a dataset in clustering mode.

    Args:
        dataset: Dataset to be clustered.
        threads: Number of threads to be used by the clustering algorithm.
        log_dir: Directory to store the logs.
    """
    if not INSTALLED[MMSEQS2]:
        raise ValueError("MMseqs is not installed.")

    parser = MultiYAMLParser(DIAMOND)
    makedb_args = parser.get_user_arguments(dataset.args, [], 0)
    blastp_args = parser.get_user_arguments(dataset.args, [], 1)

    with open("diamond.fasta", "w") as out:
        for name, seq in dataset.data.items():
            seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', 'G', seq)
            out.write(f">{name}\n{seq}\n")

    result_folder = Path("diamond_results")

    cmd = lambda x: f"mkdir {result_folder} && " \
                    f"cd {result_folder} && " \
                    f"diamond makedb --in ../diamond.fasta --db seqs.dmnd {makedb_args} {x} --threads {threads} && " \
                    f"diamond blastp --db seqs.dmnd --query ../diamond.fasta --out alis.tsv --outfmt 6 qseqid sseqid pident " \
                    f"--threads {threads} {blastp_args} {x} && " \
                    f"rm ../diamond.fasta"

    if log_dir is None:
        cmd = cmd("> /dev/null 2>&1")
    else:
        cmd = cmd(f">> {(Path(log_dir) / f'{dataset.get_name()}_mmseqspp.log').resolve()}")

    if result_folder.exists():
        cmd = f"rm -rf {result_folder} && " + cmd

    LOGGER.info("Start DIAMOND")
    LOGGER.info(cmd)
    os.system(cmd)

    if not (result_folder / "alis.tsv").is_file():
        raise ValueError("Something went wrong with DIAMOND alignment. The output file does not exist.")

    df = pd.read_csv(result_folder / "alis.tsv", sep="\t")
    df.columns = ["query", "target", "pident"]
    df["fident"] = df["pident"] / 100
    rev = df.copy(deep=True)
    rev.columns = ["target", "query", "pident", "fident"]
    df = pd.concat([df, rev])
    df = df.groupby(["query", "target"]).agg({"fident": "mean"}).reset_index()
    table = df.pivot(index="query", columns="target", values="fident").fillna(0)

    shutil.rmtree(result_folder, ignore_errors=True)

    dataset.cluster_names = table.index.tolist()
    dataset.cluster_map = {n: n for n in dataset.cluster_names}
    dataset.cluster_similarity = table.to_numpy()
