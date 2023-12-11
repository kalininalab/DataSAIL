import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import shutil

import numpy as np
import pandas as pd

from datasail.cluster.utils import extract_fasta
from datasail.parsers import MultiYAMLParser
from datasail.reader.utils import DataSet
from datasail.settings import LOGGER, MMSEQS2, INSTALLED, MMSEQSPP


def run_mmseqspp(
        dataset: DataSet,
        threads: int,
        log_dir: Optional[Path] = None,
) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    if not INSTALLED[MMSEQS2]:
        raise ValueError("MMseqs is not installed.")

    parser = MultiYAMLParser(MMSEQSPP)
    prefilter_args = parser.get_user_arguments(dataset.args, [], 0)
    align_args = parser.get_user_arguments(dataset.args, [], 1)
    extract_fasta(dataset)

    result_folder = Path("mmseqspp_results")

    cmd = lambda x: f"mkdir {result_folder} && " \
                    f"cd {result_folder} && " \
                    f"mmseqs createdb {str(Path('..') / dataset.location)} seqs.db {x} && " \
                    f"mmseqs prefilter seqs.db seqs.db seqs.pref --threads {threads} {prefilter_args} {x} && " \
                    f"mmseqs align seqs.db seqs.db seqs.pref seqs.ali -e inf --threads {threads} {align_args} {x} && " \
                    f"mmseqs convertalis seqs.db seqs.db seqs.ali alis.tsv --format-mode 4 --format-output query,target,fident --threads {threads} {x}"

    if log_dir is None:
        cmd = cmd("> /dev/null 2>&1")
    else:
        cmd = cmd(f">> {(Path(log_dir) / f'{dataset.get_name()}_mmseqspp.log').resolve()}")

    if result_folder.exists():
        cmd = f"rm -rf {result_folder} && " + cmd

    LOGGER.info("Start MMseqs2 Align")
    LOGGER.info(cmd)
    os.system(cmd)

    if not (result_folder / "alis.tsv").is_file():
        raise ValueError("Something went wront with mmseqs alignment. The output file does not exist.")

    df = pd.read_csv(result_folder / "alis.tsv", sep="\t")
    table = df.pivot(index="query", columns="target", values="fident").fillna(0).to_numpy()

    shutil.rmtree(result_folder, ignore_errors=True)

    return dataset.names, {n: n for n in dataset.names}, table
