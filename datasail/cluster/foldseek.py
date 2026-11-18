import csv
import itertools
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from pyarrow.csv import ConvertOptions

from datasail.parsers import MultiYAMLParser
from datasail.reader.utils import DataSet
from datasail.settings import LOGGER, FOLDSEEK, INSTALLED


_NULL_VALUES = frozenset(ConvertOptions().null_values)


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
    sort_executable = _check_sort()
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
    if os.system(cmd) != 0:
        raise ValueError("Foldseek command failed; check its search output.")

    if not (results_folder / "aln.m8").exists():
        raise ValueError("Something went wrong with foldseek. The output file does not exist.")

    cluster_sim = _read_foldseek_similarity(
        results_folder / "aln.m8", dataset, tmp_dir=tmp_dir, threads=threads,
        sort_executable=sort_executable,
    )
    shutil.rmtree(results_folder, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)

    dataset.cluster_names = dataset.names
    dataset.cluster_map = dict((n, n) for n in dataset.names)
    dataset.cluster_similarity = cluster_sim


def extract(tmp):
    """Extract the filename stem and optional chain from a Foldseek PDB label."""
    label = "_".join(tmp)
    if label.endswith(".pdb"):
        return label[:-4], "?"
    if ".pdb_" in label:
        return tuple(label.rsplit(".pdb_", 1))
    if len(tmp) == 1:
        return tmp[0], "?"
    else:
        return "_".join(tmp[:-1]), tmp[-1]


def _alignment_rows(alignment):
    """Read the five Foldseek columns without retaining earlier rows."""
    with open(alignment) as source:
        for number, line in enumerate(source, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 5:
                raise ValueError(f"{alignment}: row {number}: expected query,target,fident,qlen,lddt (five tab-separated columns)")
            if '"' in line:
                try:
                    parsed = next(csv.reader([line], delimiter="\t", strict=True))
                except csv.Error as error:
                    raise ValueError(f"{alignment}: row {number}: {error}") from error
                if len(parsed) != 5:
                    raise ValueError(f"{alignment}: row {number}: expected five tab-separated columns")
                # The reference raw-value pass uses the first three fields
                # literally; Arrow unquotes the remaining numeric columns.
                fields[3:] = parsed[3:]
            yield number, fields


def _number(token, integer=False):
    if token in _NULL_VALUES:
        return None
    return int(token) if integer else float(token)


def _scan_alignment(alignment, dataset, matrix, index):
    """Infer numeric column types globally, matching the corrected CSV reader."""
    # Arrow's whole-table reader promotes an entire numeric column to double
    # when any nonnull token is not an int64. Streaming CSV inference alone
    # freezes the first block's types, changing later conversion/accumulation.
    integers = [True, True, True]
    filename_ids = {name for name in dataset.names if Path(dataset.data[name]).name == name}
    count = 0
    for number, fields in _alignment_rows(alignment):
        try:
            for column, token in enumerate(fields[2:]):
                if token in _NULL_VALUES:
                    continue
                float(token)
                if integers[column]:
                    try:
                        value = int(token)
                        integers[column] = -(2 ** 63) <= value < 2 ** 63 and not token.lstrip().startswith("+")
                    except ValueError:
                        integers[column] = False
            ids = []
            for value in (fields[0].lstrip(), fields[1]):
                entity = value if value in filename_ids else extract(value.split("_"))[0]
                if entity not in index:
                    raise ValueError(f"unknown identifier {value!r} (resolved to {entity!r})")
                ids.append(index[entity])
            # Preserve the reference's last-row symmetric fallback, including
            # pairs for which only one directional aggregate is available.
            matrix[ids[0], ids[1]] = matrix[ids[1], ids[0]] = float(fields[2])
        except (ValueError, OverflowError) as error:
            raise ValueError(f"{alignment}: row {number}: {error}") from error
        count = number
    if not count:
        raise ValueError(f"{alignment}: empty Foldseek alignment")
    return integers, count


def _rank(value):
    # Arrow: ordinary numbers (including infinities), then NaNs, then nulls.
    # Normalize NaN sort tokens so fident breaks their ties, then --stable
    # retains original row order even across external-sort merge files.
    if value is None:
        return 2, "0"
    if isinstance(value, float) and math.isnan(value):
        return 1, "0"
    return 0, repr(value)


def _stage_alignment(alignment, spool, index, integers):
    with spool.open("x") as output:
        for number, fields in _alignment_rows(alignment):
            query, target = fields[:2]
            qi, qc = extract(query.split("_"))
            ti, tc = extract(target.split("_"))
            # Unused legacy aliases still undergo greedy arithmetic in the
            # reference. Tag them on disk without retaining an alias dictionary.
            qi = str(index[qi]) if qi in index else "u" + qi
            ti = str(index[ti]) if ti in index else "u" + ti
            fident = _number(fields[2], integers[0])
            quality = _number(fields[4], integers[2])
            qr, qv = _rank(quality)
            fr, fv = _rank(fident)
            output.write(f"{qi}\t{ti}\t{qc}\t{tc}\t{qr}\t{qv}\t{fr}\t{fv}\t{fields[2]}\t{fields[3]}\t{number}\n")


def _sort_alignment(spool, sorted_path, scratch, threads, sort_memory, sort_executable, *, timeout=None):
    try:
        subprocess.run(
            [sort_executable, "--stable", "--field-separator=\t", "-k1,1", "-k2,2",
             "-k5,5n", "-k6,6gr", "-k7,7n", "-k8,8gr", f"--parallel={threads}",
             f"--buffer-size={sort_memory}", f"--temporary-directory={scratch}",
             "-o", str(sorted_path), str(spool)],
            check=True, stderr=subprocess.PIPE, text=True, timeout=timeout,
            env={**os.environ, "LC_ALL": "C"},
        )
    except FileNotFoundError as error:
        raise RuntimeError("GNU sort (coreutils) is required for Foldseek alignment processing") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Foldseek alignment sorting failed (GNU sort is required): {error.stderr.strip()}") from error


def _check_sort():
    """Resolve sort and exercise the options needed by alignment processing."""
    requirement = (
        "Foldseek alignment processing requires GNU sort from coreutils. "
        "Install GNU coreutils and put its sort executable on PATH."
    )
    executable = shutil.which("sort")
    if executable is None:
        raise RuntimeError(requirement)
    executable = str(Path(executable).resolve())
    rows = [
        "0\t1\tB\tY\t0\t2\t0\t2\tfirst\n",
        "0\t1\tA\tX\t0\t2\t0\t2\tsecond\n",
        "0\t1\tC\tZ\t0\t2\t0\t10\thigher\n",
        "0\t1\tD\tW\t0\t10\t0\t1\tbest\n",
        "0\t1\tE\tV\t1\t0\t0\t0.99\tnan\n",
        "0\t1\tF\tU\t2\t0\t0\t1\tnull\n",
    ]
    with tempfile.TemporaryDirectory(prefix="datasail-sort-check-") as directory:
        scratch = Path(directory)
        spool, output = scratch / "input.tsv", scratch / "output.tsv"
        spool.write_text("".join(rows))
        try:
            _sort_alignment(spool, output, scratch, 1, "64K", executable, timeout=10)
            observed = output.read_text()
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"{requirement} Compatibility check failed: {error}") from error
        if observed != "".join(rows[i] for i in (3, 2, 0, 1, 4, 5)):
            raise RuntimeError(f"{requirement} Compatibility check returned incorrect output.")
    return executable


def _chain_scores(sorted_path, alignment, integers):
    """Keep only one entity pair's original greedy chain-selection state."""
    with sorted_path.open() as source:
        rows = (line.rstrip("\n").split("\t") for line in source)
        for (qi, ti), group in itertools.groupby(rows, key=lambda row: (row[0], row[1])):
            queries, targets, numerator, denominator = "", "", 0, 0
            for row in group:
                qc, tc = row[2:4]
                # Preserve the reference's substring test and concatenation,
                # rather than silently selecting different chain semantics.
                if qc in queries or tc in targets:
                    continue
                fident, length = _number(row[8], integers[0]), _number(row[9], integers[1])
                if fident is None or length is None:
                    raise ValueError(f"{alignment}: row {row[10]}: selected chain has null fident or qlen")
                queries += qc
                targets += tc
                numerator += fident * length
                denominator += length
            yield qi, ti, (queries, targets, numerator, denominator)


def read_foldseek_similarity(
    alignment: Path, dataset: DataSet, *, tmp_dir: Optional[Path] = None,
    threads: int = 1, sort_memory: str = "256M",
) -> np.ndarray:
    """Build similarities from a five-column alignment without running Foldseek.

    Args:
        alignment: Existing query,target,fident,qlen,lddt TSV. Read in place.
        dataset: Requested matrix order and original source filenames.
        tmp_dir: Parent for a private temporary workspace; defaults to TMPDIR.
        threads: GNU sort parallelism.
        sort_memory: GNU sort buffer size, e.g. "64M". This is not an RSS cap.

    Returns:
        Dense float64 similarity matrix in dataset.names order. Its N squared
        storage remains in memory; parsing and sorting spill to temporary disk.
    """
    return _read_foldseek_similarity(
        alignment, dataset, tmp_dir=tmp_dir, threads=threads, sort_memory=sort_memory,
        sort_executable=_check_sort(),
    )


def _read_foldseek_similarity(
    alignment, dataset, *, tmp_dir=None, threads=1, sort_memory="256M", sort_executable,
):
    """Process an alignment with an already checked sort executable."""
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if not re.fullmatch(r"[1-9][0-9]*[KMG]?", str(sort_memory)):
        raise ValueError("sort_memory must be a positive size in bytes, K, M or G")
    names = dataset.names
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Foldseek entity names must be nonempty strings")
    index = {name: i for i, name in enumerate(names)}
    if len(index) != len(names):
        raise ValueError("Duplicate Foldseek entity names")
    if dataset.data is None or any(name not in dataset.data for name in names):
        raise ValueError("Source filenames are required for all Foldseek entity names")
    alignment = Path(alignment)
    matrix = np.zeros((len(names), len(names)), dtype=np.float64)
    LOGGER.info("Read Foldseek alignment %s", alignment)
    integers, count = _scan_alignment(alignment, dataset, matrix, index)
    with tempfile.TemporaryDirectory(prefix="datasail-foldseek-", dir=tmp_dir) as directory:
        scratch = Path(directory)
        spool, sorted_path = scratch / "chains.tsv", scratch / "chains.sorted.tsv"
        _stage_alignment(alignment, spool, index, integers)
        LOGGER.info("Sort %s Foldseek rows with buffer %s and %s threads", count, sort_memory, threads)
        _sort_alignment(spool, sorted_path, scratch, threads, sort_memory, sort_executable)
        LOGGER.info("Aggregate Foldseek chain scores")
        for qi, ti, record in _chain_scores(sorted_path, alignment, integers):
            if not qi.startswith("u") and not ti.startswith("u") and qi != ti:
                a, b = int(qi), int(ti)
                if record[3] == 0:
                    raise ValueError(f"{alignment}: selected chain lengths sum to zero for {names[a]!r} -> {names[b]!r}")
                matrix[a, b] = record[2] / record[3]
    # Two O(N) rows preserve operand order (including NaN payloads) without
    # allocating a second N by N matrix during symmetrization.
    for i in range(len(names)):
        upper = (matrix[i, i + 1:] + matrix[i + 1:, i]) / 2
        lower = (matrix[i + 1:, i] + matrix[i, i + 1:]) / 2
        matrix[i, i + 1:], matrix[i + 1:, i] = upper, lower
        matrix[i, i] = 1
    return matrix
