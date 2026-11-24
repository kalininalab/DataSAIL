"""Small-fixture adapters for the frozen reference and streaming parser."""
from pathlib import Path
from unittest.mock import patch

import numpy as np

import datasail.cluster.foldseek as foldseek
from datasail.reader.utils import DataSet
from tests import foldseek_reference as reference


def dataset(names, suffix=".pdb"):
    return DataSet(names=list(names), data={n: Path(n + suffix) for n in names})


def reference_matrix(alignment, data, work):
    """Use the unchanged implementation; replace only structure staging/search."""
    work.mkdir()

    def search(command):
        folder = work / "fs_results"
        folder.mkdir()
        (folder / "aln.m8").symlink_to(alignment.resolve())
        return 0

    with patch.object(reference.os, "system", search), \
            patch.object(reference.shutil, "copy", return_value=None), \
            patch.dict(reference.INSTALLED, {reference.FOLDSEEK: True}):
        reference.run_foldseek(data, tmp_dir=work)
    return data.cluster_similarity


def assert_bits(actual, expected):
    assert actual.dtype == expected.dtype == np.float64
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def read_scores(alignment, root, memory="1K"):
    """Expose the streaming per-pair records for existing independent expectations."""
    sort_executable = foldseek._check_sort()
    names = sorted({foldseek.extract(fields[i].split("_"))[0]
                    for _, fields in foldseek._alignment_rows(alignment) for i in (0, 1)})
    index = {n: i for i, n in enumerate(names)}
    matrix = np.zeros((len(names), len(names)))
    spool, sorted_path = root / "spool", root / "sorted"
    integers, _ = foldseek._prepare_alignment(alignment, spool, dataset(names), matrix, index)
    foldseek._sort_alignment(spool, sorted_path, root, 1, memory, sort_executable, integers=integers)
    records = {}
    for q, t, record in foldseek._chain_scores(sorted_path, alignment, integers):
        records.setdefault(names[int(q)], {})[names[int(t)]] = list(record)
    return records
