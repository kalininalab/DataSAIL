"""One-pass staging must retain global types, stable ties and bounded caches."""
from pathlib import Path

import numpy as np
import pytest

import datasail.cluster.foldseek as foldseek
from tests.foldseek_helpers import assert_bits, dataset, reference_matrix
from tests.test_foldseek_identifiers import MULTICHAIN_ALIGNMENT


@pytest.mark.parametrize('threads,memory', [(1, '1K'), (4, '64K')])
@pytest.mark.parametrize('late_float,expected', [(False, 1.5), (True, 2.0)])
def test_integer_rank_keys_and_late_promotion(tmp_path, threads, memory, late_float, expected):
    alignment = tmp_path / 'rank.m8'
    rows = ('P.pdb_A\tQ.pdb_X\t1\t1\t9007199254740993\n'
            'P.pdb_B\tQ.pdb_X\t2\t1\t9007199254740992\n')
    if late_float:
        rows += 'P.pdb_A\tP.pdb_A\t1\t1\t0.5\n'
    alignment.write_text(rows)
    observed = foldseek.read_foldseek_similarity(alignment, dataset(['P', 'Q']),
                                                tmp_dir=tmp_path, threads=threads, sort_memory=memory)
    assert_bits(observed, np.array([[1., expected], [expected, 1.]]))
    assert_bits(observed, reference_matrix(alignment, dataset(['P', 'Q']), tmp_path / 'reference'))


@pytest.mark.parametrize('size', [1, 2, 7, 8192])
def test_cache_evictions_do_not_change_results(tmp_path, monkeypatch, size):
    original = foldseek._bounded_cache
    monkeypatch.setattr(foldseek, '_bounded_cache', lambda fn, capacity, length: original(fn, size, length))
    alignment = tmp_path / 'input.m8'
    alignment.write_text(MULTICHAIN_ALIGNMENT * 300)
    names = ['protein_R_single', 'protein_P_multi', 'protein_Q_multi']
    observed = foldseek.read_foldseek_similarity(alignment, dataset(names), tmp_dir=tmp_path, sort_memory='1K', threads=2)
    assert_bits(observed, reference_matrix(alignment, dataset(names), tmp_path / 'reference'))


def test_alignment_is_read_once(tmp_path, monkeypatch):
    alignment = tmp_path / 'input.m8'
    alignment.write_text(MULTICHAIN_ALIGNMENT)
    original = foldseek._alignment_rows
    calls = []
    def rows(path):
        calls.append(Path(path))
        yield from original(path)
    monkeypatch.setattr(foldseek, '_alignment_rows', rows)
    names = ['protein_P_multi', 'protein_Q_multi', 'protein_R_single']
    observed = foldseek.read_foldseek_similarity(alignment, dataset(names), tmp_dir=tmp_path)
    assert calls == [alignment]
    assert_bits(observed, reference_matrix(alignment, dataset(names), tmp_path / 'reference'))


def test_long_cache_keys_are_not_retained():
    calls = []
    def function(value):
        calls.append(value)
        return value
    cached = foldseek._bounded_cache(function, 2, 3)
    for value in ['abc', 'abc', 'long', 'long']:
        assert cached(value) == value
    assert calls == ['abc', 'long', 'long']


@pytest.mark.parametrize('threads,memory', [(1, '1K'), (4, '64K')])
def test_cached_bare_refined_structure_identifiers(tmp_path, threads, memory):
    names = ['5abc_LIG_A_101_protein_refined', '6def_LIG_B_202_protein_refined']
    alignment = tmp_path / 'bare.m8'
    alignment.write_text(f'{names[0]}\t{names[1]}\t0.25\t4\t0.9\n'
                         f'{names[1]}\t{names[0]}\t0.75\t2\t0.8\n')
    observed = foldseek.read_foldseek_similarity(alignment, dataset(names),
                                                tmp_dir=tmp_path, threads=threads, sort_memory=memory)
    assert_bits(observed, np.array([[1., 0.5], [0.5, 1.]]))
    assert_bits(observed, reference_matrix(alignment, dataset(names), tmp_path / 'reference'))
