"""Matrix placement tests, including an explicitly isolated indexing regression."""
from collections import defaultdict

import numpy as np
import pytest

import datasail.cluster.foldseek as foldseek
from datasail.reader.read_proteins import read_protein_data
from datasail.settings import FOLDSEEK
from tests.test_foldseek_identifiers import MULTICHAIN_ALIGNMENT


def run_alignment(tmp_path, monkeypatch, names, alignment, suffix=".pdb"):
    """Run the actual reader and matrix builder, stubbing only the external search."""
    inputs = tmp_path / "pdbs"
    inputs.mkdir()
    for name in names:
        (inputs / f"{name}{suffix}").write_text("END\n")
    source = inputs if suffix else {name: inputs / name for name in names}
    dataset = read_protein_data(source, sim="foldseek", detect_duplicates=False)
    assert set(dataset.names) == set(names)
    dataset.names = names

    def search(command):
        results = tmp_path / "fs_results"
        results.mkdir()
        (results / "aln.m8").write_text(alignment)
        return 0

    monkeypatch.setitem(foldseek.INSTALLED, FOLDSEEK, True)
    monkeypatch.setattr(foldseek.os, "system", search)
    foldseek.run_foldseek(dataset, tmp_dir=tmp_path)
    assert dataset.cluster_names == names
    assert dataset.cluster_map == {name: name for name in names}
    assert not (tmp_path / "fs_results").exists()
    assert not (tmp_path / "fs_tmp").exists()
    return dataset.cluster_similarity


def test_isolated_indexing_places_distinct_scores(tmp_path, monkeypatch):
    # Controlled canonical aggregation isolates the index error from the label error.
    scores = defaultdict(foldseek.outer_dict)
    rows = []
    for a, b, value in [("P", "Q", 0.2), ("P", "R", 0.4), ("Q", "R", 0.6)]:
        for query, target in [(a, b), (b, a)]:
            scores[query][target] = ["A", "A", value, 1]
            rows.append(f"{query}.pdb_A\t{target}.pdb_A\t{value}\t100\t0.9\n")
    monkeypatch.setattr(foldseek, "read_with_pyarrow", lambda path, names=(): scores)
    matrix = run_alignment(tmp_path, monkeypatch, ["P", "Q", "R"], "".join(rows))
    np.testing.assert_array_equal(matrix, [[1, 0.2, 0.4], [0.2, 1, 0.6], [0.4, 0.6, 1]])


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1), (1, 2, 0)])
def test_multichain_matrix_through_actual_parser(tmp_path, monkeypatch, order):
    names = ["protein_P_multi", "protein_Q_multi", "protein_R_single"]
    # Independently derived directional scores: P->Q=.6, Q->P=.42;
    # P->R=.4, R->P=.2; Q->R=.6, R->Q=.8. Average each direction pair.
    expected = np.array([[1, 0.51, 0.3], [0.51, 1, 0.7], [0.3, 0.7, 1]])
    matrix = run_alignment(tmp_path, monkeypatch, [names[i] for i in order], MULTICHAIN_ALIGNMENT)
    np.testing.assert_allclose(matrix, expected[np.ix_(order, order)], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(matrix.diagonal(), [1, 1, 1])


def test_single_protein_is_unaffected(tmp_path, monkeypatch):
    matrix = run_alignment(tmp_path, monkeypatch, ["P"], "P.pdb\tP.pdb\t1\t100\t1\n")
    np.testing.assert_array_equal(matrix, [[1]])


@pytest.mark.parametrize("names", [("legacy_name", "other_name"),
                                  ("protein.pdb_copy", "other.pdb_copy"),
                                  ("protein.pdb", "other.pdb")])
def test_bare_foldseek_headers_preserve_exact_entity_ids(tmp_path, monkeypatch, names):
    # Real Foldseek emits these headers for extensionless PDB input files.
    query, target = names
    alignment = f"{query}\t{target}\t0.7\t178\t0.94\n{target}\t{query}\t0.7\t166\t0.96\n"
    matrix = run_alignment(tmp_path, monkeypatch, list(names), alignment, suffix="")
    np.testing.assert_array_equal(matrix, [[1, 0.7], [0.7, 1]])


def test_pdb_chain_header_does_not_alias_a_different_entity(tmp_path, monkeypatch):
    # P.pdb_A is chain A of P, even when an entity named P.pdb_A also exists.
    alignment = "P.pdb_A\tP.pdb_A.pdb\t0.2\t100\t0.9\nP.pdb_A.pdb\tP.pdb_A\t0.4\t80\t0.9\n"
    matrix = run_alignment(tmp_path, monkeypatch, ["P", "P.pdb_A"], alignment)
    np.testing.assert_allclose(matrix, [[1, 0.3], [0.3, 1]], rtol=0, atol=1e-15)
