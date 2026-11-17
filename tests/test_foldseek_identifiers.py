"""Foldseek labels must map back to the PDB filename stems used by the reader."""
import pytest

import datasail.cluster.foldseek as foldseek
from datasail.cluster.foldseek import extract, read_with_pyarrow
from datasail.reader.read_proteins import read_protein_data
from datasail.settings import FOLDSEEK


# P: chains A (100 residues), B (50); Q: X (80), Y (20); R: one chain (60).
# Each row is query chain, target chain, fident, query-chain length, lddt.
MULTICHAIN_ALIGNMENT = """protein_P_multi.pdb_A\tprotein_Q_multi.pdb_X\t0.8\t100\t0.9
protein_P_multi.pdb_A\tprotein_Q_multi.pdb_Y\t0.9\t100\t0.8
protein_P_multi.pdb_B\tprotein_Q_multi.pdb_X\t0.95\t50\t0.7
protein_P_multi.pdb_B\tprotein_Q_multi.pdb_Y\t0.2\t50\t0.6
protein_Q_multi.pdb_X\tprotein_P_multi.pdb_A\t0.5\t80\t0.9
protein_Q_multi.pdb_X\tprotein_P_multi.pdb_B\t0.9\t80\t0.8
protein_Q_multi.pdb_Y\tprotein_P_multi.pdb_A\t0.9\t20\t0.7
protein_Q_multi.pdb_Y\tprotein_P_multi.pdb_B\t0.1\t20\t0.6
protein_P_multi.pdb_B\tprotein_R_single.pdb\t0.4\t50\t0.95
protein_P_multi.pdb_A\tprotein_R_single.pdb\t0.9\t100\t0.8
protein_R_single.pdb\tprotein_P_multi.pdb_A\t0.2\t60\t0.95
protein_R_single.pdb\tprotein_P_multi.pdb_B\t0.7\t60\t0.8
protein_Q_multi.pdb_X\tprotein_R_single.pdb\t0.6\t80\t0.9
protein_Q_multi.pdb_Y\tprotein_R_single.pdb\t0.1\t20\t0.8
protein_R_single.pdb\tprotein_Q_multi.pdb_X\t0.8\t60\t0.9
protein_R_single.pdb\tprotein_Q_multi.pdb_Y\t0.3\t60\t0.8
protein_P_multi.pdb_A\tprotein_P_multi.pdb_A\t1\t100\t1
protein_P_multi.pdb_B\tprotein_P_multi.pdb_B\t1\t50\t1
protein_Q_multi.pdb_X\tprotein_Q_multi.pdb_X\t1\t80\t1
protein_Q_multi.pdb_Y\tprotein_Q_multi.pdb_Y\t1\t20\t1
protein_R_single.pdb\tprotein_R_single.pdb\t1\t60\t1
"""


@pytest.mark.parametrize("label, expected", [
    ("protein.pdb_A", ("protein", "A")),
    ("protein.pdb", ("protein", "?")),
    ("protein_with_underscores.pdb_B", ("protein_with_underscores", "B")),
    ("protein_with_underscores.pdb", ("protein_with_underscores", "?")),
    ("protein.pdb_copy.pdb_A", ("protein.pdb_copy", "A")),
    ("protein.pdb_copy.pdb", ("protein.pdb_copy", "?")),
    ("protein", ("protein", "?")),
    ("protein_A", ("protein", "A")),
])
def test_extract_identifier(label, expected):
    assert extract(label.split("_")) == expected


def test_folder_names_match_aggregation(tmp_path):
    inputs = tmp_path / "pdbs"
    inputs.mkdir()
    names = {"protein_P_multi", "protein_Q_multi", "protein_R_single"}
    for name in names:
        (inputs / f"{name}.pdb").write_text("END\n")
    dataset = read_protein_data(inputs, sim="foldseek", detect_duplicates=False)
    alignment = tmp_path / "aln.m8"
    alignment.write_text(MULTICHAIN_ALIGNMENT)
    scores = read_with_pyarrow(alignment)
    assert set(dataset.names) == names
    assert set(scores) == names
    assert all(set(targets) == names for targets in scores.values())


def test_query_length_weighting_and_greedy_chain_non_reuse(tmp_path):
    alignment = tmp_path / "aln.m8"
    alignment.write_text(MULTICHAIN_ALIGNMENT)
    scores = read_with_pyarrow(alignment)
    # Lddt ranks A-X before the competing A-Y and B-X, then accepts B-Y.
    # P->Q: (0.8*100 + 0.2*50)/150 = 0.6; Q->P: (0.5*80 + 0.1*20)/100 = 0.42.
    expected = {
        ("protein_P_multi", "protein_Q_multi"): ["AB", "XY", 90, 150],
        ("protein_Q_multi", "protein_P_multi"): ["XY", "AB", 42, 100],
        # A single target chain blocks the other query chain: no unmatched-chain penalty.
        ("protein_P_multi", "protein_R_single"): ["B", "?", 20, 50],
        ("protein_R_single", "protein_P_multi"): ["?", "A", 12, 60],
        ("protein_Q_multi", "protein_R_single"): ["X", "?", 48, 80],
        ("protein_R_single", "protein_Q_multi"): ["?", "X", 48, 60],
    }
    for (query, target), record in expected.items():
        assert scores[query][target] == record


def test_equal_lddt_uses_descending_fident(tmp_path):
    alignment = tmp_path / "aln.m8"
    alignment.write_text(
        "P.pdb_A\tQ.pdb_X\t0.8\t100\t0.9\n"
        "P.pdb_A\tQ.pdb_Y\t0.9\t100\t0.9\n"
        "P.pdb_B\tQ.pdb_X\t0.2\t50\t0.8\n"
    )
    # A-Y wins the lddt tie, allowing B-X: numerator 90+10, denominator 100+50.
    assert read_with_pyarrow(alignment)["P"]["Q"] == ["AB", "YX", 100, 150]


@pytest.mark.parametrize("reverse, expected", [
    (False, ["A", "X", 80, 100]),
    (True, ["AB", "YX", 90, 150]),
])
def test_exact_rank_ties_preserve_row_order(tmp_path, reverse, expected):
    alignment = tmp_path / "aln.m8"
    tied = ["P.pdb_A\tQ.pdb_X\t0.8\t100\t0.9\n", "P.pdb_A\tQ.pdb_Y\t0.8\t100\t0.9\n"]
    if reverse:
        tied.reverse()
    alignment.write_text("".join(tied) + "P.pdb_B\tQ.pdb_X\t0.2\t50\t0.8\n")
    assert read_with_pyarrow(alignment)["P"]["Q"] == expected


@pytest.mark.parametrize("name", ["protein_with_underscores", "protein.pdb_copy"])
def test_single_chain_filename_in_matrix_lookup(tmp_path, monkeypatch, name):
    inputs = tmp_path / "pdbs"
    inputs.mkdir()
    (inputs / f"{name}.pdb").write_text("END\n")
    dataset = read_protein_data(inputs, sim="foldseek", detect_duplicates=False)

    def search(command):
        results = tmp_path / "fs_results"
        results.mkdir()
        (results / "aln.m8").write_text(f"{name}.pdb\t{name}.pdb\t1\t60\t1\n")
        return 0

    monkeypatch.setitem(foldseek.INSTALLED, FOLDSEEK, True)
    monkeypatch.setattr(foldseek.os, "system", search)
    foldseek.run_foldseek(dataset, tmp_dir=tmp_path)
    assert dataset.cluster_similarity.tolist() == [[1]]
