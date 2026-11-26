"""Storage changes must preserve the complete corrected Foldseek matrix."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import datasail.cluster.foldseek as foldseek
from datasail.reader.utils import DataSet
from tests.foldseek_helpers import assert_bits, dataset, reference_matrix
from tests.test_foldseek_identifiers import MULTICHAIN_ALIGNMENT


@pytest.mark.parametrize("memory,threads", [("1K", 1), ("64K", 2), ("1M", 1)])
@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
def test_multichain_reference_and_independent_values(tmp_path, memory, threads, order):
    alignment = tmp_path / "saved.m8"
    alignment.write_text(MULTICHAIN_ALIGNMENT)
    names = ["protein_P_multi", "protein_Q_multi", "protein_R_single"]
    ordered = [names[i] for i in order]
    pq = ((.8 * 100 + .2 * 50) / 150 + (.5 * 80 + .1 * 20) / 100) / 2
    pr, qr = (.4 + .2) / 2, (.6 + .8) / 2
    expected = np.array([[1, pq, pr], [pq, 1, qr], [pr, qr, 1]])
    actual = foldseek.read_foldseek_similarity(alignment, dataset(ordered), tmp_dir=tmp_path,
                                              sort_memory=memory, threads=threads)
    assert_bits(actual, reference_matrix(alignment, dataset(ordered), tmp_path / "reference"))
    assert_bits(actual, expected[np.ix_(order, order)])


@pytest.mark.parametrize("memory", ["1K", "64K", "1M"])
@pytest.mark.parametrize("reverse", [False, True])
def test_exact_ties_survive_external_sort_merges(tmp_path, memory, reverse, monkeypatch):
    # Repeated competing rows force actual spill/merge files with the small buffers.
    tied = ["P.pdb_A\tQ.pdb_X\t0.8\t100\t0.9\n", "P.pdb_B\tQ.pdb_X\t0.8\t50\t0.9\n"]
    if reverse:
        tied.reverse()
    alignment = tmp_path / "ties.m8"
    alignment.write_text(tied[0] + "P.pdb_A\tQ.pdb_Y\t0.2\t100\t0.8\n" * 4000
                         + tied[1] + tied[0] * 4000 + "P.pdb_B\tQ.pdb_Y\t0.1\t50\t0.7\n")
    calls = []
    original = subprocess.run

    def external(command, **kwargs):
        calls.append(command)
        return original(command, **kwargs)

    monkeypatch.setattr(foldseek.subprocess, "run", external)
    observed = foldseek.read_foldseek_similarity(alignment, dataset(["P", "Q"]), tmp_dir=tmp_path, sort_memory=memory)
    target = reference_matrix(alignment, dataset(["P", "Q"]), tmp_path / "reference")
    assert_bits(observed, target)
    # Missing reverse direction retains the final raw .1 fallback.
    score = (40 + 20) / 150 if reverse else (80 + 5) / 150
    assert_bits(observed, np.array([[1, (score + .1) / 2], [(score + .1) / 2, 1]]))
    assert len(calls) == 2 and all("--stable" in command for command in calls)
    assert calls[0][0] == calls[1][0]
    assert list(tmp_path.glob("datasail-foldseek-*")) == []


@pytest.mark.parametrize("quality", ["-NAN", "NAN", "NaN", "NULL", "", "inf", "-inf", "-0.0"])
def test_nan_null_and_infinite_ranking_matches_reference(tmp_path, quality):
    alignment = tmp_path / "rank.m8"
    alignment.write_text(f"P.pdb_A\tQ.pdb_X\t0.2\t100\t{quality}\n"
                         "P.pdb_A\tQ.pdb_Y\t0.9\t100\tNaN\n"
                         "P.pdb_B\tQ.pdb_Z\t0.1\t50\t0.000\n")
    actual = foldseek.read_foldseek_similarity(alignment, dataset(["P", "Q"]), tmp_dir=tmp_path, sort_memory="1K")
    assert_bits(actual, reference_matrix(alignment, dataset(["P", "Q"]), tmp_path / "reference"))


@pytest.mark.parametrize("rows", [
    "P.pdb_A\tQ.pdb_X\tNaN\t100\t0\nP.pdb_A\tQ.pdb_X\t0.8\t100\t1\n",
    "P.pdb_A\tQ.pdb_X\t0.1\tNA\t0\nP.pdb_A\tQ.pdb_X\t0.8\t100\t1\n",
    "P.pdb_A\tQ.pdb_X\t-NAN\t100\t1\nQ.pdb_X\tP.pdb_A\tNAN\t100\t1\n",
    "P.pdb_A\tQ.pdb_X\t0.8\t100\t0.9\nP.pdb_B\tQ.pdb_Y\t0.2\t0.5\t0.8\n",
    "P.pdb_A\tQ.pdb_X\t9007199254740993\t1\t1\nP.pdb_B\tQ.pdb_Y\t1\t1\t0\n",
    "P.pdb_A\tQ.pdb_X\t0.3\t100\t9007199254740992\nP.pdb_A\tQ.pdb_Y\t0.9\t100\t9007199254740993\n",
    "P.pdb_A\tQ.pdb_X\t-0.0\t100\t1\n",
    "P.pdb_A\tQ.pdb_X\tinf\t100\t1\nQ.pdb_X\tP.pdb_A\t-inf\t100\t1\n",
])
def test_numeric_types_and_accumulation_order_are_preserved(tmp_path, rows):
    alignment = tmp_path / "numeric.m8"
    alignment.write_text(rows)
    actual = foldseek.read_foldseek_similarity(alignment, dataset(["P", "Q"]), tmp_dir=tmp_path, sort_memory="1K")
    assert_bits(actual, reference_matrix(alignment, dataset(["P", "Q"]), tmp_path / "reference"))


def test_late_float_promotes_entire_column(tmp_path):
    # Cross Arrow's default 1 MiB block, where a streaming CSV reader freezes types.
    alignment = tmp_path / "types.m8"
    alignment.write_text("P.pdb_A\tQ.pdb_X\t1\t100\t1\n" * 50000 + "P.pdb_B\tQ.pdb_Y\t0.5\t0.5\t0.5\n")
    actual = foldseek.read_foldseek_similarity(alignment, dataset(["P", "Q"]), tmp_dir=tmp_path, sort_memory="64K")
    assert_bits(actual, reference_matrix(alignment, dataset(["P", "Q"]), tmp_path / "reference"))


@pytest.mark.parametrize("names,suffix,rows", [
    (["legacy_name", "other_name"], "", "legacy_name\tother_name\t0.7\t178\t0.94\n"),
    (["protein.pdb_copy", "other.pdb_copy"], "", "protein.pdb_copy\tother.pdb_copy\t0.7\t178\t0.94\n"),
    (["protein.pdb", "other.pdb"], "", "protein.pdb\tother.pdb\t0.7\t178\t0.94\n"),
    (["P", "P.pdb_A"], ".pdb", "P.pdb_A\tP.pdb_A.pdb\t0.2\t100\t0.9\nP.pdb_A.pdb\tP.pdb_A\t0.4\t80\t0.9\n"),
    (["P", "Q", "absent"], ".pdb", "P.pdb_A\tQ.pdb_X\t0.8\t100\t0.9\nP.pdb_B\tQ.pdb_Y\t0.2\t50\t0.8\n"),
])
def test_filename_mapping_and_missing_direction_fallback(tmp_path, names, suffix, rows):
    alignment = tmp_path / "identifiers.m8"
    alignment.write_text(rows)
    actual = foldseek.read_foldseek_similarity(alignment, dataset(names, suffix), tmp_dir=tmp_path, sort_memory="1K")
    assert_bits(actual, reference_matrix(alignment, dataset(names, suffix), tmp_path / "reference"))


@pytest.mark.parametrize("memory", ["1K", "1M"])
def test_real_foldseek_nan_fixture(tmp_path, memory):
    fixtures = Path(__file__).parent / "fixtures"
    alignment = fixtures / "foldseek_nan_alignment.m8"
    expected = json.loads((fixtures / "foldseek_nan_expected.json").read_text())
    actual = foldseek.read_foldseek_similarity(alignment, dataset(expected["names"]), tmp_dir=tmp_path, sort_memory=memory)
    assert_bits(actual, reference_matrix(alignment, dataset(expected["names"]), tmp_path / "reference"))
    assert_bits(actual, np.array(expected["expected"]))


def test_reuse_never_executes_foldseek_or_modifies_input(tmp_path, monkeypatch):
    alignment = tmp_path / "supplied.m8"
    alignment.write_text("P.pdb\tQ.pdb\t0.7\t100\t0.9\n")
    alignment.chmod(0o444)
    before = (alignment.read_bytes(), alignment.stat().st_mode, alignment.stat().st_mtime_ns)
    sentinel = tmp_path / "fs_results"
    sentinel.mkdir()
    (sentinel / "owned").write_text("caller")
    monkeypatch.setattr(foldseek.os, "system", lambda command: pytest.fail("Foldseek must not run"))
    monkeypatch.setitem(foldseek.INSTALLED, foldseek.FOLDSEEK, False)
    actual = foldseek.read_foldseek_similarity(alignment, dataset(["P", "Q"]), tmp_dir=tmp_path)
    assert_bits(actual, np.array([[1, .7], [.7, 1]]))
    assert before == (alignment.read_bytes(), alignment.stat().st_mode, alignment.stat().st_mtime_ns)
    assert (sentinel / "owned").read_text() == "caller"
    assert list(tmp_path.glob("datasail-foldseek-*")) == []


@pytest.mark.parametrize("stage,error", [("_sort_alignment", RuntimeError), ("_sort_alignment", KeyboardInterrupt),
                                         ("_chain_scores", RuntimeError), ("_chain_scores", KeyboardInterrupt)])
def test_failure_and_cancellation_clean_only_private_files(tmp_path, monkeypatch, stage, error):
    alignment = tmp_path / "supplied.m8"
    alignment.write_text("P.pdb\tQ.pdb\t0.7\t100\t0.9\n")
    before = hashlib.sha256(alignment.read_bytes()).digest()
    owned = tmp_path / "caller-owned"
    owned.write_text("keep")

    original = getattr(foldseek, stage)

    def fail(*args, **kwargs):
        if stage == "_sort_alignment" and kwargs.get("timeout") == 10:
            return original(*args, **kwargs)
        raise error("original")

    monkeypatch.setattr(foldseek, stage, fail)
    with pytest.raises(error, match="original"):
        foldseek.read_foldseek_similarity(alignment, dataset(["P", "Q"]), tmp_dir=tmp_path)
    assert hashlib.sha256(alignment.read_bytes()).digest() == before
    assert owned.read_text() == "keep"
    assert list(tmp_path.glob("datasail-foldseek-*")) == []


@pytest.mark.parametrize("rows,message", [
    ("P\tQ\t0.5\n", "row 1.*five tab-separated"),
    ("P.pdb\tQ.pdb\t0.5\t100\t1\nunknown.pdb\tQ.pdb\t0.5\t100\t1\n", "row 2.*unknown identifier.*unknown.pdb"),
    ("P.pdb\tQ.pdb\tbroken\t100\t1\n", "row 1"),
    ("P.pdb\tQ.pdb\t0.5\tbroken\t1\n", "row 1"),
    ("P.pdb\tQ.pdb\t0.5\t100\tbroken\n", "row 1"),
    ("P.pdb\tQ.pdb\tNaN\t100\t1\n", "row 1.*null fident"),
    ("", "empty Foldseek alignment"),
])
def test_invalid_alignment_is_actionable_and_preserved(tmp_path, rows, message):
    alignment = tmp_path / "bad.m8"
    alignment.write_text(rows)
    with pytest.raises(ValueError, match=message):
        foldseek.read_foldseek_similarity(alignment, dataset(["P", "Q"]), tmp_dir=tmp_path)
    assert alignment.read_text() == rows
    assert list(tmp_path.glob("datasail-foldseek-*")) == []


def test_mapping_validation(tmp_path):
    for data, message in [(dataset(["P", "P"]), "Duplicate"),
                          (DataSet(names=["P"], data={}), "Source filenames"),
                          (dataset([]), "nonempty")]:
        with pytest.raises(ValueError, match=message):
            foldseek.read_foldseek_similarity(tmp_path / "unused", data, tmp_dir=tmp_path)


def test_missing_or_failed_sort_has_actionable_error(tmp_path, monkeypatch):
    executable = foldseek._check_sort()
    monkeypatch.setattr(foldseek, "_check_sort", lambda: executable)
    alignment = tmp_path / "input.m8"
    alignment.write_text("P.pdb\tQ.pdb\t0.7\t100\t0.9\n")
    for error in (FileNotFoundError(), subprocess.CalledProcessError(2, "sort", stderr="disk full")):
        def fail(*args, **kwargs):
            raise error
        monkeypatch.setattr(foldseek.subprocess, "run", fail)
        with pytest.raises(RuntimeError, match="sort"):
            foldseek.read_foldseek_similarity(alignment, dataset(["P", "Q"]), tmp_dir=tmp_path)
        assert alignment.exists() and list(tmp_path.glob("datasail-foldseek-*")) == []


@pytest.mark.skipif(shutil.which("foldseek") is None, reason="Foldseek executable unavailable")
def test_native_single_and_multichain_execution_and_reuse(tmp_path):
    from datasail.reader.read_proteins import read_protein_data

    inputs = tmp_path / "pdbs"
    inputs.mkdir()
    fixtures = Path(__file__).parent / "data/pipeline/pdbs"
    first = (fixtures / "1CYN_A.pdb").read_text()
    second = (fixtures / "1XO7_A.pdb").read_text()
    chain_a = [line for line in first.splitlines() if line.startswith(("ATOM  ", "HETATM"))]
    chain_b = [line[:21] + "B" + line[22:] for line in second.splitlines() if line.startswith(("ATOM  ", "HETATM"))]
    (inputs / "single_with_underscores.pdb").write_text(first)
    (inputs / "protein.pdb_copy.pdb").write_text(second)
    (inputs / "multi_with_underscores.pdb").write_text("\n".join(chain_a + ["TER"] + chain_b + ["TER", "END"]) + "\n")
    data = read_protein_data(inputs, sim="foldseek", detect_duplicates=False)
    original = foldseek.os.system
    alignment = tmp_path / "saved.m8"

    def search(command):
        status = original(command)
        assert status == 0
        shutil.copyfile(tmp_path / "fs_results/aln.m8", alignment)
        return status

    with patch.object(foldseek.os, "system", search):
        foldseek.run_foldseek(data, threads=1, tmp_dir=tmp_path, log_dir=tmp_path)
    actual = data.cluster_similarity
    assert_bits(actual, foldseek.read_foldseek_similarity(alignment, data, tmp_dir=tmp_path, sort_memory="1K"))
    assert_bits(actual, reference_matrix(alignment, dataset(data.names), tmp_path / "reference"))
    # Independently: shared exact chains score 1; the two distinct single chains score .7.
    expected = np.ones((3, 3))
    a, b = (data.names.index(n) for n in ("single_with_underscores", "protein.pdb_copy"))
    expected[a, b] = expected[b, a] = .7
    assert_bits(actual, expected)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("memory", ["1K", "16K"])
def test_complete_random_matrices_preserve_greedy_string_matching(tmp_path, seed, memory):
    rng = np.random.default_rng(seed)
    names = ["protein_with_underscores", "protein.pdb_copy", "P.pdb_A"]
    names = [names[i] for i in rng.permutation(3)]
    rows = []
    for _ in range(90):
        q, t = (names[i] for i in rng.integers(0, 3, 2))
        # Overlapping multi-character labels exercise the original substring rule.
        qc, tc = rng.choice(["A", "B", "AB"], 2)
        identity = rng.choice([.1, .2, .5, 1.0])
        length = rng.choice([1, 50, 99])
        quality = rng.choice(["-inf", "-NAN", "NaN", "", "0.8", "0.8", "inf"])
        rows.append(f"{q}.pdb_{qc}\t{t}.pdb_{tc}\t{identity}\t{length}\t{quality}\n")
    alignment = tmp_path / "random.m8"
    alignment.write_text("".join(rows))
    observed = foldseek.read_foldseek_similarity(alignment, dataset(names), tmp_dir=tmp_path, sort_memory=memory)
    assert_bits(observed, reference_matrix(alignment, dataset(names), tmp_path / "reference"))


def test_mocked_search_and_explicit_reuse_are_identical(tmp_path, monkeypatch):
    from tests.test_foldseek_matrix import run_alignment

    names = ["protein_P_multi", "protein_Q_multi", "protein_R_single"]
    normal = tmp_path / "normal"
    normal.mkdir()
    matrix = run_alignment(normal, monkeypatch, names, MULTICHAIN_ALIGNMENT)
    alignment = tmp_path / "saved.m8"
    alignment.write_text(MULTICHAIN_ALIGNMENT)
    observed = foldseek.read_foldseek_similarity(alignment, dataset(names), tmp_dir=tmp_path, sort_memory="1K")
    assert_bits(matrix, observed)


@pytest.mark.parametrize("rows", [
    'P.pdb_A\tQ.pdb_X\t0.8\t"100"\t"0.9"\nP.pdb_B\tQ.pdb_Y\t0.2\t"0.5"\t"NaN"\n',
    'P.pdb_A\tQ.pdb_X\t9007199254740993\t1\t1\nP.pdb_B\tQ.pdb_Y\t+1\t1\t0\n',
    'P.pdb_A\tQ.pdb_X\t0.8\t100\t9007199254740992\nP.pdb_B\tQ.pdb_X\t0.8\t50\t+9007199254740993\nP.pdb_A\tQ.pdb_Y\t0.2\t100\t0\nP.pdb_B\tQ.pdb_Y\t0.1\t50\t-1\n',
])
def test_csv_numeric_quoting_and_plus_sign_promotions(tmp_path, rows):
    alignment = tmp_path / "csv-numeric.m8"
    alignment.write_text(rows)
    actual = foldseek.read_foldseek_similarity(alignment, dataset(["P", "Q"]), tmp_dir=tmp_path, sort_memory="1K")
    assert_bits(actual, reference_matrix(alignment, dataset(["P", "Q"]), tmp_path / "reference"))


def test_zero_length_self_comparison_keeps_unit_diagonal(tmp_path):
    alignment = tmp_path / "self.m8"
    alignment.write_text("P.pdb_A\tP.pdb_A\t0.8\t0\t1\n")
    actual = foldseek.read_foldseek_similarity(alignment, dataset(["P"]), tmp_dir=tmp_path)
    assert_bits(actual, reference_matrix(alignment, dataset(["P"]), tmp_path / "reference"))
    assert_bits(actual, np.ones((1, 1)))


def test_unused_legacy_aggregates_still_validate_selected_numbers(tmp_path):
    alignment = tmp_path / "legacy.m8"
    alignment.write_text("legacy_name\tother_name\t0.7\tNA\t1\n")
    names = ["legacy_name", "other_name"]
    with pytest.raises(TypeError):
        reference_matrix(alignment, dataset(names, ""), tmp_path / "reference")
    with pytest.raises(ValueError, match="row 1.*null fident or qlen"):
        foldseek.read_foldseek_similarity(alignment, dataset(names, ""), tmp_dir=tmp_path)
    assert alignment.exists() and list(tmp_path.glob("datasail-foldseek-*")) == []


@pytest.mark.parametrize("first,second", [("NaN", "NAN"), ("NaN", "inf")])
def test_secondary_fident_nan_and_null_order(tmp_path, first, second):
    alignment = tmp_path / "secondary-rank.m8"
    alignment.write_text(f"P.pdb_A\tQ.pdb_X\t{first}\t100\t1\nP.pdb_A\tQ.pdb_Y\t{second}\t100\t1\n")
    actual = foldseek.read_foldseek_similarity(alignment, dataset(["P", "Q"]), tmp_dir=tmp_path, sort_memory="1K")
    assert_bits(actual, reference_matrix(alignment, dataset(["P", "Q"]), tmp_path / "reference"))


def test_failed_search_does_not_process_leftover_alignment(tmp_path, monkeypatch):
    from datasail.reader.read_proteins import read_protein_data

    inputs = tmp_path / "pdbs"
    inputs.mkdir()
    (inputs / "P.pdb").write_text("END\n")
    data = read_protein_data(inputs, sim="foldseek", detect_duplicates=False)
    old_results = tmp_path / "fs_results"
    old_results.mkdir()
    alignment = old_results / "aln.m8"
    alignment.write_text("P.pdb\tP.pdb\t1\t100\t1\n")
    before = alignment.read_bytes()
    monkeypatch.setitem(foldseek.INSTALLED, foldseek.FOLDSEEK, True)
    monkeypatch.setattr(foldseek.os, "system", lambda command: 1)
    monkeypatch.setattr(foldseek, "_read_foldseek_similarity", lambda *args, **kwargs: pytest.fail("Failed search must not reuse an alignment"))
    with pytest.raises(ValueError, match="Foldseek command failed"):
        foldseek.run_foldseek(data, tmp_dir=tmp_path)
    assert alignment.read_bytes() == before
    assert data.cluster_similarity is None


@pytest.mark.parametrize("availability", ["missing", "unsupported-options", "incorrect-output"])
@pytest.mark.parametrize("normal_search", [False, True])
def test_sort_preflight_blocks_expensive_work_and_preserves_files(tmp_path, monkeypatch, availability, normal_search):
    alignment = tmp_path / "supplied.m8"
    alignment.write_text("P.pdb\tQ.pdb\t0.7\t100\t0.9\n")
    alignment.chmod(0o444)
    before = (alignment.read_bytes(), alignment.stat().st_mode, alignment.stat().st_mtime_ns)
    owned = tmp_path / "caller-owned"
    owned.write_text("keep")
    if availability == "missing":
        monkeypatch.setattr(foldseek.shutil, "which", lambda name: None)
    else:
        monkeypatch.setattr(foldseek.shutil, "which", lambda name: str(tmp_path / "sort"))

        def incompatible(command, **kwargs):
            if availability == "unsupported-options":
                raise subprocess.CalledProcessError(2, command, stderr="unsupported --parallel")
            Path(command[command.index("-o") + 1]).write_text("wrong sorting\n")
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr(foldseek.subprocess, "run", incompatible)
    monkeypatch.setitem(foldseek.INSTALLED, foldseek.FOLDSEEK, True)

    def forbidden(*args, **kwargs):
        pytest.fail("Incompatible sort must prevent Foldseek, allocation, scanning and staging")

    monkeypatch.setattr(foldseek.os, "system", forbidden)
    monkeypatch.setattr(foldseek.np, "zeros", forbidden)
    monkeypatch.setattr(foldseek, "_prepare_alignment", forbidden)
    with pytest.raises(RuntimeError, match="GNU sort from coreutils"):
        if normal_search:
            # Missing sources also prove the check precedes structure copying.
            foldseek.run_foldseek(dataset(["P", "Q"]), tmp_dir=tmp_path)
        else:
            foldseek.read_foldseek_similarity(alignment, dataset(["P", "Q"]), tmp_dir=tmp_path)
    assert before == (alignment.read_bytes(), alignment.stat().st_mode, alignment.stat().st_mtime_ns)
    assert owned.read_text() == "keep"
    assert set(tmp_path.iterdir()) == {alignment, owned}


def test_compatible_sort_is_resolved_and_reused_once(tmp_path, monkeypatch):
    original_run = subprocess.run
    actual_sort = Path(shutil.which("sort")).resolve()
    linked = tmp_path / "sort-link"
    linked.symlink_to(actual_sort)
    monkeypatch.setattr(foldseek.shutil, "which", lambda name: str(linked))
    alignment = tmp_path / "input.m8"
    alignment.write_text("P.pdb\tQ.pdb\t0.7\t100\t0.9\n")
    calls = []

    def checked(command, **kwargs):
        result = original_run(command, **kwargs)
        calls.append((command, kwargs))
        if len(calls) == 1:
            # Later PATH resolution would select an incompatible executable.
            monkeypatch.setenv("PATH", "")
            monkeypatch.setattr(foldseek.shutil, "which", lambda name: pytest.fail("Sort must be resolved once"))
        return result

    monkeypatch.setattr(foldseek.subprocess, "run", checked)
    result = foldseek.read_foldseek_similarity(alignment, dataset(["P", "Q"]), tmp_dir=tmp_path)
    assert_bits(result, np.array([[1, .7], [.7, 1]]))
    assert len(calls) == 2
    assert calls[0][0][0] == calls[1][0][0] == str(actual_sort)
    assert calls[0][1]["timeout"] == 10 and calls[1][1]["timeout"] is None


def test_normal_search_checks_sort_once(tmp_path, monkeypatch):
    from tests.test_foldseek_matrix import run_alignment

    original = foldseek._check_sort
    checks = []

    def checked():
        executable = original()
        checks.append(executable)
        return executable

    monkeypatch.setattr(foldseek, "_check_sort", checked)
    result = run_alignment(tmp_path, monkeypatch, ["P"], "P.pdb\tP.pdb\t1\t100\t1\n")
    assert_bits(result, np.ones((1, 1)))
    assert len(checks) == 1


@pytest.mark.parametrize("error", [PermissionError("not executable"), subprocess.TimeoutExpired("sort", 10)])
def test_sort_preflight_execution_errors_are_actionable(tmp_path, monkeypatch, error):
    monkeypatch.setattr(foldseek.shutil, "which", lambda name: str(tmp_path / "sort"))

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(foldseek.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="GNU sort from coreutils.*Compatibility check failed"):
        foldseek._check_sort()


def test_sort_preflight_rejects_ignored_stability(tmp_path, monkeypatch):
    original = subprocess.run

    def unstable(command, **kwargs):
        return original([argument for argument in command if argument != "--stable"], **kwargs)

    monkeypatch.setattr(foldseek.subprocess, "run", unstable)
    with pytest.raises(RuntimeError, match="Compatibility check returned incorrect output"):
        foldseek._check_sort()
