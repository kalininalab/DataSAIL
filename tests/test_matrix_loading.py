"""Compare streaming file loading with the isolated pre-change implementation."""
import builtins
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from datasail.cluster.clustering import cluster
from datasail.reader import utils
from datasail.reader.read_genomes import read_genome_data
from datasail.reader.read_molecules import read_molecule_data
from datasail.reader.read_other import read_other_data
from datasail.reader.read_proteins import read_protein_data
from datasail.settings import KW_CACHE, KW_LINKAGE, KW_LOGDIR, KW_THREADS, KW_TMPDIR
from tests import matrix_reference as reference


def assert_same(before, after):
    before_names, before_matrix = before
    names, matrix = after
    assert names == before_names
    assert matrix.shape == before_matrix.shape
    assert matrix.dtype == before_matrix.dtype == np.float64
    np.testing.assert_array_equal(matrix.view(np.uint64), before_matrix.view(np.uint64))
    np.testing.assert_array_equal(np.isnan(matrix), np.isnan(before_matrix))


def write_matrix(path, names, values, sep="\t", corner="id", newline="\n", final_newline=True):
    lines = [sep.join([corner, *names])]
    lines.extend(sep.join([name, *row]) for name, row in zip(names, values))
    path.write_bytes((newline.join(lines) + (newline if final_newline else "")).encode())
    return path


@pytest.mark.parametrize("sep", ["\t", ",", "||"])
@pytest.mark.parametrize("newline,final_newline", [("\n", True), ("\r\n", False)])
@pytest.mark.parametrize("names", [["only"], ["z", "a"], ["protein.pdb_copy", '"quoted ID"', "β_chain_A"], ["001", "nan", "0"]])
def test_complete_identifiers_values_and_order(tmp_path, sep, newline, final_newline, names):
    n = len(names)
    tokens = ["-0.0", "0.10000000000000001", "1.0000000000000002", "5e-324", "-2.2250738585072014e-308",
              "1.7976931348623157e308", "9007199254740993", "1_2.5", "1e-4000"]
    values = [[tokens[(i * n + j) % len(tokens)] for j in range(n)] for i in range(n)]
    path = write_matrix(tmp_path / "matrix.data", names, values, sep, "", newline, final_newline)
    before = reference.read_clustering_file(path, sep)
    after = utils.read_clustering_file(path, sep)
    assert_same(before, after)
    assert after[0] == names
    expected = np.array([[float(token) for token in row] for row in values])
    np.testing.assert_array_equal(after[1].view(np.uint64), expected.view(np.uint64))
    assert after[1].flags.c_contiguous and after[1].flags.owndata and after[1].flags.writeable


def test_special_values_are_preserved_bitwise(tmp_path):
    tokens = ["nan", "-nan", "NaN", "NAN", "inf", "+Infinity", "-inf", "-Infinity", "0.0", "-0.0", "1e4000"]
    names = [f"e{i}" for i in range(len(tokens))]
    values = [[tokens[(i + j) % len(tokens)] for j in range(len(tokens))] for i in range(len(tokens))]
    path = write_matrix(tmp_path / "special.tsv", names, values)
    before = reference.read_clustering_file(path)
    after = utils.read_clustering_file(path)
    assert_same(before, after)
    assert np.isposinf(after[1]).any() and np.isneginf(after[1]).any()
    assert np.signbit(after[1][0, 1])  # Negative NaN sign is not normalized away.


@pytest.mark.parametrize("relative", ["pipeline/drug_sim.tsv", "pipeline/drug_dist.tsv",
                                      "pipeline/prot_sim.tsv", "pipeline/prot_dist.tsv",
                                      "rw_data/distance_matrix.tsv"])
def test_existing_real_matrix_files(tmp_path, relative):
    path = Path(__file__).parent / "data" / relative
    assert_same(reference.read_clustering_file(path), utils.read_clustering_file(path))
    # Default routing remains tab-separated, regardless of extension.
    arbitrary = tmp_path / "matrix.csv"
    arbitrary.write_bytes(path.read_bytes())
    assert_same(reference.read_matrix_input(str(arbitrary)), utils.read_matrix_input(str(arbitrary)))
    assert_same(reference.read_matrix_input(arbitrary), utils.read_matrix_input(arbitrary))


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
@pytest.mark.parametrize("callable_input", [False, True])
def test_in_memory_inputs_keep_identity_dtype_and_layout(dtype, callable_input):
    names = ["B", "A"]
    matrix = np.asfortranarray(np.array([[1, 2], [3, 4]], dtype=dtype))
    source = (lambda: (names, matrix)) if callable_input else (names, matrix)
    for reader in (reference.read_matrix_input, utils.read_matrix_input):
        result = reader(source)
        assert result[0] is names and result[1] is matrix
        assert result[1].dtype == dtype and result[1].flags.f_contiguous


def test_memory_mapped_tuple_is_not_copied(tmp_path):
    names = ["A", "B"]
    matrix = np.lib.format.open_memmap(tmp_path / "input.npy", mode="w+", dtype=np.float64, shape=(2, 2))
    matrix[:] = [[1, .5], [.5, 1]]
    assert utils.read_matrix_input((names, matrix))[1] is matrix
    assert reference.read_matrix_input((names, matrix))[1] is matrix


@pytest.mark.parametrize("reader,data", [
    (read_other_data, {"A": "one", "B": "two", "C": "three"}),
    (read_protein_data, {"A": "AAAA", "B": "CCCC", "C": "GGGG"}),
    (read_genome_data, {"A": "AAAA", "B": "CCCC", "C": "GGGG"}),
    (read_molecule_data, {"A": "C", "B": "CC", "C": "CCC"}),
])
@pytest.mark.parametrize("kind", ["sim", "dist"])
def test_all_entity_readers_use_the_shared_loader(tmp_path, reader, data, kind):
    names = ["C", "A", "B"]
    matrix = np.array([[1, .2, .4], [.2, 1, .3], [.4, .3, 1]])
    if kind == "dist":
        matrix = 1 - matrix
    path = write_matrix(tmp_path / "input.tsv", names, [[repr(float(x)) for x in row] for row in matrix])
    with patch.object(utils, "read_clustering_file", reference.read_clustering_file):
        before = reader(data=data, detect_duplicates=False, **{kind: path})
    after = reader(data=data, detect_duplicates=False, **{kind: path})
    assert before.names == after.names == names
    assert before.weights == after.weights
    assert_same((before.names, getattr(before, "similarity" if kind == "sim" else "distance")),
                (after.names, getattr(after, "similarity" if kind == "sim" else "distance")))


@pytest.mark.parametrize("kind", ["sim", "dist"])
@pytest.mark.parametrize("linkage", ["average", "single", "complete"])
def test_downstream_real_clustering_is_unchanged(tmp_path, kind, linkage):
    path = Path(__file__).parent / "data/pipeline" / f"drug_{kind}.tsv"
    names, _ = reference.read_clustering_file(path)
    data = {name: str(i) for i, name in enumerate(reversed(names))}
    weights = {name: float(1 + i % 4) for i, name in enumerate(names)}
    results = []
    for loader in (reference.read_clustering_file, utils.read_clustering_file):
        with patch.object(utils, "read_clustering_file", loader):
            dataset = read_other_data(data=data, weights=weights, num_clusters=4, **{kind: path})
        np.random.seed(42)
        results.append(cluster(dataset, **{KW_CACHE: False, KW_THREADS: 1, KW_LOGDIR: None,
                                           KW_LINKAGE: linkage, KW_TMPDIR: tmp_path}))
    before, after = results
    assert list(before.cluster_names) == list(after.cluster_names)
    assert before.cluster_map == after.cluster_map
    assert before.cluster_weights == after.cluster_weights
    assert before.names == after.names == names
    field = "cluster_similarity" if kind == "sim" else "cluster_distance"
    assert_same((list(before.cluster_names), getattr(before, field)),
                (list(after.cluster_names), getattr(after, field)))
    assert after.cluster_weights == {c: sum(weights[n] for n in names if after.cluster_map[n] == c)
                                    for c in after.cluster_names}


@pytest.mark.parametrize("text,message", [
    ("", "empty matrix file"),
    ("id\n", "header requires"),
    ("id\tA\t\n", "nonempty entity labels"),
    ("id\tA\tA\nA\t1\t0\nA\t0\t1\n", "duplicate entity labels"),
    ("id\tA\tB\nA\t1\t0\n", "expected 2 matrix rows, got 1"),
    ("id\tA\nA\t1\nB\t0\n", "row 3.*too many rows"),
    ("id\tA\tB\nA\t1\nB\t0\t1\n", "row 2.*expected 2 values, got 1"),
    ("id\tA\nA\t1\t0\n", "row 2.*expected 1 values, got 2"),
    ("id\tA\tB\nB\t1\t0\nA\t0\t1\n", "row 2.*label 'B'.*header label 'A'"),
    ("id\tA\tB\nA\t1\t0\nA\t0\t1\n", "row 3.*label 'A'.*header label 'B'"),
    ("id\tA\nA\tnot-a-number\n", "row 2.*invalid numeric value.*not-a-number"),
    ("id\tA\nA\t\n", "row 2.*expected 1 values"),
    ("id\tA\nA\t1\n\n", "row 3.*too many rows"),
])
def test_malformed_inputs_are_actionable_and_preserved(tmp_path, text, message):
    path = tmp_path / "bad.tsv"
    path.write_text(text)
    before = (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_mode)
    with pytest.raises(ValueError, match=message):
        utils.read_clustering_file(path)
    assert before == (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_mode)


def test_inconsistent_labels_are_a_deliberate_error_improvement(tmp_path):
    path = tmp_path / "mislabeled.tsv"
    path.write_text("id\tA\tB\nB\t1\t0\nA\t0\t1\n")
    assert reference.read_clustering_file(path)[0] == ["B", "A"]
    with pytest.raises(ValueError, match="does not match header"):
        utils.read_clustering_file(path)


def test_streaming_does_not_read_the_complete_file(tmp_path, monkeypatch):
    path = write_matrix(tmp_path / "input.tsv", ["B", "A"], [["1", ".5"], [".5", "1"]])
    before = (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_mode)
    path.chmod(0o444)
    real_open = builtins.open

    class IncrementalFile:
        def __enter__(self):
            self.file = real_open(path, "r")
            return self

        def __exit__(self, *args):
            self.file.close()

        def readline(self):
            return self.file.readline()

        def __iter__(self):
            return iter(self.file)

        def read(self, *args):
            pytest.fail("Whole-file reading is forbidden")

        def readlines(self, *args):
            pytest.fail("Whole-file reading is forbidden")

    monkeypatch.setattr(utils, "open", lambda *args, **kwargs: IncrementalFile(), raising=False)
    names, matrix = utils.read_clustering_file(path)
    assert names == ["B", "A"]
    np.testing.assert_array_equal(matrix, [[1, .5], [.5, 1]])
    assert path.read_bytes() == before[0] and path.stat().st_mtime_ns == before[1]
    assert path.stat().st_mode & 0o777 == 0o444
    assert list(tmp_path.iterdir()) == [path]


def test_real_foldseek_expected_matrix(tmp_path):
    fixture = json.loads((Path(__file__).parent / "fixtures/foldseek_nan_expected.json").read_text())
    expected = np.array(fixture["expected"], dtype=np.float64)
    path = write_matrix(tmp_path / "foldseek.tsv", fixture["names"],
                        [[repr(float(value)) for value in row] for row in expected])
    result = utils.read_clustering_file(path)
    assert_same(reference.read_clustering_file(path), result)
    assert result[0] == fixture["names"]
    np.testing.assert_array_equal(result[1].view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("kind", ["sim", "dist"])
def test_evaluation_file_input_path_is_unchanged(tmp_path, kind):
    from datasail.eval import eval_split

    path = Path(__file__).parent / "data/pipeline" / f"drug_{kind}.tsv"
    names, _ = reference.read_clustering_file(path)
    kwargs = dict(datatype="O", data={name: str(i) for i, name in enumerate(names)},
                  weights={name: 1.0 for name in names}, similarity=path if kind == "sim" else None,
                  distance=path if kind == "dist" else None, dist_conv=1.0,
                  split_assignment={name: "train" if i < 24 else "test" for i, name in enumerate(names)},
                  return_matrix=True)
    with patch.object(utils, "read_clustering_file", reference.read_clustering_file):
        before = eval_split(**kwargs, tmp_dir=tmp_path / "before")
    after = eval_split(**kwargs, tmp_dir=tmp_path / "after")
    np.testing.assert_array_equal(np.array(before[:3]).view(np.uint64), np.array(after[:3]).view(np.uint64))
    assert_same((names, before[3]), (names, after[3]))
