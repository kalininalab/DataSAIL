"""Aggregation must preserve exact ordered reductions and downstream mathematics."""
import copy
import json
from pathlib import Path

import cvxpy as cp
import numpy as np
import pytest
from scipy import sparse

import datasail.cluster.aggregation as aggregation
import datasail.cluster.clustering as clustering
from datasail.reader.utils import DataSet, read_clustering_file
from datasail.solver.cluster_2d import solve_c2
from tests.aggregation_reference import labels2clusters as reference


LINKAGES = ["average", "single", "complete"]


def dataset(matrix, mode="similarity", counts=True, names=None):
    n = len(matrix)
    names = list(names) if names is not None else [f"cluster-{i}" for i in range(n)][::-1]
    result = DataSet(
        names=[f"entity-{i}" for i in range(2 * n)], cluster_names=names,
        cluster_map={f"entity-{i}": names[i // 2] for i in range(2 * n)},
        cluster_weights={name: (i + 1) / 8 for i, name in enumerate(names)},
        classes={"a": 0, "b": 1}, interaction_classes={"x": 0, "y": 1, "z": 2},
        num_clusters=min(3, n),
    )
    setattr(result, "cluster_" + mode, matrix)
    if counts:
        result.cluster_stratification = {name: np.array([i, n - i], dtype=float)
                                         for i, name in enumerate(names)}
        result.cluster_interaction_stratification = {name: np.array([i + 1, 2 * i, 3], dtype=float)
                                                     for i, name in enumerate(names)}
    return result


def assert_bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    np.testing.assert_array_equal(np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8))


def assert_datasets(a, b):
    for field in ["cluster_names", "cluster_map", "cluster_weights"]:
        assert getattr(a, field) == getattr(b, field)
    for field in ["cluster_similarity", "cluster_distance"]:
        left, right = getattr(a, field), getattr(b, field)
        if left is None:
            assert right is None
        else:
            assert_bits(left, right)
    for field in ["cluster_stratification", "cluster_interaction_stratification"]:
        left, right = getattr(a, field), getattr(b, field)
        if left is None:
            assert right is None
        else:
            assert list(left) == list(right)
            for key in left:
                assert_bits(left[key], right[key])


def compare(matrix, labels, linkage, mode="similarity", counts=True, names=None):
    original = dataset(matrix, mode, counts, names)
    before = matrix.copy()
    target = reference(labels, copy.deepcopy(original), matrix, linkage)
    actual = clustering.labels2clusters(labels, copy.deepcopy(original), matrix, linkage)
    assert_datasets(actual, target)
    assert_bits(matrix, before)
    return actual


@pytest.mark.parametrize("linkage", LINKAGES)
@pytest.mark.parametrize("mode", ["similarity", "distance"])
@pytest.mark.parametrize("block_bytes", [8, 40, 1024**2])
def test_unequal_reordered_groups_and_repeated_aggregation(monkeypatch, linkage, mode, block_bytes):
    monkeypatch.setattr(aggregation, "_BLOCK_BYTES", block_bytes)
    rng = np.random.default_rng(471)
    matrix = rng.random((13, 13))
    labels = [2, 0, 2, 1, 2, 0, 2, 2, 0, 2, 0, 2, 2]
    actual = compare(matrix, labels, linkage, mode)
    reduced = getattr(actual, "cluster_" + mode)
    expected = reference([1, 0, 1], copy.deepcopy(actual), reduced, linkage)
    actual = clustering.labels2clusters([1, 0, 1], actual, reduced, linkage)
    assert_datasets(actual, expected)


@pytest.mark.parametrize("linkage", LINKAGES)
def test_independent_contribution_selection(linkage):
    matrix = np.array([[10 * i + j for j in range(4)] for i in range(4)], dtype=float)
    # Off-diagonal scores within a group are intentionally ignored.
    matrix[0, 2] = matrix[2, 0] = matrix[1, 3] = matrix[3, 1] = 9999
    expected = {
        "average": [[22, 21], [12, 11]],
        "single": [[11, 10], [1, 0]],
        "complete": [[33, 32], [23, 22]],
    }
    assert_bits(aggregation.aggregate_cluster_matrix([1, 0, 1, 0], matrix, linkage),
                np.array(expected[linkage], dtype=float))
    for mode in ["similarity", "distance"]:
        actual = compare(matrix, [1, 0, 1, 0], linkage, mode)
        independent = np.array(expected[linkage], dtype=float)
        if mode == "similarity":
            independent = np.maximum(independent, np.eye(2))
        else:
            independent = np.minimum(independent, 1 - np.eye(2))
        assert_bits(getattr(actual, "cluster_" + mode), independent)


@pytest.mark.parametrize("linkage", LINKAGES)
@pytest.mark.parametrize("mode", ["similarity", "distance"])
@pytest.mark.parametrize("labels", [[0], [0, 0, 0, 0], [3, 0, 2, 1], [-1, 0, -1, 0], []])
def test_singletons_diagonals_and_empty(monkeypatch, linkage, mode, labels):
    monkeypatch.setattr(aggregation, "_BUFFER_BYTES", 0)
    n = len(labels)
    compare(np.arange(n * n, dtype=float).reshape(n, n) / 16 - .5, labels, linkage, mode, counts=False)


@pytest.mark.parametrize("linkage", LINKAGES)
@pytest.mark.parametrize("mode", ["similarity", "distance"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int64, ">f8"])
def test_numeric_dtypes_and_noncontiguous_inputs(linkage, mode, dtype):
    matrix = np.arange(100, dtype=float).reshape(10, 10).astype(dtype)[::2, ::2]
    compare(matrix, [1, 0, 1, 2, 0], linkage, mode)


@pytest.mark.parametrize("linkage", LINKAGES)
@pytest.mark.parametrize("mode", ["similarity", "distance"])
@pytest.mark.parametrize("block_bytes", [8, 56, 1024**2])
@pytest.mark.parametrize("spill", [False, True])
def test_special_values_and_reduction_order(monkeypatch, linkage, mode, block_bytes, spill):
    monkeypatch.setattr(aggregation, "_BLOCK_BYTES", block_bytes)
    monkeypatch.setattr(aggregation, "_BUFFER_BYTES", 0 if spill else 8 * 1024**2)
    rng = np.random.default_rng(74)
    matrix = rng.choice([1e16, -1e16, 1, -1, 1e-16, -1e-16, 0., -0.], (41, 41))
    labels = (np.arange(41) % 3)[::-1]
    with np.errstate(invalid="ignore", over="ignore"):
        compare(matrix, labels, linkage, mode)
        matrix.flat[[2, 3, 4, 6, 7, 9]] = [np.inf, -np.inf, np.nan, -np.nan, -0., 0.]
        matrix[1, 2] = np.array([0x7ff8000000000031], dtype=np.uint64).view(np.float64)[0]
        compare(matrix, labels, linkage, mode)


def test_average_preserves_numpy_pairwise_summation(monkeypatch):
    monkeypatch.setattr(aggregation, "_BLOCK_BYTES", 24)
    monkeypatch.setattr(aggregation, "_BUFFER_BYTES", 0)
    labels = np.arange(34) % 2
    matrix = np.zeros((34, 34))
    pairs = [(i, j) if labels[i] == 0 else (j, i)
             for i in range(34) for j in range(i + 1, 34) if labels[i] != labels[j]]
    values = [1e16] + [1.] * (len(pairs) - 2) + [-1e16]
    for (i, j), value in zip(pairs, values):
        matrix[i, j] = value
    # A running sum loses the intervening units; NumPy's reference reduction does not.
    assert np.mean(values) != sum(values) / len(values)
    actual = compare(matrix, labels, "average")
    assert actual.cluster_similarity[0, 1] == np.mean(values)


@pytest.mark.parametrize("linkage", LINKAGES)
@pytest.mark.parametrize("mode", ["similarity", "distance"])
def test_real_saved_fixture(monkeypatch, linkage, mode):
    fixture = json.loads((Path(__file__).parent / "fixtures/foldseek_nan_expected.json").read_text())
    matrix = np.array(fixture["expected"])
    if mode == "distance":
        matrix = 1 - matrix
    order = [7, 2, 10, 0, 9, 3, 4, 1, 6, 8, 5]
    monkeypatch.setattr(aggregation, "_BUFFER_BYTES", 16)
    compare(matrix[np.ix_(order, order)], [2, 1, 0, 2, 0, 1, 2, 2, 0, 2, 2], linkage, mode, names=[fixture["names"][i] for i in order])


@pytest.mark.parametrize("linkage", LINKAGES)
@pytest.mark.parametrize("relative,mode", [("pipeline/drug_sim.tsv", "similarity"),
                                           ("pipeline/drug_dist.tsv", "distance"),
                                           ("pipeline/prot_sim.tsv", "similarity"),
                                           ("pipeline/prot_dist.tsv", "distance"),
                                           ("rw_data/distance_matrix.tsv", "distance")])
def test_existing_real_matrix_files(linkage, relative, mode):
    names, matrix = read_clustering_file(Path(__file__).parent / "data" / relative)
    order = np.random.default_rng(94).permutation(len(names))
    compare(matrix[np.ix_(order, order)], np.arange(len(names)) % 5, linkage, mode,
            names=[names[i] for i in order])


@pytest.mark.parametrize("linkage", LINKAGES)
@pytest.mark.parametrize("mode", ["similarity", "distance"])
def test_additional_then_forced_clustering(monkeypatch, linkage, mode):
    rng = np.random.default_rng(81)
    matrix = rng.random((18, 18))
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 1)
    if mode == "distance":
        matrix = 1 - matrix
    source = dataset(matrix, mode)
    with monkeypatch.context() as m:
        m.setattr(clustering, "labels2clusters", reference)
        target = clustering.additional_clustering(copy.deepcopy(source), 7, linkage)
        target.num_clusters = 3
        target = clustering.force_clustering(target, linkage)
    actual = clustering.additional_clustering(copy.deepcopy(source), 7, linkage)
    actual.num_clusters = 3
    actual = clustering.force_clustering(actual, linkage)
    assert_datasets(actual, target)


@pytest.mark.parametrize("linkage", LINKAGES)
@pytest.mark.parametrize("relative,mode", [("pipeline/drug_sim.tsv", "similarity"),
                                           ("pipeline/drug_dist.tsv", "distance"),
                                           ("pipeline/prot_sim.tsv", "similarity"),
                                           ("pipeline/prot_dist.tsv", "distance"),
                                           ("rw_data/distance_matrix.tsv", "distance")])
def test_real_additional_then_forced_clustering(monkeypatch, linkage, relative, mode):
    names, matrix = read_clustering_file(Path(__file__).parent / "data" / relative)
    source = dataset(matrix, mode, names=names)
    with monkeypatch.context() as m:
        m.setattr(clustering, "labels2clusters", reference)
        target = clustering.additional_clustering(copy.deepcopy(source), 7, linkage)
        target.num_clusters = 3
        target = clustering.force_clustering(target, linkage)
    actual = clustering.additional_clustering(copy.deepcopy(source), 7, linkage)
    actual.num_clusters = 3
    actual = clustering.force_clustering(actual, linkage)
    assert_datasets(actual, target)


def test_file_backed_storage_cleanup_and_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(aggregation, "_BUFFER_BYTES", 0)
    original_file = aggregation.tempfile.TemporaryFile
    files = []

    def create():
        result = original_file(dir=tmp_path)
        files.append(result)
        return result

    monkeypatch.setattr(aggregation.tempfile, "TemporaryFile", create)
    matrix = np.ones((10, 10))
    compare(matrix, list(np.arange(10) % 2), "average")
    assert len(files) == 1 and files[0].closed and not list(tmp_path.iterdir())
    original_mean = np.mean
    calls = []

    def fail(values):
        calls.append(values.shape)
        raise KeyboardInterrupt("original interruption")

    monkeypatch.setattr(aggregation.np, "mean", fail)
    with pytest.raises(KeyboardInterrupt, match="original interruption"):
        aggregation.aggregate_cluster_matrix(np.arange(10) % 2, matrix, "average")
    monkeypatch.setattr(aggregation.np, "mean", original_mean)
    assert calls and all(f.closed for f in files) and not list(tmp_path.iterdir())


def solver_arguments(data, mode):
    matrix = getattr(data, "cluster_" + mode)
    weights = np.array([data.cluster_weights[c] for c in data.cluster_names])
    entity = np.stack([data.cluster_stratification[c] for c in data.cluster_names])
    interaction = np.stack([data.cluster_interaction_stratification[c] for c in data.cluster_names])
    result = dict(delta=.5, epsilon=.5, max_sec=10, solver="SCIP", log_file=None)
    for axis in ["e", "f"]:
        result.update({axis + "_clusters": data.cluster_names, axis + "_s_matrix": entity,
                       axis + "is_matrix": interaction, axis + "_similarities": matrix if mode == "similarity" else None,
                       axis + "_distances": matrix if mode == "distance" else None,
                       axis + "_weights": weights, axis + "_splits": [.5, .5], axis + "_names": ["train", "test"]})
    return result


@pytest.mark.parametrize("linkage", LINKAGES)
@pytest.mark.parametrize("mode", ["similarity", "distance"])
def test_solver_problem_and_feasible_objectives_unchanged(monkeypatch, linkage, mode):
    if "SCIP" not in cp.installed_solvers():
        pytest.skip("SCIP canonicalization is unavailable")
    matrix = (np.arange(64, dtype=float).reshape(8, 8) % 31 + 1) / 32
    labels = [3, 0, 2, 1, 3, 0, 2, 1]
    source = dataset(matrix if mode == "similarity" else 1 - matrix, mode)
    expected = reference(labels, copy.deepcopy(source), getattr(source, "cluster_" + mode), linkage)
    actual = clustering.labels2clusters(labels, source, getattr(source, "cluster_" + mode), linkage)
    problems = []

    def capture(loss, constraints, *args):
        problems.append(cp.Problem(cp.Minimize(loss), constraints))
        return None

    monkeypatch.setattr("datasail.solver.cluster_2d.solve", capture)
    solve_c2(**solver_arguments(expected, mode))
    solve_c2(**solver_arguments(actual, mode))
    left, right = [p.get_problem_data(cp.SCIP)[0] for p in problems]
    assert left.keys() == right.keys()
    for key in left:
        a, b = left[key], right[key]
        if sparse.issparse(a):
            assert a.shape == b.shape
            for field in ["data", "indices", "indptr"]:
                assert_bits(getattr(a, field), getattr(b, field))
        elif isinstance(a, np.ndarray):
            assert_bits(a, b)
        elif key == "param_prob":
            # Compare the materialized parameter program, not object addresses.
            assert type(a) is type(b)
            for x, y in zip(a.apply_parameters(), b.apply_parameters()):
                if sparse.issparse(x):
                    assert x.shape == y.shape
                    for field in ["data", "indices", "indptr"]:
                        assert_bits(getattr(x, field), getattr(y, field))
                else:
                    assert_bits(np.asarray(x), np.asarray(y))
        else:
            assert str(a) == str(b), key
    assignment = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    evaluations = []
    for problem in problems:
        for variable in problem.variables():
            variable.value = assignment
        evaluations.append((problem.objective.value, [np.asarray(c.violation()) for c in problem.constraints]))
    assert_bits(np.asarray(evaluations[0][0]), np.asarray(evaluations[1][0]))
    for a, b in zip(evaluations[0][1], evaluations[1][1]):
        assert_bits(a, b)
        assert np.max(a) == 0


@pytest.mark.parametrize("mode", ["similarity", "distance"])
def test_native_c2_before_and_after(monkeypatch, mode):
    if "SCIP" not in cp.installed_solvers():
        pytest.skip("SCIP is unavailable")
    matrix = np.array([[((i + j) % 7 + 1) / 8 for j in range(8)] for i in range(8)])
    source = dataset(matrix if mode == "similarity" else 1 - matrix, mode)
    source.cluster_weights = dict.fromkeys(source.cluster_names, 1)
    source.cluster_stratification = {n: np.array([1., 1.]) for n in source.cluster_names}
    source.cluster_interaction_stratification = {n: np.array([1., 2., 1.]) for n in source.cluster_names}
    labels = [3, 0, 2, 1, 3, 0, 2, 1]
    old = reference(labels, copy.deepcopy(source), getattr(source, "cluster_" + mode), "average")
    new = clustering.labels2clusters(labels, source, getattr(source, "cluster_" + mode), "average")
    from datasail.solver.utils import solve
    solved = []

    def observe(loss, constraints, max_sec, solver, log_file):
        problem = solve(loss, constraints, max_sec, solver, log_file)
        assert problem is not None and problem.status == cp.OPTIMAL
        assert max(float(np.max(c.violation())) for c in constraints) < 1e-7
        solved.append(problem.value)
        return problem

    monkeypatch.setattr("datasail.solver.cluster_2d.solve", observe)
    for data in [old, new]:
        result = solve_c2(**solver_arguments(data, mode))
        assert result is not None and all(set(axis.values()) == {"train", "test"} for axis in result)
    # Native solvers use numerical tolerances; exact coefficients are tested above.
    assert solved[0] == pytest.approx(solved[1], abs=1e-8, rel=0)
