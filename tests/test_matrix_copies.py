"""Matrix ownership and layout contracts at preparation/clustering boundaries."""
import numpy as np
import pytest

from datasail.cluster.clustering import additional_clustering, finish_clustering
from datasail.reader.read_molecules import remove_duplicate_values
from datasail.reader.utils import DataSet


def assert_bits(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(
        np.ascontiguousarray(actual).view(np.uint8),
        np.ascontiguousarray(expected).view(np.uint8),
    )


def matrix_input(values, layout, tmp_path):
    if layout == "view":
        owner = np.empty(tuple(2 * n for n in values.shape), dtype=values.dtype)
        result = owner[::2, ::2]
        result[:] = values
    elif layout == "reversed":
        result = values[::-1, ::-1].copy()[::-1, ::-1]
    elif layout == "memmap":
        path = tmp_path / "matrix.npy"
        np.save(path, values)
        result = np.load(path, mmap_mode="r")
    else:
        result = np.array(values, order="F" if layout.endswith("F") else "C")
        if layout.startswith("readonly"):
            result.flags.writeable = False
    return result


LAYOUTS = ["C", "F", "view", "reversed", "readonlyC", "readonlyF", "memmap"]


@pytest.mark.parametrize("field", ["similarity", "distance"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("remove", [False, True])
def test_deduplication_preserves_values_layout_and_detachment(tmp_path, field, layout, remove):
    values = np.arange(16, dtype=np.float64).reshape(4, 4)
    values[0, 0] = -0.0
    values.view(np.uint64)[2, 3] = 0xFFF8000000000017
    source = matrix_input(values, layout, tmp_path)
    original = source.copy()
    writable = source.flags.writeable
    names = ["b", "a", "d", "c"]
    data = {name: ("shared" if remove and i < 2 else name) for i, name in enumerate(names)}
    dataset = DataSet(names=names.copy(), data=data.copy(), weights=dict.fromkeys(names, 1))
    setattr(dataset, field, source)
    result = remove_duplicate_values(dataset, data)
    removed = [1] if remove else []
    # This is the pre-change NumPy deletion contract, including its F-order view.
    expected = np.delete(np.delete(source, removed, axis=0), removed, axis=1)
    actual = getattr(result, field)
    assert_bits(actual, expected)
    assert actual.strides == expected.strides
    assert actual.flags.c_contiguous == expected.flags.c_contiguous
    assert actual.flags.f_contiguous == expected.flags.f_contiguous
    assert actual.flags.writeable and not np.shares_memory(actual, source)
    assert type(actual) is np.ndarray
    assert source.flags.writeable == writable
    assert_bits(source, original)
    assert result.names == (["b", "d", "c"] if remove else names)
    assert result.id_map == ({"b": "b", "a": "b", "d": "d", "c": "c"} if remove else dict(zip(names, names)))
    assert result.weights == ({"b": 2, "d": 1, "c": 1} if remove else dict.fromkeys(names, 1))
    actual[0, 0] = 91
    assert_bits(source, original)


@pytest.mark.parametrize("shape", [(0, 0), (1, 1), (1, 3), (3, 1)])
@pytest.mark.parametrize("order", ["C", "F"])
def test_empty_or_small_noop_deduplication(shape, order):
    source = np.arange(np.prod(shape), dtype=np.float64).reshape(shape, order=order)
    names = list(map(str, range(shape[0])))
    dataset = DataSet(names=names, data=dict(zip(names, names)), weights=dict.fromkeys(names, 1), similarity=source)
    actual = remove_duplicate_values(dataset, dataset.data).similarity
    expected = np.delete(np.delete(source, [], axis=0), [], axis=1)
    assert_bits(actual, expected)
    assert actual.strides == expected.strides
    assert actual.flags.writeable
    assert actual.flags.owndata == expected.flags.owndata


@pytest.mark.parametrize("field", ["cluster_similarity", "cluster_distance"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("missing", [False, True])
def test_finish_clustering_retains_independent_float64_result(tmp_path, field, layout, missing):
    values = np.array([[1, -.0], [.25, 1]], dtype=np.float32)
    source = matrix_input(values, layout, tmp_path)
    original = source.copy()
    names = ["b", "a"] + (["c"] if missing else [])
    dataset = DataSet(names=names, cluster_names=["b", "a"], cluster_map={"b": "b", "a": "a"},
                      weights=dict.fromkeys(names, 1), classes={0: 0, 1: 1},
                      stratification={name: np.array([1., 0.]) for name in names},
                      interaction_classes={0: 0, 1: 1},
                      interaction_stratification={name: np.array([0., 2.]) for name in names})
    setattr(dataset, field, source)
    finish_clustering(dataset)
    actual = getattr(dataset, field)
    expected = (np.zeros if field == "cluster_similarity" else np.ones)((len(names), len(names)))
    expected[:2, :2] = source
    assert_bits(actual, expected)
    assert actual.dtype == np.float64 and actual.flags.c_contiguous and actual.flags.writeable
    assert not np.shares_memory(actual, source)
    assert_bits(source, original)
    assert dataset.cluster_names == names
    assert dict(dataset.cluster_weights) == dict.fromkeys(names, 1.)
    for name in names:
        assert_bits(dataset.cluster_stratification[name], np.array([1., 0.]))
        assert_bits(dataset.cluster_interaction_stratification[name], np.array([0., 2.]))


@pytest.mark.parametrize("mode", ["similarity", "distance"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("linkage", ["average", "single", "complete"])
def test_clustering_reads_inputs_without_mutation_or_output_aliasing(tmp_path, mode, layout, linkage):
    random = np.random.RandomState(42)
    values = random.uniform(.05, .9, (18, 18))
    values = (values + values.T) / 2
    np.fill_diagonal(values, 1.)
    if mode == "distance":
        values = 1 - values
    source = matrix_input(values, layout, tmp_path)
    original = source.copy()
    writable = source.flags.writeable
    names = list(map(str, range(len(source))))
    dataset = DataSet(names=names.copy(), cluster_names=names.copy(),
                      cluster_map=dict(zip(names, names)), cluster_weights=dict.fromkeys(names, 1.),
                      weights=dict.fromkeys(names, 1.), classes={0: 0, 1: 1},
                      cluster_stratification={n: np.array([1., 0.]) for n in names},
                      interaction_classes={0: 0, 1: 1},
                      cluster_interaction_stratification={n: np.array([0., 2.]) for n in names})
    setattr(dataset, mode, source)
    setattr(dataset, "cluster_" + mode, source)
    result = additional_clustering(dataset, 4, linkage)
    first = getattr(result, "cluster_" + mode)
    first_values = first.copy()
    assert not np.shares_memory(first, source)
    assert first.flags.writeable and first.dtype == np.float64
    assert source.flags.writeable == writable
    assert_bits(source, original)
    assert sum(result.cluster_weights.values()) == len(names)
    result = additional_clustering(result, 2, linkage)
    assert not np.shares_memory(getattr(result, "cluster_" + mode), first)
    assert_bits(first, first_values)
    assert_bits(source, original)
    # A retained original is still usable by later techniques/runs.
    if writable:
        final = getattr(result, "cluster_" + mode).copy()
        source[0, 1] = .75
        assert_bits(getattr(result, "cluster_" + mode), final)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, ">f8"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("mode", ["similarity", "distance"])
def test_fit_input_preserves_legacy_conversion_and_layout(tmp_path, monkeypatch, dtype, layout, mode):
    from datasail.cluster import clustering

    values = np.arange(36, dtype=np.float64).reshape(6, 6).astype(dtype)
    source = matrix_input(values, layout, tmp_path)
    legacy = np.array(source, dtype=float)
    original = source.copy()
    names = list(map(str, range(6)))
    dataset = DataSet(names=names.copy(), cluster_names=names.copy(), cluster_map=dict(zip(names, names)),
                      weights=dict.fromkeys(names, 1.), cluster_weights=dict.fromkeys(names, 1.))
    setattr(dataset, "cluster_" + mode, source)

    class InspectEstimator:
        def __init__(self, **kwargs):
            assert kwargs["n_clusters"] == 3
            if mode == "similarity":
                assert kwargs["random_state"] == 42
                assert kwargs["affinity"] == "precomputed"
            else:
                assert kwargs["linkage"] == "average"

        def fit_predict(self, matrix):
            assert_bits(matrix, legacy)
            assert matrix.strides == legacy.strides
            assert not matrix.flags.writeable
            borrowable = source.dtype == np.dtype(float) and (
                source.flags.c_contiguous or source.flags.f_contiguous
            )
            assert np.shares_memory(matrix, source) == borrowable
            return np.arange(6) % 3

    monkeypatch.setattr(clustering, "SpectralClustering" if mode == "similarity" else "AgglomerativeClustering", InspectEstimator)
    additional_clustering(dataset, 3, "average")
    assert_bits(source, original)


@pytest.mark.parametrize("mode", ["similarity", "distance"])
def test_rejected_fit_does_not_mutate_input_or_flags(mode):
    source = np.eye(6)
    source[0, 1] = np.nan
    original = source.copy()
    names = list(map(str, range(6)))
    dataset = DataSet(names=names.copy(), cluster_names=names.copy(), cluster_map=dict(zip(names, names)),
                      weights=dict.fromkeys(names, 1.), cluster_weights=dict.fromkeys(names, 1.))
    setattr(dataset, "cluster_" + mode, source)
    with pytest.raises(ValueError):
        additional_clustering(dataset, 3, "average")
    assert_bits(source, original)
    assert source.flags.writeable
    assert getattr(dataset, "cluster_" + mode) is source


@pytest.mark.parametrize("constructor", [np.matrix, np.ma.MaskedArray])
@pytest.mark.parametrize("remove", [False, True])
def test_deduplication_keeps_array_subclass_contract(constructor, remove):
    source = constructor(np.arange(16, dtype=np.float64).reshape(4, 4))
    names = ["a", "b", "c", "d"]
    data = {name: ("shared" if remove and i < 2 else name) for i, name in enumerate(names)}
    dataset = DataSet(names=names.copy(), data=data.copy(), weights=dict.fromkeys(names, 1), similarity=source)
    expected = np.delete(np.delete(source, [1] if remove else [], axis=0), [1] if remove else [], axis=1)
    actual = remove_duplicate_values(dataset, data).similarity
    assert type(actual) is type(expected)
    assert_bits(actual, expected)
    assert actual.strides == expected.strides
    assert not np.shares_memory(actual, source)
    if np.ma.isMaskedArray(expected):
        np.testing.assert_array_equal(actual.mask, expected.mask)


@pytest.mark.parametrize("dtype", [np.float32, ">f8"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_noop_deduplication_retains_dtype(dtype, order):
    source = np.array([[1, -.0], [.25, 1]], dtype=dtype, order=order)
    dataset = DataSet(names=["a", "b"], data={"a": "a", "b": "b"}, weights={"a": 1, "b": 1},
                      distance=source)
    actual = remove_duplicate_values(dataset, dataset.data).distance
    expected = np.delete(np.delete(source, [], axis=0), [], axis=1)
    assert_bits(actual, expected)
    assert actual.strides == expected.strides
    assert not np.shares_memory(actual, source)


def test_failed_second_axis_deletion_keeps_first_axis_result():
    source = np.arange(4, dtype=np.float64)
    data = {n: n for n in "abcd"}
    dataset = DataSet(names=list(data), data=data.copy(), weights=dict.fromkeys(data, 1), similarity=source)
    with pytest.raises(np.exceptions.AxisError, match="axis 1"):
        remove_duplicate_values(dataset, data)
    assert_bits(dataset.similarity, source)
    assert not np.shares_memory(dataset.similarity, source)
