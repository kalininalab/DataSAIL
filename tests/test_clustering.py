import numpy as np

from datasail.cluster.clustering import additional_clustering
from datasail.reader.utils import DataSet


def test_additional_clustering():
    names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    base_map = dict((n, n) for n in names)
    similarity = np.asarray([
        [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
        [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
        [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
        [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
        [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
        [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 5, 5, 5, 5],
        [0, 0, 0, 0, 0, 0, 5, 5, 5, 5],
        [0, 0, 0, 0, 0, 0, 5, 5, 5, 5],
        [0, 0, 0, 0, 0, 0, 5, 5, 5, 5],
    ])
    distance = np.asarray([
        [0, 0, 0, 0, 0, 0, 4, 4, 4, 4],
        [0, 0, 0, 0, 0, 0, 4, 4, 4, 4],
        [0, 0, 0, 0, 0, 0, 4, 4, 4, 4],
        [0, 0, 0, 0, 0, 0, 4, 4, 4, 4],
        [0, 0, 0, 0, 0, 0, 4, 4, 4, 4],
        [0, 0, 0, 0, 0, 0, 4, 4, 4, 4],
        [4, 4, 4, 4, 4, 4, 0, 0, 0, 0],
        [4, 4, 4, 4, 4, 4, 0, 0, 0, 0],
        [4, 4, 4, 4, 4, 4, 0, 0, 0, 0],
        [4, 4, 4, 4, 4, 4, 0, 0, 0, 0],
    ])
    weights = dict((n, (6 if n in names[:6] else 4)) for n in names)

    s_dataset = DataSet()
    s_dataset.cluster_names = names
    s_dataset.cluster_map = base_map
    s_dataset.cluster_weights = weights
    s_dataset.cluster_similarity = similarity
    s_dataset.cluster_distance = None
    d_dataset = DataSet()
    d_dataset.cluster_names = names
    d_dataset.cluster_map = base_map
    d_dataset.cluster_weights = weights
    d_dataset.cluster_similarity = None
    d_dataset.cluster_distance = distance

    s_dataset = additional_clustering(s_dataset)
    assert len(s_dataset.cluster_names) < 10
    assert set(s_dataset.cluster_names) == set(s_dataset.cluster_map.values())
    assert set(s_dataset.cluster_names) == set(s_dataset.cluster_weights.keys())
    assert set(names) == set(s_dataset.cluster_map.keys())
    assert len(set(s_dataset.cluster_map[x] for x in names[:6]).intersection(set(
        s_dataset.cluster_map[x] for x in names[6:]
    ))) == 0
    assert np.min(s_dataset.cluster_similarity) == 0
    assert np.max(s_dataset.cluster_similarity) == 5
    assert s_dataset.cluster_distance is None
    assert [s_dataset.cluster_weights[i] for i in s_dataset.cluster_names] == [18, 12, 6, 12, 4]

    d_dataset = additional_clustering(d_dataset)
    assert len(d_dataset.cluster_names) < 10
    assert set(d_dataset.cluster_names) == set(d_dataset.cluster_map.values())
    assert set(d_dataset.cluster_names) == set(d_dataset.cluster_weights.keys())
    assert set(names) == set(d_dataset.cluster_map.keys())
    assert len(set(d_dataset.cluster_map[x] for x in names[:6]).intersection(set(
        d_dataset.cluster_map[x] for x in names[6:]
    ))) == 0
    assert d_dataset.cluster_similarity is None
    assert np.min(d_dataset.cluster_distance) == 0
    assert np.max(d_dataset.cluster_distance) == 4
    assert [d_dataset.cluster_weights[i] for i in d_dataset.cluster_names] == [16, 36]
