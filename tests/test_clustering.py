import numpy as np

from scala.bqp.clustering import additional_clustering


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

    cluster_names, cluster_map, cluster_similarity, cluster_distance, cluster_weights = additional_clustering(
        names, base_map, similarity, None, weights
    )
    assert len(cluster_names) < 10
    assert len(set(cluster_map[x] for x in names[:6]).intersection(set(cluster_map[x] for x in names[6:]))) == 0
    assert np.min(cluster_similarity) == 0
    assert np.max(cluster_similarity) == 5
    assert cluster_distance is None
    assert [cluster_weights[i] for i in cluster_names] == [18, 12, 6, 12, 4]

    cluster_names, cluster_map, cluster_similarity, cluster_distance, cluster_weights = additional_clustering(
        names, base_map, None, distance, weights
    )
    assert len(cluster_names) < 10
    assert len(set(cluster_map[x] for x in names[:6]).intersection(set(cluster_map[x] for x in names[6:]))) == 0
    assert cluster_similarity is None
    assert np.min(cluster_distance) == 0
    assert np.max(cluster_distance) == 4
    assert [cluster_weights[i] for i in cluster_names] == [16, 36]
