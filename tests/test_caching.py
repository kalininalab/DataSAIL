import copy
import os

import pytest

from datasail.cluster.caching import store_to_cache, load_from_cache
from datasail.cluster.clustering import cluster
from datasail.reader.read_molecules import read_molecule_data
from datasail.reader.utils import read_csv


@pytest.mark.parametrize("size", ["perf_7_3", "perf_70_30"])
def test_caching(size):
    os.makedirs(f"data/{size}/splits/", exist_ok=True)
    dataset = read_molecule_data(
        data="data/perf_7_3/lig.tsv",
        sim="data/perf_7_3/lig_sim.tsv",
        inter=list(read_csv("data/perf_7_3/inter.tsv")),
        index=0,
    )

    dataset = cluster(dataset, **{"output": "data/perf_7_3/splits/", "threads": 1, "log_dir": "log.txt"})

    store_to_cache(dataset, **{"cache": True, "cache_dir": "./test_cache/"})

    assert load_from_cache(dataset, **{"cache": True, "cache_dir": "./test_cache/"}) is not None


@pytest.mark.parametrize("size", ["perf_7_3", "perf_70_30"])
def test_shuffling(size):
    os.makedirs(f"data/{size}/splits/", exist_ok=True)
    original_dataset = read_molecule_data(
        data="data/perf_7_3/lig.tsv",
        sim="data/perf_7_3/lig_sim.tsv",
        inter=list(read_csv("data/perf_7_3/inter.tsv")),
        index=0,
    )

    original_dataset = cluster(original_dataset, **{"output": "data/perf_7_3/splits/", "threads": 1, "log_dir": "log.txt"})
    shuffle_dataset = copy.deepcopy(original_dataset)
    for i in range(5):
        shuffle_dataset.shuffle()
        check_true_equality(original_dataset, shuffle_dataset)
        assert original_dataset.names != shuffle_dataset.names
        assert original_dataset.cluster_names != shuffle_dataset.cluster_names


def check_true_equality(truth, shuffled):
    assert truth.type == shuffled.type
    assert truth.format == shuffled.format
    assert truth.args == shuffled.args
    assert sorted(truth.names) == sorted(shuffled.names)
    assert truth.id_map == shuffled.id_map
    assert sorted(truth.cluster_names) == sorted(shuffled.cluster_names)
    assert truth.data == shuffled.data
    assert truth.cluster_map == shuffled.cluster_map
    assert truth.location == shuffled.location
    assert truth.weights == shuffled.weights
    assert truth.cluster_weights == shuffled.cluster_weights
    assert truth.threshold == shuffled.threshold

    if truth.similarity is not None:
        check_matrices(truth.names, truth.similarity, shuffled.names, shuffled.similarity)
    else:
        assert shuffled.similarity is None

    if truth.distance is not None:
        check_matrices(truth.names, truth.distance, shuffled.names, shuffled.distance)
    else:
        assert shuffled.distance is None

    if truth.cluster_similarity is not None:
        check_matrices(truth.cluster_names, truth.cluster_similarity, shuffled.cluster_names, shuffled.cluster_similarity)
    else:
        assert shuffled.cluster_similarity is None

    if truth.cluster_distance is not None:
        check_matrices(truth.cluster_names, truth.cluster_distance, shuffled.cluster_names, shuffled.cluster_distance)
    else:
        assert shuffled.cluster_distance is None


def check_matrices(names_a, matrix_a, names_b, matrix_b):
    names_map = dict((n, i) for i, n in enumerate(names_b))
    for i, name_a in enumerate(names_a):
        for j, name_b in enumerate(names_a):
            assert abs(matrix_a[i, j] - matrix_b[names_map[name_a], names_map[name_b]]) < 1e-10
