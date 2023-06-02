import os

import pytest

from datasail.cluster.caching import store_to_cache, load_from_cache
from datasail.cluster.clustering import cluster, similarity_clustering
from datasail.reader.read_molecules import read_molecule_data
from datasail.reader.utils import read_csv


@pytest.mark.parametrize("size", ["perf_7_3", "perf_70_30"])
def test_caching(size):
    os.makedirs(f"data/{size}/splits/", exist_ok=True)
    dataset, _ = read_molecule_data(
        data="data/perf_7_3/lig.tsv",
        sim="data/perf_7_3/lig_sim.tsv",
        inter=list(read_csv("data/perf_7_3/inter.tsv")),
        index=0,
    )

    dataset = cluster(dataset, **{"output": "data/perf_7_3/splits/", "threads": 1, "log_dir": "log.txt"})

    store_to_cache(dataset, **{"cache": True, "cache_dir": "./test_cache/"})

    assert load_from_cache(dataset, **{"cache": True, "cache_dir": "./test_cache/"}) is not None
