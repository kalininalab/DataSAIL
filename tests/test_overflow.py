from pathlib import Path
import random
import numpy as np
import pytest

from datasail.sail import datasail
from datasail.reader.read_proteins import read_protein_data
from datasail.cluster.clustering import cluster
from datasail.settings import KW_THREADS, KW_LOGDIR, KW_LINKAGE, KW_TMPDIR
from datasail.solver.overflow import check_dataset


def test_overflow_break():
    random.seed(42)
    np.random.seed(42)
    Path("tmp").mkdir(exist_ok=True, parents=True)
    dataset = read_protein_data(
        Path("data") / "rw_data" / "overflow_data" / "goldstandard.protein_sequences.fasta",
        Path("data") / "rw_data" / "overflow_data" / "weight_map_for_datasail.tsv",
        sim="mmseqs",
        num_clusters=50,
    )
    dataset = cluster(dataset, **{KW_THREADS: 1, KW_LOGDIR: None, KW_LINKAGE: "average", KW_TMPDIR: Path("tmp")})
    dataset, pre_e_name_split_map, pre_e_cluster_split_map, e_split_ratios, e_split_names = check_dataset(
        dataset, 
        [1/11] * 11, 
        ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11"], 
        "break", 
        "single", 
        ["C1e"],
    )
    assert len(dataset.cluster_names) == 51
    assert len(dataset.cluster_weights) == 51
    assert dataset.cluster_similarity.shape == (51, 51)

    assert "C1e" in pre_e_name_split_map
    assert len(pre_e_name_split_map["C1e"]) == 0
    assert "C1e" in pre_e_cluster_split_map
    assert len(pre_e_cluster_split_map["C1e"]) == 0
    
    assert "C1e" in e_split_ratios
    assert all(np.isclose(e_split_ratios["C1e"], [1/11] * 11))
    assert e_split_names == {"C1e": ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11"]}


def test_overflow_assign():
    random.seed(42)
    np.random.seed(42)
    Path("tmp").mkdir(exist_ok=True, parents=True)
    dataset = read_protein_data(
        Path("data") / "rw_data" / "overflow_data" / "goldstandard.protein_sequences.fasta",
        Path("data") / "rw_data" / "overflow_data" / "weight_map_for_datasail.tsv",
        sim="mmseqs",
        num_clusters=50,
    )
    dataset = cluster(dataset, **{KW_THREADS: 1, KW_LOGDIR: None, KW_LINKAGE: "average", KW_TMPDIR: Path("tmp")})
    dataset, pre_e_name_split_map, pre_e_cluster_split_map, e_split_ratios, e_split_names = check_dataset(
        dataset, 
        [1/11] * 11, 
        ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11"], 
        "assign", 
        "average", 
        ["C1e"],
    )
    assert len(dataset.cluster_names) == 49
    assert len(dataset.cluster_weights) == 49
    assert dataset.cluster_similarity.shape == (49, 49)

    assert "C1e" in pre_e_name_split_map
    assert len(pre_e_name_split_map["C1e"].keys()) > 0
    assert "C1e" in pre_e_cluster_split_map
    assert len(pre_e_cluster_split_map["C1e"]) == 1

    assert "C1e" in e_split_ratios
    assert all(np.isclose(e_split_ratios["C1e"], [0.1] * 10))
    assert e_split_names == {"C1e": ["S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11"]}


@pytest.mark.parametrize("overflow", ["break", "assign"])
def test_overflow_full(overflow):
    e_splits, _, _ = datasail(
        techniques=["C1e"],
        e_type="P",
        e_data=Path("data") / "rw_data" / "overflow_data" / "goldstandard.protein_sequences.fasta",
        e_weights=Path("data") / "rw_data" / "overflow_data" / "weight_map_for_datasail.tsv",
        e_sim="mmseqs",
        splits=[2, 2, 2, 2, 2],
        overflow=overflow,
    )
    assert len(e_splits["C1e"][0]) > 240
