from pathlib import Path
import pytest

from datasail.sail import datasail
from datasail.reader.read_proteins import read_protein_data
from datasail.cluster.clustering import cluster
from datasail.settings import KW_THREADS, KW_LOGDIR, KW_LINKAGE
from datasail.solver.overflow import check_dataset


def test_overflow_break():
    dataset = read_protein_data(
        Path("data") / "rw_data" / "overflow_data" / "goldstandard.protein_sequences.fasta",
        Path("data") / "rw_data" / "overflow_data" / "weight_map_for_datasail.tsv",
        sim="mmseqspp",
        num_clusters=50,
    )
    dataset = cluster(dataset, **{KW_THREADS: 1, KW_LOGDIR: None, KW_LINKAGE: "average"})
    dataset, pre_e_name_split_map, pre_e_cluster_split_map, e_split_ratios, e_split_names = check_dataset(
        dataset, 
        [0.2, 0.2, 0.2, 0.2, 0.2], 
        ["S1", "S2", "S3", "S4", "S5"], 
        "break", 
        "single", 
        None, 
        False,
        "C1e",
        False,
    )
    assert len(dataset.cluster_names) == 52
    assert "C1e" in pre_e_name_split_map
    assert len(pre_e_name_split_map["C1e"].keys()) == 0
    assert "C1e" in pre_e_cluster_split_map
    assert len(pre_e_cluster_split_map["C1e"].keys()) == 0
    assert e_split_ratios == {"C1e": [0.2, 0.2, 0.2, 0.2, 0.2]}
    assert e_split_names == {"C1e": ["S1", "S2", "S3", "S4", "S5"]}


def test_overflow_assign():
    dataset = read_protein_data(
        Path("data") / "rw_data" / "overflow_data" / "goldstandard.protein_sequences.fasta",
        Path("data") / "rw_data" / "overflow_data" / "weight_map_for_datasail.tsv",
        sim="mmseqspp",
        num_clusters=50,
    )
    dataset = cluster(dataset, **{KW_THREADS: 1, KW_LOGDIR: None, KW_LINKAGE: "average"})
    dataset, pre_e_name_split_map, pre_e_cluster_split_map, e_split_ratios, e_split_names = check_dataset(
        dataset, 
        [0.2, 0.2, 0.2, 0.2, 0.2], 
        ["S1", "S2", "S3", "S4", "S5"], 
        "assign", 
        "average", 
        None, 
        False,
        "C1e",
        False,
    )
    assert len(dataset.cluster_names) == 49
    assert "C1e" in pre_e_name_split_map
    assert len(pre_e_name_split_map["C1e"].keys()) > 0
    assert "C1e" in pre_e_cluster_split_map
    assert len(pre_e_cluster_split_map["C1e"]) == 1
    assert e_split_ratios == {"C1e": [0.25, 0.25, 0.25, 0.25]}
    assert e_split_names == {"C1e": ["S2", "S3", "S4", "S5"]}


@pytest.mark.parametrize("overflow", ["break", "assign"])
def test_overflow_full(overflow):
    e_splits, _, _ = datasail(
        techniques=["C1e"],
        e_type="P",
        e_data=Path("data") / "rw_data" / "overflow_data" / "goldstandard.protein_sequences.fasta",
        e_weights=Path("data") / "rw_data" / "overflow_data" / "weight_map_for_datasail.tsv",
        e_sim="mmseqspp",
        splits=[2, 2, 2, 2, 2],
        overflow=overflow,
    )
    assert len(e_splits["C1e"][0]) > 240
