import numpy as np
import pytest

from datasail.cluster.cdhit import run_cdhit
from datasail.cluster.clustering import additional_clustering
from datasail.cluster.ecfp import run_ecfp
from datasail.cluster.foldseek import run_foldseek
from datasail.cluster.mash import run_mash
from datasail.cluster.mmseqs2 import run_mmseqs
from datasail.reader.read_proteins import parse_fasta, read_pdb_folder
from datasail.reader.utils import DataSet, read_csv


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


@pytest.fixture
def protein_fasta_data():
    data = parse_fasta("data/pipeline/seqs.fasta")
    return DataSet(type="M", data=data, names=list(sorted(data.keys())))


@pytest.fixture
def protein_pdb_data():
    data = dict((k, v) for k, v in read_pdb_folder("data/pipeline/pdbs"))
    return DataSet(type="M", data=data, names=list(sorted(data.keys())))


@pytest.fixture
def molecule_data():
    data = dict((k, v) for k, v in read_csv("data/pipeline/drugs.tsv"))
    return DataSet(type="M", data=data, names=list(sorted(data.keys())))


@pytest.fixture
def genome_fasta_data():
    data = parse_fasta("data/pipeline/seqs.fasta")
    return DataSet(type="M", data=data, names=list(sorted(data.keys())))


def test_cdhit_protein(protein_fasta_data):
    check_clustering(*run_cdhit(protein_fasta_data), protein_fasta_data)


@pytest.mark.todo
def test_cdhit_genome(genome_fasta_data):
    check_clustering(*run_cdhit(genome_fasta_data), genome_fasta_data)


def test_ecfp_molecule(molecule_data):
    check_clustering(*run_ecfp(molecule_data), molecule_data)


def test_foldseek_protein(protein_pdb_data):
    check_clustering(*run_foldseek(protein_pdb_data), protein_pdb_data)


@pytest.mark.todo
def test_mash_genomic(genome_fasta_data):
    check_clustering(*run_mash(genome_fasta_data), genome_fasta_data)


def test_mmseqs2_protein(protein_fasta_data):
    check_clustering(*run_mmseqs(protein_fasta_data), protein_fasta_data)


def test_wlkernel_protein(protein_pdb_data):
    check_clustering(*run_mmseqs(protein_pdb_data), protein_pdb_data)


def test_wlkernel_molecule(molecule_data):
    check_clustering(*run_mmseqs(molecule_data), molecule_data)


def check_clustering(names, mapping, matrix, dataset):
    assert list(sorted(names)) == list(sorted(set(mapping.values())))
    assert list(sorted(mapping.keys())) == list(sorted(dataset.names))
    assert tuple(matrix.shape) == (len(names), len(names))
    assert np.min(matrix) >= 0
    assert np.max(matrix) <= 1
    assert len(names) <= len(dataset.names)
