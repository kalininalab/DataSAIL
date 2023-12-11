import platform
from pathlib import Path

import numpy as np
import pytest

from datasail.cluster.cdhit import run_cdhit
from datasail.cluster.clustering import additional_clustering, cluster
from datasail.cluster.ecfp import run_ecfp
from datasail.cluster.foldseek import run_foldseek
from datasail.cluster.mash import run_mash
from datasail.cluster.mmseqs2 import run_mmseqs
from datasail.cluster.mmseqspp import run_mmseqspp
from datasail.cluster.tmalign import run_tmalign
from datasail.cluster.wlk import run_wlk
from datasail.reader.read_proteins import parse_fasta, read_folder
from datasail.reader.utils import DataSet, read_csv
from datasail.reader.validate import check_cdhit_arguments, check_foldseek_arguments, check_mmseqs_arguments, \
    check_mash_arguments, check_mmseqspp_arguments
from datasail.settings import P_TYPE, FORM_FASTA, MMSEQS, CDHIT, KW_LOGDIR, KW_THREADS, FOLDSEEK, TMALIGN, MMSEQSPP


@pytest.mark.todo
def test_additional_clustering():
    names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    base_map = dict((n, n) for n in names)
    similarity = np.asarray([
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ])
    distance = np.asarray([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
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

    s_dataset = additional_clustering(s_dataset, n_clusters=5)
    assert len(s_dataset.cluster_names) == 5
    assert set(s_dataset.cluster_names) == set(s_dataset.cluster_map.values())
    assert set(s_dataset.cluster_names) == set(s_dataset.cluster_weights.keys())
    assert set(names) == set(s_dataset.cluster_map.keys())
    assert len(set(s_dataset.cluster_map[x] for x in names[:6]).intersection(set(
        s_dataset.cluster_map[x] for x in names[6:]
    ))) == 0
    assert np.min(s_dataset.cluster_similarity) == 0
    assert np.max(s_dataset.cluster_similarity) == 1
    assert s_dataset.cluster_distance is None
    # assert [s_dataset.cluster_weights[i] for i in s_dataset.cluster_names] == [18, 12, 6, 12, 4]

    d_dataset = additional_clustering(d_dataset, n_clusters=5)
    assert len(d_dataset.cluster_names) == 5
    assert set(d_dataset.cluster_names) == set(d_dataset.cluster_map.values())
    assert set(d_dataset.cluster_names) == set(d_dataset.cluster_weights.keys())
    assert set(names) == set(d_dataset.cluster_map.keys())
    assert len(set(d_dataset.cluster_map[x] for x in names[:6]).intersection(set(
        d_dataset.cluster_map[x] for x in names[6:]
    ))) == 0
    assert d_dataset.cluster_similarity is None
    assert np.min(d_dataset.cluster_distance) == 0
    assert np.max(d_dataset.cluster_distance) == 1
    # assert [d_dataset.cluster_weights[i] for i in d_dataset.cluster_names] == [16, 36]


def protein_fasta_data(algo):
    data = parse_fasta(Path("data") / "pipeline" / "seqs.fasta")
    if algo == CDHIT:
        args = check_cdhit_arguments("")
    elif algo == MMSEQS:
        args = check_mmseqs_arguments("")
    elif algo == MMSEQSPP:
        args = check_mmseqspp_arguments("")
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    return DataSet(
        type="M",
        data=data,
        names=list(sorted(data.keys())),
        location=Path("data") / "pipeline" / "seqs.fasta",
        args=args,
    )


def protein_pdb_data(algo):
    data = dict((k, v) for k, v in read_folder(Path("data") / "pipeline" / "pdbs", "pdb"))
    return DataSet(
        type="M",
        data=data,
        names=list(sorted(data.keys())),
        location=Path("data") / "pipeline" / "pdbs",
        args=check_foldseek_arguments() if algo == FOLDSEEK else None,
    )


@pytest.fixture
def molecule_data():
    data = dict((k, v) for k, v in read_csv(Path("data") / "pipeline" / "drugs.tsv"))
    return DataSet(
        type="M",
        data=data,
        names=list(sorted(data.keys())),
        location=Path("data") / "pipeline" / "drugs.tsv",
    )


@pytest.fixture
def genome_fasta_data():
    data = dict((k, v) for k, v in read_folder(Path("data") / "genomes", "fna"))
    return DataSet(
        type="M",
        data=data,
        names=list(sorted(data.keys())),
        location=Path("data") / "genomes",
        args=check_mash_arguments(""),
    )


@pytest.mark.nowin
def test_cdhit_protein():
    data = protein_fasta_data(CDHIT)
    if platform.system() == "Windows":
        pytest.skip("CD-HIT is not supported on Windows")
    check_clustering(*run_cdhit(data, 1, Path()), dataset=data)


@pytest.mark.todo
@pytest.mark.nowin
def test_cdhit_genome(genome_fasta_data):
    if platform.system() == "Windows":
        pytest.skip("CD-HIT is not supported on Windows")
    output = run_cdhit(genome_fasta_data, 1, Path())
    check_clustering(*output, dataset=genome_fasta_data)


def test_ecfp_molecule(molecule_data):
    check_clustering(*run_ecfp(molecule_data), dataset=molecule_data)


@pytest.mark.nowin
def test_foldseek_protein():
    data = protein_pdb_data(FOLDSEEK)
    if platform.system() == "Windows":
        pytest.skip("Foldseek is not supported on Windows")
    check_clustering(*run_foldseek(data, 1, Path()), dataset=data)


@pytest.mark.nowin
def test_mash_genomic(genome_fasta_data):
    if platform.system() == "Windows":
        pytest.skip("MASH is not supported on Windows")
    check_clustering(*run_mash(genome_fasta_data, 1, Path()), dataset=genome_fasta_data)


@pytest.mark.nowin
def test_mmseqs2_protein():
    data = protein_fasta_data(MMSEQS)
    if platform.system() == "Windows":
        pytest.skip("MMseqs2 is not supported on Windows")
    check_clustering(*run_mmseqs(data, 1, Path()), dataset=data)


@pytest.mark.nowin
def test_mmseqspp_protein():
    data = protein_fasta_data(MMSEQSPP)
    if platform.system() == "Windows":
        pytest.skip("MMseqs2 is not supported on Windows")
    check_clustering(*run_mmseqspp(data, 1, Path()), dataset=data)


@pytest.mark.nowin
@pytest.mark.todo
def test_tmalign_protein():
    data = protein_pdb_data(TMALIGN)
    if platform.system() == "Windows":
        pytest.skip("TM-align is not supported on Windows")
    check_clustering(*run_tmalign(data), dataset=data)


@pytest.mark.nowin
def test_wlkernel_protein():
    protein_data = protein_pdb_data(FOLDSEEK)
    check_clustering(*run_wlk(protein_data), dataset=protein_data)


def test_wlkernel_molecule(molecule_data):
    check_clustering(*run_wlk(molecule_data), dataset=molecule_data)


@pytest.mark.parametrize("algo", [CDHIT, MMSEQS])
def test_force_clustering(algo):
    base = Path("data") / "rw_data"
    dataset = cluster(DataSet(
        type=P_TYPE,
        format=FORM_FASTA,
        names=[f"Seq{i + 1:04d}" for i in range(len(open(base / "pdbbind_clean.fasta", "r").readlines()))],
        location=base / "pdbbind_clean.fasta",
        similarity=algo,
        args=check_cdhit_arguments("") if algo == CDHIT else check_mmseqs_arguments(""),

    ), **{KW_THREADS: 1, KW_LOGDIR: Path()})
    assert len(dataset.cluster_names) <= 100


def check_clustering(names, mapping, matrix, dataset):
    assert list(sorted(names)) == list(sorted(set(mapping.values())))
    assert list(sorted(mapping.keys())) == list(sorted(dataset.names))
    assert tuple(matrix.shape) == (len(names), len(names))
    assert np.min(matrix) >= 0
    assert np.max(matrix) <= 1
    assert len(names) <= len(dataset.names)
