import platform
from pathlib import Path
from typing import get_args

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from datasail.cluster.cdhit import run_cdhit
from datasail.cluster.clustering import additional_clustering, cluster, force_clustering
from datasail.cluster.diamond import run_diamond
from datasail.cluster.ecfp import run_ecfp
from datasail.cluster.foldseek import run_foldseek
from datasail.cluster.mash import run_mash
from datasail.cluster.mmseqs2 import run_mmseqs
from datasail.cluster.mmseqspp import run_mmseqspp
from datasail.cluster.vectors import run_vector, SIM_OPTIONS
from datasail.cluster.tmalign import run_tmalign
from datasail.cluster.wlk import run_wlk

from datasail.reader.read_proteins import read_folder
from datasail.reader.utils import DataSet, read_csv, parse_fasta
from datasail.reader.validate import check_cdhit_arguments, check_foldseek_arguments, check_mmseqs_arguments, \
    check_mash_arguments, check_mmseqspp_arguments, check_diamond_arguments
from datasail.sail import datasail
from datasail.settings import P_TYPE, FORM_FASTA, MMSEQS, CDHIT, KW_LOGDIR, KW_THREADS, FOLDSEEK, TMALIGN, MMSEQSPP, \
    DIAMOND, MASH


@pytest.fixture()
def md_calculator():
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    return MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)


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
    s_dataset.classes = {0: 0}
    s_dataset.cluster_stratification = {n: np.array([0]) for n in names}
    d_dataset = DataSet()
    d_dataset.cluster_names = names
    d_dataset.cluster_map = base_map
    d_dataset.cluster_weights = weights
    d_dataset.cluster_similarity = None
    d_dataset.cluster_distance = distance
    d_dataset.classes = {0: 0}
    d_dataset.cluster_stratification = {n: np.array([0]) for n in names}

    s_dataset = additional_clustering(s_dataset, n_clusters=5, linkage="average")
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

    d_dataset = additional_clustering(d_dataset, n_clusters=5, linkage="average")
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


def test_force_clustering():
    # Create a mock DataSet object
    dataset = DataSet()
    dataset.cluster_names = ["cluster1", "cluster2", "cluster3", "cluster4", "cluster5"]
    dataset.cluster_map = {"item1": "cluster1", "item2": "cluster2", "item3": "cluster3", "item4": "cluster4", "item5": "cluster5"}
    dataset.cluster_weights = {"cluster1": 10, "cluster2": 20, "cluster3": 30, "cluster4": 40, "cluster5": 50}
    dataset.cluster_similarity = np.array([
        [1, 0.5, 0.3, 0.2, 0.1],
        [0.5, 1, 0.4, 0.3, 0.2],
        [0.3, 0.4, 1, 0.5, 0.4],
        [0.2, 0.3, 0.5, 1, 0.6],
        [0.1, 0.2, 0.4, 0.6, 1]
    ])
    dataset.classes = {0: 0}
    dataset.cluster_stratification = {"cluster1": np.array([0]), "cluster2": np.array([0]), "cluster3": np.array([0]), "cluster4": np.array([0]), "cluster5": np.array([0])}
    dataset.num_clusters = 3

    # Call the force_clustering function
    result_dataset = force_clustering(dataset)

    # Assert that the number of clusters in the returned DataSet object is equal to the expected number of clusters
    assert len(result_dataset.cluster_names) == dataset.num_clusters


def protein_fasta_data(algo):
    data = parse_fasta(Path("data") / "pipeline" / "seqs.fasta")
    if algo == CDHIT:
        args = check_cdhit_arguments("")
    elif algo == MMSEQS:
        args = check_mmseqs_arguments("")
    elif algo == MMSEQSPP:
        args = check_mmseqspp_arguments("")
    elif algo == DIAMOND:
        args = check_diamond_arguments("")
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


def molecule_data():
    data = dict((k, v) for k, v in read_csv(Path("data") / "pipeline" / "drugs.tsv", "\t"))
    return DataSet(
        type="M",
        data=data,
        names=list(sorted(data.keys())),
        location=Path("data") / "pipeline" / "drugs.tsv",
    )


def genome_fasta_data(mode):
    data = dict((k, v) for k, v in read_folder(Path("data") / "genomes", "fna"))
    return DataSet(
        type="M",
        data=data,
        names=list(sorted(data.keys())),
        location=Path("data") / "genomes",
        args=check_mash_arguments("") if mode == MASH else check_cdhit_arguments(""),
    )


@pytest.mark.full
def test_cdhit_protein():
    data = protein_fasta_data(CDHIT)
    if platform.system() == "Windows":
        pytest.skip("CD-HIT is not supported on Windows")
    run_cdhit(data, 1, Path())
    check_clustering(data)


@pytest.mark.full
@pytest.mark.todo
def test_cdhit_genome():
    data = genome_fasta_data(CDHIT)
    if platform.system() == "Windows":
        pytest.skip("CD-HIT is not supported on Windows")
    run_cdhit(data, 1, Path())
    check_clustering(data)


def test_ecfp_molecule():
    data = molecule_data()
    run_ecfp(data)
    check_clustering(data)


@pytest.mark.full
def test_foldseek_protein():
    data = protein_pdb_data(FOLDSEEK)
    if platform.system() == "Windows":
        pytest.skip("Foldseek is not supported on Windows")
    run_foldseek(data, 1, Path())
    check_clustering(data)


@pytest.mark.full
def test_mash_genomic():
    data = genome_fasta_data(MASH)
    if platform.system() == "Windows":
        pytest.skip("MASH is not supported on Windows")
    run_mash(data, 1, Path())
    check_clustering(data)


@pytest.mark.full
def test_diamond_protein():
    data = protein_fasta_data(DIAMOND)
    if platform.system() == "Windows":
        pytest.skip("DIAMOND is not supported on Windows")
    run_diamond(data, 1, Path())
    check_clustering(data)


@pytest.mark.full
def test_mmseqs2_protein():
    data = protein_fasta_data(MMSEQS)
    if platform.system() == "Windows":
        pytest.skip("MMseqs2 is not supported on Windows")
    run_mmseqs(data, 1, Path())
    check_clustering(data)


@pytest.mark.full
def test_mmseqspp_protein():
    data = protein_fasta_data(MMSEQSPP)
    if platform.system() == "Windows":
        pytest.skip("MMseqs2 is not supported on Windows")
    run_mmseqspp(data, 1, Path())
    check_clustering(data)


@pytest.mark.parametrize("algo", ["FP", "MD"])
@pytest.mark.parametrize("in_type", ["Original", "List", "Numpy"])
@pytest.mark.parametrize("method", [
    "allbit", "asymmetric", "braunblanquet", "cosine", "dice", "kulczynski", "onbit", "rogotgoldberg", "russel",
    "sokal", "tanimoto", "canberra", "hamming", "jaccard", "matching", "rogerstanimoto", "sokalmichener", "yule"
])
def test_vector(md_calculator, algo, in_type, method) :
    data = molecule_data()
    if algo == "FP":
        embed = lambda x: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2, nBits=1024)
    else:
        embed = lambda x: md_calculator.CalcDescriptors(Chem.MolFromSmiles(x))
    if in_type == "Original":
        wrap = lambda x: x
    elif in_type == "List":
        wrap = lambda x: [int(y) for y in x]
    else:
        wrap = lambda x: np.array([max(-2_147_483_648, min(2_147_483_647, int(y))) for y in x])
    data.data = dict((k, wrap(embed(v))) for k, v in data.data.items())
    if (algo == "MD" and in_type == "Original" and method in get_args(SIM_OPTIONS)) or method == "mahalanobis":
        with pytest.raises(ValueError):
            run_vector(data, method)
    else:
        run_vector(data, method)
        check_clustering(data)


@pytest.mark.parametrize("method", [
    "allbit", "asymmetric", "braunblanquet", "cosine", "dice", "kulczynski", "onbit", "rogotgoldberg", "russel",
    "sokal", "canberra", "hamming", "jaccard", "matching", "rogerstanimoto", "sokalmichener", "yule"
])
def test_vector_edge(method):
    dataset = DataSet(
        names=["A", "B", "C", "D", "E", "F", "G", "H"],
        data={
            "A": np.array([1, 1, 1]),
            "B": np.array([1, 1, 0]),
            "C": np.array([1, 0, 1]),
            "D": np.array([0, 1, 1]),
            "E": np.array([1, 0, 0]),
            "F": np.array([0, 1, 0]),
            "G": np.array([0, 0, 1]),
            "H": np.array([0, 0, 0]),
        },
    )
    run_vector(dataset, method)
    check_clustering(dataset)


@pytest.mark.full
@pytest.mark.todo
def test_tmalign_protein():
    data = protein_pdb_data(TMALIGN)
    if platform.system() == "Windows":
        pytest.skip("TM-align is not supported on Windows")
    run_tmalign(data)
    check_clustering(data)


@pytest.mark.full
def test_wlkernel_protein():
    protein_data = protein_pdb_data(FOLDSEEK)
    run_wlk(protein_data)
    check_clustering(protein_data)


def test_wlkernel_molecule():
    data = molecule_data()
    run_wlk(data)
    check_clustering(data)


@pytest.mark.full
@pytest.mark.parametrize("algo", [CDHIT, MMSEQS])
def test_clustering(algo):
    base = Path("data") / "rw_data"
    seqs = parse_fasta(base / "pdbbind_clean.fasta")
    dataset = cluster(
        DataSet(
            type=P_TYPE,
            format=FORM_FASTA,
            names=list(seqs.keys()),
            data=seqs,
            weights={k: 1 for k in seqs.keys()},
            location=base / "pdbbind_clean.fasta",
            similarity=algo,
            stratification={k: 0 for k in seqs.keys()},
            class_oh=np.eye(1),
            classes={0: 0},
            args=check_cdhit_arguments("") if algo == CDHIT else check_mmseqs_arguments(""),
        ),
        num_clusters=50,
        linkage="average",
        **{KW_THREADS: 1, KW_LOGDIR: Path()})
    assert len(dataset.cluster_names) <= 100


def test_distance_input():
    names = list(pd.read_csv(Path("data") / "rw_data" / "distance_matrix.tsv", sep="\t", header=0, index_col=0).columns)
    e_splits, _, _ = datasail(
        techniques=["C1e"],
        e_type=P_TYPE,
        e_data=[(name, "A" * i) for i, name in enumerate(names)],
        e_dist=Path("data") / "rw_data" / "distance_matrix.tsv",
        splits=[0.7, 0.3],
        names=["train", "test"],
    )
    assert "C1e" in e_splits


def check_clustering(dataset):
    if dataset.cluster_similarity is not None:
        matrix = dataset.cluster_similarity
    elif dataset.cluster_distance is not None:
        matrix = dataset.cluster_distance
    else:
        raise ValueError("No similarity or distance matrix found")

    assert list(sorted(dataset.cluster_names)) == list(sorted(set(dataset.cluster_map.values())))
    assert list(sorted(dataset.cluster_map.keys())) == list(sorted(dataset.names))
    assert tuple(matrix.shape) == (len(dataset.cluster_names), len(dataset.cluster_names))
    assert np.min(matrix) >= 0
    assert np.max(matrix) <= 1
    assert len(dataset.cluster_names) <= len(dataset.names)
