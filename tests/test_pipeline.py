import os.path
import shutil

import pytest

from datasail.sail import sail
from tests.utils import check_folder, run_sail


@pytest.mark.parametrize("data", [
    (True, False, None, None, None, False, None, None, False, "ICSf"),
    (True, False, "wlk", None, None, False, None, None, False, "ICSf"),
    (False, False, None, None, None, False, None, None, False, "ICSf"),
    # (False, False, "mmseqs", None, None, False, None, None, False, "ICP"),
    (False, False, "data/pipeline/prot_sim.tsv", None, None, False, None, None, False, "ICSf"),
    (False, False, None, "data/pipeline/prot_dist.tsv", None, False, None, None, False, "ICSf"),
    (False, True, None, None, None, False, None, None, False, "ICSf"),
    (None, False, None, None, "data/pipeline/drugs.tsv", False, None, None, False, "ICSe"),
    (False, False, None, None, "data/pipeline/drugs.tsv", False, None, None, False, "ICSe"),
    (False, False, None, None, "data/pipeline/drugs.tsv", True, None, None, False, "ICSe"),
    (False, False, None, None, "data/pipeline/drugs.tsv", False, "data/pipeline/drug_sim.tsv", None, False, "ICSe"),
    (False, False, None, None, "data/pipeline/drugs.tsv", True, "wlk", None, False, "ICSe"),  # <-- 10/11
    (False, False, None, None, "data/pipeline/drugs.tsv", False, None, "data/pipeline/drug_dist.tsv", False, "ICSe"),
    (True, False, "wlk", None, "data/pipeline/drugs.tsv", False, "wlk", None, True, "ICSf"),
    # (False, False, None, None, "data/pipeline/drugs.tsv", False, None, "data/pipeline/drug_dist.tsv", False, "CCSe"),
    # (False, False, "data/pipeline/prot_sim.tsv", None, "data/pipeline/drugs.tsv", False, None,
    #  "data/pipeline/drug_dist.tsv", False, "CCSf"),
    # (False, False, None, None, "data/pipeline/drugs.tsv", False, None, "data/pipeline/drug_dist.tsv", False, "CCSe"),
])
def test_pipeline(data):
    pdb, prot_weights, prot_sim, prot_dist, drugs, drug_weights, drug_sim, drug_dist, inter, mode = data
    shutil.rmtree("data/pipeline/out/", ignore_errors=True)

    sail(
        inter="data/pipeline/inter.tsv" if inter else None,
        output="data/pipeline/out/",
        max_sec=10,
        max_sol=10,
        verbosity="I",
        techniques=[mode],
        vectorized=False,
        splits=[0.67, 0.33] if mode in ["IC", "CC"] else [0.7, 0.3],
        names=["train", "test"],
        epsilon=0.25,
        runs=1,
        e_type=None if drugs is None else "M",
        e_data=drugs,
        e_weights="data/pipeline/drug_weights.tsv" if drug_weights else None,
        e_sim=drug_sim,
        e_dist=drug_dist,
        e_max_sim=1,
        e_max_dist=1,
        e_args="",
        f_type=None if pdb is None else "P",
        f_data=None if pdb is None else ("data/pipeline/pdbs" if pdb else "data/pipeline/seqs.fasta"),
        f_weights="data/pipeline/prot_weights.tsv" if prot_weights else None,
        f_sim=prot_sim,
        f_dist=prot_dist,
        f_max_sim=1,
        f_max_dist=1,
        f_args="",
        cache=False,
        cache_dir=None,
        solver="SCIP",
        threads=1,
    )

    check_folder(
        output_root="data/pipeline/out/" + mode,
        epsilon=0.25,
        e_weight="data/pipeline/drug_weights.tsv" if drug_weights else None,
        f_weight="data/pipeline/prot_weights.tsv" if prot_weights else None,
        e_filename="Molecule_drugs_splits.tsv" if mode[-1] == "e" else None,
        f_filename=f"Protein_{'pdbs' if pdb else 'seqs'}_splits.tsv" if mode[-1] == "f" else None,
    )


def test_report():
    shutil.rmtree("data/perf_7_3/out", ignore_errors=True)

    run_sail(
        inter="data/perf_7_3/inter.tsv",
        output="data/perf_7_3/out/",
        max_sec=100,
        techniques=["R", "ICSe", "ICSf", "ICD", "CCSe", "CCSf", "CCD"],
        splits=[0.7, 0.3],
        names=["train", "test"],
        epsilon=0.25,
        e_type="M",
        e_data="data/perf_7_3/lig.tsv",
        e_sim="data/perf_7_3/lig_sim.tsv",
        f_type="P",
        f_data="data/perf_7_3/prot.fasta",
        f_sim="data/perf_7_3/prot_sim.tsv",
        solver="SCIP",
    )

    assert os.path.isfile("data/perf_7_3/out/lig_similarity.png")
    assert os.path.isfile("data/perf_7_3/out/prot_similarity.png")

    assert os.path.isdir("data/perf_7_3/out/R")
    assert len(os.listdir("data/perf_7_3/out/R")) == 1
    check_assignment_tsv("data/perf_7_3/out/R/inter.tsv")

    assert os.path.isdir("data/perf_7_3/out/ICSe")
    assert len(os.listdir("data/perf_7_3/out/ICSe")) == 2
    check_assignment_tsv("data/perf_7_3/out/ICSe/Molecule_lig_splits.tsv")
    check_assignment_tsv("data/perf_7_3/out/ICSe/inter.tsv")

    assert os.path.isdir("data/perf_7_3/out/ICSf")
    assert len(os.listdir("data/perf_7_3/out/ICSf")) == 2
    check_assignment_tsv("data/perf_7_3/out/ICSf/Protein_prot_splits.tsv")
    check_assignment_tsv("data/perf_7_3/out/ICSf/inter.tsv")

    assert os.path.isdir("data/perf_7_3/out/ICD")
    assert len(os.listdir("data/perf_7_3/out/ICD")) == 3
    check_assignment_tsv("data/perf_7_3/out/ICD/inter.tsv")
    check_assignment_tsv("data/perf_7_3/out/ICD/Molecule_lig_splits.tsv")
    check_assignment_tsv("data/perf_7_3/out/ICD/Protein_prot_splits.tsv")

    assert os.path.isdir("data/perf_7_3/out/CCSe")
    assert len(os.listdir("data/perf_7_3/out/CCSe")) == 6
    assert os.path.isfile("data/perf_7_3/out/CCSe/Molecule_lig_clusters.png")
    assert os.path.isfile("data/perf_7_3/out/CCSe/Molecule_lig_splits.png")
    assert os.path.isfile("data/perf_7_3/out/CCSe/Molecule_lig_cluster_hist.png")
    check_assignment_tsv("data/perf_7_3/out/CCSe/inter.tsv")
    check_identity_tsv("data/perf_7_3/out/CCSe/Molecule_lig_clusters.tsv")
    check_assignment_tsv("data/perf_7_3/out/CCSe/Molecule_lig_splits.tsv")

    assert os.path.isdir("data/perf_7_3/out/CCSf")
    assert len(os.listdir("data/perf_7_3/out/CCSf")) == 6
    assert os.path.isfile("data/perf_7_3/out/CCSf/Protein_prot_clusters.png")
    assert os.path.isfile("data/perf_7_3/out/CCSf/Protein_prot_splits.png")
    assert os.path.isfile("data/perf_7_3/out/CCSf/Protein_prot_cluster_hist.png")
    check_assignment_tsv("data/perf_7_3/out/CCSf/inter.tsv")
    check_identity_tsv("data/perf_7_3/out/CCSf/Protein_prot_clusters.tsv")
    check_assignment_tsv("data/perf_7_3/out/CCSf/Protein_prot_splits.tsv")

    assert os.path.isdir("data/perf_7_3/out/CCD")
    assert len(os.listdir("data/perf_7_3/out/CCD")) == 11
    assert os.path.isfile("data/perf_7_3/out/CCD/Molecule_lig_clusters.png")
    assert os.path.isfile("data/perf_7_3/out/CCD/Molecule_lig_splits.png")
    assert os.path.isfile("data/perf_7_3/out/CCD/Molecule_lig_cluster_hist.png")
    assert os.path.isfile("data/perf_7_3/out/CCD/Protein_prot_clusters.png")
    assert os.path.isfile("data/perf_7_3/out/CCD/Protein_prot_splits.png")
    assert os.path.isfile("data/perf_7_3/out/CCD/Protein_prot_cluster_hist.png")
    check_assignment_tsv("data/perf_7_3/out/CCD/inter.tsv")
    check_identity_tsv("data/perf_7_3/out/CCD/Molecule_lig_clusters.tsv")
    check_assignment_tsv("data/perf_7_3/out/CCD/Molecule_lig_splits.tsv")
    check_identity_tsv("data/perf_7_3/out/CCD/Protein_prot_clusters.tsv")
    check_assignment_tsv("data/perf_7_3/out/CCD/Protein_prot_splits.tsv")


def test_report_ICD():
    shutil.rmtree("data/perf_7_3/out", ignore_errors=True)

    run_sail(
        inter="data/perf_7_3/inter.tsv",
        output="data/perf_7_3/out/",
        max_sec=100,
        techniques=["ICD"],
        splits=[0.7, 0.3],
        names=["train", "test"],
        epsilon=0.25,
        e_type="M",
        e_data="data/perf_7_3/lig.tsv",
        e_sim="data/perf_7_3/lig_sim.tsv",
        f_type="P",
        f_data="data/perf_7_3/prot.fasta",
        f_sim="data/perf_7_3/prot_sim.tsv",
        solver="SCIP",
    )
    assert os.path.isdir("data/perf_7_3/out/ICD")
    assert len(os.listdir("data/perf_7_3/out/ICD")) == 3
    check_assignment_tsv("data/perf_7_3/out/ICD/inter.tsv")
    check_assignment_tsv("data/perf_7_3/out/ICD/Molecule_lig_splits.tsv")
    check_assignment_tsv("data/perf_7_3/out/ICD/Protein_prot_splits.tsv")


@pytest.mark.todo
def test_report_repeat():
    shutil.rmtree("data/perf_7_3/out", ignore_errors=True)

    run_sail(
        inter="data/perf_7_3/inter.tsv",
        output="data/perf_7_3/out/",
        max_sec=100,
        techniques=["R", "ICSe", "ICSf", "ICD", "CCSe", "CCSf", "CCD"],
        splits=[0.7, 0.3],
        names=["train", "test"],
        epsilon=0.25,
        e_type="M",
        e_data="data/perf_7_3/lig.tsv",
        e_sim="data/perf_7_3/lig_sim.tsv",
        f_type="P",
        f_data="data/perf_7_3/prot.fasta",
        f_sim="data/perf_7_3/prot_sim.tsv",
        solver="SCIP",
        runs=3,
    )

    assert os.path.isfile("data/perf_7_3/out/lig_similarity.png")
    assert os.path.isfile("data/perf_7_3/out/prot_similarity.png")

    for i in range(1, 4):

        assert os.path.isdir(f"data/perf_7_3/out/R_{i}")
        assert len(os.listdir(f"data/perf_7_3/out/R_{i}")) == 1

        assert os.path.isdir(f"data/perf_7_3/out/ICSe_{i}")
        assert len(os.listdir(f"data/perf_7_3/out/ICSe_{i}")) == 2

        assert os.path.isdir(f"data/perf_7_3/out/ICSf_{i}")
        assert len(os.listdir(f"data/perf_7_3/out/ICSf_{i}")) == 2

        assert os.path.isdir(f"data/perf_7_3/out/ICD_{i}")
        assert len(os.listdir(f"data/perf_7_3/out/ICD_{i}")) == 3

        assert os.path.isdir(f"data/perf_7_3/out/CCSe_{i}")
        assert len(os.listdir(f"data/perf_7_3/out/CCSe_{i}")) == 6

        assert os.path.isdir(f"data/perf_7_3/out/CCSf_{i}")
        assert len(os.listdir(f"data/perf_7_3/out/CCSf_{i}")) == 6

        assert os.path.isdir(f"data/perf_7_3/out/CCD_{i}")
        assert len(os.listdir(f"data/perf_7_3/out/CCD_{i}")) == 11


@pytest.mark.todo
def test_genomes():
    sail(
        inter=None,
        output="data/genomes/out/",
        max_sec=100,
        max_sol=10,
        verbosity="I",
        techniques=["ICSe", "CCSe"],
        vectorized=True,
        splits=[0.7, 0.3],
        names=["train", "test"],
        epsilon=0.25,
        runs=1,
        e_type="M",
        e_data="data/perf_7_3/lig.tsv",
        e_weights=None,
        e_sim="data/perf_7_3/lig_sim.tsv",
        e_dist=None,
        e_max_sim=1,
        e_max_dist=1,
        f_type="P",
        f_data="data/perf_7_3/prot.fasta",
        f_weights=None,
        f_sim="data/perf_7_3/prot_sim.tsv",
        f_dist=None,
        f_max_sim=1,
        f_max_dist=1,
        solver="SCIP",
        cache=False,
        cache_dir=None,
    )

    shutil.rmtree("data/genomes/out")


def check_identity_tsv(filename):
    assert os.path.isfile(filename)
    with open(filename, "r") as data:
        for line in data.readlines()[1:]:
            parts = line.strip().split("\t")
            assert len(parts) == 2
            assert parts[0] == parts[1]


def check_assignment_tsv(filename):
    assert os.path.isfile(filename)
    with open(filename, "r") as data:
        for line in data.readlines()[1:]:
            parts = line.strip().split("\t")
            assert len(parts) == (3 if "inter" in filename else 2)
            assert parts[0] not in ["train", "test", "not_selected"]
            if "inter" in filename:
                assert parts[0] != parts[1]
                assert parts[1] not in ["train", "test", "not_selected"]
            assert parts[-1] in ["train", "test", "not selected"]


@pytest.mark.issue
def test_issue1():
    test_pipeline(False, False, "data/pipeline/prot_sim.tsv", None, None, False, None, None, False, "CCSe")


@pytest.mark.issue
def test_issue2():
    test_pipeline(True, False, "data/pipeline/prot_sim.tsv", None, "data/pipeline/drugs.tsv", False, None,
                  "data/pipeline/drug_dist.tsv", True, "CCD")
