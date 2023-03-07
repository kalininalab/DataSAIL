import os.path
import shutil

import pytest

from datasail.sail import sail
from tests.utils import check_folder


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
        e_type=None if drugs is None else "M",
        e_data=drugs,
        e_weights="data/pipeline/drug_weights.tsv" if drug_weights else None,
        e_sim=drug_sim,
        e_dist=drug_dist,
        e_max_sim=1,
        e_max_dist=1,
        f_type=None if pdb is None else "P",
        f_data=None if pdb is None else ("data/pipeline/pdbs" if pdb else "data/pipeline/seqs.fasta"),
        f_weights="data/pipeline/prot_weights.tsv" if prot_weights else None,
        f_sim=prot_sim,
        f_dist=prot_dist,
        f_max_sim=1,
        f_max_dist=1,
        solver="SCIP",
    )

    check_folder(
        output_root="data/pipeline/out/" + mode,
        epsilon=0.25,
        e_weight="data/pipeline/drug_weights.tsv" if drug_weights else None,
        f_weight="data/pipeline/prot_weights.tsv" if prot_weights else None,
        e_filename="Molecule_drugs_splits.tsv" if mode[-1] == "e" else None,
        f_filename=f"Protein_{'pdbs' if pdb else 'seqs'}_splits.tsv" if mode[-1] == "f" else None,
    )


@pytest.mark.todo
def test_pipeline_tsne():
    pdb, prot_weights, prot_sim, prot_dist, drugs, drug_weights, drug_sim, drug_dist, inter, mode = (
        False, False, "../tests/data/pipeline/prot_sim.tsv", None, None, False, None,
        None, False, "CCS"
    )

    sail(
        inter="../tests/data/pipeline/inter.tsv" if inter else None,
        output="../tests/data/pipeline/out/",
        max_sec=10,
        max_sol=10,
        verbosity="I",
        techniques=[mode],
        vectorized=False,
        splits=[0.67, 0.33] if mode in ["IC", "CC"] else [0.7, 0.3],
        names=["train", "test"],
        epsilon=0.25,
        e_type=None if drugs is None else "M",
        e_data=drugs,
        e_weights="../tests/data/pipeline/drug_weights.tsv" if drug_weights else None,
        e_sim=drug_sim,
        e_dist=drug_dist,
        e_max_sim=1,
        e_max_dist=1,
        f_type=None if pdb is None else "P",
        f_data=None if pdb is None else ("../tests/data/pipeline/pdbs" if pdb else "../tests/data/pipeline/seqs.fasta"),
        f_weights="../tests/data/pipeline/prot_weights.tsv" if prot_weights else None,
        f_sim=prot_sim,
        f_dist=prot_dist,
        f_max_sim=1,
        f_max_dist=1,
    )

    check_folder(
        "../tests/data/pipeline/out",
        0.25,
        "../tests/data/pipeline/prot_weights.tsv" if prot_weights else None,
        "../tests/data/pipeline/drug_weights.tsv" if drug_weights else None,
        "Molecule_drugs_splits.tsv",
        None if pdb is None else f"Protein_{'pdbs' if pdb else 'seqs'}_splits.tsv",
    )


@pytest.mark.todo
def test_report():
    shutil.rmtree("../tests/data/pipeline/out", ignore_errors=True)
    sail(
        inter="../tests/data/pipeline/inter.tsv",
        output="../tests/data/pipeline/out/",
        max_sec=100,
        max_sol=10,
        verbosity="I",
        techniques=["R", "ICSe", "ICSf", "ICD", "CCSe", "CCSf"],
        vectorized=True,
        splits=[0.7, 0.3],
        names=["train", "test"],
        epsilon=0.25,
        e_type="M",
        e_data="data/pipeline/drugs.tsv",
        e_weights=None,
        e_sim="data/pipeline/drug_sim.tsv",
        e_dist=None,  # "data/pipeline/drug_dist.tsv",
        e_max_sim=1,
        e_max_dist=1,
        f_type="P",
        f_data="../tests/data/pipeline/pdbs",  # "../tests/data/pipeline/seqs.fasta"),
        f_weights=None,  # "../tests/data/pipeline/prot_weights.tsv"
        f_sim="data/pipeline/prot_sim.tsv",
        f_dist=None,  # "data/pipeline/drug_dist.tsv",
        f_max_sim=1,
        f_max_dist=1,
        solver="MOSEK",
    )

    assert os.path.isdir("../tests/data/pipeline/out/R")
    assert len(os.listdir("../tests/data/pipeline/out/R")) == 1
    check_assignment_tsv("../tests/data/pipeline/out/R/inter.tsv")

    assert os.path.isdir("../tests/data/pipeline/out/ICSe")
    assert len(os.listdir("../tests/data/pipeline/out/ICSe")) == 2
    check_assignment_tsv("../tests/data/pipeline/out/ICSe/Molecule_drugs_splits.tsv")
    check_assignment_tsv("../tests/data/pipeline/out/ICSe/inter.tsv")

    assert os.path.isdir("../tests/data/pipeline/out/ICSf")
    assert len(os.listdir("../tests/data/pipeline/out/ICSf")) == 2
    check_assignment_tsv("../tests/data/pipeline/out/ICSf/Protein_pdbs_splits.tsv")
    check_assignment_tsv("../tests/data/pipeline/out/ICSf/inter.tsv")

    assert os.path.isdir("../tests/data/pipeline/out/ICD")
    assert len(os.listdir("../tests/data/pipeline/out/ICD")) == 3
    check_assignment_tsv("../tests/data/pipeline/out/ICD/inter.tsv")
    check_assignment_tsv("../tests/data/pipeline/out/ICD/Molecule_drugs_splits.tsv")
    check_assignment_tsv("../tests/data/pipeline/out/ICD/Protein_pdbs_splits.tsv")

    assert os.path.isdir("../tests/data/pipeline/out/CCSe")
    assert len(os.listdir("../tests/data/pipeline/out/CCSe")) == 5
    assert os.path.isfile("../tests/data/pipeline/out/CCSe/Molecule_drugs_clusters.png")
    assert os.path.isfile("../tests/data/pipeline/out/CCSe/Molecule_drugs_splits.png")
    check_assignment_tsv("../tests/data/pipeline/out/CCSe/inter.tsv")
    check_identity_tsv("../tests/data/pipeline/out/CCSe/Molecule_drugs_clusters.tsv")
    check_assignment_tsv("../tests/data/pipeline/out/CCSe/Molecule_drugs_splits.tsv")

    assert os.path.isdir("../tests/data/pipeline/out/CCSf")
    assert len(os.listdir("../tests/data/pipeline/out/CCSf")) == 5
    assert os.path.isfile("../tests/data/pipeline/out/CCSf/Protein_pdbs_clusters.png")
    assert os.path.isfile("../tests/data/pipeline/out/CCSf/Protein_pdbs_splits.png")
    check_assignment_tsv("../tests/data/pipeline/out/CCSf/inter.tsv")
    check_identity_tsv("../tests/data/pipeline/out/CCSf/Protein_pdbs_clusters.tsv")
    check_assignment_tsv("../tests/data/pipeline/out/CCSf/Protein_pdbs_splits.tsv")

    # assert os.path.isdir("../tests/data/pipeline/out/CCD")


def check_identity_tsv(filename):
    assert os.path.isfile(filename)
    with open(filename, "r") as data:
        for line in data.readlines():
            parts = line.strip().split("\t")
            assert len(parts) == 2
            assert parts[0] == parts[1]


def check_assignment_tsv(filename):
    assert os.path.isfile(filename)
    with open(filename, "r") as data:
        for line in data.readlines():
            parts = line.strip().split("\t")
            assert len(parts) == (3 if "inter" in filename else 2)
            assert parts[0] not in ["train", "test"]
            if "inter" in filename:
                assert parts[0] != parts[1]
                assert parts[1] not in ["train", "test"]
            assert parts[-1] in ["train", "test"]


@pytest.mark.issue
def test_issue1():
    test_pipeline(False, False, "data/pipeline/prot_sim.tsv", None, None, False, None, None, False, "CCS")


@pytest.mark.issue
def test_issue2():
    test_pipeline(True, False, "data/pipeline/prot_sim.tsv", None, "data/pipeline/drugs.tsv", False, None,
                  "data/pipeline/drug_dist.tsv", True, "CCD")
