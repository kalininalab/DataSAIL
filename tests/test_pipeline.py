import pytest

from datasail.sail import sail
from tests.utils import check_folder


@pytest.mark.parametrize("data", [
    (True, False, None, None, None, False, None, None, False, "ICS"),
    (True, False, "wlk", None, None, False, None, None, False, "ICS"),
    (False, False, None, None, None, False, None, None, False, "ICS"),
    # (False, False, "mmseqs", None, None, False, None, None, False, "ICP"),
    (False, False, "data/pipeline/prot_sim.tsv", None, None, False, None, None, False, "ICS"),
    (False, False, None, "data/pipeline/prot_dist.tsv", None, False, None, None, False, "ICS"),
    (False, True, None, None, None, False, None, None, False, "ICS"),
    (None, False, None, None, "data/pipeline/drugs.tsv", False, None, None, False, "ICS"),
    (False, False, None, None, "data/pipeline/drugs.tsv", False, None, None, False, "ICS"),
    (False, False, None, None, "data/pipeline/drugs.tsv", True, None, None, False, "ICS"),
    (False, False, None, None, "data/pipeline/drugs.tsv", False, "data/pipeline/drug_sim.tsv", None, False, "ICS"),
    (False, False, None, None, "data/pipeline/drugs.tsv", True, "wlk", None, False, "ICS"),  # <-- 10/11
    (False, False, None, None, "data/pipeline/drugs.tsv", False, None, "data/pipeline/drug_dist.tsv", False, "ICS"),
    (True, False, "wlk", None, "data/pipeline/drugs.tsv", False, "wlk", None, True, "ICS"),
    (False, False, None, None, "data/pipeline/drugs.tsv", False, None, "data/pipeline/drug_dist.tsv", False, "CCS"),
    (False, False, None, None, "data/pipeline/drugs.tsv", False, None, "data/pipeline/drug_dist.tsv", False, "CCS"),
    (False, False, "data/pipeline/prot_sim.tsv", None, "data/pipeline/drugs.tsv", False, None,
     "data/pipeline/drug_dist.tsv", False, "CCS"),
    (False, False, None, None, "data/pipeline/drugs.tsv", False, None, "data/pipeline/drug_dist.tsv", False, "CCS"),
])
def test_pipeline(data):
    pdb, prot_weights, prot_sim, prot_dist, drugs, drug_weights, drug_sim, drug_dist, inter, mode = data

    sail(
        inter="data/pipeline/inter.tsv" if inter else None,
        output="data/pipeline/out/",
        max_sec=10,
        max_sol=10,
        verbosity="I",
        technique=mode,
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
    )

    check_folder(
        "data/pipeline/out",
        0.25,
        "data/pipeline/prot_weights.tsv" if prot_weights else None,
        "data/pipeline/drug_weights.tsv" if drug_weights else None,
    )


def test_pipeline_tsne():
    pdb, prot_weights, prot_sim, prot_dist, drugs, drug_weights, drug_sim, drug_dist, inter, mode = (
        False, False, "../tests/data/pipeline/prot_sim.tsv", None, None, False, None,
        None, False, "CCS"
    )

    sail(
        inter="../tests/data/pipeline/inter.tsv" if inter else None,
        output="../tests/data/pipeline/out/",
        max_sec=100,
        max_sol=10,
        verbosity="I",
        technique=mode,
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
    )


@pytest.mark.issue
def test_issue1():
    test_pipeline(False, False, "data/pipeline/prot_sim.tsv", None, None, False, None, None, False, "CCS")


@pytest.mark.issue
def test_issue2():
    test_pipeline(True, False, "data/pipeline/prot_sim.tsv", None, "data/pipeline/drugs.tsv", False, None,
                  "data/pipeline/drug_dist.tsv", True, "CCD")
