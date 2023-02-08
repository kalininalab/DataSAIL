import pytest

from datasail.sail import sail
from tests.utils import check_folder


@pytest.mark.parametrize("data", [
    (True, False, None, None, None, False, None, None, False, "ICP"),
    (True, False, "wlk", None, None, False, None, None, False, "ICP"),
    (False, False, None, None, None, False, None, None, False, "ICP"),
    # (False, False, "mmseqs", None, None, False, None, None, False, "ICP"),
    (False, False, "data/pipeline/prot_sim.tsv", None, None, False, None, None, False, "ICP"),
    (False, False, None, "data/pipeline/prot_dist.tsv", None, False, None, None, False, "ICP"),
    (False, True, None, None, None, False, None, None, False, "ICP"),
    (None, False, None, None, "data/pipeline/drugs.tsv", False, None, None, False, "ICD"),
    (False, False, None, None, "data/pipeline/drugs.tsv", False, None, None, False, "ICD"),
    (False, False, None, None, "data/pipeline/drugs.tsv", True, None, None, False, "ICD"),
    (False, False, None, None, "data/pipeline/drugs.tsv", False, "data/pipeline/drug_sim.tsv", None, False, "ICD"),
    (False, False, None, None, "data/pipeline/drugs.tsv", True, "wlk", None, False, "ICD"),
    (False, False, None, None, "data/pipeline/drugs.tsv", False, None, "data/pipeline/drug_dist.tsv", False, "ICD"),
    (True, False, "wlk", None, "data/pipeline/drugs.tsv", False, "wlk", None, True, "ICD"),
    (False, False, "data/pipeline/prot_sim.tsv", None, None, False, None, None, False, "CCP"),
    (False, False, "data/pipeline/prot_sim.tsv", None, "data/pipeline/drugs.tsv", False, None, "data/pipeline/drug_dist.tsv", False, "CCP"),
    (False, False, None, None, "data/pipeline/drugs.tsv", False, None, "data/pipeline/drug_dist.tsv", False, "CCD"),
    (True, False, "data/pipeline/prot_sim.tsv", None, "data/pipeline/drugs.tsv", False, None,
     "data/pipeline/drug_dist.tsv", True, "CCD"),
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
        splits=[0.65, 0.35],  # [0.67, 0.33] if mode in ["IC", "CC"] else [0.7, 0.3],
        names=["train", "test"],
        limit=0.25,
        protein_data=None if pdb is None else ("data/pipeline/pdbs" if pdb else "data/pipeline/seqs.fasta"),
        protein_weights="data/pipeline/prot_weights.tsv" if prot_weights else None,
        protein_sim=prot_sim,
        protein_dist=prot_dist,
        protein_max_sim=1,
        protein_max_dist=1,
        ligand_data=drugs,
        ligand_weights="data/pipeline/drug_weights.tsv" if drug_weights else None,
        ligand_sim=drug_sim,
        ligand_dist=drug_dist,
        ligand_max_sim=1,
        ligand_max_dist=1,
    )

    check_folder("data/pipeline/out", 0.25)
