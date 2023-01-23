import logging
import os
from typing import Optional

import pytest

from scala.bqp.run import bqp_main
from tests.test_bqp import read_tsv


@pytest.mark.mosek
@pytest.mark.parametrize("pdb", [True, False])
@pytest.mark.parametrize("prot_weights", [True, False])
@pytest.mark.parametrize("prot_sim", ["data/pipeline/prot_sim.tsv", "wlk", "mmseqs", None])
@pytest.mark.parametrize("drugs", ["data/pipeline/drugs.tsv", None])
@pytest.mark.parametrize("drug_weights", [True, False])
@pytest.mark.parametrize("drug_sim", ["data/pipeline/drug_sim.tsv", "wlk", None])
@pytest.mark.parametrize("inter", [True, False])
# @pytest.mark.parametrize("mode", ["ICD", "ICP", "IC", "CCD", "CCP", "CC"])
@pytest.mark.parametrize("mode", ["ICD", "ICP", "CCD", "CCP"])  # , "CC"])
def test_pipeline(
        pdb: bool,
        prot_weights: bool,
        prot_sim: Optional[str],
        drugs: Optional[str],
        drug_weights: bool,
        drug_sim: Optional[str],
        inter: bool,
        mode: str,
        out_folder: str = "out",
):
    if (pdb and prot_sim == "mmseqs") or \
            (not pdb and prot_sim == "wlk") or \
            (mode in ["ICD", "CCD"] and drugs is None) or \
            (mode == "CCP" and prot_sim is None) or \
            (mode == "CCD" and drug_sim is None) or \
            (mode in ["IC", "CC"] and not inter) or \
            (inter and drugs is None) or \
            (prot_sim == "mmseqs"):
        pytest.skip("reason")

    base = "data/pipeline"
    limit = 0.25

    bqp_main(
        output=f"{base}/{out_folder}/",
        method="ilp",
        verbosity="I",
        input=f"{base}/pdbs" if pdb else f"{base}/seqs.fasta",
        weight_file=f"{base}/prot_weights.tsv" if prot_weights else None,
        prot_sim=prot_sim,
        prot_dist=None,
        drugs=drugs,
        drug_weights=f"{base}/drug_weights.tsv" if drug_weights else None,
        drug_sim=drug_sim,
        drug_dist=None,
        inter=f"{base}/inter.tsv" if inter else None,
        technique=mode,
        header=None,
        sep="\t",
        names=["train", "test"],
        splits=[0.67, 0.33] if mode in ["IC", "CC"] else [0.7, 0.3],
        limit=limit,
        max_sec=10,
        max_sol=-1,
    )

    inter = read_tsv(f"{base}/out/inter.tsv")

    _, _, splits = list(zip(*inter))
    trains, tests = splits.count("train"), splits.count("test")
    train_frac, test_frac = trains / (trains + tests), tests / (trains + tests)
    assert 0.7 * (1 - limit) <= train_frac <= 0.7 * (1 + limit)
    assert 0.3 * (1 - limit) <= test_frac <= 0.3 * (1 + limit)

    for filename in os.listdir(f"{base}/out"):
        os.remove(f"{base}/out/{filename}")

@pytest.mark.mosek
def test_detail(
        pdb: bool = True,
        prot_weights: bool = True,
        prot_sim: Optional[str] = 'data/pipeline/prot_sim.tsv',
        drugs: Optional[str] = 'data/pipeline/drugs.tsv',
        drug_weights: bool = True,
        drug_sim: Optional[str] = 'data/pipeline/drug_sim.tsv',
        inter: bool = True,
        mode: str = 'ICD',
        out_folder: str = "out",
):
    base = "data/pipeline"
    limit = 0.25
    bqp_main(
        output=f"{base}/{out_folder}/",
        method="ilp",
        verbosity="I",
        input=f"{base}/pdbs" if pdb else f"{base}/seqs.fasta",
        weight_file=f"{base}/prot_weights.tsv" if prot_weights else None,
        prot_sim=prot_sim,
        drugs=drugs,
        drug_weights=f"{base}/drug_weights.tsv" if drug_weights else None,
        drug_sim=drug_sim,
        inter=f"{base}/inter.tsv" if inter else None,
        technique=mode,
        header=None,
        sep="\t",
        names=["train", "test"],
        splits=[0.67, 0.33] if mode in ["IC", "CC"] else [0.7, 0.3],
        limit=limit,
        max_sec=10,
        max_sol=100,
    )
    assert os.path.isfile(f"{base}/out/inter.tsv")
    inter = read_tsv(f"{base}/out/inter.tsv")

    _, _, splits = list(zip(*inter))
    trains, tests = splits.count("train"), splits.count("test")
    train_frac, test_frac = trains / (trains + tests), tests / (trains + tests)
    assert 0.7 * (1 - limit) <= train_frac <= 0.7 * (1 + limit)
    assert 0.3 * (1 - limit) <= test_frac <= 0.3 * (1 + limit)
