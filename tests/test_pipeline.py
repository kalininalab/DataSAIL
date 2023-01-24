import os
import shutil

import pytest

from scala.bqp.run import bqp_main
from tests.test_bqp import read_tsv


@pytest.mark.mosek
@pytest.mark.parametrize("data", [
    (True, False, None, None, None, False, None, None, False, "ICP"),
    (True, False, "wlk", None, None, False, None, None, False, "ICP"),
    (False, False, None, None, None, False, None, None, False, "ICP"),
    (False, False, "mmseqs", None, None, False, None, None, False, "ICP"),
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
])
def test_pipeline(data):
    pdb, prot_weights, prot_sim, prot_dist, drugs, drug_weights, drug_sim, drug_dist, inter, mode = data

    bqp_main(
        output="data/pipeline/out/",
        method="ilp",
        verbosity="I",
        input=None if pdb is None else ("data/pipeline/pdbs" if pdb else "data/pipeline/seqs.fasta"),
        weight_file="data/pipeline/prot_weights.tsv" if prot_weights else None,
        prot_sim=prot_sim,
        prot_dist=prot_dist,
        drugs=drugs,
        drug_weights="data/pipeline/drug_weights.tsv" if drug_weights else None,
        drug_sim=drug_sim,
        drug_dist=drug_dist,
        inter="data/pipeline/inter.tsv" if inter else None,
        technique=mode,
        header=None,
        sep="\t",
        names=["train", "test"],
        splits=[0.67, 0.33] if mode in ["IC", "CC"] else [0.7, 0.3],
        limit=0.25,
        max_sec=10,
        max_sol=-1,
    )

    split_data = []
    if os.path.exists("data/pipeline/out/inter.tsv"):
        split_data.append(read_tsv("data/pipeline/out/inter.tsv"))
    if os.path.exists("data/pipeline/out/inter.tsv"):
        split_data.append(read_tsv("data/pipeline/out/proteins.tsv"))
    if os.path.exists("data/pipeline/out/inter.tsv"):
        split_data.append(read_tsv("data/pipeline/out/drugs.tsv"))

    assert len(split_data) > 0

    for data in split_data:
        splits = list(zip(*data))[-1]
        trains, tests = splits.count("train"), splits.count("test")
        train_frac, test_frac = trains / (trains + tests), tests / (trains + tests)
        assert 0.7 * (1 - 0.25) <= train_frac <= 0.7 * (1 + 0.25)
        assert 0.3 * (1 - 0.25) <= test_frac <= 0.3 * (1 + 0.25)

    shutil.rmtree("data/pipeline/out")


def test_algos():
    bqp_main(
        output="data/pipeline/out/",
        method="ilp",
        verbosity="I",
        input="data/pipeline/pdbs",
        weight_file="data/pipeline/prot_weights.tsv",
        prot_sim="data/pipeline/prot_sim.tsv",
        prot_dist=None,
        drugs="data/pipeline/drugs.tsv",
        drug_weights="data/pipeline/drug_weights.tsv",
        # drug_sim="data/pipeline/drug_sim.tsv",
        # drug_dist=None,
        drug_sim=None,
        drug_dist="data/pipeline/drug_dist.tsv",
        inter="data/pipeline/inter.tsv",
        technique="CCD",
        header=None,
        sep="\t",
        names=["train", "test"],
        splits=[0.7, 0.3],
        limit=0.25,
        max_sec=5,
        max_sol=-1,
    )

    split_data = []
    if os.path.exists("data/pipeline/out/inter.tsv"):
        split_data.append(read_tsv("data/pipeline/out/inter.tsv"))
    if os.path.exists("data/pipeline/out/inter.tsv"):
        split_data.append(read_tsv("data/pipeline/out/proteins.tsv"))
    if os.path.exists("data/pipeline/out/inter.tsv"):
        split_data.append(read_tsv("data/pipeline/out/drugs.tsv"))

    assert len(split_data) > 0

    for data in split_data:
        splits = list(zip(*data))[-1]
        trains, tests = splits.count("train"), splits.count("test")
        train_frac, test_frac = trains / (trains + tests), tests / (trains + tests)
        assert 0.7 * (1 - 0.25) <= train_frac <= 0.7 * (1 + 0.25)
        assert 0.3 * (1 - 0.25) <= test_frac <= 0.3 * (1 + 0.25)

    shutil.rmtree("data/pipeline/out")
