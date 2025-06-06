from pathlib import Path
from typing import Optional

from pytest_cases import lazy_value

from datasail.reader.utils import read_data
from datasail.dataset import DataSet
from datasail.sail import datasail
from tests.pipeline_package_fixtures import *


base = Path("data") / "pipeline"


@pytest.mark.full
@pytest.mark.parametrize("data", [
    (True, False, None, None, None, False, None, None, False, "I1f"),
    (True, False, "wlk", None, None, False, None, None, False, "I1f"),  # <-- 1/14
    (False, False, None, None, None, False, None, None, False, "I1f"),
    (False, False, "mmseqspp", None, None, False, None, None, False, "I1f"),  # <-- 3/14
    (False, False, base / "prot_sim.tsv", None, None, False, None, None, False, "I1f"),
    (False, False, None, base / "prot_dist.tsv", None, False, None, None, False, "I1f"),
    (False, True, None, None, None, False, None, None, False, "I1f"),
    (None, False, None, None, base / "drugs.tsv", False, None, None, False, "I1e"),
    (False, False, None, None, base / "drugs.tsv", False, None, None, False, "I1e"),
    (False, False, None, None, base / "drugs.tsv", True, None, None, False, "I1e"),
    (False, False, None, None, base / "drugs.tsv", False, base / "drug_sim.tsv", None, False, "I1e"),
    (False, False, None, None, base / "drugs.tsv", True, "wlk", None, False, "I1e"),  # <-- 11/14
    (False, False, None, None, base / "drugs.tsv", False, None, base / "drug_dist.tsv", False, "I1e"),
    (True, False, "wlk", None, base / "drugs.tsv", False, "wlk", None, True, "I1f"),  # <-- 13/14
    (False, False, None, None, base / "drugs.tsv", False, None, base / "drug_dist.tsv", False, "C1e"),
    # (False, False, "data/pipeline/prot_sim.tsv", None, "data/pipeline/drugs.tsv", False, None,
    #  "data/pipeline/drug_dist.tsv", False, "C1f"),
    # (False, False, None, None, "data/pipeline/drugs.tsv", False, None, "data/pipeline/drug_dist.tsv", False, "C1e"),
])
def test_pipeline(data):
    pdb, prot_weights, prot_sim, prot_dist, drugs, drug_weights, drug_sim, drug_dist, inter, mode = data

    e_name_split_map, f_name_split_map, inter_split_map = datasail(
        inter=base / "inter.tsv" if inter else None,
        max_sec=10,
        techniques=[mode],
        splits=[0.67, 0.33] if mode in ["IC", "CC"] else [0.7, 0.3],
        names=["train", "test"],
        epsilon=0.25,
        e_type=None if drugs is None else "M",
        e_data=drugs,
        e_weights=(base / "drug_weights.tsv") if drug_weights else None,
        e_sim=drug_sim,
        e_dist=drug_dist,
        f_type=None if pdb is None else "P",
        f_data=None if pdb is None else ((base / "pdbs") if pdb else (base / "seqs.fasta")),
        f_weights=(base / "prot_weights.tsv") if prot_weights else None,
        f_sim=prot_sim,
        f_dist=prot_dist,
        solver="SCIP",
    )

    assert any(mode in x for x in [e_name_split_map, f_name_split_map, inter_split_map])


@pytest.mark.parametrize(
    "ligand_type,ligand_data,ligand_weights,ligand_sim,protein_type,protein_data,protein_weights,protein_sim,interactions,combo",
    [
        ("M", lazy_value(mibig_dict), None, None, None, None, None, None, None, "e|mibig|_|_"),
        (None, None, None, None, "M", lazy_value(mibig_returner), None, None, None, "_|_|f|mibig"),
        ("M", lazy_value(mibig_generator), None, None, None, None, None, None, None, "e|mibig|_|_"),
        ("P", lazy_value(mave_dict), lazy_value(mave_weights_returner), None, None, None, None, None, None,
         "e|mave|_|_"),
        (None, None, None, None, "P", lazy_value(mave_returner), lazy_value(mave_weights_generator), None, None,
         "_|_|f|mave"),
        ("P", lazy_value(mave_generator), lazy_value(mave_weights_dict), None, None, None, None, None, None,
         "e|mave|_|_"),
        ("P", lazy_value(sabdab_ag_dict), None, None, "P", lazy_value(sabdab_vh_returner), None, None,
         lazy_value(sabdab_inter_generator), "e|sabdab_ag|f|sabdab_vh"),
        ("P", lazy_value(sabdab_ag_returner), None, None, "P", lazy_value(sabdab_vh_generator), None, None,
         lazy_value(sabdab_inter_list), "e|sabdab_ag|f|sabdab_vh"),
        ("P", lazy_value(sabdab_ag_generator), None, None, "P", lazy_value(sabdab_vh_dict), None, None,
         lazy_value(sabdab_inter_returner), "e|sabdab_ag|f|sabdab_vh"),
    ])
def test_pipeline_inputs(
        ligand_type, ligand_data, ligand_weights, ligand_sim, protein_type, protein_data, protein_weights, protein_sim,
        interactions, combo,
        mave_dataset, mibig_dataset, sabdab_ag_dataset, sabdab_vh_dataset,
):
    def read_data_sub(
            inter=None, e_type=None, e_data=None, e_weights=None, e_sim=None, f_type=None, f_data=None,
            f_weights=None, f_sim=None,
    ) -> tuple[DataSet, DataSet, Optional[list[tuple]]]:
        kwargs = dict(
            inter=inter, e_type=e_type, e_data=e_data, e_weights=e_weights, e_sim=e_sim, e_dist=None, e_args="",
            e_strat=None, e_clusters=50, f_type=f_type, f_data=f_data, f_weights=f_weights, f_sim=f_sim, f_dist=None,
            f_args="", f_strat=None, f_clusters=50,
        )
        # read e-entities and f-entities
        return read_data(**kwargs)

    def reference_dataset(name: str):
        if name == "mave":
            return mave_dataset
        if name == "mibig":
            return mibig_dataset
        return sabdab_ag_dataset if name == "sabdab_ag" else sabdab_vh_dataset

    e_dataset, f_dataset, interactions = read_data_sub(
        inter=interactions, e_type=ligand_type, e_data=ligand_data, e_weights=ligand_weights, e_sim=ligand_sim,
        f_type=protein_type, f_data=protein_data, f_weights=protein_weights, f_sim=protein_sim,
    )
    if e_dataset.type is not None:
        e_dataset = cluster(e_dataset, threads=1, logdir="", linkage="average")
    if f_dataset.type is not None:
        f_dataset = cluster(f_dataset, threads=1, logdir="", linkage="average")

    parts = combo.split("|")
    if parts[0] == "e":
        reference = reference_dataset(parts[1])
        e_dataset.location = reference.location
        assert e_dataset == reference
    if parts[2] == "f":
        reference = reference_dataset(parts[3])
        f_dataset.location = reference.location
        assert f_dataset == reference


def test_report():
    root = Path("data") / "perf_7_3"
    e_name_split_map, f_name_split_map, inter_split_map = datasail(
        inter=root / "inter.tsv",
        max_sec=100,
        techniques=["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"],
        splits=[0.7, 0.3],
        names=["train", "test"],
        epsilon=0.25,
        e_type="M",
        e_data=root / "lig.tsv",
        e_sim=root / "lig_sim.tsv",
        f_type="P",
        f_data=root / "prot.fasta",
        f_sim=root / "prot_sim.tsv",
        solver="SCIP",
    )

    assert "I1e" in e_name_split_map
    assert "I1f" in f_name_split_map
    assert "I1e" in inter_split_map
    assert "I1f" in inter_split_map
    assert "I2" in e_name_split_map
    assert "I2" in f_name_split_map
    assert "I2" in inter_split_map
    assert "C1e" in e_name_split_map
    assert "C1f" in f_name_split_map
    assert "C1e" in inter_split_map
    assert "C1f" in inter_split_map
    assert "C2" in e_name_split_map
    assert "C2" in f_name_split_map
    assert "C2" in inter_split_map


@pytest.mark.todo
def test_genomes():
    e_name_split_map, f_name_split_map, inter_split_map = datasail(
        max_sec=100,
        techniques=["I1e", "C1e"],
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

    assert "I1e" in e_name_split_map
    assert "C1e" in e_name_split_map
    assert len(f_name_split_map) == 0
    assert len(inter_split_map) == 0


def check_identity_tsv(pairs):
    for parts in pairs:
        assert len(parts) == 2
        assert parts[0] == parts[1]


def check_assignment_tsv(pairs):
    assert pairs is not None
    for parts in pairs:
        assert len(parts) in (2, 3)
        assert parts[0] not in ["train", "test", "not_selected"]
        if len(parts) == 3:
            assert parts[0] != parts[1]
            assert parts[1] not in ["train", "test", "not_selected"]
        assert parts[-1] in ["train", "test", "not selected"]
