import pytest

from datasail.sail import datasail


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

    e_name_split_map, f_name_split_map, inter_split_map = datasail(
        inter="data/pipeline/inter.tsv" if inter else None,
        max_sec=10,
        max_sol=10,
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
        cache=False,
        cache_dir=None,
        solver="SCIP",
    )

    assert any(mode[:3] in x for x in [e_name_split_map, f_name_split_map, inter_split_map])


def test_report():
    e_name_split_map, f_name_split_map, inter_split_map = datasail(
        inter="data/perf_7_3/inter.tsv",
        max_sec=100,
        max_sol=10,
        techniques=["R", "ICSe", "ICSf", "ICD", "CCSe", "CCSf", "CCD"],
        vectorized=True,
        splits=[0.7, 0.3],
        names=["train", "test"],
        epsilon=0.25,
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

    assert "ICS" in e_name_split_map
    assert "ICS" in f_name_split_map
    assert "ICSe" in inter_split_map
    assert "ICSf" in inter_split_map
    assert "ICD" in e_name_split_map
    assert "ICD" in f_name_split_map
    assert "ICD" in inter_split_map
    assert "CCS" in e_name_split_map
    assert "CCS" in f_name_split_map
    assert "CCSe" in inter_split_map
    assert "CCSf" in inter_split_map
    assert "CCD" in e_name_split_map
    assert "CCD" in f_name_split_map
    assert "CCD" in inter_split_map


@pytest.mark.todo
def test_genomes():
    e_name_split_map, f_name_split_map, inter_split_map = datasail(
        inter=None,
        max_sec=100,
        max_sol=10,
        techniques=["ICSe", "CCSe"],
        vectorized=True,
        splits=[0.7, 0.3],
        names=["train", "test"],
        epsilon=0.25,
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

    assert "ICS" in e_name_split_map
    assert "CCS" in e_name_split_map
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
