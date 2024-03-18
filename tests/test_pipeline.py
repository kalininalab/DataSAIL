import pickle
import shutil
from pathlib import Path

import h5py
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from datasail.reader.read_molecules import read_molecule_data
from datasail.sail import sail, datasail
from tests.utils import check_folder, run_sail


base = Path("data") / "pipeline"
@pytest.mark.parametrize("data", [
    (True, False, None, None, None, False, None, None, False, "I1f"),
    (True, False, "wlk", None, None, False, None, None, False, "I1f"),
    (False, False, None, None, None, False, None, None, False, "I1f"),
    # (False, False, "mmseqs", None, None, False, None, None, False, "ICP"),
    (False, False, base / "prot_sim.tsv", None, None, False, None, None, False, "I1f"),
    (False, False, None, base / "prot_dist.tsv", None, False, None, None, False, "I1f"),
    # (False, True, None, None, None, False, None, None, False, "I1f"),  # <-- 5/12
    (None, False, None, None, base / "drugs.tsv", False, None, None, False, "I1e"),
    (False, False, None, None, base / "drugs.tsv", False, None, None, False, "I1e"),
    (False, False, None, None, base / "drugs.tsv", True, None, None, False, "I1e"),  # <-- 8/12
    (False, False, None, None, base / "drugs.tsv", False, base / "drug_sim.tsv", None, False, "I1e"),
    (False, False, None, None, base / "drugs.tsv", True, "wlk", None, False, "I1e"),  # <-- 10/12
    (False, False, None, None, base / "drugs.tsv", False, None, base / "drug_dist.tsv", False, "I1e"),
    (True, False, "wlk", None, base / "drugs.tsv", False, "wlk", None, True, "I1f"),
    # (False, False, None, None, "data/pipeline/drugs.tsv", False, None, "data/pipeline/drug_dist.tsv", False, "C1e"),
    # (False, False, "data/pipeline/prot_sim.tsv", None, "data/pipeline/drugs.tsv", False, None,
    #  "data/pipeline/drug_dist.tsv", False, "C1f"),
    # (False, False, None, None, "data/pipeline/drugs.tsv", False, None, "data/pipeline/drug_dist.tsv", False, "C1e"),
])
def test_pipeline(data):
    pdb, prot_weights, prot_sim, prot_dist, drugs, drug_weights, drug_sim, drug_dist, inter, mode = data
    base = Path("data") / "pipeline"
    shutil.rmtree(out := base / "out", ignore_errors=True)

    sail(
        inter=(base / "inter.tsv") if inter else None,
        output=out,
        max_sec=10,
        max_sol=10,
        verbosity="I",
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
        f_data=None if pdb is None else (base / ("pdbs" if pdb else "seqs.fasta")),
        f_weights=(base / "prot_weights.tsv") if prot_weights else None,
        f_sim=prot_sim,
        f_dist=prot_dist,
    )

    check_folder(
        output_root=out / mode,
        epsilon=0.25,
        e_weight=(base / "drug_weights.tsv") if drug_weights else None,
        f_weight=(base / "prot_weights.tsv") if prot_weights else None,
        e_filename="Molecule_drugs_splits.tsv" if mode[-1] == "e" else None,
        f_filename=f"Protein_{'pdbs' if pdb else 'seqs'}_splits.tsv" if mode[-1] == "f" else None,
    )


def test_report():
    base = Path("data/perf_7_3")
    shutil.rmtree(base / "out", ignore_errors=True)

    run_sail(
        inter=base / "inter.tsv",
        output=(out := base / "out"),
        max_sec=100,
        techniques=["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"],
        splits=[0.7, 0.3],
        names=["train", "test"],
        epsilon=0.25,
        e_type="M",
        e_data=base / "lig.tsv",
        e_sim=base / "lig_sim.tsv",
        f_type="P",
        f_data=base / "prot.fasta",
        f_sim=base / "prot_sim.tsv",
        solver="SCIP",
    )

    assert (out / "lig_similarity.png").is_file()
    assert (out / "prot_similarity.png").is_file()

    assert (out_r := out / "R").is_dir()
    assert len(list(out_r.iterdir())) == 1
    check_assignment_tsv(out_r / "inter.tsv")

    assert (i1e := out / "I1e").is_dir()
    assert len(list(i1e.iterdir())) == 2
    check_assignment_tsv(i1e / "Molecule_lig_splits.tsv")
    check_assignment_tsv(i1e / "inter.tsv")

    assert (i1f := out / "I1f").is_dir()
    assert len(list(i1f.iterdir())) == 2
    check_assignment_tsv(i1f / "Protein_prot_splits.tsv")
    check_assignment_tsv(i1f / "inter.tsv")

    assert (i2 := out / "I2").is_dir()
    assert len(list(i2.iterdir())) == 3
    check_assignment_tsv(i2 / "inter.tsv")
    check_assignment_tsv(i2 / "Molecule_lig_splits.tsv")
    check_assignment_tsv(i2 / "Protein_prot_splits.tsv")

    assert (c1e := out / "C1e").is_dir()
    assert len(list(c1e.iterdir())) == 6
    assert (c1e / "Molecule_lig_clusters.png").is_file()
    assert (c1e / "Molecule_lig_splits.png").is_file()
    assert (c1e / "Molecule_lig_cluster_hist.png").is_file()
    check_assignment_tsv(c1e / "inter.tsv")
    check_identity_tsv(c1e / "Molecule_lig_clusters.tsv")
    check_assignment_tsv(c1e / "Molecule_lig_splits.tsv")

    assert (c1f := out / "C1f").is_dir()
    assert len(list(c1f.iterdir())) == 6
    assert (c1f / "Protein_prot_clusters.png").is_file()
    assert (c1f / "Protein_prot_splits.png").is_file()
    assert (c1f / "Protein_prot_cluster_hist.png").is_file()
    check_assignment_tsv(c1f / "inter.tsv")
    check_identity_tsv(c1f / "Protein_prot_clusters.tsv")
    check_assignment_tsv(c1f / "Protein_prot_splits.tsv")

    assert (c2 := out / "C2").is_dir()
    assert len(list(c2.iterdir())) == 11
    assert (c2 / "Molecule_lig_clusters.png").is_file()
    assert (c2 / "Molecule_lig_splits.png").is_file()
    assert (c2 / "Molecule_lig_cluster_hist.png").is_file()
    assert (c2 / "Protein_prot_clusters.png").is_file()
    assert (c2 / "Protein_prot_splits.png").is_file()
    assert (c2 / "Protein_prot_cluster_hist.png").is_file()
    check_assignment_tsv(c2 / "inter.tsv")
    check_identity_tsv(c2 / "Molecule_lig_clusters.tsv")
    check_assignment_tsv(c2 / "Molecule_lig_splits.tsv")
    check_identity_tsv(c2 / "Protein_prot_clusters.tsv")
    check_assignment_tsv(c2 / "Protein_prot_splits.tsv")


def test_report_I2():
    base = Path("data") / "perf_7_3"
    shutil.rmtree(out := (base / "out"), ignore_errors=True)

    run_sail(
        inter=base / "inter.tsv",
        output=out,
        max_sec=100,
        techniques=["I2"],
        splits=[0.7, 0.3],
        names=["train", "test"],
        epsilon=0.25,
        e_type="M",
        e_data=base / "lig.tsv",
        e_sim=base / "lig_sim.tsv",
        f_type="P",
        f_data=base / "prot.fasta",
        f_sim=base / "prot_sim.tsv",
        solver="SCIP",
    )
    assert (i2 := out / "I2").is_dir()
    assert len(list(i2.iterdir())) == 3
    check_assignment_tsv(i2 / "inter.tsv")
    check_assignment_tsv(i2 / "Molecule_lig_splits.tsv")
    check_assignment_tsv(i2 / "Protein_prot_splits.tsv")


@pytest.mark.todo
def test_report_repeat():
    base = Path("data") / "perf_7_3"
    shutil.rmtree(out := (base / "out"), ignore_errors=True)

    run_sail(
        inter=base / "inter.tsv",
        output=out,
        max_sec=100,
        techniques=["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"],
        splits=[0.7, 0.3],
        names=["train", "test"],
        epsilon=0.25,
        e_type="M",
        e_data=base / "lig.tsv",
        e_sim=base / "lig_sim.tsv",
        f_type="P",
        f_data=base / "prot.fasta",
        f_sim=base / "prot_sim.tsv",
        solver="SCIP",
        runs=3,
    )

    assert Path(out / "lig_similarity.png").is_file()
    assert Path(out / "prot_similarity.png").is_file()

    for i in range(1, 4):

        assert (r := out / f"R_{i}").is_dir()
        assert len(list(r.iterdir())) == 1

        assert (i1e := out / f"I1e_{i}").is_dir()
        assert len(list(i1e.iterdir())) == 2

        assert (i1f := out / f"I1f_{i}").is_dir()
        assert len(list(i1f.iterdir())) == 2

        assert (i2 := out / f"I2_{i}").is_dir()
        assert len(list(i2.iterdir())) == 3

        assert (c1e := out / f"C1e_{i}").is_dir()
        assert len(list(c1e.iterdir())) == 6

        assert (c1f := out / f"C1f_{i}").is_dir()
        assert len(list(c1f.iterdir())) == 6

        assert (c2 := out / f"C2_{i}").is_dir()
        assert len(list(c2.iterdir())) == 11


@pytest.fixture()
def md_calculator():
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    return MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)


@pytest.mark.parametrize("mode", ["CSV", "TSV", "PKL", "H5PY", "SDF"])
def test_input_formats(mode, md_calculator):
    base = Path("data") / "pipeline"
    drugs = pd.read_csv(base / "drugs.tsv", sep="\t")
    ddict = {row["Drug_ID"]: row["SMILES"] for index, row in drugs.iterrows()}
    (base / "input_forms").mkdir(exist_ok=True, parents=True)

    if mode == "CSV":
        filepath = base / "input_forms" / "drugs.csv"
        drugs.to_csv(filepath, sep=",", index=False)
    elif mode == "TSV":
        filepath = base / "input_forms" / "drugs.tsv"
        drugs.to_csv(filepath, sep="\t", index=False)
    elif mode == "PKL":
        data = {}
        for k, v in ddict.items():
            data[k] = AllChem.MolToSmiles(Chem.MolFromSmiles(v))
        filepath = base / "input_forms" / "drugs.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
    elif mode == "H5PY":
        filepath = base / "input_forms" / "drugs.h5"
        with h5py.File(filepath, "w") as f:
            for k, v in ddict.items():
                f[k] = list(md_calculator.CalcDescriptors(Chem.MolFromSmiles(v)))
    elif mode == "SDF":
        filepath = base / "input_forms" / "drugs.sdf"
        with Chem.SDWriter(str(filepath)) as w:
            for k, v in ddict.items():
                mol = Chem.MolFromSmiles(v)
                mol.SetProp("_Name", k)
                w.write(mol)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    dataset = read_molecule_data(filepath)

    shutil.rmtree(base / "input_forms", ignore_errors=True)

    assert set(dataset.names) == set(ddict.keys())


@pytest.mark.todo
def test_genomes():
    base = Path("data") / "genomes"
    sail(
        inter=None,
        output=base / "out",
        max_sec=100,
        verbosity="I",
        techniques=["I1e", "C1e"],
        splits=[0.7, 0.3],
        names=["train", "test"],
        e_type="G",
        e_data=base,
        solver="SCIP",
    )

    shutil.rmtree(base / "out", ignore_errors=True)


def test_rna():
    e_splits, _, _ = datasail(
        inter=None,
        max_sec=100,
        verbose="I",
        techniques=["I1e", "C1e"],
        splits=[0.7, 0.3],
        names=["train", "test"],
        e_type="G",
        e_data="data/rw_data/RBD/RBD_small.fasta",
    )
    assert "I1e" in e_splits
    assert "C1e" in e_splits
    assert len(e_splits["I1e"]) == 1
    assert len(e_splits["C1e"]) == 1
    assert len(e_splits["I1e"][0]) == 25
    assert len(e_splits["C1e"][0]) == 25
    assert 15 < sum(x == "train" for x in e_splits["I1e"][0].values()) < 20
    assert 5 < sum(x == "test" for x in e_splits["I1e"][0].values()) < 10
    assert 15 < sum(x == "train" for x in e_splits["C1e"][0].values()) < 20
    assert 5 < sum(x == "test" for x in e_splits["C1e"][0].values()) < 10


def check_identity_tsv(filename):
    assert Path(filename).is_file()
    with open(filename, "r") as data:
        for line in data.readlines()[1:]:
            parts = line.strip().split("\t")
            assert len(parts) == 2
            assert parts[0] == parts[1]


def check_assignment_tsv(filename: Path):
    assert filename.is_file()
    with open(filename, "r") as data:
        for line in data.readlines()[1:]:
            parts = line.strip().split("\t")
            assert len(parts) == (3 if "inter" in filename.stem else 2)
            assert parts[0] not in ["train", "test", "not_selected"]
            if "inter" in filename.stem:
                assert parts[0] != parts[1]
                assert parts[1] not in ["train", "test", "not_selected"]
            assert parts[-1] in ["train", "test", "not selected"]


@pytest.mark.issue
def test_issue1():
    test_pipeline(False, False, "data/pipeline/prot_sim.tsv", None, None, False, None, None, False, "C1e")


@pytest.mark.issue
def test_issue2():
    test_pipeline(True, False, "data/pipeline/prot_sim.tsv", None, "data/pipeline/drugs.tsv", False, None,
                  "data/pipeline/drug_dist.tsv", True, "C2")
