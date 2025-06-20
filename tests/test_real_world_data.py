import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from datasail.reader.utils import read_csv, parse_fasta
from datasail.settings import NOT_ASSIGNED
from tests.utils import run_sail

base = Path("data") / "rw_data"
mave = base / "mave"
mibig = base / "mibig"
sabdab_f = base / "sabdab_full"
sabdab_d = base / "sabdab_domains"


@pytest.mark.real
@pytest.mark.todo
@pytest.mark.full
@pytest.mark.parametrize(
    "ligand_data,ligand_weights,protein_data,protein_weights,interactions,output", [
        (None, None, mave / "mave_db_gold_standard_only_sequences.fasta",
         mave / "mave_db_gold_standard_weights.tsv", None, base / "mave_splits"),
        (mibig / "compounds.tsv", None, None, None, None, base / "mibig_splits"),
        (sabdab_f / "ag.fasta", None, sabdab_f / "vh.fasta", None, sabdab_f / "inter.tsv", base / "sabdab_full_splits"),
        (sabdab_f / "ag.fasta", None, sabdab_f / "vh.fasta", None, None, base / "sabdab_full2_splits"),
        (sabdab_d / "ag.fasta", None, sabdab_d / "CDR-H1_48_seqlen.tsv", None, sabdab_d / "inter.tsv",
         base / "sabdab_dom1_splits"),
        (sabdab_d / "ag.fasta", None, sabdab_d / "CDR-H1_48_seqlen_1000.tsv", None, sabdab_d / "inter.tsv",
         base / "sabdab_dom2_splits"),
        (sabdab_d / "ag.fasta", None, sabdab_d / "CDR-H1_seq_all.tsv", None, sabdab_d / "inter.tsv",
         base / "sabdab_dom3_splits"),
        (sabdab_d / "ag.fasta", None, sabdab_d / "CDR-H2_48_seqlen.tsv", None, sabdab_d / "inter.tsv",
         base / "sabdab_dom4_splits"),
        (sabdab_d / "ag.fasta", None, sabdab_d / "CDR-H2_48_seqlen_1000.tsv", None, sabdab_d / "inter.tsv",
         base / "sabdab_dom5_splits"),
        (sabdab_d / "ag.fasta", None, sabdab_d / "CDR-H2_seq_all.tsv", None, sabdab_d / "inter.tsv",
         base / "sabdab_dom6_splits"),
        (sabdab_d / "ag.fasta", None, sabdab_d / "CDR-H3_48_seqlen.tsv", None, sabdab_d / "inter.tsv",
         base / "sabdab_dom7_splits"),
        (sabdab_d / "ag.fasta", None, sabdab_d / "CDR-H3_48_seqlen_1000.tsv", None, sabdab_d / "inter.tsv",
         base / "sabdab_dom8_splits"),
        (sabdab_d / "ag.fasta", None, sabdab_d / "CDR-H3_seq_all.tsv", None, sabdab_d / "inter.tsv",
         base / "sabdab_dom9_splits"),
    ])
def test_full_single_colds(ligand_data, ligand_weights, protein_data, protein_weights, interactions, output):
    techniques = []
    if ligand_data is not None:
        techniques += ["I1e", "C1e"]
    if protein_data is not None:
        techniques += ["I1f", "C1f"]

    run_sail(
        inter=interactions,
        output=output,
        techniques=techniques,
        splits=[0.7, 0.3],
        names=["train", "test"],
        epsilon=0.2,
        e_type=None if ligand_data is None else ("P" if "sabdab" in str(ligand_data) else "M"),
        e_data=ligand_data,
        e_weights=ligand_weights,
        f_type=None if protein_data is None else "P",
        f_data=protein_data,
        f_weights=protein_weights,
        solver="SCIP",
        threads=1
    )

    if ligand_data is not None:
        name_prefix = "Protein_e_seqs" if "sabdab" in str(ligand_data) else "Molecule_e_smiles"
        assert (output / "I1e").is_dir()
        if interactions is not None:
            assert (output / "I1e" / "inter.tsv").is_file()
            assert check_inter_completeness(interactions, output / "I1e" / "inter.tsv", ["train", "test"])
        assert (output / "I1e" / (name_prefix + "_splits.tsv")).is_file()
        assert check_split_completeness(ligand_data, output / "I1e" / (name_prefix + "_splits.tsv"), ["train", "test"])

        assert (output / "C1e").is_dir()
        if interactions is not None:
            assert (output / "C1e" / "inter.tsv").is_file()
            assert check_inter_completeness(interactions, output / "C1e" / "inter.tsv", ["train", "test"])
        assert (output / "C1e" / (name_prefix + "_cluster_hist.png")).is_file()
        assert (output / "C1e" / (name_prefix + "_clusters.png")).is_file()
        assert (output / "C1e" / (name_prefix + "_clusters.tsv")).is_file()
        assert (output / "C1e" / (name_prefix + "_splits.tsv")).is_file()
        assert check_split_completeness(ligand_data, output / "C1e" / (name_prefix + "_splits.tsv"), ["train", "test"])

    if protein_data is not None:
        assert (output / "I1f").is_dir()
        if interactions is not None:
            assert (output / "I1f" / "inter.tsv").is_file()
            assert check_inter_completeness(interactions, output / "I1f" / "inter.tsv", ["train", "test"])
        assert (split_path := (output / "I1f" / f"Protein_{protein_data.stem}_splits.tsv")).is_file()
        assert check_split_completeness(protein_data, split_path, ["train", "test"])

        assert (output / "C1f").is_dir()
        if interactions is not None:
            assert (output / "C1f" / "inter.tsv").is_file()
            assert check_inter_completeness(interactions, output / "C1f" / "inter.tsv", ["train", "test"])
        assert (output / "C1f" / f"Protein_{protein_data.stem}_cluster_hist.png").is_file()
        assert (output / "C1f" / f"Protein_{protein_data.stem}_clusters.png").is_file()
        assert (output / "C1f" / f"Protein_{protein_data.stem}_clusters.tsv").is_file()
        assert (split_path := (output / "C1f" / f"Protein_{protein_data.stem}_splits.tsv")).is_file()
        assert check_split_completeness(protein_data, split_path, ["train", "test"])

    assert (output / "logs").is_dir()
    # assert (output / "tmp").is_dir()

    shutil.rmtree(output, ignore_errors=True)


@pytest.mark.full
def test_pdbbind_splits():
    base = Path("data") / "rw_data"
    df = pd.read_csv(base / "LP_PDBBind.csv").iloc[:1000, :]
    run_sail(
        inter=[(x[0], x[0]) for x in df[["ids"]].values.tolist()],
        output=(out := base / "pdbbind_splits"),
        techniques=["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"],
        splits=[0.8, 0.2],
        names=["train", "test"],
        epsilon=0.2,
        e_type="M",
        e_data=df[["ids", "Ligand"]].values.tolist(),
        e_sim="mmseqs",
        f_type="P",
        f_data=df[["ids", "Target"]].values.tolist(),
        solver="SCIP",
        threads=1
    )

    assert out.is_dir()
    assert (r := out / "R").is_dir()
    assert (i1e := out / "I1e").is_dir()
    assert (i1f := out / "I1f").is_dir()
    assert (i2 := out / "I2").is_dir()
    assert (c1e := out / "C1e").is_dir()
    assert (c1f := out / "C1f").is_dir()
    assert (c2 := out / "C2").is_dir()

    assert (r / "inter.tsv")
    assert (i1e / "inter.tsv")
    assert (i1f / "inter.tsv")
    assert (i2 / "inter.tsv")
    assert (c1e / "inter.tsv")
    assert (c1f / "inter.tsv")
    assert (c2 / "inter.tsv")

    for technique in ["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"]:
        df = pd.read_csv(f"data/rw_data/pdbbind_splits/{technique}/inter.tsv", sep="\t")
        # assert df.shape > (19120, 3)
        assert set(df.columns).issubset({"E_ID", "F_ID", "Split"})
        assert set(df["Split"].unique()).issubset({"train", "test", NOT_ASSIGNED})
        vc = df["Split"].value_counts().to_dict()
        if technique[-1] != "2" and "train" in vc and "test" in vc:
            assert vc["train"] > vc["test"]
            assert vc["test"] > 0

    shutil.rmtree("data/rw_data/pdbbind_splits", ignore_errors=True)


def check_inter_completeness(input_inter_filename, split_inter_filename, split_names):
    split_names = set(split_names + ["not selected"])
    with open(split_inter_filename, "r") as inter:
        split_inter = dict()
        for line in inter.readlines()[1:]:
            a, b, split = line.strip().split("\t")[:3]
            split_inter[(a, b)] = split

    input_inter_count = 0
    with open(input_inter_filename, "r") as inter:
        for line in inter.readlines()[1:]:
            a, b = line.strip().split("\t")[:2]
            if (a, b) not in split_inter:
                return False
            if split_inter[(a, b)] not in split_names:
                return False
            input_inter_count += 1
    return input_inter_count == len(split_inter)


def check_split_completeness(input_data, split_names_filename, split_names):
    split_names = set(split_names + ["not selected"])
    with open(split_names_filename, "r") as data:
        names_split = dict()
        for line in data.readlines()[1:]:
            key, value = line.strip().split("\t")[:2]
            names_split[key] = value

    names_count = 0
    if input_data.is_dir():
        for filename in os.listdir(input_data):
            if filename not in names_split:
                return False
            if names_split[filename] not in split_names:
                return False
            names_count += 1
    elif input_data.is_file():
        if input_data.suffix in {".fasta", ".fa", ".fna"}:
            data = parse_fasta(input_data)
        elif input_data.suffix in {".tsv"}:
            data = dict(read_csv(input_data, "\t"))
        else:
            return False
        for item in data:
            if item not in names_split:
                return False
            if names_split[item] not in split_names:
                return False
            names_count += 1
    if names_count != len(names_split):
        return False
    return names_count == len(names_split)
