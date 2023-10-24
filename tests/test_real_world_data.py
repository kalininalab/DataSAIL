import os
import shutil
from os.path import isdir, isfile, join

import pandas as pd
import pytest

from datasail.reader.read_proteins import parse_fasta
from datasail.reader.utils import read_csv
from datasail.settings import NOT_ASSIGNED
from tests.utils import run_sail


@pytest.mark.real
@pytest.mark.todo
@pytest.mark.parametrize(
    "ligand_data,ligand_weights,protein_data,protein_weights,interactions,output", [
        (None, None, "data/rw_data/mave/mave_db_gold_standard_only_sequences.fasta",
         "data/rw_data/mave/mave_db_gold_standard_weights.tsv", None, "data/rw_data/mave_splits"),
        ("data/rw_data/mibig/compounds.tsv", None, None, None, None, "data/rw_data/mibig_splits"),
        ("data/rw_data/sabdab_full/ag.fasta", None, "data/rw_data/sabdab_full/vh.fasta", None,
         "data/rw_data/sabdab_full/interactions.tsv", "data/rw_data/sabdab_full_splits"),
        ("data/rw_data/sabdab_full/ag.fasta", None, "data/rw_data/sabdab_full/vh.fasta", None, None,
         "data/rw_data/sabdab_full2_splits"),
        ("data/rw_data/sabdab_domains/ag.fasta", None, "data/rw_data/sabdab_domains/CDR-H1_48_seqlen.tsv", None,
         "data/rw_data/sabdab_domains/interactions.tsv", "data/rw_data/sabdab_dom1_splits"),
        ("data/rw_data/sabdab_domains/ag.fasta", None, "data/rw_data/sabdab_domains/CDR-H1_48_seqlen_1000.tsv", None,
         "data/rw_data/sabdab_domains/interactions.tsv", "data/rw_data/sabdab_dom2_splits"),
        ("data/rw_data/sabdab_domains/ag.fasta", None, "data/rw_data/sabdab_domains/CDR-H1_seq_all.tsv", None,
         "data/rw_data/sabdab_domains/interactions.tsv", "data/rw_data/sabdab_dom3_splits"),
        ("data/rw_data/sabdab_domains/ag.fasta", None, "data/rw_data/sabdab_domains/CDR-H2_48_seqlen.tsv", None,
         "data/rw_data/sabdab_domains/interactions.tsv", "data/rw_data/sabdab_dom4_splits"),
        ("data/rw_data/sabdab_domains/ag.fasta", None, "data/rw_data/sabdab_domains/CDR-H2_48_seqlen_1000.tsv", None,
         "data/rw_data/sabdab_domains/interactions.tsv", "data/rw_data/sabdab_dom5_splits"),
        ("data/rw_data/sabdab_domains/ag.fasta", None, "data/rw_data/sabdab_domains/CDR-H2_seq_all.tsv", None,
         "data/rw_data/sabdab_domains/interactions.tsv", "data/rw_data/sabdab_dom6_splits"),
        ("data/rw_data/sabdab_domains/ag.fasta", None, "data/rw_data/sabdab_domains/CDR-H3_48_seqlen.tsv", None,
         "data/rw_data/sabdab_domains/interactions.tsv", "data/rw_data/sabdab_dom7_splits"),
        ("data/rw_data/sabdab_domains/ag.fasta", None, "data/rw_data/sabdab_domains/CDR-H3_48_seqlen_1000.tsv", None,
         "data/rw_data/sabdab_domains/interactions.tsv", "data/rw_data/sabdab_dom8_splits"),
        ("data/rw_data/sabdab_domains/ag.fasta", None, "data/rw_data/sabdab_domains/CDR-H3_seq_all.tsv", None,
         "data/rw_data/sabdab_domains/interactions.tsv", "data/rw_data/sabdab_dom9_splits"),
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
        epsilon=0.1,
        e_type=None if ligand_data is None else ("P" if "sabdab" in ligand_data else "M"),
        e_data=ligand_data,
        e_weights=ligand_weights,
        f_type=None if protein_data is None else "P",
        f_data=protein_data,
        f_weights=protein_weights,
        solver="SCIP",
        threads=1
    )

    if ligand_data is not None:
        name_prefix = "Protein_e_seqs" if "sabdab" in ligand_data else "Molecule_e_smiles"
        assert isdir(join(output, "I1e"))
        if interactions is not None:
            assert isfile(join(output, "I1e", "inter.tsv"))
            assert check_inter_completeness(interactions, join(output, "I1e", "inter.tsv"), ["train", "test"])
        assert isfile(join(output, "I1e", name_prefix + "_splits.tsv"))
        assert check_split_completeness(ligand_data, join(output, "I1e", name_prefix + "_splits.tsv"), ["train", "test"])

        assert isdir(join(output, "C1e"))
        if interactions is not None:
            assert isfile(join(output, "C1e", "inter.tsv"))
            assert check_inter_completeness(interactions, join(output, "C1e", "inter.tsv"), ["train", "test"])
        assert isfile(join(output, "C1e", name_prefix + "_cluster_hist.png"))
        assert isfile(join(output, "C1e", name_prefix + "_clusters.png"))
        assert isfile(join(output, "C1e", name_prefix + "_clusters.tsv"))
        assert isfile(join(output, "C1e", name_prefix + "_splits.tsv"))
        assert check_split_completeness(ligand_data, join(output, "C1e", name_prefix + "_splits.tsv"), ["train", "test"])

    if protein_data is not None:
        assert isdir(join(output, "I1f"))
        if interactions is not None:
            assert isfile(join(output, "I1f", "inter.tsv"))
            assert check_inter_completeness(interactions, join(output, "I1f", "inter.tsv"), ["train", "test"])
        assert isfile(join(output, "I1f", "Protein_f_seqs_splits.tsv"))
        assert check_split_completeness(protein_data, join(output, "I1f", "Protein_f_seqs_splits.tsv"), ["train", "test"])

        assert isdir(join(output, "C1f"))
        if interactions is not None:
            assert isfile(join(output, "C1f", "inter.tsv"))
            assert check_inter_completeness(interactions, join(output, "C1f", "inter.tsv"), ["train", "test"])
        assert isfile(join(output, "C1f", "Protein_f_seqs_cluster_hist.png"))
        assert isfile(join(output, "C1f", "Protein_f_seqs_clusters.png"))
        assert isfile(join(output, "C1f", "Protein_f_seqs_clusters.tsv"))
        assert isfile(join(output, "C1f", "Protein_f_seqs_splits.tsv"))
        assert check_split_completeness(protein_data, join(output, "C1f", "Protein_f_seqs_splits.tsv"), ["train", "test"])

    assert isdir(join(output, "logs"))
    assert isdir(join(output, "tmp"))

    shutil.rmtree(output, ignore_errors=True)


def test_pdbbind_splits():
    df = pd.read_csv("data/rw_data/LP_PDBBind.csv")  # .iloc[:1000, :]
    run_sail(
        inter=[(x[0], x[0]) for x in df[["ids"]].values.tolist()],
        output="data/rw_data/pdbbind_splits",
        techniques=["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"],
        splits=[0.8, 0.2],
        names=["train", "test"],
        epsilon=0.1,
        e_type="M",
        e_data=df[["ids", "Ligand"]].values.tolist(),
        f_type="P",
        f_data=df[["ids", "Target"]].values.tolist(),
        solver="SCIP",
        threads=1
    )

    assert isdir("data/rw_data/pdbbind_splits")
    assert isdir("data/rw_data/pdbbind_splits/R")
    assert isdir("data/rw_data/pdbbind_splits/I1e")
    assert isdir("data/rw_data/pdbbind_splits/I1f")
    assert isdir("data/rw_data/pdbbind_splits/I2")
    assert isdir("data/rw_data/pdbbind_splits/C1e")
    assert isdir("data/rw_data/pdbbind_splits/C1f")
    assert isdir("data/rw_data/pdbbind_splits/C2")

    assert isfile("data/rw_data/pdbbind_splits/R/inter.tsv")
    assert isfile("data/rw_data/pdbbind_splits/I1e/inter.tsv")
    assert isfile("data/rw_data/pdbbind_splits/I1f/inter.tsv")
    assert isfile("data/rw_data/pdbbind_splits/I2/inter.tsv")
    assert isfile("data/rw_data/pdbbind_splits/C1e/inter.tsv")
    assert isfile("data/rw_data/pdbbind_splits/C1f/inter.tsv")
    assert isfile("data/rw_data/pdbbind_splits/C2/inter.tsv")

    for technique in ["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"]:
        print(technique)
        df = pd.read_csv(f"data/rw_data/pdbbind_splits/{technique}/inter.tsv", sep="\t")
        assert df.shape > (19120, 3)
        assert set(df.columns).issubset({"E_ID", "F_ID", "Split"})
        assert set(df["Split"].unique()).issubset({"train", "test", NOT_ASSIGNED})
        vc = df["Split"].value_counts().to_dict()
        assert vc["train"] / (vc["train"] + vc["test"]) > 0.69
        assert vc["test"] > 10

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
    if isdir(input_data):
        for filename in os.listdir(input_data):
            if filename not in names_split:
                return False
            if names_split[filename] not in split_names:
                return False
            names_count += 1
    elif isfile(input_data):
        if input_data.split(".")[-1].lower() in {"fasta", "fa", "fna"}:
            data = parse_fasta(input_data)
        elif input_data.split(".")[-1].lower() in {"tsv"}:
            data = dict(read_csv(input_data))
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
