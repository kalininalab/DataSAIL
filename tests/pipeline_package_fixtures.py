from pathlib import Path

import pytest

from datasail.cluster.clustering import cluster
from datasail.reader.utils import read_csv

from datasail.reader.read_molecules import read_molecule_data
from datasail.reader.read_proteins import read_protein_data, parse_fasta

base = Path("data") / "rw_data"

def sabdab_inter_list():
    return list(read_csv(base / "sabdab_full" / "inter.tsv", "\t"))


def sabdab_inter_returner():
    return lambda: list(read_csv(base / "sabdab_full" / "inter.tsv", "\t"))


def sabdab_inter_generator():
    for x in list(read_csv(base / "sabdab_full" / "inter.tsv", "\t")):
        yield x


@pytest.fixture
def sabdab_ag_dataset():
    return cluster(
        read_protein_data(base / "sabdab_full" / "ag.fasta", None, None, None, None,
                          list(read_csv(base / "sabdab_full" / "inter.tsv", "\t")), 0, 50, ""),
        num_clusters=50,
        threads=1,
        logdir=Path(),
        linkage="average",
    )


@pytest.fixture
def sabdab_vh_dataset():
    return cluster(
        read_protein_data(base / "sabdab_full" / "vh.fasta", None, None, None, None,
                          list(read_csv(base / "sabdab_full" / "inter.tsv", "\t")), 1, 50, ""),
        num_clusters=50,
        threads=1,
        logdir=Path(),
        linkage="average",
    )


@pytest.fixture
def mave_dataset():
    return cluster(
        read_protein_data(base / "mave" / "mave_db_gold_standard_only_sequences.fasta",
                          base / "mave" / "mave_db_gold_standard_weights.tsv",
                          None, None, None, None, None, 50, ""
        ),
        num_clusters=50,
        threads=1,
        logdir=Path(),
        linkage="average",
    )


@pytest.fixture
def mibig_dataset():
    return cluster(
        read_molecule_data(
            base / "mibig" / "compounds.tsv", None, None, None, None, None, None, 50, ""
        ),
        num_clusters=50,
        threads=1,
        logdir=Path(),
        linkage="average",
    )


def mibig_dict():
    return dict(read_csv(base / "mibig" / "compounds.tsv", "\t"))


def mibig_returner():
    return lambda: dict(read_csv(base / "mibig" / "compounds.tsv", "\t"))


def mibig_generator():
    for x in list(read_csv(base / "mibig" / "compounds.tsv", "\t")):
        yield x


def mave_weights_dict():
    return dict((n, float(w)) for n, w in read_csv(base / "mave" / "mave_db_gold_standard_weights.tsv", "\t"))


def mave_weights_returner():
    return lambda: dict((n, float(w)) for n, w in read_csv(base / "mave" / "mave_db_gold_standard_weights.tsv", "\t"))


def mave_weights_generator():
    for x, y in list(read_csv(base / "mave" / "mave_db_gold_standard_weights.tsv", "\t")):
        yield x, float(y)


def mave_dict():
    return parse_fasta(base / "mave" / "mave_db_gold_standard_only_sequences.fasta")


def mave_returner():
    return lambda: parse_fasta(base / "mave" / "mave_db_gold_standard_only_sequences.fasta")


def mave_generator():
    for x in list(parse_fasta(base / "mave" / "mave_db_gold_standard_only_sequences.fasta").items()):
        yield x


def sabdab_ag_dict():
    return parse_fasta(base / "sabdab_full" / "ag.fasta")


def sabdab_ag_returner():
    return lambda: parse_fasta(base / "sabdab_full" / "ag.fasta")


def sabdab_ag_generator():
    for x in list(parse_fasta(base / "sabdab_full" / "ag.fasta").items()):
        yield x


def sabdab_vh_dict():
    return parse_fasta(base / "sabdab_full" / "vh.fasta")


def sabdab_vh_returner():
    return lambda: parse_fasta(base / "sabdab_full" / "vh.fasta")


def sabdab_vh_generator():
    for x in list(parse_fasta(base / "sabdab_full" / "vh.fasta").items()):
        yield x
