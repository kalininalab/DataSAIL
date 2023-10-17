import pytest

from datasail.cluster.clustering import cluster
from datasail.reader.utils import read_csv

from datasail.reader.read_molecules import read_molecule_data
from datasail.reader.read_proteins import read_protein_data, parse_fasta


def sabdab_inter_list():
    return list(read_csv("data/rw_data/sabdab_full/interactions.tsv"))


def sabdab_inter_returner():
    return lambda: list(read_csv("data/rw_data/sabdab_full/interactions.tsv"))


def sabdab_inter_generator():
    for x in list(read_csv("data/rw_data/sabdab_full/interactions.tsv")):
        yield x


@pytest.fixture
def sabdab_ag_dataset():
    return cluster(
        read_protein_data("data/rw_data/sabdab_full/ag.fasta", None, None, None, 1, 1, list(read_csv("data/rw_data/sabdab_full/interactions.tsv")), 0),
        threads=1,
        logdir="",
    )


@pytest.fixture
def sabdab_vh_dataset():
    return cluster(
        read_protein_data("data/rw_data/sabdab_full/vh.fasta", None, None, None, 1, 1, list(read_csv("data/rw_data/sabdab_full/interactions.tsv")), 1),
        threads=1,
        logdir="",
    )


@pytest.fixture
def mave_dataset():
    return cluster(
        read_protein_data(
            "data/rw_data/mave/mave_db_gold_standard_only_sequences.fasta",
            "data/rw_data/mave/mave_db_gold_standard_weights.tsv", None, None, 1, 1, None, None
        ),
        threads=1,
        logdir="",
    )


@pytest.fixture
def mibig_dataset():
    return cluster(
        read_molecule_data(
            "data/rw_data/mibig/compounds.tsv", None, None, None, 1, 1, None, None
        ),
        threads=1,
        logdir="",
    )


def mibig_dict():
    return dict(read_csv("data/rw_data/mibig/compounds.tsv"))


def mibig_returner():
    return lambda: dict(read_csv("data/rw_data/mibig/compounds.tsv"))


def mibig_generator():
    for x in list(read_csv("data/rw_data/mibig/compounds.tsv")):
        yield x


def mave_weights_dict():
    return dict((n, float(w)) for n, w in read_csv("data/rw_data/mave/mave_db_gold_standard_weights.tsv"))


def mave_weights_returner():
    return lambda: dict((n, float(w)) for n, w in read_csv("data/rw_data/mave/mave_db_gold_standard_weights.tsv"))


def mave_weights_generator():
    for x, y in list(read_csv("data/rw_data/mave/mave_db_gold_standard_weights.tsv")):
        yield x, float(y)


def mave_dict():
    return parse_fasta("data/rw_data/mave/mave_db_gold_standard_only_sequences.fasta")


def mave_returner():
    return lambda: parse_fasta("data/rw_data/mave/mave_db_gold_standard_only_sequences.fasta")


def mave_generator():
    for x in list(parse_fasta("data/rw_data/mave/mave_db_gold_standard_only_sequences.fasta").items()):
        yield x


def sabdab_ag_dict():
    return parse_fasta("data/rw_data/sabdab_full/ag.fasta")


def sabdab_ag_returner():
    return lambda: parse_fasta("data/rw_data/sabdab_full/ag.fasta")


def sabdab_ag_generator():
    for x in list(parse_fasta("data/rw_data/sabdab_full/ag.fasta").items()):
        yield x


def sabdab_vh_dict():
    return parse_fasta("data/rw_data/sabdab_full/vh.fasta")


def sabdab_vh_returner():
    return lambda: parse_fasta("data/rw_data/sabdab_full/vh.fasta")


def sabdab_vh_generator():
    for x in list(parse_fasta("data/rw_data/sabdab_full/vh.fasta").items()):
        yield x
