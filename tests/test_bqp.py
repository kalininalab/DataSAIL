import os

from scala.bqp.run import bqp_main


def test_random():
    bqp_main(
        input="data/perf_8_2/prot.tsv",
        output="data/perf_8_2/random/",
        method="ilp",
        verbosity="I",
        weight_file=None,
        inter="data/perf_8_2/inter.tsv",
        drugs="data/perf_8_2/lig.tsv",
        splits=[0.7, 0.3],
        limit=0.05,
        drug_weights=None,
        technique="R",
        prot_sim=None,
        drug_sim=None,
        header=None,
        sep="\t",
        names=["train", "test"],
    )

    assert os.path.exists("data/perf_8_2/random/inter.tsv")
    with open("data/perf_8_2/random/inter.tsv", "r") as data:
        inter = [line.strip().split("\t") for line in data.readlines()]

    drugs, targets, splits = list(zip(*inter))
    trains, tests = splits.count("train"), splits.count("test")
    train_frac = trains / (trains + tests)
    test_frac = tests / (trains + tests)
    assert 0.7 * 0.95 <= train_frac <= 0.7 * 1.05
    assert 0.3 * 0.95 <= test_frac <= 0.3 * 1.05

