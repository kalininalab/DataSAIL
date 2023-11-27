import os
import shutil

from datasail.sail import sail


def test_cdhit_cargs_1():
    sail([
        "-o", "data/pipeline/output",
        "-t", "C1e",
        "--e-type", "P",
        "--e-data", "data/pipeline/seqs.fasta",
        "--e-sim", "cdhit",
        "--e-args", "-c 0.55 -n 3"
    ])

    assert os.path.exists("data/pipeline/output")
    assert os.path.exists("data/pipeline/output/C1e")
    assert os.path.exists("data/pipeline/output/logs")
    assert os.path.exists("data/pipeline/output/logs/seqs_cdhit_c_0.55_n_3_l_2.log")

    shutil.rmtree("data/pipeline/output", ignore_errors=True)


def test_cdhit_cargs_2():
    sail([
        "-o", "data/pipeline/output",
        "-t", "C1e",
        "--e-type", "P",
        "--e-data", "data/pipeline/seqs.fasta",
        "--e-sim", "cdhit",
        "--e-args", "-c 0.8 -g 1"
    ])

    assert os.path.exists("data/pipeline/output")
    assert os.path.exists("data/pipeline/output/C1e")
    assert os.path.exists("data/pipeline/output/logs")
    assert os.path.exists("data/pipeline/output/logs/seqs_cdhit_c_0.8_n_5_l_4_g_1.log")

    shutil.rmtree("data/pipeline/output", ignore_errors=True)


def test_cdhitest_cargs_1():
    sail([
        "-o", "data/pipeline/output",
        "-t", "C1e",
        "--e-type", "G",
        "--e-data", "data/rw_data/RBD/RBD_small.fasta",
        "--e-sim", "cdhit_est",
        "--e-args", "-c 0.95 -n 9"
    ])

    assert os.path.exists("data/pipeline/output")
    assert os.path.exists("data/pipeline/output/C1e")
    assert os.path.exists("data/pipeline/output/logs")
    assert os.path.exists("data/pipeline/output/logs/RBD_small_cdhit_est_c_0.95_n_9_l_8.log")

    shutil.rmtree("data/pipeline/output", ignore_errors=True)


def test_cdhitest_cargs_2():
    sail([
        "-o", "data/pipeline/output",
        "-t", "C1e",
        "--e-type", "G",
        "--e-data", "data/rw_data/RBD/RBD_small.fasta",
        "--e-sim", "cdhit_est",
        "--e-args", "-c 0.95 -g 1"
    ])

    assert os.path.exists("data/pipeline/output")
    assert os.path.exists("data/pipeline/output/C1e")
    assert os.path.exists("data/pipeline/output/logs")
    assert os.path.exists("data/pipeline/output/logs/RBD_small_cdhit_est_c_0.95_n_10_l_9_g_1.log")

    shutil.rmtree("data/pipeline/output", ignore_errors=True)


def test_foldseek_cargs():
    sail([
        "--output", "data/pipeline/output",
        "-t", "C1e",
        "--e-type", "P",
        "--e-data", "data/pipeline/pdbs",
        "--e-sim", "foldseek",
        "--e-args", "-s 1"
    ])

    assert os.path.exists("data/pipeline/output")
    assert os.path.exists("data/pipeline/output/C1e")
    assert os.path.exists("data/pipeline/output/logs")
    assert os.path.exists("data/pipeline/output/logs/pdbs_foldseek.log")

    shutil.rmtree("data/pipeline/output", ignore_errors=True)


def test_mash_cargs():
    sail([
        "--output", "data/pipeline/output",
        "-t", "C1e",
        "--e-type", "G",
        "--e-data", "data/genomes",
        "--e-dist", "mash",
        "--e-args", "-v 0.9 -s 10000"
    ])

    assert os.path.exists("data/pipeline/output")
    assert os.path.exists("data/pipeline/output/C1e")
    assert os.path.exists("data/pipeline/output/logs")
    assert os.path.exists("data/pipeline/output/logs/genomes_mash.log")

    shutil.rmtree("data/pipeline/output", ignore_errors=True)


def test_mmseqs_cargs():
    sail([
        "--output", "data/pipeline/output",
        "-t", "C1e",
        "--e-type", "P",
        "--e-data", "data/pipeline/seqs.fasta",
        "--e-sim", "mmseqs",
        "--e-args", "-c 0.9 --cov-mode 1"
    ])

    assert os.path.exists("data/pipeline/output")
    assert os.path.exists("data/pipeline/output/C1e")
    assert os.path.exists("data/pipeline/output/logs")
    assert os.path.exists("data/pipeline/output/logs/seqs_mmseqs_c_0.9_covmode_1.log")

    shutil.rmtree("data/pipeline/output", ignore_errors=True)


def test_mmseqspp_cargs():
    sail([
        "--output", "data/pipeline/output",
        "-t", "C1e",
        "--e-type", "P",
        "--e-data", "data/pipeline/seqs.fasta",
        "--e-sim", "mmseqspp",
        "--e-args", "--mask-lower-case 1 --alignment-mode 1 --cov-mode 1"
    ])

    assert os.path.exists("data/pipeline/output")
    assert os.path.exists("data/pipeline/output/C1e")
    assert os.path.exists("data/pipeline/output/logs")
    assert os.path.exists("data/pipeline/output/logs/seqs_mmseqspp.log")

    shutil.rmtree("data/pipeline/output", ignore_errors=True)
