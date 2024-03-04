import shutil
from pathlib import Path

from datasail.sail import sail


def test_cdhit_cargs_1():
    out = Path("data") / "pipeline" / "output"
    sail([
        "-o", str(out),
        "-t", "C1e",
        "-s", "0.7", "0.3",
        "--e-type", "P",
        "--e-data", str(Path("data") / "pipeline" / "seqs.fasta"),
        "--e-sim", "cdhit",
        "--e-args", "-c 0.55 -n 3",
    ])

    assert out.is_dir()
    assert (out / "C1e").is_dir()
    assert (out / "logs").is_dir()
    assert (out / "logs" / "seqs_cdhit_c_0.55_n_3_l_2.log").is_file()

    shutil.rmtree(out, ignore_errors=True)


def test_cdhit_cargs_2():
    out = Path("data") / "pipeline" / "output"
    sail([
        "-o", str(out),
        "-t", "C1e",
        "-s", "0.7", "0.3",
        "--e-type", "P",
        "--e-data", str(Path("data") / "pipeline" / "seqs.fasta"),
        "--e-sim", "cdhit",
        "--e-args", "-c 0.8 -g 1",
    ])

    assert out.is_dir()
    assert (out / "C1e").is_dir()
    assert (out / "logs").is_dir()
    assert (out / "logs" / "seqs_cdhit_c_0.8_n_5_l_4_g_1.log").is_file()

    shutil.rmtree(out, ignore_errors=True)


def test_cdhitest_cargs_1():
    out = Path("data") / "pipeline" / "output"
    sail([
        "-o", str(out),
        "-t", "C1e",
        "-s", "0.7", "0.3",
        "--e-type", "G",
        "--e-data", str(Path("data") / "rw_data" / "RBD" / "RBD_small.fasta"),
        "--e-sim", "cdhit_est",
        "--e-args", "-c 0.95 -n 9",
    ])

    assert out.is_dir()
    assert (out / "C1e").is_dir()
    assert (out / "logs").is_dir()
    assert (out / "logs" / "RBD_small_cdhit_est_c_0.95_n_9_l_8.log")

    shutil.rmtree(out, ignore_errors=True)


def test_cdhitest_cargs_2():
    out = Path("data/pipeline/output")
    sail([
        "-o", str(out),
        "-t", "C1e",
        "-s", "0.7", "0.3",
        "--e-type", "G",
        "--e-data", str(Path("data") / "rw_data" / "RBD" / "RBD_small.fasta"),
        "--e-sim", "cdhit_est",
        "--e-args", "-c 0.95 -g 1",
    ])

    assert out.is_dir()
    assert (out / "C1e").is_dir()
    assert (out / "logs").is_dir()
    assert (out / "logs" / "RBD_small_cdhit_est_c_0.95_n_10_l_9_g_1.log")

    shutil.rmtree(out, ignore_errors=True)


def test_foldseek_cargs():
    out = Path("data/pipeline/output")
    sail([
        "--output", str(out),
        "-t", "C1e",
        "-s", "0.7", "0.3",
        "--e-type", "P",
        "--e-data", str(Path("data") / "pipeline" / "pdbs"),
        "--e-sim", "foldseek",
        "--e-args", "-s 1",
    ])

    assert out.is_dir()
    assert (out / "C1e").is_dir()
    assert (out / "logs").is_dir()
    assert (out / "logs" / "pdbs_foldseek.log")

    shutil.rmtree(out, ignore_errors=True)


def test_mash_cargs():
    out = Path("data/pipeline/output")
    sail([
        "--output", str(out),
        "-t", "C1e",
        "-s", "0.7", "0.3",
        "--e-type", "G",
        "--e-data", str(Path("data") / "genomes"),
        "--e-dist", "mash",
        "--e-args", "-v 0.9 -s 10000",
    ])

    assert out.is_dir()
    assert (out / "C1e").is_dir()
    assert (out / "logs").is_dir()
    assert (out / "logs" / "genomes_mash.log")

    shutil.rmtree(out, ignore_errors=True)


def test_diamond_cargs():
    out = Path("data/pipeline/output")
    sail([
        "-o", str(out),
        "-t", "C1e",
        "-s", "0.7", "0.3",
        "--e-type", "P",
        "--e-data", str(Path('data') / 'pipeline' / 'seqs.fasta'),
        "--e-sim", "diamond",
        "--e-args", "--faster"
    ])

    assert out.is_dir()
    assert (out / "C1e").is_dir()
    assert (out / "logs").is_dir()
    assert (out / "logs" / "seqs_diamond_faster.log")

    shutil.rmtree(out, ignore_errors=True)


def test_mmseqs_cargs():
    out = Path("data/pipeline/output")
    sail([
        "-o", str(out),
        "-t", "C1e",
        "-s", "0.7", "0.3",
        "--e-type", "P",
        "--e-data", str(Path('data') / 'pipeline' / 'seqs.fasta'),
        "--e-sim", "mmseqs",
        "--e-args", "-c 0.9 --cov-mode 1"
    ])

    assert out.is_dir()
    assert (out / "C1e").is_dir()
    assert (out / "logs").is_dir()
    assert (out / "logs" / "seqs_mmseqs_c_0.9_covmode_1.log")

    shutil.rmtree(out, ignore_errors=True)


def test_mmseqspp_cargs():
    out = Path("data/pipeline/output")
    sail([
        "-o", str(out),
        "-t", "C1e",
        "-s", "0.7", "0.3",
        "--e-type", "P",
        "--e-data", str(Path("data") / "pipeline" / "seqs.fasta"),
        "--e-sim", "mmseqspp",
        "--e-args", "--mask-lower-case 1 --alignment-mode 1 --cov-mode 1",
    ])

    assert out.is_dir()
    assert (out / "C1e").is_dir()
    assert (out / "logs").is_dir()
    assert (out / "logs" / "seqs_mmseqspp.log")

    shutil.rmtree(out, ignore_errors=True)
