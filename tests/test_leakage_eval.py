from pathlib import Path

import pandas as pd

from datasail.eval import eval_single_split


def test_eval_split():
    seqs = pd.read_csv(Path("data") / "pipeline" / "seqs.tsv", sep="\t")["ID"].values.tolist()
    scaled_dl, total_dl, max_dl = eval_single_split(
        "P",
        Path("data") / "pipeline" / "seqs.fasta",
        Path("data") / "pipeline" / "prot_weights.tsv",
        similarity="mmseqspp",
        distance=None,
        split_assignment={seq: ("test" if i % 4 == 0 else "train") for i, seq in enumerate(seqs)},
    )
    assert 0 < scaled_dl and scaled_dl < 1
    assert 0 < total_dl
    assert 0 < max_dl
    assert total_dl < max_dl
