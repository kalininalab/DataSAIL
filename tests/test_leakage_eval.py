from pathlib import Path

import pandas as pd

from datasail.eval import eval_split
from datasail.settings import M_TYPE


def test_eval_split():
    seqs = pd.read_csv(Path("data") / "pipeline" / "seqs.tsv", sep="\t")["ID"].values.tolist()
    scaled_dl, total_dl, max_dl = eval_split(
        "P",
        Path("data") / "pipeline" / "seqs.fasta",
        Path("data") / "pipeline" / "prot_weights.tsv",
        similarity="mmseqspp",
        distance=None,
        dist_conv=None,
        split_assignment={seq: ("test" if i % 4 == 0 else "train") for i, seq in enumerate(seqs)},
    )
    assert 0 < scaled_dl and scaled_dl < 1
    assert 0 < total_dl
    assert 0 < max_dl
    assert total_dl < max_dl


def test_esol():
    PATH = Path("data") / "rw_data" / "esol"
    df_i_tr = pd.read_csv(PATH / "I1e" / "train.csv")
    df_i_te = pd.read_csv(PATH / "I1e" / "test.csv")
    df_c_tr = pd.read_csv(PATH / "C1e" / "train.csv")
    df_c_te = pd.read_csv(PATH / "C1e" / "test.csv")
    df_i_tr["split"] = "train"
    df_i_te["split"] = "test"
    df_c_tr["split"] = "train"
    df_c_te["split"] = "test"
    df_i = pd.concat([df_i_tr, df_i_te], axis=0)
    df_c = pd.concat([df_c_tr, df_c_te], axis=0)
    data = dict(df_i[["ID", "SMILES"]].values.tolist())
    i_assi = dict(df_i[["ID", "split"]].values.tolist())
    c_assi = dict(df_c[["ID", "split"]].values.tolist())
    sim_i = eval_split(
        datatype=M_TYPE,
        data=data,
        weights=None,
        similarity="tanimoto",
        distance=None,
        dist_conv=None,
        split_assignment=i_assi,
    )
    sim_c = eval_split(
        datatype=M_TYPE,
        data=data,
        weights=None,
        similarity="tanimoto",
        distance=None,
        dist_conv=None,
        split_assignment=c_assi,
    )
    dist_i = eval_split(
        datatype=M_TYPE,
        data=data,
        weights=None,
        similarity=None,
        distance="jaccard",
        dist_conv=None,
        split_assignment=i_assi,
    )
    dist_c = eval_split(
        datatype=M_TYPE,
        data=data,
        weights=None,
        similarity=None,
        distance="jaccard",
        dist_conv=None,
        split_assignment=c_assi,
    )

    assert 0 < sim_c[0] < sim_i[0] <= 1
    assert 0 < dist_c[0] < dist_i[0] < 1