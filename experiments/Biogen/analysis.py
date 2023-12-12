from pathlib import Path

import scikit_posthocs as sp
import pandas as pd


def datasail():
    root = Path("experiments") / "Biogen" / "datasail" / "HLM"
    df = pd.read_csv(root / "test_metrics.tsv")
    df = df.transpose().iloc[1:]
    df["tech"] = df.index.str.split("_").str[0]
    df.columns = ["mae", "tech"]
    print(df)

    print(sp.posthoc_wilcoxon(df, val_col="mae", group_col="tech"))


if __name__ == '__main__':
    datasail()
