import sys
from pathlib import Path

import pandas as pd

from experiments.MPP.train import train_tech


def main(base_path: Path):
    dfs = []
    for tool in ["datasail/d_0.2_e_0.2", "deepchem"]:
        for model in ["rf", "svm", "xgb", "mlp", "d-mpnn"]:
            perf = train_tech(base_path / tool, base_path / "data", model, tool.split("/")[0], "tox21")
            df = pd.DataFrame(list(perf.items()), columns=["name", "perf"])
            df["model"] = model
            df["tool"] = tool.split("/")[0]
            df["run"] = df["name"].apply(lambda x: x.split("_")[1])
            dfs.append(df)
    pd.concat(dfs).to_csv(base_path / "results.csv", index=False)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
