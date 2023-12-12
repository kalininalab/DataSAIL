import os
from pathlib import Path

import chemprop
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from experiments.utils import RUNS, MPP_EPOCHS


def train(tool, subset, techniques):
    dfs = {"val": pd.DataFrame({"rows": list(range(50))}), "test": pd.DataFrame({"rows": [0]})}
    # store the results in training, validation, and test files
    root = Path("experiments") / "Biogen" / tool / subset
    for tech in techniques:
        for run in range(RUNS):
            for seed in [0, 1, 42, 1234, 1337]:
                print(tech, "-", run)
                try:
                    path = root / tech / f"split_{run}"

                    # train the D-MPNN model
                    targets = list(pd.read_csv(path / "train.csv").columns)
                    targets.remove("SMILES")
                    targets.remove("ID")
                    arguments = [
                        "--data_path", str(path / "train.csv"),
                        "--separate_val_path", str(path / "test.csv"),
                        "--separate_test_path", str(path / "test.csv"),
                        "--dataset_type", "regression",
                        "--save_dir", str(path / f"seed_{seed}"),
                        "--quiet",
                        "--epochs", str(MPP_EPOCHS),
                        "--smiles_columns", "SMILES",
                        "--target_columns", *targets,
                        "--metric", "mae",
                        "--seed", str(seed),
                    ]
                    args = chemprop.args.TrainArgs().parse_args(arguments)
                    chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
                    del targets
                    del arguments
                    del args

                    # extract the data and save them in a CSV file
                    tb_path = path / f"seed_{seed}" / "fold_0" / "model_0"
                    tb_file = tb_path / list(sorted(filter(lambda x: x.startswith("events"), os.listdir(tb_path))))[-1]
                    ea = EventAccumulator(str(tb_file))
                    ea.Reload()
                    for long, short in [("validation_", "val"), ("test_", "test")]:
                        for metric in filter(lambda x: x.startswith(long), ea.Tags()["scalars"]):
                            dfs[short][f"{tech}_{run}_{seed}"] = [x.value for x in ea.Scalars(metric)]

                except Exception as e:
                    print("=" * 80 + f"\n{e}\n" + "=" * 80)
    for split, df in dfs.items():
        save_path = root / f"{split}_metrics.tsv"
        print("Saving:", df.shape, "to", save_path)
        df.to_csv(save_path, index=False)


train("datasail", "HLM", ["I1e", "C1e"])
