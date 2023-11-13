import os
import shutil
from pathlib import Path

import chemprop
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from experiments.utils import mpp_datasets, RUNS, MPP_EPOCHS, telegram

count = 0


def train(model, name):
    dfs = {"val": pd.DataFrame({"rows": list(range(50))}), "test": pd.DataFrame({"rows": [0]})}
    # store the results in training, validation, and test files
    cpath = Path("experiments") / "MPP" / model / "cdata" / name
    for tech in ["I1e"]:  # x for x in os.listdir(cpath) if os.path.isdir(cpath / x)]:
        for run in range(RUNS):
            print(tech, "-", run)
            try:
                path = cpath / tech / f"split_{run}"
                train_df = pd.read_csv(path / "train.csv")
                test_df = pd.read_csv(path / "test.csv")
                train_nunique = train_df.nunique()
                test_nunique = test_df.nunique()
                train_dropable = train_nunique[train_nunique == 1].index
                test_dropable = test_nunique[test_nunique == 1].index
                print(train_dropable)
                print(test_dropable)
                train_df.drop(train_dropable, axis=1, inplace=True)
                test_df.drop(train_dropable, axis=1, inplace=True)
                train_df.drop(test_dropable, axis=1, inplace=True)
                test_df.drop(test_dropable, axis=1, inplace=True)
                train_df.to_csv(path / "train.csv", index=False)
                test_df.to_csv(path / "test.csv", index=False)

                # train the D-MPNN model
                targets = list(pd.read_csv(path / "train.csv").columns)
                targets.remove("SMILES")
                targets.remove("ID")
                arguments = [
                    "--data_path", str(path / "train.csv"),
                    "--separate_val_path", str(path / "test.csv"),
                    "--separate_test_path", str(path / "test.csv"),
                    "--dataset_type", mpp_datasets[name][1],
                    "--save_dir", str(path),
                    "--quiet", "--epochs", str(MPP_EPOCHS),
                    "--smiles_columns", "SMILES",
                    "--target_columns", *targets,
                    "--metric", mpp_datasets[name][2],
                ]
                args = chemprop.args.TrainArgs().parse_args(arguments)
                chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
                del targets
                del arguments
                del args

                # extract the data and save them in a CSV file
                tb_path = Path("experiments") / "MPP" / model / "cdata" / name / tech / f"split_{run}" / "fold_0" / \
                    "model_0"
                tb_file = tb_path / list(sorted(filter(
                    lambda x: x.startswith("events"), os.listdir(tb_path)
                )))[-1]
                print("File:", tb_file)
                ea = EventAccumulator(str(tb_file))
                ea.Reload()
                for long, short in [("validation_", "val"), ("test_", "test")]:
                    print([m for m in filter(lambda x: x.startswith(long), ea.Tags()["scalars"])])
                    for metric in filter(lambda x: x.startswith(long), ea.Tags()["scalars"]):
                        print("metric", [e.value for e in ea.Scalars(metric)])
                        dfs[short][f"{tech}_{metric}_split_{run}"] = [e.value for e in ea.Scalars(metric)]
                del tb_file
                del ea

                global count
                count += 1
                telegram(f"[MPP {count} / 10] Training finished for MPP - lohi - {name} - Run {run + 1} / 5")
            except Exception as e:
                print(e)
    for split, df in dfs.items():
        save_path = Path("experiments") / "MPP" / model / "cdata" / name / f"new_{split}_metrics.tsv"
        print("Saving:", df.shape, "to", save_path)
        df.to_csv(save_path, sep="\t", index=False)


# for dataset in sorted(list(mpp_datasets.keys()), key=lambda x: mpp_datasets[x][3]):
#     if dataset in {"qm9", "muv", "bace"}:
#         continue
#     print(dataset, "-", "lohi")
#     train("lohi", dataset)
train("datasail", "muv")
train("datasail", "qm9")
