import os
import shutil

import chemprop
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from experiments.utils import mpp_datasets, RUNS, MPP_EPOCHS, telegram


count = 0


def train(model, name):
    dfs = {"val": pd.DataFrame({"rows": list(range(50))}), "test": pd.DataFrame({"rows": [0]})}
    # store the results in training, validation, and test files
    for tech in [x for x in os.listdir(f"experiments/MPP/{model}/cdata/{name}") if os.path.isdir(f"experiments/MPP/{model}/cdata/{name}/{x}")]:
        for run in range(RUNS):
            print(tech, "-", run)
            try:
                print("Check folder:", f"experiments/MPP/{model}/cdata/{name}/{tech}/split_{run}/fold_0/", end="\t")
                print(os.path.exists("experiments/MPP/{model}/cdata/{name}/{tech}/split_{run}/fold_0/"))
                if os.path.exists("experiments/MPP/{model}/cdata/{name}/{tech}/split_{run}/fold_0/"):
                    print("Delete folder")
                    shutil.rmtree("experiments/MPP/{model}/cdata/{name}/{tech}/split_{run}/fold_0/", ignore_errors=True)

                train_df = pd.read_csv(f"experiments/MPP/{model}/cdata/{name}/{tech}/split_{run}/train.csv")
                test_df = pd.read_csv(f"experiments/MPP/{model}/cdata/{name}/{tech}/split_{run}/test.csv")
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
                train_df.to_csv(f"experiments/MPP/{model}/cdata/{name}/{tech}/split_{run}/train.csv", index=False)
                test_df.to_csv(f"experiments/MPP/{model}/cdata/{name}/{tech}/split_{run}/test.csv", index=False)
                
                # train the D-MPNN model
                targets = list(pd.read_csv(f"experiments/MPP/{model}/cdata/{name}/{tech}/split_{run}/train.csv").columns)
                targets.remove("SMILES")
                targets.remove("ID")
                arguments = [
                    "--data_path", f"experiments/MPP/{model}/cdata/{name}/{tech}/split_{run}/train.csv",
                    "--separate_val_path", f"experiments/MPP/{model}/cdata/{name}/{tech}/split_{run}/test.csv",
                    "--separate_test_path", f"experiments/MPP/{model}/cdata/{name}/{tech}/split_{run}/test.csv",
                    "--dataset_type", mpp_datasets[name][1],
                    "--save_dir", f"experiments/MPP/{model}/cdata/{name}/{tech}/split_{run}/",
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
                tb_file = f"experiments/MPP/{model}/cdata/{name}/{tech}/split_{run}/fold_0/model_0/" + list(sorted(filter(
                    lambda x: x.startswith("events"), os.listdir(f"experiments/MPP/{model}/cdata/{name}/{tech}/split_{run}/fold_0/model_0/")
                )))[-1]
                print("File:", tb_file)
                ea = EventAccumulator(tb_file)
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
                telegram(f"[MPP {count} / 105] Training finished for MPP - {model} - {name} - {tech} - Run {run}/4")
            except Exception as e:
                print(e)
    for split, df in dfs.items():
        print("Saving:", df.shape, "to", f"experiments/MPP/{model}/cdata/{name}/{split}_metrics.tsv")
        df.to_csv(f"experiments/MPP/{model}/cdata/{name}/{split}_metrics.tsv", sep="\t", index=False)


# for dataset in ["freesolv", "esol", "sider", "clintox", "bace", "bbbp", "lipophilicity", "qm7", "tox21", "toxcast", "qm8", "hiv", "muv", "qm9"]:
for dataset in ["qm7", "qm8", "qm9"]:
    # for dataset in ["tox21", "toxcast", "hiv", "muv"]:
    for tool in ["datasail", "deepchem"]:
        print(dataset, "-", tool)
        train(tool, dataset)

