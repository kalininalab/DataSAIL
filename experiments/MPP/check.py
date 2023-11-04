from pathlib import Path
import os

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# extract the data and save them in a CSV file
tb_path = Path("experiments") / "MPP" / "lohi" / "cdata" / "freesolv" / "lohi" / f"split_0" / "fold_0" / \
          "model_0"
tb_file = tb_path / list(sorted(filter(
    lambda x: str(x).startswith("events"), os.listdir(tb_path)
)))[-1]
print("File:", tb_file)
ea = EventAccumulator(str(tb_file))
ea.Reload()
for long, short in [("validation_", "val"), ("test_", "test")]:
    print([m for m in filter(lambda x: x.startswith(long), ea.Tags()["scalars"])])
    for metric in filter(lambda x: x.startswith(long), ea.Tags()["scalars"]):
        print("metric", [e.value for e in ea.Scalars(metric)])
        # dfs[short][f"{tech}_{metric}_split_{run}"] = [e.value for e in ea.Scalars(metric)]
# print(df)
