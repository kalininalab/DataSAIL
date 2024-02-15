# MPP Experiments

-------------

Here, we explore how to rerun the experiments for the Molecular Property Prediction (MPP) benchmark. The experiments are 
conducted on the most of the datasets contained in MoleculeNet, except for the QM7b, PCBA, and PDBBind datasets.

## Splitting the data

To split the data for the experiments, run the following command from the main directory of the repository:

```bash
python -m experiments.MPP.split <path/to/save/MPP> [<name>] [<solver>]
```

The second argument defines the name of the dataset to split. If omitted, the data will be split for all datasets. The 
third argument defines the solver to use for the splitting. If omitted, the DataSAIL will use GUROBI. Make sure to have 
a valid license in this case.

This will save the data in the specified directory in the following format:

```
path/to/save/MPP
    ├── datasail/
    │    ├── qm7/
    │    │    ├── C1e/
    │    │    │    ├── split_0/
    │    │    │    │    ├── train.csv
    │    │    │    │    └── test.csv
    │    │    │    ├── split_1/
    │    │    │    ├── split_2/
    │    │    │    ├── split_3/
    │    │    │    └── split_4/
    │    │    ├── I1e/
    │    │    └── time.csv
    │    ├── qm8/
    │    ├── ...
    │    └── clintox/
    ├── deepchem/
    │    ├── qm7/
    │    │    ├── Butina/
    │    │    ├── Fingerprint/
    │    │    ├── MaxMin/
    │    │    ├── Scaffold/
    │    │    ├── Weight/
    │    │    └── time.csv
    │    ├── qm8/
    │    ├── ...
    │    └── clintox/
    └── lohi/
         ├── qm7/
         │    ├── lohi/
         │    └── time.csv
         ├── qm8/
         ├── ...
         └── clintox/
```

As visualized, every tool has a sub-folder for every dataset we used from the MoleculeNet collection. Also, every 
dataset sub-folder has folders for all techniques and every technique folder has sub-folders for each split, and each 
split has a `train.csv` and `test.csv` (as denoted for `datasail/qm7/C1e/`). Furthermore, every dataset has a 
`time.csv` file, which contains the time it took to run the tool from calling it to receiving the data splits.

Each of the `.csv` files has the following structure:

```csv
ids, smiles, target_1, target_2, ...
6r8o, CSc1ccccc1..., 8.22, 0.271, ...
...
```

where `target_i` is the name of the `i`-th target in the dataset.

-------------

## Training the model

To train the model, run the following command from the main directory of the repository:

```bash
python -m experiments.MPP.train <path/to/save/MPP> [<name>]
```

The second argument defines the name of the dataset to train on. If omitted, the model will be trained on all datasets.

This adds `results.csv` files to all dataset folders.

```
path/to/save/MPP/
    ├── data/
    ├── datasail/
    │    ├── qm7/
    │    │    ├── C1e/
    │    │    ├── I1e/
    │    │    ├── results.csv
    │    │    └── time.txt
    │    ├── ...
    │    └── clintox/
    ├── deepchem/
    ├── graphpart/
    └── lohi/
```

with each `results.csv` having the following structure

```csv
name, perf, model, tool, tech, run, dataset
...
```

where `name` is `<tech>_<run>`, `perf` is the RMSE, `model` is the model name (one of `rf`, `svm`, `xgb`, `mlp`, 
`deepdta`), `tool` is the tool used (one of `datasail`, `deepchem`, `lohi`), `tech` is the technique used (one of `R`, 
`I1e`, `I1f`, `I2`, `C1e`, `C1f`, `C2`, `Butina`, `Fingerprint`, `MaxMin`, `Scaffold`, `Weight`, and `lohi` (depending 
on what's available for the tool)), and `run` is the run number. Lastly, `dataset` is the name of the dataset.

The structure of sub-folders as introduced by `split` is maintained but for simplicity omitted.

-------------

## Visualizing the results

To visualize the results from the paper, run the following command from the main directory of the repository:

```bash
python -m experiments.MPP.visualize <path/to/save/MPP>
```

This will save the visualizations in the specified directory in the following format:

```
path/to/save/MPP/
    ├── data/
    ├── datasail/
    ├── deepchem/
    ├── graphpart/
    ├── lohi/
    └── plots/
         ├── MoleculeNet_comp.png
         └── QM8_Tox21.png
```
