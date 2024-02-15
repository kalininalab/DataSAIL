# DTI Experiments

-------------

Here, we explore how to rerun the experiments for the Drug-Target Interaction (DTI) prediction. The experiments are 
conducted on the LP-PDBBind dataset. To run the experiments, make sure, you cloned the LP-PDBBind repo into the 
`datasail/experiments/lppdbind` directory.

## Splitting the data

To split the data for the experiments, run the following command from the main directory of the repository:

```bash
python -m experiments.DTI.split <path/to/save/DTI>
```

This will save the data in the specified directory in the following format:

```
path/to/save/DTI
    ├── datasail/
    │    ├── C1e/
    │    │    ├── split_0/
    │    │    │    ├── train.csv
    │    │    │    └── test.csv
    │    │    ├── split_1/
    │    │    ├── split_2/
    │    │    ├── split_3/
    │    │    └── split_4/
    │    ├── C1f/
    │    ├── C2/
    │    ├── I1f/
    │    ├── I1e/
    │    ├── I2/
    │    └── R/
    └── deepchem/
    │    ├── Butina/
    │    ├── Fingerprint/
    │    ├── MaxMin/
    │    ├── Scaffold/
    │    └── Weight/
    ├── graphpart/
    │    └── graphpart/
    └── lohi/
         └── lohi/
```

As visualized for `C1e` all techniques have folders for each split, and each split has a `train.csv` and `test.csv` 
(as denoted for `C1e/split_0`).

Each of the `.csv` files has the following structure:

```csv
ids, smiles, seq, value
6r8o, CSc1ccccc1..., GNPLVYLDVD..., 8.22
...
```

-------------

## Training the model

To train the model, run the following command from the main directory of the repository:

```bash
python -m experiments.DTI.train <path/to/save/DTI>
```

Format of output:

```
path/to/save/DTI/
    ├── data/
    ├── datasail/
    ├── datasail.csv
    ├── deepchem/
    ├── deepchem.csv
    ├── graphpart/
    ├── graphpart.csv
    ├── lohi/
    └── lohi.csv
```

with each CSV having the following structure
```csv
name, perf, model, tool, tech, run
...
```
where `name` is `<tech>_<run>`, `perf` is the RMSE, `model` is the model name (one of `rf`, `svm`, `xgb`, `mlp`, 
`deepdta`), `tool` is the tool used (one of `datasail`, `deepchem`, `lohi`), `tech` is the technique used (one of `R`, 
`I1e`, `I1f`, `I2`, `C1e`, `C1f`, `C2`, `Butina`, `Fingerprint`, `MaxMin`, `Scaffold`, `Weight`, and `lohi` (depending 
on what's available for the tool)), and `run` is the run number. Lastly, `dataset`

The structure of sub-folders as introduced by `split` is maintained but for simplicity omitted.

-------------

## Visualizing the results

To visualize the results from the paper, run the following command from the main directory of the repository:

```bash
python -m experiments.DTI.visualize <path/to/save/DTI>
```

This will save the visualizations in the specified directory in the following format:

```
path/to/save/DTI/
    ├── data/
    ├── datasail/
    ├── datasail.csv
    ├── deepchem/
    ├── deepchem.csv
    ├── graphpart/
    ├── graphpart.csv
    ├── lohi/
    ├── lohi.csv
    └── plots/
        ├── PDBBind_tsne_3x3.csv
        ├── PDBBind_CD_tsne.csv
        └── PDBBind_CT_tsne.csv
```