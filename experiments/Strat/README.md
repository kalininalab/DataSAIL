# MPP Experiments

-------------

Here, we explore how to rerun the experiments for the Molecular Property Prediction (MPP) benchmark. The experiments are 
conducted on the most of the datasets contained in MoleculeNet, except for the QM7b, PCBA, and PDBBind datasets.

## Splitting the data

To split the data for the experiments, run the following command from the main directory of the repository:

```bash
python -m experiments.Strat.split <path/to/save/Strat>
```

This will save the data in the specified directory in the following format:

```
path/to/save/Strat
    ├── datasail/
    │    ├── d_0.3_e_0.3/
    │    │    ├── split_0/
    │    │    │    ├── train.csv
    │    │    │    └── test.csv
    │    │    ├── split_1/
    │    │    ├── split_2/
    │    │    ├── split_3/
    │    │    └── split_4/
    │    ├── d_3.1_e_0.25/
    │    ├── ...
    │    └── d_0.05_e_0.05/
    └── deepchem/
         ├── split_0/
         ├── split_1/
         ├── split_2/
         ├── split_3/
         └── split_4/
```

Here, we not only create splits for the stratification experiment but also for the ablation study on the influence of 
different values of delta and epsilon on the quality of splits. Therefore, deepchem lists all splits in the known 
format and in datasail, we have 30 folders with different splits for different combinations of delta and epsilon.

Each of the `.csv` files has the following structure:

```csv
ID, SMILES, SR-ARE
Comp000001, CCOc1ccc2nc(S(N)(=O)=O)sc2c1, 1
Comp000001, CCN1C(=O)NC(c2ccccc2)C1=O, 0
...
```

where `target_i` is the name of the `i`-th target in the dataset.

-------------

## Training the model

To train the model, run the following command from the main directory of the repository:

```bash
python -m experiments.Strat.train <path/to/save/Strat>
```

This adds `results.csv` files to all dataset folders.

```
path/to/save/Strat/
    ├── data/
    ├── datasail/
    ├── deepchem/
    └── results.csv
```

with `results.csv` having the following structure

```csv
name, perf, model, tool, run
...
```

where `name` is `<tech>_<run>`, `perf` is the AUROC, `model` is the model name (one of `rf`, `svm`, `xgb`, `mlp`, 
`deepdta`), `tool` is the tool used (`datasail` or `deepchem`), and `run` is the run number.

The structure of sub-folders as introduced by `split` is maintained but for simplicity omitted.

-------------

## Visualizing the results

To visualize the results from the paper, run the following command from the main directory of the repository:

```bash
python -m experiments.Strat.visualize <path/to/save/Strat>
```

This will save the visualizations in the specified directory in the following format:

```
path/to/save/Strat/
    ├── data/
    ├── datasail/
    ├── deepchem/
    ├── graphpart/
    ├── lohi/
    └── Strat.png
```
