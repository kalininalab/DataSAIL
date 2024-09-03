# Experiments

-------------

For the publication, we have conducted several experiments:

 1. Splitting of data for drug-target interaction data,
 2. Splitting of data for Molecular Property Prediction,
 3. Splitting of data with samples belonging to either of two classes for stratified splits, 

and some ablation studies based on above's data. The experiments cover all possible applications of DataSAIL. Each 
experiments-folder is structured in the same way:

 1. `split.py`: Contains the code used for splitting using DataSAIL or baselines tools.
 2. `train.py`: Contains the code to train the different models on the split data.
 3. `visualize.py`: Contains the code to visualize the results of the training.

All can be executed in the same way:
    
```shell
python -m experiments.<experiment>.<script> <path/to/storage-folder>
```

where `<experiment>` is the name of the experiment type (`DTI`, `MPP`, or `Strat`) and `<script>` is the name of the 
script (`split`, `train`, or `visualize`). Lastly <path/to/storage-folder> is the path to a folder where the results 
from the previous step can be found and new results shall be stored. Because the scripts rely on the results from the 
previous step, it is necessary to run them in order. For example, to run the entire DTI experiment pipeline, you need 
to run:

```shell
python -m experiments.DTI.split scratch/DataSAIL_results/DTI
python -m experiments.DTI.train scratch/DataSAIL_results/DTI
python -m experiments.DTI.visualize scratch/DataSAIL_results/DTI
```

where the path can be exchanged with any other path.
