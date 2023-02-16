# DataSAIL: Data Splitting Against Information Leaking 

![testing](https://github.com/kalininalab/glyles/actions/workflows/test.yaml/badge.svg)
[![codecov](https://codecov.io/gh/kalininalab/DataSAIL/branch/main/graph/badge.svg)](https://codecov.io/gh/kalininalab/DataSAIL)

DataSAIL is a tool that splits data while minimizing the information leakage. This tool formulates the splitting of a 
dataset as constrained minimization problem and computes the assignment of data points to splits while minimizing the 
objective function that accounts for information leakage.

Internally, DataSAIL uses disciplined quasi-convex programming and binary quadratic programs to formulate the 
optimization task. To solve this DataSAIL relies on [SCIP](https://scipopt.org/), one of the fastest non-commercial 
solvers for this type of problems.

Apart from the here presented short overview, you can find a more detailed explanation of the tool on 
[ReadTheDocs](https://datasail.readthedocs.io/en/stable/index.html). 

## Installation

**This is currently not possible as DataSAIL is not uploaded to conda yet.**

DataSAIL is installable from `conda` (`mamba` works equivalently) using

````shell
conda create -n sail -c kalininalab python=3.8 datasail
conda activate sail
````

to install it into a new empty environment or

````shell
conda install -c kalininalab datasail
````

to install DataSAIL in an already existing environment. Due to dependencies of the clustering algorithms, the latter 
might lead to package conflicts with the already installed packages and requirements.

## Usage

DataSAIL is installed as a commandline tool. So, in the conda environment DataSAIL has been installed to, you can run 

````shell
sail --e-type P --e-data <path_to_fasta> --e-sim mmseqs --output <path_to_output_path> --technique CCS
````

to split a set of proteins that have been clustered using mmseqs. For a full list of arguments run `sail -h` and 
checkout [ReadTheDocs](https://datasail.readthedocs.io/en/stable/index.html).

## Splitting techniques

DataSAIL allows to split multiple types of data in different ways minimizing various sources of data leakage.

### `--technique R`: Random split

Randomly splitting a list of interactions into different splits. This option is for completeness. It does not account 
for any type of data leakage.

### `--technique ICS`: Identity-based single-cold split

Split a dataset based on the IDs of the datapoints. This ensures that every datapoint is present in exactly one split. 
This is especially useful together with weighting of the datapoints. Then, the splits are optimized towards the 
requested sizes taking the weights into account.

### `--technique ICD`: Identity-based double-cold split

Split a dataset of pairwise interactions based on the IDs of both sets of datapoints. This ensures any 
datapoints of neither of the two interacting datasets is present in more than one split.

### `--technique CCS`: Cluster-based single-cold split

Split a dataset based on clusters of datapoints. This further reduces the information leakage between two splits but 
not only ensuring that the same protein is not present in more than one split but also to ensure that no two datapoints 
with pairwise similarity above a certain threshold are present in the same split. For example, consider a dataset 
containing proteins from different families, all proteins from one family are similar in their sequence and structure. 
Therefore, they should be in the same split to prevent information leakage.

### `--technique CCD`: Cluster-based double-cold split

Split a dataset of pairwise interactions based on clusters in both sets of datapoints. This is the combination of ICD 
and CCS combining both advantages and features.