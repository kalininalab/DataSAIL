# DataSAIL: Data Splitting Against Information Leaking 

![testing](https://github.com/kalininalab/glyles/actions/workflows/test.yaml/badge.svg)
[![docs-image](https://readthedocs.org/projects/glyles/badge/?version=latest)](https://datasail.readthedocs.io/en/latest/index.html)
[![codecov](https://codecov.io/gh/kalininalab/DataSAIL/branch/main/graph/badge.svg)](https://codecov.io/gh/kalininalab/DataSAIL)
[![anaconda](https://anaconda.org/kalininalab/datasail/badges/version.svg)](https://anaconda.org/kalininalab/datasail)
[![update](https://anaconda.org/kalininalab/datasail/badges/latest_release_date.svg)](https://anaconda.org/kalininalab/datasail)
[![platforms](https://anaconda.org/kalininalab/datasail/badges/platforms.svg)](https://anaconda.org/kalininalab/datasail)
[![license](https://anaconda.org/kalininalab/datasail/badges/license.svg)](https://anaconda.org/kalininalab/datasail)
[![downloads](https://anaconda.org/kalininalab/datasail/badges/downloads.svg)](https://anaconda.org/kalininalab/datasail)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

DataSAIL is a tool that splits data while minimizing the information leakage. This tool formulates the splitting of a 
dataset as constrained minimization problem and computes the assignment of data points to splits while minimizing the 
objective function that accounts for information leakage.

Internally, DataSAIL uses disciplined quasi-convex programming and binary quadratic programs to formulate the 
optimization task. To solve this DataSAIL relies on [SCIP](https://scipopt.org/), one of the fastest non-commercial 
solvers for this type of problems and [MOSEK](https://mosek.com), a commercial solver that distributes free licenses 
for academic use.

Apart from the here presented short overview, you can find a more detailed explanation of the tool on 
[ReadTheDocs](https://datasail.readthedocs.io/en/latest/index.html). 

## Installation

**This is currently not possible as DataSAIL is not uploaded to conda yet.**

DataSAIL is installable from `conda` (`mamba` works equivalently) ([Link](https://anaconda.org/kalininalab/datasail)) 
using

````shell
conda create -n sail -c conda-forge -c kalininalab -c mosek -c bioconda datasail
conda activate sail
pip install grakel
````

to install it into a new empty environment or

````shell
conda install -c conda-forge -c kalininalab -c mosek -c bioconda datasail
pip install grakel
````

to install DataSAIL in an already existing environment. Due to dependencies of the clustering algorithms, the latter 
might lead to package conflicts with the already installed packages and requirements.

## Usage

DataSAIL is installed as a commandline tool. So, in the conda environment DataSAIL has been installed to, you can run 

````shell
sail --e-type P --e-data <path_to_fasta> --e-sim mmseqs --output <path_to_output_path> --technique CCS
````

to split a set of proteins that have been clustered using mmseqs. For a full list of arguments run `sail -h` and 
checkout [ReadTheDocs](https://datasail.readthedocs.io/en/latest/index.html).

## When to use DataSAIL and when not to use

One can distinguish two main ways to train a machine learning model on biological data. 
* Either the model shall be applied to data that is substantially different from the data to train on. In this case it 
  is important to have test cases that model this real world application scenario properly by being as dissimilar as 
  possible to the training data. 
* Or the training dataset already covers the full space of possible samples shown to the model.

DataSAIL is created to compute complex splits of the data by separating data based on similarities. This creates 
complex data-splits for the first scenario. Therefore, use DataSAIL when your model is applied to data that is 
different from your training data but not if the data in application is more or less the same as in the training.

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
