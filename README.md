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

DataSAIL is installable from [conda](https://anaconda.org/kalininalab/datasail) using
[mamba](https://mamba.readthedocs.io/en/latest/installation.html#existing-conda-install>).
using

````shell
conda create -n sail -c conda-forge -c kalininalab -c bioconda -c mosek DataSAIL
conda activate sail
pip install grakel
````

to install it into a new empty environment or

````shell
conda install -c conda-forge -c kalininalab -c bioconda -c mosek DataSAIL
pip install grakel
````

to install DataSAIL in an already existing environment. Due to dependencies of the clustering algorithms, the latter 
might lead to package conflicts with the already installed packages and requirements.

DataSAIL is available from Python 3.8 and newer.

## Usage

DataSAIL is installed as a commandline tool. So, in the conda environment DataSAIL has been installed to, you can run 

````shell
sail --e-type P --e-data <path_to_fasta> --e-sim mmseqs --output <path_to_output_path> --technique C1e
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
