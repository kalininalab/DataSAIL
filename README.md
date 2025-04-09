# DataSAIL: Data Splitting Against Information Leaking 

![testing](https://github.com/kalininalab/glyles/actions/workflows/test.yaml/badge.svg)
[![docs-image](https://readthedocs.org/projects/glyles/badge/?version=latest)](https://datasail.readthedocs.io/en/latest/index.html)
[![codecov](https://codecov.io/gh/kalininalab/DataSAIL/branch/main/graph/badge.svg)](https://codecov.io/gh/kalininalab/DataSAIL)
[![anaconda](https://anaconda.org/kalininalab/datasail/badges/version.svg)](https://anaconda.org/kalininalab/datasail)
[![update](https://anaconda.org/kalininalab/datasail/badges/latest_release_date.svg)](https://anaconda.org/kalininalab/datasail)
[![license](https://anaconda.org/kalininalab/datasail/badges/license.svg)](https://anaconda.org/kalininalab/datasail)
[![downloads](https://anaconda.org/kalininalab/datasail/badges/downloads.svg)](https://anaconda.org/kalininalab/datasail)
![Python 3](https://img.shields.io/badge/python-3-blue.svg)
[![DOI](https://zenodo.org/badge/598109632.svg)](https://doi.org/10.5281/zenodo.13938602)

DataSAIL: [![platforms](https://anaconda.org/kalininalab/datasail/badges/platforms.svg)](https://anaconda.org/kalininalab/datasail)
DataSAIL-lite: [![platforms](https://anaconda.org/kalininalab/datasail-lite/badges/platforms.svg)](https://anaconda.org/kalininalab/datasail-lite)

DataSAIL is a tool that splits data while minimizing Information Leakage. This tool formulates the splitting of a 
dataset as a constrained minimization problem and computes the assignment of data points to splits while minimizing the 
objective function that accounts for information leakage.

Internally, DataSAIL uses disciplined quasi-convex programming and binary quadratic programs to formulate the 
optimization task. DataSAIL utilizes solves like [SCIP](https://scipopt.org/), one of the fastest non-commercial 
solvers for this type of problem, and [MOSEK](https://mosek.com), a commercial solver that distributes free licenses 
for academic use. There are other options; please check the documentation.

Apart from the short overview, you can find a more detailed explanation of the tool on 
[ReadTheDocs](https://datasail.readthedocs.io/en/latest/index.html). 

## Installation

DataSAIL is installable from [conda](https://anaconda.org/kalininalab/datasail) using
[mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>).
using

````shell
mamba create -n sail -c conda-forge -c kalininalab -c bioconda datasail
conda activate sail
pip install grakel
````

to install it into a new empty environment or

````shell
mamba install -c conda-forge -c kalininalab -c bioconda -c mosek datasail
pip install grakel
````

to install DataSAIL in an already existing environment. Alternatively, one can install DataSAIL-lite from conda. 
DataSAIL-lite is a version of DataSAIL that does not install all clustering algorithms as the standard DataSAIL.
Installing either package usually takes less than 5 minutes.

DataSAIL is available for Python 3.9 to Python 3.12.

## Usage

DataSAIL is installed as a command-line tool. So, in the conda environment, DataSAIL has been installed to, you can run 

````shell
datasail --e-type P --e-data <path_to_fasta> --e-sim mmseqs --output <path_to_output_path> --technique C1e
````

to split a set of proteins that have been clustered using mmseqs. For a full list of arguments, run `datasail -h` and 
checkout [ReadTheDocs](https://datasail.readthedocs.io/en/latest/index.html). There is a more detailed explanation of the arguments and example notebooks. The runtime largy depends on the number and type of splits to be computed and the size of the dataset. For small datasets (less then 10k samples) DataSAIL finished within minutes. On large datasets (more than 100k samples) it can take several hours to complete.

## When to use DataSAIL and when not to use

One can distinguish two main ways to train a machine-learning model on biological data. 
* Either the model shall be applied to data substantially different from the data to train on. In this case, it 
  is essential to have test cases that correctly model this real-world application scenario by being as dissimilar as 
  possible to the training data. 
* Or the training dataset already covers the whole space of possible samples shown to the model.

DataSAIL is created to compute complex data splits by separating data based on similarities. This makes 
complex data splits for the first scenario. So, you can use DataSAIL when your model is applied to data  
different from your training data but not if the data in the application is more or less the same as in the training.

## Citation

If you used DataSAIL to split your data, please cite DataSAIL in your publication.
````
@article{joeres2025datasail,
  title={Data splitting to avoid information leakage with DataSAIL},
  author={Joeres, Roman and Blumenthal, David B. and Kalinina, Olga V},
  journal={Nature Communications},
  volume={16},
  pages={3337},
  year={2025},
  doi={10.1038/s41467-025-58606-8},
}
````
