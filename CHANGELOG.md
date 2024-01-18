# Change Log

## [Planned - Long-term project ideas]

- [ ] Normalization of objectives for better comparability and better splits in solvers
- [ ] Time-limit and solution limit for all solvers
- [ ] Multi-threading support for pre-solving (Snakemake as backbone)
- [ ] Make (more) deterministic ([Issue #6](https://github.com/kalininalab/DataSAIL/issues/6))
- [ ] Reports of results with plots and tables a PDF and or HTML
- [ ] Generalization to R-dimensional datasets (see [paper](https://doi.org/10.1101/2023.11.15.566305))
- [ ] Input from config files
- [X] Stratified splits
- [ ] Replace GraKel with something "modern" and fully "conda-installable" to make DataSAIL fully conda-installable

## v0.3.0 (2024-01-??)

- Stratified splits
- Extensive checks of available solvers
- Time and Space limits for all solvers
- Runtime experiments and experiments on a [stratified dataset](LINK)
- Bugs and Docu fixed

## v0.2.2 (2023-12-11)

- DataSAIL citation
- Support for MMseqs2 to compute similarity matrices
- More tests
- Bugs fixed
- Added CHANGELOG
- Switched from string-based paths to pathlib

## v0.2.1.beta (2023-11-06)

- Support for Windows and OSX-ARM (fixing [Issue #7](https://github.com/kalininalab/DataSAIL/issues/7))
- Support for Python 3.12
- Updated documentation

## v0.2.1 (2023-10-26)

- Renaming of splitting techniques to align with preprint to be I1/C1/I2/C2
- More tests to better cover the supposed functionality
- Now supports for Python 3.8 to Python 3.11
- Experiments on [MoleculeNet](https://doi.org/10.1039/C7SC02664A) and [LP-PDBBind](https://doi.org/10.48550/arXiv.2308.09639)
- Bugs fixed

## v0.2.0 (2023-09-27)

- Linear problem formulations
- Addition of more solvers due to updated problem formulations
- Bugs fixed

## v0.1.0 (2023-09-22)

- Support for Python 3.7
- Pandas in the backend
- Bug fixes

## v0.0.X (until 2023-08-15)

Initial development of DataSAIL and frequent bug removal and feature extension. It operated on Linux and OSX only and 
had quadratic problem specifications. 
