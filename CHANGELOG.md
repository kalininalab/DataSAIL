# Change Log

## [Planned - Long-term project ideas]

- [ ] Multi-threading support for pre-solving (Snakemake as backbone)
- [ ] Make (more) deterministic ([Issue #6](https://github.com/kalininalab/DataSAIL/issues/6))
- [ ] Reports of results with plots and tables a PDF and or HTML
- [ ] Generalization to R-dimensional datasets (see [paper](https://doi.org/10.1101/2023.11.15.566305))
- [ ] Input from config files
- [ ] Replace GraKel with something "modern" and fully "conda-installable" to make DataSAIL fully conda-installable
- [ ] Include [MashMap3](https://github.com/marbl/MashMap)
- [ ] Include MASH for amino acid sequences
- [ ] Custom clustering methods ([Issue #25](https://github.com/kalininalab/DataSAIL/issues/25))

## v1.2.2 (2025-10-14)

- Bug fixed in the evaluation module.

## v1.2.1 (2025-08-19)

- Improved stratification and testing thereof to better handle mutliclass and multilabel-multiclass stratification

## v1.2.0 (2025-07-20)

- New Features!!!
  - DataSAIL can now handle clusters that are too big for one split. The new parameter `overflow` takes `break` or `assign` as arguments.
  - It is now possible to quantify the similarity-induced leakage with DataSAIL in the `datasail.eval` module.
- Parallel publication of PyPI and conda packages
- Major update of documentation

## v1.1.2 and 1.1.3 (2025-06-06) [PyPI only]

- Publication of DataSAIL-lite as `datasail` on the Python Package Index

## v1.1.1 (2025-05-03)

- Bug fix in C2 splitting

## v1.1.0 (2025-0?-??)

- Support terminated for Python v3.8. Now, DataSAIL supports Python v3.9 to v3.12.
- Major rework of two-dimensional splitting
- Bug fixes

## v1.0.1 (2024-05-08) till v1.0.7 (2024-06-27)

- Bug fixes in stratification

## v1.0.0 (2024-04-04)

- Stratified splits
- Extensive checks of available solvers
- Time and Space limits for all solvers
- Runtime experiments and experiments on a the Tox21 SR-ARE target
- Improvement and extension of the documentation
- Bug fixes

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
