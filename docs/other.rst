Other Initiatives
=================

In recent years, many datasets have been published with data splits that put special focus on minimizing similarity-induced information leakage between the 
splits. These splits and the underlying algorithms are often very specific to the dataset. 

Here, we compare the similarity-induced information leakage in these splits to DataSAIL splits. We measure the leakage of a split by the scaled :math:`L(\pi)` 
metric as defined in the main manuscript:

.. math::

    \text{scaled L}(\pi):=\frac{\sum_{xx^\prime\in\binom{\mathcal{D}}{2}}[\pi(x)\neq\pi(x^\prime)]\cdot\text{sim}(x,x^\prime)\cdot\kappa(x)\cdot\kappa(x^\prime)}{\sum\nolimits_{x,x^\prime\in\mathcal{D}}\text{sim}(x,x^\prime)}

Here :math:`\pi:\mathcal{D}\rightarrow [k]` is a data splitting function mapping samples :math:`x` of the dataset :math:`\mathcal{D}` to one of :math:`k` splits. 
:math:`\text{sim}:\mathcal{D}\times\mathcal{D}\rightarrow [0,1]` is a similarity function between samples :math:`x` and :math:`x^\prime` of :math:`\mathcal{D}`. 
:math:`\kappa:\mathcal{D}\rightarrow\mathbb{R}_{\geq 0}` is a weighting function that can be used to put more emphasis on certain samples. 
This is especially useful if :math:`x` represents clusters or has multiple interactions in a drug-target interaction dataset and potentially leaks information multiple times.

MoleculeNet
-----------
| Zu et al. (2018)
| DOI: `10.1039/C7SC02664A <https://doi.org/10.1039/C7SC02664A>`_

This benchmark suite provides multiple datasets for molecular property prediction with different properties to predict. Each dataset 
contains a predefined split, some of which are scaffold-based or time-based, but most are random. Here, we compare these default split to similarity-based DataSAIL splits.

*comparison coming soon*

Leak Proof PDBBind (LP-PDBBind)
----------------------------------
| Li et al. (2023)
| DOI: `10.48550/arXiv.2308.09639 <https://doi.org/10.48550/arXiv.2308.09639>`_

This work improves the PDBBind dataset by defining a new datasplit that reduces data leakage between train, validation and test sets.
The resulting LP-PDBBind dataset ensures that the train set has a maximum sequence similarity of 0.5 and maximum ligand similarity of 
0.99 to both validation and test sets. Between the validation and test set, those guarantees are 0.9 for protein similarity and 0.99 
for ligand similarity. Protein similarity was measured as the percentage of matching residues after a Needleman-Wunsch alignment, 
while ligand similarity was measured as the Dice similarity between Morgan fingerprints.

.. raw:: html
    :file: tables/lppdbbind.html

|

Gold Standard Human Proteome Dataset for sequence-based PPI prediction
----------------------------------------------------------------------
| Bernett et al. (2023)
| DOI: `10.1093/bib/bbae076 <https://doi.org/10.1093/bib/bbae076>`_

The authors first show that all sequence-based protein-protein interaction (PPI) predictors they evaluated perform no better than random when sequence similarity 
between splits is removed. They further develop a PPI dataset based on the human proteome where they separate the proteins into three blocks 
using KaHIP over SIMAP2 bitscores. Then, the PPIs are assigned to the blocks if and only if the interacting proteins are both in the corresponding block. In 
a last step, CDHIT is used to remove redundancy (max 40% sequence similarity) within each block.

*comparison coming soon*

Protein Ligand INteraction Dataset and Evaluation Resource (PLINDER)
--------------------------------------------------------------------
| Durairaj et al. (2024)
| DOI: `10.1101/2024.07.17.603955 <https://doi.org/10.1101/2024.07.17.603955>`_

This work introduces PLINDER, a dataset for protein-ligand interaction prediction, extracted from the PDB and intesively annotated.
The authors provide three different data splits. The most complex one is PLINDER-PL50, which was created by combining mutliple similarity metrics: 
(i) sequence identity for proteins, (ii) pocket-level Jaccard similarity using pharmacophores, (iii) interaction-level similarity using PLIP features, 
and (iii) ligand-level similarity using Tanimoto similarity on ECFP4 fingerprints. The algorithm then identifies clusters of similar protein-ligand systems. 
Finally, the test set is constructed to contain systems from clusters that have no or minimal similarity to any systems in the training or validation sets.
Along this, there are two simpler splits: PLINDER-TIME, which is a time-based split, and PLINDER-ECOD, which is based on ECOD topologies.

.. raw:: html
    :file: tables/plinder.html

|

Protein INteraction Dataset and Evaluation Resource (PINDER)
------------------------------------------------------------
| Kovtun et al. (2024)
| DOI: `10.1101/2024.07.17.603980 <https://doi.org/10.1101/2024.07.17.603980>`_

*coming soon*
