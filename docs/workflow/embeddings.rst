########################
Clustering of Embeddings
########################

.. _embeddings-label:

DataSAIL offers different clustering algorithms implemented in SciPy and RDKit to cluster the embeddings.
The clustering algorithms are:

.. list-table:: Title
    :widths: 30 15 15 15 15 15
    :header-rows: 1

    * - Algorithm
      - Sim or Dist
      - Boolean
      - Integer
      - Float
      - RDKit or SciPy
    * - AllBit
      - Sim
      - X
      - \-
      - \-
      - RDKit
    * - Asymmetric
      - Sim
      - X
      - \-
      - \-
      - RDKit
    * - Braun-Blanquet
      - Sim
      - X
      - \-
      - \-
      - RDKit
    * - Canberra
      - Dist
      - X
      - X
      - X
      - SciPy
    * - Dice
      - Sim
      - X
      - X
      - \-
      - RDKit
    * - Hamming
      - Dist
      - X
      - X
      - X
      - SciPy
    * - Kulczynski
      - Sim
      - X
      - \-
      - \-
      - RDKit
    * - Jaccard
      - Dist
      - X
      - \-
      - \-
      - SciPy
    * - Matching
      - Dist
      - X
      - X
      - X
      - SciPy
    * - OnBit
      - Sim
      - X
      - \-
      - \-
      - RDKit
    * - Rogers-Tanimoto
      - Dist
      - X
      - \-
      - \-
      - SciPy
    * - Rogot-Goldberg
      - Sim
      - X
      - \-
      - \-
      - RDKit
    * - Russel
      - Sim
      - X
      - \-
      - \-
      - RDKit
    * - Sokal
      - Sim
      - X
      - \-
      - \-
      - RDKit
    * - Sokal-Michener
      - Dist
      - X
      - \-
      - \-
      - SciPy
    * - Tanimoto
      - Sim
      - X
      - X
      - \-
      - RDKit
    * - Yule
      - Dist
      - X
      - \-
      - \-
      - SciPy

Individual Algorithms
#####################

In the following, we will describe the individual algorithms in more detail and with the mathematical formula that
computes the respective metric between two vectors :math:`u` and :math:`v` of length :math:`n`. Depending on the method
used, :math:`u` and :math:`v` can be float-vectors but may also be restricted to be int-vectors or bit-vectors.

.. note::
    We will use the `Iverson bracket <https://en.wikipedia.org/wiki/Iverson_bracket>`__ notation :math:`[P]` to
    denote the indicator function that is 1 if the predicate :math:`P` is true and 0 otherwise.

AllBit
======

This is the ratio of equal bits in the two bit vectors :math:`u` and :math:`v`.

.. math::

    \text{AllBit}(u, v) = \frac{\sum_{i=1}^{n} [u[i] = v[i]]}{n}

Asymmetric
==========

The Asymmetric similarity is the ratio of equal bits in the two bit vectors :math:`u` and :math:`v` divided by the
minimum number of bits set in either of the two vectors. The implementation is given in `RDKit <https://github.com/rdkit/rdkit/blob/722cbba894736bf3adbe792e7158fba26b5f8e6f/Code/DataStructs/BitOps.cpp#L520>`__.

.. math::

    & u_1 = \sum_{i=1}^{n} [u[i]]\\
    & v_1 = \sum_{i=1}^{n} [v[i]]\\
    & \text{Asymmetric}(u, v) = \begin{cases}
        0, &\text{if} \min(u_1,v_1) = 0,\\
        \frac{\sum_{i=1}^{n} [u[i] = v[i] = 1]}{\min(u_1, v_1)} &\text{otherwise}
    \end{cases}

Braun-Blanquet
==============

The Braun-Blanquet similarity is the ratio of equal bits in the two bit vectors :math:`u` and :math:`v` divided by the
maximum number of bits set in either of the two vectors. The implementation is given in `RDKit <https://github.com/rdkit/rdkit/blob/722cbba894736bf3adbe792e7158fba26b5f8e6f/Code/DataStructs/BitOps.cpp#L409>`__.

.. math::

    & u_1 = \sum_{i=1}^{n} [u[i]]\\
    & v_1 = \sum_{i=1}^{n} [v[i]]\\
    & \text{Braun-Blanquet}(u, v) = \begin{cases}
        0, &\text{if} \max(u_1,v_1) = 0,\\
        \frac{\sum_{i=1}^{n} [u[i] = v[i] = 1]}{\max(u_1, v_1)} &\text{otherwise}
    \end{cases}

Canberra
========

The Canberra distance is the sum of the absolute differences of the two vectors :math:`u` and :math:`v` divided by the
sum of the absolute values of the two vectors. The implementation is given in `SciPy <https://github.com/scipy/scipy/blob/7dcd8c59933524986923cde8e9126f5fc2e6b30b/scipy/spatial/distance.py#L1131>`__.

.. math::

    \text{Canberra}(u, v) = \sum_{i=1}^{n} \frac{|u[i] - v[i]|}{|u[i]| + |v[i]|}

Dice
====

The Dice similarity is the ratio of equal bits in the two bit vectors :math:`u` and :math:`v` divided by the sum of the
number of bits set in either of the two vectors. The implementation is given in `RDKit <https://github.com/rdkit/rdkit/blob/722cbba894736bf3adbe792e7158fba26b5f8e6f/Code/DataStructs/BitOps.cpp#L333>`__.

.. math::

    \text{Dice}(u, v) = \frac{2 \sum_{i=1}^{n} [u[i] = v[i] = 1]}{\sum_{i=1}^{n} [u[i]] + \sum_{i=1}^{n} [v[i]]}

Hamming or Matching
===================

The Hamming distance (a.k.a. Matching distance) is the number of bits that are different in the two bit vectors
:math:`u` and :math:`v`. The implementation is given in `SciPy <https://github.com/scipy/scipy/blob/7dcd8c59933524986923cde8e9126f5fc2e6b30b/scipy/spatial/distance.py#L697>`__.

.. math::

    \text{Hamming}(u, v) = \sum_{i=1}^{n} [u[i] \neq v[i]]

Jaccard
=======

The Jaccard distance is the number of bits that are different in the two bit vectors :math:`u` and :math:`v` divided by
the number of equal one-bits in the two bit vectors :math:`u` and :math:`v` plus the number of bits that are different
in the two bit vectors :math:`u` and :math:`v`. The implementation is given in `SciPy <https://github.com/scipy/scipy/blob/7dcd8c59933524986923cde8e9126f5fc2e6b30b/scipy/spatial/distance.py#L755>`__.

.. math::

    \text{Jaccard}(u, v) = \frac{\sum_{i=1}^{n} [u[i] \neq v[i]]}{n}

Kulczynski
==========

The Kulczynski similarity is the number of equal one-bits in the two bit vectors :math:`u` and :math:`v` multiplied
with the sum of ones in both vectors divided by twice the sum of ones in both vectors multiplied. The implementation is
given in `RDKit <https://github.com/rdkit/rdkit/blob/722cbba894736bf3adbe792e7158fba26b5f8e6f/Code/DataStructs/BitOps.cpp#L317>`__.

.. math::

    & u_1 = \sum_{i=1}^{n} [u[i]]\\
    & v_1 = \sum_{i=1}^{n} [v[i]]\\
    & \text{Kulczynski}(u, v) = \begin{cases}
        0, &\text{if} u_1 \cdot v_1 = 0,\\
        \frac{(\sum_{i=1}^{n} [u[i] = v[i] = 1]) \cdot (u_1 + v_1)}{2 \cdot u_1 \cdot v_1)} &\text{otherwise}
    \end{cases}

Matching
========

see Hamming

OnBit
=====

The OnBit similarity is the ratio of equal one-bits in the two bit vectors :math:`u` and :math:`v` divided by the sum
of the one-bits in the two bit vectors :math:`u` and :math:`v`. The similarity is 0 if the latter sum is 0. The
implementation is given in `RDKit <https://github.com/rdkit/rdkit/blob/722cbba894736bf3adbe792e7158fba26b5f8e6f/Code/DataStructs/BitOps.cpp#L463>`__.

.. math::

    \text{OnBit}(u, v) = \begin{cases}
        0, &\text{if} \sum_{i=1}^{n} [u[i] \lor v[i]] = 0,\\
        \frac{(\sum_{i=1}^{n} [u[i] = v[i] = 1])}{\sum_{i=1}^{n} [u[i] \lor v[i]]} &\text{otherwise}
    \end{cases}

Rogers-Tanimoto
===============

The Rogers-Tanimoto distance is twice the number of bits that are different in the two bit vectors :math:`u` and
:math:`v` divided by the sum of the number of bits that are different in the two bit vectors :math:`u` and :math:`v`
plus the number of bits that are equal in the vectors. The implementation is given in `SciPy <https://github.com/scipy/scipy/blob/7dcd8c59933524986923cde8e9126f5fc2e6b30b/scipy/spatial/distance.py#L1389>`__.

.. math::

    \text{Rogers-Tanimoto}(u, v) = \frac{2 \cdot \sum_{i=1}^{n} [u[i] \neq v[i]]}{\sum_{i=1}^{n} [u[i] \neq v[i]] + \sum_{i=1}^{n} [u[i] = v[i]]}

Rogot-Goldberg
==============

The Rogot-Goldberg similarity is the ratio of equal one-bits in the two bit vectors :math:`u` and :math:`v` divided by
the sum of the one-bits in the two bit vectors :math:`u` and :math:`v` plus the number of bits that are different in
the two bit vectors :math:`u` and :math:`v`. The implementation is given in `RDKit <https://github.com/rdkit/rdkit/blob/722cbba894736bf3adbe792e7158fba26b5f8e6f/Code/DataStructs/BitOps.cpp#L434>`__.

.. math::

    & x = \sum_{i=1}^{n} [u[i] = v[i] = 1]\\
    & y = \sum_{i=1}^{n} [u[i]]\\
    & z = \sum_{i=1}^{n} [u[i]]\\
    & d = n - y - z + x\\
    & \text{Rogot-Goldberg}(u, v) = \begin{cases}
        1, &\text{if} x = n \lor d = n,\\
        \frac{x}{x + z} + \frac{d}{2 \cdot n - y - z} &\text{otherwise}
    \end{cases}

Russel
======

The Russel similarity is the ratio of equal one-bits in the two bit vectors :math:`u` and :math:`v` divided by the
number of one-bits in the two bit vectors :math:`u` and :math:`v`. The implementation is given in `RDKit <https://github.com/rdkit/rdkit/blob/722cbba894736bf3adbe792e7158fba26b5f8e6f/Code/DataStructs/BitOps.cpp#L425>`__.

.. math::

    \text{Russel}(u, v) = \frac{\sum_{i=1}^{n} [u[i] = v[i] = 1]}{n}

Sokal
=====

The Sokal similarity is the ratio of equal one-bits in the two bit vectors :math:`u` and :math:`v` divided by the sum
of the one-bits in the two bit vectors :math:`u` and :math:`v` minus the number of equal one-bits in the two bit
vectors :math:`u` and :math:`v`. The implementation is given in `RDKit <https://github.com/rdkit/rdkit/blob/722cbba894736bf3adbe792e7158fba26b5f8e6f/Code/DataStructs/BitOps.cpp#L349>`__.

.. math::

    \text{Sokal}(u, v) = \frac{\sum_{i=1}^{n} [u[i] = v[i] = 1]}{2 \cdot \sum_{i=1}^{n} [u[i]] + [v[i]] - \sum_{i=1}^{n} [u[i] = v[i] = 1]}

Sokal-Michener
==============

The Sokal-Michener distance is twice the number of bits that are different in the two bit vectors :math:`u` and
:math:`v` divided by the sum of the number of bits that are different in the two bit vectors :math:`u` and :math:`v`
plus the number of bits that are equal in the vectors. The implementation is given in `SciPy <https://github.com/scipy/scipy/blob/7dcd8c59933524986923cde8e9126f5fc2e6b30b/scipy/spatial/distance.py#L1496>`__.

.. math::

    \text{Sokal-Michener}(u, v) = \frac{2 \cdot \sum_{i=1}^{n} [u[i] \neq v[i]]}{\sum_{i=1}^{n} 2 \cdot [u[i] \neq v[i]] + [u[i] = v[i]]}

Tanimoto
========

The Tanimoto similarity is the ratio of equal one-bits in the two bit vectors :math:`u` and :math:`v` divided by the
sum of the one-bits in the two bit vectors :math:`u` and :math:`v` minus the number of equal one-bits in the two bit
vectors :math:`u` and :math:`v`. The implementation is given in `RDKit <https://github.com/rdkit/rdkit/blob/722cbba894736bf3adbe792e7158fba26b5f8e6f/Code/DataStructs/BitOps.cpp#L270>`__.

.. math::

    & t = \sum_{i=1}^{n} [u[i]] + [v[i]]\\
    & c = \sum_{i=1}^{n} [u[i] = v[i] = 1]\\
    & \text{Tanimoto}(u, v) = \begin{cases}
        1, &\text{if} t = 0,\\
        \frac{c}{t - c} &\text{otherwise}
    \end{cases}

Yule
====

The Yule distance is twice the number of bits that are different in the two bit vectors :math:`u` and :math:`v` divided
by the sum of the number of bits that are different in the two bit vectors :math:`u` and :math:`v` plus the number of
bits that are equal in the vectors. The implementation is given in `SciPy <https://github.com/scipy/scipy/blob/7dcd8c59933524986923cde8e9126f5fc2e6b30b/scipy/spatial/distance.py#L1274>`__.

.. math::

    \text{Yule}(u, v) = \frac{2 \cdot \sum_{i=1}^{n} [u[i] = v[i] = 1]}{\sum_{i=1}^{n} [u[i] = v[i]] + \sum_{i=1}^{n} [u[i] = v[i] = 1]}
