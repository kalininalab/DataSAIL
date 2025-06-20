.. _eval-dl-label:

##########################
Evaluation of Data Leakage
##########################

From version 1.3.0 (or a dev-build from github) onwards, DataSAIL allows you to easily quantify the similarity-induced data leakage for a given datasplit.
This feature is currently only available for the python interface and can be used as follows:

.. code-block:: python

    from datasail.eval import eval_splits

    scaled_dl, total_dl, max_dl = eval_splits("P", path_to_data, path_to_weights, similarity, distance, split_assignments)

The arguments are the same as for the :ref:`python interface<doc-label>` to DataSAIL. The full documentation of this function is given below. 
The output is a tuple containing the following elements:

- :code:`scaled_dl` The scaled data leakage, which is the total data leakage divided by the total pairwise similarity (or distance) in the dataset.
- :code:`total_dl`: The total, unscaled data leakage.
- :code:`max_dl` The total pairwise similarity (or distance) in the dataset.

.. module:: eval_splits
.. autofunction:: datasail.eval.eval_splits
