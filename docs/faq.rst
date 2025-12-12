.. _faq-label:

Frequently Asked Questions
==========================

Many questions are already answered in the Workflow section of this documentation. Examples of how to use DataSAIL as a package or commandline tool are 
given in the Example section. Here, we collect and answer some frequently asked questions that are not covered in the other sections and arose on 
conference discussions, GitHub issues, or other occasions. If you don't find help here, check the 
`GitHub Issue Tracker <https://github.com/kalininalab/datasail/issues?q=is%3Aissue>`_. and consider opening a new issue if your question is not covered.

Theoretical and Conceptional Questions
--------------------------------------

1. Does training on DataSAIL splits produce better generalizing models?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Yes, training on DataSAIL splits generally leads to better generalizing models. The DataSAIL splits are designed to reduce information leakage between splits.
Therefore, when used for hyperparameter tuning, they help in selecting models (and their hyperparameter) that generalize better to unseen data.

2. What are the limitations of DataSAIL?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The most time and memory consuming step in DataSAIL is the clustering of the data. For most datatypes, this is done by third-party programms such as FoldSeek, 
DIAMOND, or MASH. In that case, DataSAIL has no influence on the runtime and memory consumption. The user may provide their own commandline arguments to these 
programs. 

Practical Questions
-------------------

1. How can I relax the split constraints if DataSAIL fails to find a solution?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sometimes, DataSAIL is unable to solve the split problem and might output a message like:

.. code-block:: shell

    GUROBI cannot solve the problem. Please consider relaxing split restrictions, e.g., less splits, or a higher tolerance level for exceeding cluster limits.

DataSAIL compiles your input into multiple variables and constraints that for a constrained optimization problem. There are some options to solve this problem:

- Check the DataSAIL version. In :code:`v1.2.0` we added handling for too large clusters. For example, 80% of your data is in cluster A but you want a 5 splits 
  with 20% of the data for a 5-fold cross-validation. This is impossible to solve. In :code:`v1.2.0` we introduced the :code:`overflow` option to either 

  - :code:`break` large clusters into smaller parts to fit the splits, or
  - :code:`assign` the whole large cluster to one split and allow that split to exceed its size limit.

- If you are already on :code:`v1.2.0` or newer, you can set the :code:`epsilon` value to higher numbers. Default is :code:`0.05` but anything up to :code:`0.2` 
  or :code:`0.3` is totally reasonable. If you use stratification, you also need to set :code:`delta` to a higher value as both values are connected in that scenario.

2. DataSAIL shows a log message stating the found solution is :code:`optimal_inaccurate`. What does that mean?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This message just means, that the solver in DataSAIL found a solution, but the optimization did not finish and was terminated because of the timeout. 
Therefore, the solution is not guaranteed to be optimal, but it is still a valid solution that satisfies all constraints and is in most cases close to optimal. 
Therefore, you can use that :code:`optimal_inaccurate` solution without problems.

3. I set :code:`runs>1` but DataSAIL outputs the same splits each time. Why is that?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When you set the :code:`runs` variable to values greater than :code:`1`, DataSAIL will shuffle the dataset inbetween splitting rounds to run the optimization from different initializations.
But since many datasets have a unique optimal solution, DataSAIL might find the same solution multiple times and output it mutliple times.
