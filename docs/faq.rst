.. _faq-label:

Frequently Asked Questions
==========================

Many questions are already answered in the Workflow section of this documentation. Examples of how to use DataSAIL as a package or commandline tool are 
given in the Example section. Here, we collect and answer some frequently asked questions that are not covered in the other sections and arose on 
conference discussions, GitHub issues, or other occasions.

Does training on DataSAIL splits produce better generalizing models?
--------------------------------------------------------------------
Yes, training on DataSAIL splits generally leads to better generalizing models. The DataSAIL splits are designed to reduce information leakage between splits.
Therefore, when used for hyperparameter tuning, they help in selecting models (and their hyperparameter) that generalize better to unseen data.

What are the limitations of DataSAIL?
-------------------------------------
The most time and memory consuming step in DataSAIL is the clustering of the data. For most datatypes, this is done by third-party programms such as FoldSeek, 
DIAMOND, or MASH. In that case, DataSAIL has no influence on the runtime and memory consumption. The user may provide their own commandline arguments to these 
programs. 
