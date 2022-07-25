ReadMe - SCALA - Training-Validation Setup

A script constructing the most challenging training-test-validation dataset by hierarchically clustering a database of fasta sequencing and then separating the tree such that no previously seen similar sequence is in the test or validation set.

How to use:

"python3 scala.py -i <path_to_fasta_database> -o <directory_for_outputfiles>"

additional optional parameters:
-s : clustering steps (int) (default=4)
-f : additional fasta output (y/n)
-tr : size of training set (default=60%)
-te : size of test set (default=30%)

Output:
- lists of IDs for training, test and validation sets
- optional: fasta files split accordingly
