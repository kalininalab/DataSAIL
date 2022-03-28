ReadMe - Cluster & Split - Training-Validation Setup

A script constructing the most challenging training-test-validation dataset by hierarchically clustering a database of fasta sequencing and then separating the tree such that no previously seen similar sequence is in the test or validation set. 

How to use: 

"python3 cluster_split.py <path_to_fasta_database> <number_of_cluster_iterations (~3-4)>" 


Output: 
intermediate output (mmseqs output) saved in data/clusters
output saved in data/finalclusters 
- 3 datasets of fasta sequences (training, test, validation)
- 3 CSV files with PDB IDs for train, test and validation respectively