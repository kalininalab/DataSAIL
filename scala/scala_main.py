import numpy as np
import pandas as pd
import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import subprocess
import sys
import random
import argparse
import shutil

from scala.datastructures import Sequence_cluster_tree, Environment, group_bins, bin_list_to_prot_list
from scala.utils import parseFasta, call_mmseqs_clustering

# use mmseqs to cluster x times, cluster directory, tmp
def clustering(env):
    """
    file_dir : directory of file to be clustered
    clu_dir : directory where clusters are saved
    tmp : tmp directory needed for mmseqs
    steps : number of cluster iterations

    ---

    clustering hierarchically - result are multiple cluster files in mmseqs style
        -> .tsv, .fasta

    """
    # lowering the sequence identity for clustering with each step
    seq_id = np.linspace(0.8, 0.99, num=env.steps, dtype=float)[::-1]

    #step 1 cluster dataset
    call_mmseqs_clustering(env.input_file, output_path = f'{env.out_dir}/1', seq_id_threshold = seq_id[0])

    # do again with representatives of clusters for 'steps' iterations
    for i in range(env.steps):
        call_mmseqs_clustering(f'{env.out_dir}/{str(i+1)}_rep_seq.fasta', output_path = f'{env.out_dir}/{str(i+1)}', seq_id_threshold = seq_id[i])

    return None


def clean_and_save(env, train, test, val):

    train = list(set(train))
    test = list(set(test))
    val = list(set(val))

    finaltrain = pd.DataFrame([elem[:4] for elem in train])
    finaltrain.to_csv(env.out_dir+"/trainlist.csv")

    finaltest = pd.DataFrame([elem[:4] for elem in test])
    finaltest.to_csv(env.out_dir+"/testlist.csv")

    finalval = pd.DataFrame([elem[:4] for elem in val])
    finalval.to_csv(env.out_dir+"/vallist.csv")

    return finaltrain[0].tolist(), finaltest[0].tolist(), finalval[0].tolist()


def cleanup():
    DIR = os.getcwd()
    deleteItem=False
    for filename in os.listdir(DIR):
        if 'tmp' in filename:
            deleteItem=True
        elif '.fasta' in filename:
            deleteItem=True
            break
        if deleteItem:
            shutil.rmtree(filename)

    return None

def print_output(validation_set, train_test_pairs, seq_tree):
    print('Validation set:')
    print(bin_list_to_prot_list(validation_set, seq_tree.nodes))
    train_set, test_set = train_test_pairs[0]
    print('Test set:')
    print(bin_list_to_prot_list(test_set, seq_tree.nodes))
    print('Train set:')
    print(bin_list_to_prot_list(train_set, seq_tree.nodes))

def transform_output(validation_set, train_test_pairs, seq_tree):
    tr_validation_set = bin_list_to_prot_list(validation_set, seq_tree.nodes)
    tr_train_test_pairs = []
    for train_set, test_set in train_test_pairs:
        tr_train_test_pairs.append((bin_list_to_prot_list(train_set, seq_tree.nodes), bin_list_to_prot_list(test_set, seq_tree.nodes)))
    return tr_validation_set, tr_train_test_pairs

def core_routine(env, return_lists = False):
    sequence_map = parseFasta(path=env.input_file, check_dups = True)
    seq_tree = Sequence_cluster_tree(sequence_map, env, initial_fasta_file = env.input_file)

    if env.write_tree_file:
        seq_tree.write_dot_file(f'{env.out_dir}/tree.txt', env)

    bins = seq_tree.split_into_bins()

    validation_set, train_test_pairs = group_bins(bins, env, seq_tree)

    if return_lists:
        return transform_output(validation_set, train_test_pairs, seq_tree)

    return validation_set, train_test_pairs


###################

def main():

    parser = argparse.ArgumentParser(
        prog = 'SCALA',
        description = "this tool helps providing the most challenging dataset split for a machine learning model in order to prevent information leakage and improve generalizability",
        epilog = "enjoy :)"
    )
    # parser.add_argument("-h", help="please give directory to input dataset and output directory - other settings are optional")
    parser.add_argument("-i", help="directory to input file (FASTA/FASTQ)", required=True, dest='input', action='store')
    parser.add_argument("-o", help="directory to save the results in", required=True, dest='output', action='store')
    parser.add_argument("-tr", help="size of training set - default ~60%", default=60, dest='tr_size', action='store', type=int)
    parser.add_argument("-te", help="size of test set - default ~30%", default=30, dest='te_size', action='store', type=int)
    parser.add_argument("-st", help="sequence identity threshold for undistinguishable sequences - range: [0.00 - 1.00] -default: 1.0", default=1.0, dest='seq_id_threshold', action='store', type=float)
    parser.add_argument("-v", help="verbosity level - range: [0 - 5] - default: 1", default=1, dest='verbosity', action='store', type=int)
    parser.add_argument("-w", help="directory to weight file (.tsv) Format: [Sequence ID (corresponding to given input file)]tab[weight value]", dest='weight_file', action='store')
    parser.add_argument("-lw", help="sequence length weighting - default: False", dest='length_weighting', default = False, action='store', type=bool)
    args = parser.parse_args()

    env = Environment(args.input, args.output, args.tr_size, args.te_size, fuse_seq_id_threshold = args.seq_id_threshold, verbosity = args.verbosity, weight_file = args.weight_file, length_weighting = args.length_weighting)

    validation_set, train_test_pairs = core_routine(env)


if __name__ == "__main__":
    print("starting")
    main()
