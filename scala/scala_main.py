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
import warnings

from scala.datastructures import Sequence_cluster_tree, Environment, group_bins, bin_list_to_prot_list
from scala.utils import parseFasta, call_mmseqs_clustering



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
    parser.add_argument("-tree", help="print tree file - default: False", dest='tree_file', default=False, action='store', type=bool)
    args = parser.parse_args()

    env = Environment(args.input, args.output, args.tr_size, args.te_size, fuse_seq_id_threshold = args.seq_id_threshold, verbosity = args.verbosity, weight_file = args.weight_file, length_weighting = args.length_weighting, tree_file=env.write_tree_file)

    validation_set, train_test_pairs = core_routine(env, return_lists=True)

    valset, tpairs = pd.DataFrame(validation_set), pd.DataFrame(train_test_pairs)
    valset.to_csv(f'{args.output}/valset.csv')
    tpairs.to_csv(f'{args.output}/tpairs.csv')

if __name__ == "__main__":
    print("starting")
    main()
