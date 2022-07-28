import os

import utils


class Environment:
    def __init__(self, input_file, steps, out_dir, fasta_store, tr_size, te_size):
        self.input_file = input_file
        self.steps = steps
        self.out_dir = out_dir

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        self.fasta_store = fasta_store
        self.tr_size = tr_size
        self.te_size = te_size

        self.tmp_folder = f'{out_dir}/tmp'

        if not os.path.isdir(self.tmp_folder):
            os.mkdir(self.tmp_folder)


class Mmseqs_cluster:
    def __init__(self, cluster_file):
        f = open(cluster_file, 'r')
        lines = f.readlines()
        f.close()

        self.clusters = {}

        for line in lines:
            words = line.split('\t')
            if len(words) != 2:
                continue
            cluster_head, cluster_member = words
            if not cluster_head in self.clusters:
                self.clusters[cluster_head] = []
            self.clusters[cluster_head].append(cluster_member)


class Sequence_cluster_tree:

    def __init__(self, sequence_map, tmp_folder, initial_fasta_file = None):
        if initial_fasta_file is not None:
            fasta_file = initial_fasta_file
        else:
            rstring = utils.randomString()
            fasta_file = f'{tmp_folder}/{rstring}.fasta'

        

        if initial_fasta_file is not None:
            os.remove(fasta_file)
