import os
import time
import subprocess
import warnings
import logging

import pandas as pd
from Bio import pairwise2

from scala.utils.blossum62 import BLOSUM62
from scala.utils.mmseqs import mmseqs_clustering, call_mmseqs_clustering
from scala.utils.utils import random_string, seq_map_to_fasta, get_cov_seq_ident, parse_fasta


def scala(args):
    env = Environment(
        args.input,
        args.output,
        args.tr_size,
        args.te_size,
        fuse_seq_id_threshold=args.seq_id_threshold,
        verbosity=2,  # args.verbosity,
        weight_file=args.weight_file,
        length_weighting=args.length_weighting,
        tree_file=args.tree_file
    )

    validation_set, train_test_pairs = core_routine(env, return_lists=True)

    valset, tpairs = pd.DataFrame(validation_set), pd.DataFrame(train_test_pairs)
    valset.to_csv(f'{args.output}/valset.csv')
    tpairs.to_csv(f'{args.output}/tpairs.csv')


def print_output(validation_set, train_test_pairs, seq_tree):
    print('Validation set:')
    print(bin_list_to_prot_list(validation_set, seq_tree.nodes))
    train_set, test_set = train_test_pairs[0]
    print('Test set:')
    print(bin_list_to_prot_list(test_set, seq_tree.nodes))
    print('Train set:')
    print(bin_list_to_prot_list(train_set, seq_tree.nodes))

def transform_output(validation_set, train_test_pairs, seq_tree, subslices = None, verbosity = 0):
    tr_validation_set = bin_list_to_prot_list(validation_set, seq_tree.nodes)
    tr_train_test_pairs = []
    for train_set, test_set in train_test_pairs:
        tr_train_test_pairs.append((bin_list_to_prot_list(train_set, seq_tree.nodes), bin_list_to_prot_list(test_set, seq_tree.nodes)))

    if subslices is not None:
        transformed_subslices = []
        for subslice in subslices:
            transformed_subslices.append(bin_list_to_prot_list([subslice], seq_tree.nodes))
        if verbosity >= 3:
            print(f'SCALA output with sublices:')
            print(f'Validation set:\n{tr_validation_set}')
            print(f'Train Test Pairs:\n{tr_train_test_pairs}')
            print(f'Sublices:\ntransformed_subslices')
        return tr_validation_set, tr_train_test_pairs, transformed_subslices

    if verbosity >= 3:
        print(f'SCALA output:')
        print(f'Validation set:\n{tr_validation_set}')
        print(f'Train Test Pairs:\n{tr_train_test_pairs}')
    return tr_validation_set, tr_train_test_pairs


def core_routine(env, return_lists = False, add_subslice = False):
    if env.verbosity >= 2:
        print(f'Core routine step 1: parse input - {env.input_file}')
    sequence_map = parse_fasta(path=env.input_file, check_dups = True)

    if env.verbosity >= 2: 
        print('Core routine step 2: sequence cluster tree creation')
    seq_tree = Sequence_cluster_tree(sequence_map, env, initial_fasta_file = env.input_file)

    if env.write_tree_file:
        seq_tree.write_dot_file(f'{env.out_dir}/tree.txt', env)

    if env.verbosity >= 3:
        print('Sequence tree print out:')
        seq_tree.print_tree()

    if env.verbosity >= 2: 
        print('Core routine step 3: bin splitting')
    bins = seq_tree.split_into_bins()

    if env.verbosity >= 3:
        for bi in bins:
            bi.print_out(seq_tree.nodes)

    if env.verbosity >= 2: 
        print('Core routine step 4: bin grouping')
    if add_subslice:
        validation_set, train_test_pairs, subslices = group_bins(bins, env, seq_tree, add_subslice = True)
    else:
        validation_set, train_test_pairs = group_bins(bins, env, seq_tree, add_subslice = False)
        subslices = None

    if return_lists:
        return transform_output(validation_set, train_test_pairs, seq_tree, subslices = subslices, verbosity = env.verbosity)

    return validation_set, train_test_pairs


class Environment:
    # storing all the variables & path directories
    def __init__(
            self,
            input_file,
            out_dir,
            tr_size,
            te_size,
            fuse_seq_id_threshold=1.0,
            verbosity=1,
            weight_file=None,
            length_weighting=False,
            tree_file=False
    ):
        """

        Args:
            input_file:
            out_dir:
            tr_size:
            te_size:
            fuse_seq_id_threshold:
            verbosity:
            weight_file:
            length_weighting:
            tree_file:
        """
        self.input_file = input_file
        self.out_dir = out_dir
        self.tr_size = tr_size
        self.te_size = te_size
        self.fuse_seq_id_threshold = fuse_seq_id_threshold
        self.verbosity = verbosity
        self.weight_file = weight_file
        self.length_weighting = length_weighting
        self.write_tree_file = tree_file

        self.val_size = 100 - tr_size - te_size

        if length_weighting and weight_file is not None:
            logging.warning('Weighting based on length cannot be done, when a weight file is given!')

        self.tmp_folder = f'{out_dir}/tmp'
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        if not os.path.isdir(self.tmp_folder):
            os.makedirs(self.tmp_folder, exist_ok=True)

        self.weight_vector = None
        self.mmseqs2_path = 'mmseqs'


class MmseqsCluster:
    def __init__(self, cluster_file, seq_id_threshold):
        """
        Read MMSeqs2 result-file.

        Args:
            cluster_file (str): Path to the file containing the mmseqs2 results
            seq_id_threshold (float): Threshold for sequence identity
        """
        self.seq_id_threshold = seq_id_threshold
        self.clusters = {}

        with open(cluster_file, 'r') as f:
            for line in f.readlines():
                words = line.strip().replace('β', 'beta').split('\t')
                if len(words) != 2:
                    continue
                cluster_head, cluster_member = words

                if cluster_head not in self.clusters:
                    self.clusters[cluster_head] = []
                self.clusters[cluster_head].append(cluster_member)

    def to_dict(self):
        return self.clusters

    def print_all(self):
        for cluster in self.clusters:
            print(f'Rep: {cluster}')
            for member in self.clusters[cluster]:
                print(f'  {member}')


def get_mmseqs_cluster(env, input_file, seq_id_threshold=0.0, cleanup=True):
    """

    Args:
        env:
        input_file:
        seq_id_threshold:
        cleanup:

    Returns:

    """
    cluster_file, rep_seq_file, all_seq_file = call_mmseqs_clustering(env, input_file, seq_id_threshold = seq_id_threshold)
    cluster_obj = MmseqsCluster(cluster_file, seq_id_threshold)

    if cleanup:
        # remove old cluster files
        os.remove(cluster_file)
        os.remove(rep_seq_file)
        os.remove(all_seq_file)

    return cluster_obj


def make_fasta(sequence_map, env, subset=None):
    rstring = random_string()
    fasta_file = f'{env.tmp_folder}/{rstring}.fasta'
    seq_map_to_fasta(sequence_map, fasta_file, subset=subset)
    return fasta_file


def fill_weight_vector(sequence_map, weight_vector):
    # in case there is no weight given for a sequence
    for prot_id in sequence_map:
        if not prot_id in weight_vector:
            weight_vector[prot_id] = 0
    return weight_vector


def initialize_weighting(env, sequence_map):
    # construct a weight vector in form dict(protID: weight, ...)
    if env.verbosity >= 1:
        if env.weight_vector is None:
            l = None
        else:
            l = len(env.weight_vector)
        print(f'Initializing weight vector: Weight file - {env.weight_file}, Given vector : {l}, Number of sequences - {len(sequence_map)}')
    if env.weight_file is not None:
        weight_vector = parse_weight_file(env.weight_file)
        weight_vector = fill_weight_vector(sequence_map, weight_vector)
        return weight_vector

    if env.weight_vector is not None:
        env.weight_vector = fill_weight_vector(sequence_map, env.weight_vector)
        return env.weight_vector

    weight_vector = {}
    for prot in sequence_map:
        if env.length_weighting:
            weight_vector[prot] = len(sequence_map[prot])
        else:
            weight_vector[prot] = 1

    return weight_vector


def parse_weight_file(path):
    with open(path) as f:
        lines = f.readlines()

        weight_vector = {}

        if lines[0].find('\t'):
            for line in lines[1:]:
                targetid, weight = line.strip().split('\t')
                weight_vector[targetid] = weight

        elif lines[0].find(','):
            for line in lines[1:]:
                targetid, weight = line.strip().split(',')
                weight_vector[targetid] = weight

    return weight_vector


def fuse_check(prot_a, prot_b, sequence_map, env):
    seq_a = sequence_map[prot_a].replace('*', '')
    seq_b = sequence_map[prot_b].replace('*', '')
    if env.fuse_seq_id_threshold == 1.0:
        return seq_a == seq_b
    else:
        try:
            (target_aligned_sequence, template_aligned_sequence, a, b, c) = \
                pairwise2.align.globalds(seq_a, seq_b, BLOSUM62, -2.0, -0.5, one_alignment_only=True)[0]
        except:
            print(f'Alignment failed: {prot_a} {prot_b}\n\n{seq_a}\n\n{seq_b}')
        aln_length, seq_id = get_cov_seq_ident(len(target_aligned_sequence), target_aligned_sequence,
                                               template_aligned_sequence)

        return seq_id >= env.fuse_seq_id_threshold


class Sequence_cluster_tree:

    class Node:
        def __init__(self, label, rep, weight, children = None, fused_children = None):
            self.label = label
            self.rep = rep
            self.weight = weight
            if children is None:
                self.children = []
            else:
                self.children = children

            if fused_children is None:
                self.fused_children = []
            else:
                self.fused_children = fused_children
            self.parent = None

        def isLeaf(self):
            return len(self.children) == 0

        def get_fused_label(self):
            if len(self.fused_children) == 0:
                return self.label
            else:
                return ' & '.join(self.fused_children)

        def get_all_prot_ids(self, nodes):
            if self.isLeaf():
                if len(self.fused_children) == 0:
                    return [self.label]
                else:
                    return self.fused_children
            else:
                prot_ids = []
                for child in self.children:
                    prot_ids += nodes[child].get_all_prot_ids(nodes)
                return prot_ids

        def print_cascade(self, nodes, level = 0):
            print(f'{" "*level}{self.get_fused_label()}')
            for child in self.children:
                nodes[child].print_cascade(nodes, level = level+1)


    def __init__(self, sequence_map, env, initial_fasta_file = None):

        if env.verbosity >= 1:
            print(f'Creating new sequence cluster tree: {initial_fasta_file}')
            t0 = time.time()

        self.weight_vector = initialize_weighting(env, sequence_map)

        if initial_fasta_file is not None:
            fasta_file = initial_fasta_file
        else:
            fasta_file = make_fasta(sequence_map, env)

        cluster_obj = get_mmseqs_cluster(env, fasta_file, seq_id_threshold = (env.fuse_seq_id_threshold/2))

        if env.verbosity >= 3:
            t1 = time.time()
            cluster_obj.print_all()
            print(f'Initial mmseqs call: {t1-t0}\n')

        if initial_fasta_file is None:
            if env.verbosity >= 1:
                print(f'Removing file: {fasta_file}')
            os.remove(fasta_file)

        self.current_node_id = 0
        self.nodes = {}

        roots, broad_nodes, potential_final_root = self.process_cluster(cluster_obj, sequence_map, env)

        if env.verbosity >= 3:
            t2 = time.time()
            print(f'Initial process_cluster: {t2-t1}\n{roots}\n{broad_nodes}')

        while len(roots) > 1 or len(broad_nodes) > 0:
            if env.verbosity >= 3:
                tl0 = time.time()
            fasta_file = make_fasta(sequence_map, env, subset = self.get_reps(roots))
            cluster_obj = get_mmseqs_cluster(env, fasta_file, seq_id_threshold = 0.0)
            os.remove(fasta_file)

            new_roots, new_broad_nodes, potential_final_root = self.process_cluster(cluster_obj, sequence_map, env)

            if env.verbosity >= 4:
                print(f'New roots coming from:\n{new_roots}\n{new_broad_nodes}\n')

            for rep in broad_nodes:
                members, node_id = broad_nodes[rep]
                fasta_file = make_fasta(sequence_map, env, subset = self.get_reps(members))
                cluster_obj = get_mmseqs_cluster(env, fasta_file, seq_id_threshold = env.fuse_seq_id_threshold)
                os.remove(fasta_file)

                if env.verbosity >= 4:
                    print(f'Member clustering:\n{members}, {node_id}')
                    cluster_obj.print_all()

                even_more_new_roots, even_more_new_broad_nodes, _ = self.process_cluster(cluster_obj, sequence_map, env, add_to_node = node_id)

                if env.verbosity >= 4:
                    print(f'New roots coming from members:\n{even_more_new_roots}\n{even_more_new_broad_nodes}\n')

                new_roots += even_more_new_roots
                new_broad_nodes.update(even_more_new_broad_nodes)

            if len(new_broad_nodes) == 0 and len(roots) == len(new_roots):
                potential_final_root = self.connect_nodes(roots)

                new_broad_nodes = {}
                new_roots = [potential_final_root]

            roots = new_roots
            broad_nodes = new_broad_nodes

            if env.verbosity >= 3:
                tl1 = time.time()
                print(f'Next iteration of process_cluster: {tl1-tl0}\n{roots}\n{broad_nodes}\n')

        self.root = self.nodes[potential_final_root]

    def get_new_node_id(self):
        self.current_node_id += 1
        return self.current_node_id

    def delete_node(self, node_id):
        parent_id = self.nodes[node_id].parent
        self.remove_parent_link(node_id, parent_id)
        del self.nodes[node_id]

    def remove_parent_link(self, node_id, parent_id):
        if self.nodes[node_id].parent is not None:
            try:
                self.nodes[parent_id].children.remove(node_id)
            except:
                pass

    def set_parent(self, node_id, parent_id):
        old_parent_id = self.nodes[node_id].parent
        self.remove_parent_link(node_id, old_parent_id)
        self.nodes[node_id].parent = parent_id

    def get_reps(self, node_ids):
        reps = []
        for node_id in node_ids:
            if node_id in self.nodes:
                reps.append(self.nodes[node_id].rep)
            else:
                reps.append(node_id)
        return reps

    def fuse_check(self, prot_a, prot_b, sequence_map, env):
        seq_a = sequence_map[prot_a].replace('*','')
        seq_b = sequence_map[prot_b].replace('*','')
        if env.fuse_seq_id_threshold == 1.0:
            return seq_a == seq_b
        else:
            try:
                (target_aligned_sequence, template_aligned_sequence, a, b, c) = pairwise2.align.globalds(seq_a, seq_b, BLOSUM62, -2.0, -0.5, one_alignment_only=True)[0]
            except:
                print(f'Alignment failed: {prot_a} {prot_b}\n\n{seq_a}\n\n{seq_b}')
            (aln_length, seq_id) = getCovSI(len(target_aligned_sequence), target_aligned_sequence, template_aligned_sequence)

            return seq_id >= env.fuse_seq_id_threshold

    def create_fused_node(self, rep, members, roots, add_to_node = None):

        children_nodes = []

        fused_weight = 0

        for member in members:
            children_nodes.append(member)
            if member in self.nodes:
                fused_weight += self.nodes[member].weight
                self.delete_node(member)
            else:
                fused_weight += self.weight_vector[member]
        #Create the fused node (which is a leaf)
        new_node_id = self.get_new_node_id()
        new_node = self.Node(new_node_id, rep, fused_weight, fused_children = children_nodes)
        self.nodes[new_node_id] = new_node


        if add_to_node is not None:  #If preselected parent exists ...
            self.nodes[add_to_node].children.append(new_node_id)
            self.set_parent(new_node_id, add_to_node)
        else: #Else, send the representative to a new round of clustering with lower sequence id threshold
            roots.append(new_node_id)
        return roots

    def process_cluster(self, cluster_obj, sequence_map, env, add_to_node = None):
        roots = []
        broad_nodes = {}

        for rep in cluster_obj.clusters:

            members = cluster_obj.clusters[rep]
            if len(members) == 1: #A cluster with one member becomes a leaf node
                #Create the leaf
                if not rep in self.nodes:
                    self.nodes[rep] = self.Node(rep, rep, self.weight_vector[rep])

                if add_to_node is not None: #If preselected parent exists ...
                    #Connect new parent node to preselected parent
                    self.nodes[add_to_node].children.append(rep)
                    self.set_parent(rep, add_to_node)
                else: #Else, send the representative to a new round of clustering with lower sequence id threshold
                    roots.append(rep)

            elif cluster_obj.seq_id_threshold == env.fuse_seq_id_threshold: #A cluster with more than two members get fused, if the sequence threshold reached the fusing threshold parameter (since they are undistinguishable)
                roots = self.create_fused_node(rep, members, roots, add_to_node = add_to_node)

            elif len(members) == 2: #A cluster with exactly two members is transformed into a node with two leafs

                if self.fuse_check(members[0], members[1], sequence_map, env): #Or, if they are undistinguishable, fuse them and create a leaf
                    roots = self.create_fused_node(rep, members, roots, add_to_node = add_to_node)
                else:
                    children_nodes = []
                    new_node_id = self.get_new_node_id()

                    fused_weight = 0
                    for member in members:
                        if not member in self.nodes: #create a leave and put it to the children list
                            self.nodes[member] = self.Node(member, member, self.weight_vector[member])
                            fused_weight += self.weight_vector[member]
                        else:
                            fused_weight += self.nodes[member].weight
                        children_nodes.append(member)
                        self.set_parent(member, new_node_id)

                    #Create the parent node
                    new_node = self.Node(new_node_id, rep, fused_weight, children = children_nodes)
                    self.nodes[new_node_id] = new_node

                    if add_to_node is not None: #If preselected parent exists ...
                        #Connect new parent node to preselected parent
                        self.nodes[add_to_node].children.append(new_node_id)
                        self.set_parent(new_node_id, add_to_node)
                    else: #Else, send the representative to a new round of clustering with lower sequence id threshold
                        roots.append(new_node_id)

            else: #A cluster with more than one member gets clustered again with higher sequence identity threshold. A parent node needs to be created and handed down as preselected parent
                #Create the parent node
                children_nodes = []
                new_node_id = self.get_new_node_id()

                fused_weight = 0
                for member in members:
                    if not member in self.nodes: #create a leave and put it to the children list
                        self.nodes[member] = self.Node(member, member, self.weight_vector[member])
                        fused_weight += self.weight_vector[member]
                    else:
                        fused_weight += self.nodes[member].weight
                    children_nodes.append(member)
                    self.set_parent(member, new_node_id)

                new_node = self.Node(new_node_id, rep, fused_weight, children = children_nodes)
                self.nodes[new_node_id] = new_node

                if add_to_node is not None:  #If preselected parent exists ...
                    #Connect new parent node to preselected parent
                    self.nodes[add_to_node].children.append(new_node_id)
                    self.set_parent(new_node_id, add_to_node)

                else: #Else, send the representative to a new round of clustering with lower sequence id threshold
                    roots.append(new_node_id)

                #broad_nodes gets clustered again
                broad_nodes[rep] = members, new_node_id

        if add_to_node is not None:
            if len(self.nodes[add_to_node].children) == 1:
                child_id = self.nodes[add_to_node].children[0]
                self.nodes[add_to_node].fused_children = self.nodes[child_id].fused_children
                self.nodes[add_to_node].children = self.nodes[child_id].children
                self.delete_node(child_id)
            elif len(self.nodes[add_to_node].children) > 2:
                placeholder_root = self.connect_nodes(self.nodes[add_to_node].children)
                self.nodes[add_to_node].fused_children = self.nodes[placeholder_root].fused_children
                self.nodes[add_to_node].children = self.nodes[placeholder_root].children
                self.delete_node(placeholder_root)

        if len(roots) == 1 and add_to_node is None:
            potential_final_root = new_node_id
        else:
            potential_final_root = None
        return roots, broad_nodes, potential_final_root

    def connect_nodes(self, node_ids):
        for rep in node_ids:
            if not rep in self.nodes:
                self.nodes[rep] = self.Node(rep, rep, self.weight_vector[rep])
        rep_a = node_ids[0]
        left_node_id = rep_a

        for rep_b in node_ids[1:]:
            children_nodes = [left_node_id]

            children_nodes.append(rep_b)

            new_node_id = self.get_new_node_id()
            self.set_parent(rep_b, new_node_id)
            new_node = self.Node(new_node_id, rep_b, self.nodes[left_node_id].weight + self.nodes[rep_b].weight, children = children_nodes)
            self.nodes[new_node_id] = new_node
            left_node_id = new_node_id

        return new_node_id


    def print_tree(self):
        self.root.print_cascade(self.nodes)

    def write_dot_file(self, outfile, env):
        lines = ['graph ""\n\t{\n']
        for node_id in self.nodes:
            label = node_id
            weight_p = self.nodes[node_id].weight
            for child in self.nodes[node_id].children:
                weight_c = self.nodes[child].weight
                child_label = self.nodes[child].get_fused_label()
                lines.append(f'\t"{label} {weight_p}" -- "{child_label} {weight_c}"\n')
        lines.append('\t}\n')
        f = open(outfile, 'w')
        f.write(''.join(lines))
        f.close()

        f = open(f'{env.out_dir}/tree_{env.fuse_seq_id_threshold}.png', 'w')
        p = subprocess.Popen(['dot', '-Tpng', outfile], stdout = f)
        p.wait()
        f.close()

    def split_into_bins(self, bin_weight_variance_threshhold = 0.1, wished_amount_of_bins = 20):
        nodes_todo = [self.root]
        done = []
        bins = []
        prelim_bin = Bin()

        total_weight = self.root.weight

        ideal_bin_weight = total_weight / wished_amount_of_bins

        lower_weight_thresh = ideal_bin_weight*(1.0-bin_weight_variance_threshhold)
        upper_weight_thresh = ideal_bin_weight*(1.0+bin_weight_variance_threshhold)

        while len(nodes_todo) > 0:

            current_node = nodes_todo.pop()
            node_weight = current_node.weight

            if node_weight < lower_weight_thresh:
                prelim_bin.add_member(current_node)
                if prelim_bin.weight >= lower_weight_thresh:
                    bins.append(prelim_bin)
                    prelim_bin = Bin()

            elif node_weight < upper_weight_thresh or current_node.isLeaf():
                bins.append(Bin(members = [current_node]))

            else:
                for child in current_node.children:
                    nodes_todo.append(self.nodes[child])

        if len(prelim_bin.members) > 0:
            bins.append(prelim_bin)

        return bins


######### not the class anymore

def group_bins(bins, env, seq_tree, add_subslice = False):
    val_size = (seq_tree.root.weight * env.val_size) / 100
    test_size = (seq_tree.root.weight * env.te_size) / 100
    train_size = (seq_tree.root.weight * env.tr_size) / 100

    validation_set = []
    if val_size > 0:
        current_weight = 0
        while len(bins) > 0:
            prot_bin = bins.pop()
            if current_weight + prot_bin.weight < val_size:
                validation_set.append(prot_bin)
                current_weight += prot_bin.weight
            elif (current_weight + prot_bin.weight) - val_size < (val_size - current_weight):
                validation_set.append(prot_bin)
                current_weight += prot_bin.weight
            else:
                bins.append(prot_bin)
                break

    subslices = []
    train_test_pairs = []
    current_test_set = {}
    current_weight = 0
    for bin_number, prot_bin in enumerate(bins):
        if current_weight + prot_bin.weight < test_size:
            current_test_set[bin_number] = prot_bin
            current_weight += prot_bin.weight
        elif (current_weight + prot_bin.weight) - test_size < (test_size - current_weight):
            current_test_set[bin_number] = prot_bin
            current_weight += prot_bin.weight
        else:
            test_set = current_test_set.values()
            train_set = []
            subslice = None
            for inner_bin_number, inner_prot_bin in enumerate(bins):
                if inner_bin_number in current_test_set:
                    continue
                train_set.append(inner_prot_bin)
                if len(train_set) == 2 and subslice is None:
                    subslice = inner_prot_bin
            if add_subslice:
                subslices.append(subslice)
            train_test_pairs.append((train_set, test_set))
            current_test_set = {bin_number:prot_bin}
            current_weight = prot_bin.weight

    if len(current_test_set) > 0:
        test_set = current_test_set.values()
        train_set = []
        subslice = None
        for inner_bin_number, inner_prot_bin in enumerate(bins):
            if inner_bin_number in current_test_set:
                continue
            train_set.append(inner_prot_bin)
            if len(train_set) == 2 and subslice is None:
                subslice = inner_prot_bin
        if add_subslice:
            subslices.append(subslice)
        train_test_pairs.append((train_set, test_set))

    if add_subslice:
        return validation_set, train_test_pairs, subslices
    return validation_set, train_test_pairs


def bin_list_to_prot_list(bins, nodes):
    prot_ids = []
    for prot_bin in bins:
        if prot_bin is None:
            continue
        prot_ids += prot_bin.list_prot_ids(nodes)

    return prot_ids


class Bin:
    def __init__(self, label=None, members=None, neighbors=None):

        if label==None:
            self.label = ''
        else:
            self.label = label

        if members == None:
            self.members = []
            self.weight = 0
        else:
            self.members = members
            self.weight = sum([mem.weight for mem in self.members])

        if neighbors == None:
            self.neighbors = []
        else:
            self.neighbors = neighbors

    def get_members(self):
        return self.members

    def get_label(self):
        return self.label

    def add_member(self, member):
        self.members.append(member)
        self.weight += member.weight

    def add_members(self, new_members):
         for mem in new_members:
            self.add_member(mem)

    def list_prot_ids(self, nodes):
        prot_ids = []
        for mem in self.members:
            prot_ids += mem.get_all_prot_ids(nodes)
        return prot_ids

    def print_out(self, nodes):
        print(f'Printing status of Bin {self.label}')
        for pos, member in enumerate(self.members):
            print(f'Member {pos}: {member.get_all_prot_ids(nodes)}')
