import os
import time
import subprocess
import pandas as pd

from Bio import pairwise2

from scala.utils import randomString, seqMapToFasta, call_mmseqs_clustering, BLOSUM62, getCovSI

class Environment:
    # storing all the variables & path directories
    def __init__(self, input_file, out_dir, tr_size, te_size, fuse_seq_id_threshold = 1.0, verbosity = 1, weight_file = None, length_weighting = False):
        self.input_file = input_file

        self.out_dir = out_dir

        self.fuse_seq_id_threshold = fuse_seq_id_threshold

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        self.tr_size = tr_size
        self.te_size = te_size
        self.val_size = 100 - tr_size - te_size

        self.weight_file = weight_file
        self.weight_vector = None
        self.length_weighting = length_weighting

        if length_weighting and weight_file is not None:
            print('\nWarning: length cannot be done, when a weight file is given!\n')

        self.tmp_folder = f'{out_dir}/tmp'

        if not os.path.isdir(self.tmp_folder):
            os.mkdir(self.tmp_folder)

        self.mmseqs2_path = 'mmseqs'

        self.verbosity = verbosity
        self.write_tree_file = False

class Mmseqs_cluster:
    #make mmseqs output files accessible
    def __init__(self, cluster_file, seq_id_threshold):
        self.seq_id_threshold = seq_id_threshold
        f = open(cluster_file, 'r')
        lines = f.readlines()
        f.close()

        self.clusters = {}

        for line in lines:
            words = line[:-1].replace('β', 'beta').split('\t')
            if len(words) != 2:
                continue
            cluster_head, cluster_member = words

            if not cluster_head in self.clusters:
                self.clusters[cluster_head] = []
            self.clusters[cluster_head].append(cluster_member)

    def print_all(self):
        for cluster in self.clusters:
            print(f'Rep: {cluster}')
            for member in self.clusters[cluster]:
                print(f'  {member}')


def get_mmseqs_cluster(env, input_file, seq_id_threshold = 0.0, cleanup = True):
    cluster_file, rep_seq_file, all_seq_file = call_mmseqs_clustering(env, input_file, seq_id_threshold = seq_id_threshold)
    cluster_obj = Mmseqs_cluster(cluster_file, seq_id_threshold)

    if cleanup:
        #remove old cluster files
        os.remove(cluster_file)
        os.remove(rep_seq_file)
        os.remove(all_seq_file)

    return cluster_obj

def make_fasta(sequence_map, env, subset = None):
    rstring = randomString()
    fasta_file = f'{env.tmp_folder}/{rstring}.fasta'
    seqMapToFasta(sequence_map, fasta_file, subset = subset)
    return fasta_file


def fill_weight_vector(sequence_map, weight_vector):
    # in case there is no weight given for a sequence
    for prot_id in sequence_map:
        if not prot_id in weight_vector:
            weight_vector[prot_id] = 0
    return weight_vector



def initialize_weighting(env, sequence_map):
    # construct a weight vector in form dict(protID: weight, ...)
    if env.weight_file is not None:
        weight_vector =  parse_weight_file(env.weight_file)
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

def parse_weight_file(path_to_file):
    # parse a tab separated weight file in form ProtId \t weight

    weight_file = pd.read_csv(path_to_file, sep="\t")

    weight_vector = {}

    for entryId, weight in weight_file.iterrows():
        weight_vector[entryId]=weight

    return weight_vector


class Sequence_cluster_tree:

    class Node:
        # one Node representing one Sequence
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

        # first cluster step with initial file
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

        # analyze initial cluster results
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

    def split_into_bins(self, bin_weight_variance_threshhold = 0.15, wished_amount_of_bins = 20):
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

            elif node_weight < upper_weight_thresh:
                bins.append(Bin(members = [current_node]))

            else:
                for child in current_node.children:
                    nodes_todo.append(self.nodes[child])

        if len(prelim_bin.members) > 0:
            bins.append(prelim_bin)

        return bins

def group_bins(bins, env, seq_tree):
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
            for inner_bin_number, inner_prot_bin in enumerate(bins):
                if inner_bin_number in current_test_set:
                    continue
                train_set.append(inner_prot_bin)
            train_test_pairs.append((train_set, test_set))
            current_test_set = {bin_number:prot_bin}
            current_weight = prot_bin.weight

    if len(current_test_set) > 0:
        test_set = current_test_set.values()
        train_set = []
        for inner_bin_number, inner_prot_bin in enumerate(bins):
            if inner_bin_number in current_test_set:
                continue
            train_set.append(inner_prot_bin)
        train_test_pairs.append((train_set, test_set))

    return validation_set, train_test_pairs

def bin_list_to_prot_list(bins, nodes):
    prot_ids = []
    for prot_bin in bins:
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
