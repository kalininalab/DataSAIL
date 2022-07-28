import os
import time

from utils import randomString, seqMapToFasta, call_mmseqs_clustering

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

        self.mmseqs2_path = 'mmseqs'

        self.verbosity = 3

class Mmseqs_cluster:
    def __init__(self, cluster_file, seq_id_threshold):
        self.seq_id_threshold = seq_id_threshold
        f = open(cluster_file, 'r')
        lines = f.readlines()
        f.close()

        self.clusters = {}

        for line in lines:
            words = line[:-1].split('\t')
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
        os.remove(cluster_file)
        os.remove(rep_seq_file)
        os.remove(all_seq_file)

    return cluster_obj

def make_fasta(sequence_map, env, subset = None):
    rstring = randomString()
    fasta_file = f'{env.tmp_folder}/{rstring}.fasta'
    seqMapToFasta(sequence_map, fasta_file, subset = subset)
    return fasta_file

class Sequence_cluster_tree:

    class Node:
        def __init__(self, label, children = None, fused_children = None):
            self.label = label
            if children is None:
                self.children = []
            else:
                self.children = children

            if fused_children is None:
                self.fused_children = []
            else:
                self.fused_children = fused_children

        def isLeaf(self):
            return len(self.children) == 0

        def fuse_check(self, sequence_map):
            if self.isLeaf():
                return
            seq_a = sequence_map[self.children[0]]
            for child in self.children[1:]:
                seq_b = sequence_map[child]
                if seq_a != seq_b:
                    return
            self.fused_children = self.children[0:]
            self.children = []

        def get_fused_label(self):
            return ' & '.join([f'{self.label}'] + [str(x) for x in self.fused_children])

        def print_cascade(self, nodes, level = 0):
            print(f'{" "*level}{self.get_fused_label()}')
            for child in self.children:
                nodes[child].print_cascade(nodes, level = level+1)


    def __init__(self, sequence_map, env, initial_fasta_file = None):

        if env.verbosity >= 1:
            print(f'Creating new sequence cluster tree: {initial_fasta_file}')
            t0 = time.time()

        if initial_fasta_file is not None:
            fasta_file = initial_fasta_file
        else:
            fasta_file = make_fasta(sequence_map, env)

        cluster_obj = get_mmseqs_cluster(env, fasta_file, seq_id_threshold = 0.5)

        if initial_fasta_file is None:
            if env.verbosity >= 1:
                print(f'Removing file: {fasta_file}')
            os.remove(fasta_file)

        self.current_node_id = 0
        self.nodes = {}
        self.node_pointers = {}

        roots, broad_nodes, potential_final_root = self.process_cluster(cluster_obj, sequence_map)

        if env.verbosity >= 3:
            t1 = time.time()
            cluster_obj.print_all()
            print(f'Initial process_cluster: {t1-t0}\n{roots}\n{broad_nodes}\n')

        while len(roots) > 1 or len(broad_nodes) > 0:
            if env.verbosity >= 3:
                tl0 = time.time()
            fasta_file = make_fasta(sequence_map, env, subset = roots)
            cluster_obj = get_mmseqs_cluster(env, fasta_file, seq_id_threshold = 0.0)
            os.remove(fasta_file)

            new_roots, new_broad_nodes, potential_final_root = self.process_cluster(cluster_obj, sequence_map)

            if env.verbosity >= 4:
                print(f'New roots coming from: {roots}\n{new_roots}\n')

            for rep in broad_nodes:
                members, node_id = broad_nodes[rep]
                fasta_file = make_fasta(sequence_map, env, subset = members)
                cluster_obj = get_mmseqs_cluster(env, fasta_file, seq_id_threshold = 1.0)
                os.remove(fasta_file)

                if env.verbosity >= 4:
                    print(f'Member clustering:\n{members}')
                    cluster_obj.print_all()

                even_more_new_roots, even_more_new_broad_nodes, _ = self.process_cluster(cluster_obj, sequence_map, add_to_node = node_id)

                if env.verbosity >= 4:
                    print(f'New roots coming from members:\n{even_more_new_roots}\n')

                new_roots += even_more_new_roots
                new_broad_nodes.update(even_more_new_broad_nodes)

            if len(new_broad_nodes) == 0 and len(roots) == len(new_roots):
                for rep in roots:
                    if not rep in self.nodes:
                        self.nodes[rep] = self.Node(rep)
                rep_a = roots[0]
                if rep_a in self.node_pointers:
                    left_node_id = self.node_pointers[rep_a]
                else:
                    left_node_id = rep_a

                for rep_b in roots[1:]:
                    children_nodes = [left_node_id]

                    if rep_b in self.node_pointers:
                        children_nodes.append(self.node_pointers[rep_b])
                    else:
                        children_nodes.append(rep_b)

                    new_node_id = self.get_new_node_id()
                    new_node = self.Node(new_node_id, children = children_nodes)
                    self.nodes[new_node_id] = new_node
                    left_node_id = new_node_id

                potential_final_root = new_node_id
                break

            roots = new_roots
            broad_nodes = new_broad_nodes

            if env.verbosity >= 3:
                tl1 = time.time()
                print(f'Next iteration of process_cluster: {tl1-tl0}\n{roots}\n{broad_nodes}\n')

        self.root = self.nodes[potential_final_root]

    def get_new_node_id(self):
        self.current_node_id += 1
        return self.current_node_id

    def process_cluster(self, cluster_obj, sequence_map, add_to_node = None):
        roots = []
        broad_nodes = {}

        for rep in cluster_obj.clusters:
            
            members = cluster_obj.clusters[rep]
            if len(members) == 1:
                if not rep in self.nodes:
                    self.nodes[rep] = self.Node(rep)
                if add_to_node is not None:
                    self.nodes[add_to_node].children.append(rep)
                else:
                    roots.append(rep)
            elif len(members) == 2:
                if not members[0] in self.nodes:
                    self.nodes[members[0]] = self.Node(members[0])
                if not members[1] in self.nodes:
                    self.nodes[members[1]] = self.Node(members[1])

                children_nodes = []

                if members[0] in self.node_pointers:
                    children_nodes.append(self.node_pointers[members[0]])
                else:
                    children_nodes.append(members[0])

                if members[1] in self.node_pointers:
                    children_nodes.append(self.node_pointers[members[1]])
                else:
                    children_nodes.append(members[1])

                new_node_id = self.get_new_node_id()
                new_node = self.Node(new_node_id, children = children_nodes)
                #new_node.fuse_check(sequence_map)
                self.nodes[new_node_id] = new_node
                if add_to_node is not None:
                    self.nodes[add_to_node].children.append(new_node_id)
                    roots.append(rep)
                    self.node_pointers[rep] = new_node_id

            elif cluster_obj.seq_id_threshold == 1.0:
                children_nodes = []
                for member in members:
                    if not member in self.nodes:
                        self.nodes[member] = self.Node(member)
                    if member in self.node_pointers:
                        children_nodes.append(self.node_pointers[member])
                    else:
                        children_nodes.append(member)

                new_node_id = self.get_new_node_id()
                new_node = self.Node(new_node_id, fused_children = children_nodes)
                self.nodes[new_node_id] = new_node
                if add_to_node is not None:
                    self.nodes[add_to_node].children.append(new_node_id)

                if add_to_node is None:
                    roots.append(rep)
                    self.node_pointers[rep] = new_node_id

            else:
                
                new_node_id = self.get_new_node_id()
                new_node = self.Node(new_node_id)
                self.nodes[new_node_id] = new_node
                if add_to_node is not None:
                    self.nodes[add_to_node].children.append(rep)

                if add_to_node is None:
                    roots.append(rep)
                    self.node_pointers[rep] = new_node_id
                broad_nodes[rep] = members, new_node_id
        if len(roots) == 1 and add_to_node is None:
            potential_final_root = new_node_id
        else:
            potential_final_root = None
        return roots, broad_nodes, potential_final_root

    def print_tree(self):
        self.root.print_cascade(self.nodes)
