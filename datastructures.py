import os
import time
import subprocess

from Bio import pairwise2

from utils import randomString, seqMapToFasta, call_mmseqs_clustering, BLOSUM62, getCovSI

class Environment:
    def __init__(self, input_file, steps, out_dir, fasta_store, tr_size, te_size, fuse_seq_id_threshold = 1.0, verbosity = 1):
        self.input_file = input_file
        self.steps = steps
        self.out_dir = out_dir

        self.fuse_seq_id_threshold = fuse_seq_id_threshold

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        self.fasta_store = fasta_store
        self.tr_size = tr_size
        self.te_size = te_size

        self.tmp_folder = f'{out_dir}/tmp'

        if not os.path.isdir(self.tmp_folder):
            os.mkdir(self.tmp_folder)

        self.mmseqs2_path = 'mmseqs'

        self.verbosity = verbosity

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
        def __init__(self, label, rep, children = None, fused_children = None):
            self.label = label
            self.rep = rep
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
            reps.append(self.nodes[node_id].rep)
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

        for member in members:
            children_nodes.append(member)
            if member in self.nodes:
                self.delete_node(member)

        #Create the fused node (which is a leaf)
        new_node_id = self.get_new_node_id()
        new_node = self.Node(new_node_id, rep, fused_children = children_nodes)
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
                    self.nodes[rep] = self.Node(rep, rep)

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

                    for member in members:
                        if not member in self.nodes: #create a leave and put it to the children list
                            self.nodes[member] = self.Node(member, member)
                        children_nodes.append(member)
                        self.set_parent(member, new_node_id)

                    #Create the parent node
                    new_node = self.Node(new_node_id, rep, children = children_nodes)
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

                for member in members:
                    if not member in self.nodes: #create a leave and put it to the children list
                        self.nodes[member] = self.Node(member, member)
                    children_nodes.append(member)
                    self.set_parent(member, new_node_id)

                new_node = self.Node(new_node_id, rep, children = children_nodes)
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
                self.nodes[rep] = self.Node(rep, rep)
        rep_a = node_ids[0]
        left_node_id = rep_a

        for rep_b in node_ids[1:]:
            children_nodes = [left_node_id]

            children_nodes.append(rep_b)

            new_node_id = self.get_new_node_id()
            self.set_parent(rep_b, new_node_id)
            new_node = self.Node(new_node_id, rep_b, children = children_nodes)
            self.nodes[new_node_id] = new_node
            left_node_id = new_node_id

        return new_node_id


    def print_tree(self):
        self.root.print_cascade(self.nodes)

    def write_dot_file(self, outfile, env):
        lines = ['graph ""\n\t{\n']
        for node_id in self.nodes:
            label = node_id
            for child in self.nodes[node_id].children:
                child_label = self.nodes[child].get_fused_label()
                lines.append(f'\t{label} -- "{child_label}"\n')
        lines.append('\t}\n')
        f = open(outfile, 'w')
        f.write(''.join(lines))
        f.close()

        f = open(f'{env.out_dir}/tree_{env.fuse_seq_id_threshold}.png', 'w')
        p = subprocess.Popen(['dot', '-Tpng', outfile], stdout = f)
        p.wait()
        f.close()
