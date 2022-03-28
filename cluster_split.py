import numpy as np
import pandas as pd
import os
import requests
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import subprocess
import sys
import random
# import extract_clusters
#not sure how best to do this and whether this is even useful


if not os.path.isdir("data/cluster"):
    os.makedirs("data/cluster")
if not os.path.isdir("data/finalclusters"):
    os.makedirs("data/finalclusters")

file_dir = sys.argv[1]
steps = int(sys.argv[2])


def load_data(file_dir):
    """
    this is not really doing much
    """
    file = file_dir + " "

    return file


# use mmseqs to cluster x times, cluster directory, tmp
def clustering(file_dir, clu_dir, tmp, steps):
    """
    file_dir : directory of file to be clustered
    clu_dir : directory of results
    tmp : tmp directory needed for mmseqs
    steps : number of cluster iterations

    ---

    clustering hierarchically - result are multiple cluster files in mmseqs style
        -> .tsv, .fasta

    """
    # lowering the sequence identity for clustering with each step
    seq_id = np.linspace(0.1, 0.9, num=steps, dtype=float)[::-1]

    #step 1 cluster dataset
    cmd = "mmseqs easy-cluster " + file_dir + clu_dir + str(1) + " " + tmp + str(1) + " --min-seq-id " + str(seq_id[0])

    proc_out = subprocess.run(cmd, shell=True)
    print(proc_out)

    # do again with representatives of clusters for 'steps' iterations
    for i in range(steps):
        cmd = "mmseqs easy-cluster " + clu_dir + str(i)+"_rep_seq.fasta" + " " + clu_dir + str(i+1) + " " + tmp + str(i+1) + " --min-seq-id " + str(seq_id[i])
        subprocess.run(cmd, shell=True)

    return None


# split and backtrack clusters
def split(file, clu_dir, steps):
    """
    file: directory to fasta file
    clu_dir : directory of clusters to be split
    steps : how many clustering steps have been done

    --------

    first separates the "highest" level in the tree -
    then backtracks all members throughout the other cluster files

    - saves resulting .csv and .fasta files in data/finalclusters

    """
    x = 60 #percentage train
    y = 30 #percentage test
    z = 10 #percentage validation


    topcluster = pd.read_csv(clu_dir + str(steps) + "_cluster.tsv", sep='\t', names=['rep','mem'])

    l = len(topcluster.rep)


    reps = topcluster['rep']


        # if I split randomly, I cannot guarantee the resulting split will be most challenging
    train, test, val = np.array(reps[:int(x*l/100)]), np.array(reps[int(x*l/100):int(y*l/100)+int(x*l/100)]), np.array(reps[int(y*l/100)+int(x*l/100):])

    previous_cluster = topcluster
    lowercluster = pd.read_csv(clu_dir + str(steps-1) + "_cluster.tsv", sep='\t', names=['rep','mem'])

        # go backwards
    for step in range(steps-2, -1, -1):
                # find all members of that cluster - might be highly imbalanced !! (how fix?)
                # else separation would not be most challenging
        if step >= 1:
            for elem in previous_cluster.rep:
                members = []

                members.append(previous_cluster['mem'][previous_cluster['rep']==elem].values)
                members.append(lowercluster['rep'][lowercluster['rep']==elem].values)
                members.append(lowercluster['mem'][lowercluster['rep']==elem].values)

                if elem in train:
                    np.append(train, members)
                elif elem in test:
                    np.append(test, members)
                else:
                    np.append(val, members)

                previous_cluster = lowercluster
                lowercluster = pd.read_csv(clu_dir + str(step) + "_cluster.tsv", sep='\t', names=['rep','mem'])

    print("len of sets: ", len(train), len(test), len(val), "\n", file=open("output.txt", "a"))

    train = [elem[:4] for elem in train]
    test = [elem[:4] for elem in test]
    val = [elem[:4] for elem in val]

    split_fasta(file, "data/finalclusters/train"+".fasta", train, test, val)
    split_fasta(file, "data/finalclusters/test"+".fasta", train, test, val)
    split_fasta(file, "data/finalclusters/val"+".fasta", train, test, val)

    train, test, val = pd.DataFrame(train), pd.DataFrame(test), pd.DataFrame(val)
    train.to_csv('data/finalclusters/train'+'_seq.csv')
    test.to_csv('data/finalclusters/test'+'_seq.csv')
    val.to_csv('data/finalclusters/val'+'_seq.csv')


    return None


def split_fasta(file, out_dir, train, test, val):
    """
    splits the original fasta file into train, test, val
    according to the splits defined before

    ------------

    file: dir to original fastafile
    outdir: where should splits be saved to
    train, test, val: separated pdb ids
    """

    train_fasta = []
    test_fasta = []
    val_fasta = []

    for seq in SeqIO.parse(file, "fasta"):
        rec = SeqRecord(seq.seq, id=seq.id, description=seq.description)
        if seq.id[:4] in train:
            train_fasta.append(rec)
        if seq.id[:4] in test:
            test_fasta.append(rec)
        if seq.id[:4] in val:
            val_fasta.append(rec)

    SeqIO.write(train_fasta, out_dir, "fasta")
    SeqIO.write(test_fasta, out_dir, "fasta")
    SeqIO.write(val_fasta, out_dir, "fasta")

    return None

def main():
    """
    please pass :
        1) directory of fasta file to be split
        2) #clustering steps for mmseqs2
    """

    file = load_data(file_dir)
    clustering(file, 'data/cluster/clu', 'data/cluster/tmp', steps=steps)
    split('data/fasta/properfasta.fasta', 'data/cluster/clu', steps)



if __name__ == "__main__":
    print("starting")
    main()
