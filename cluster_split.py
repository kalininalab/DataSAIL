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
    seq_id = np.linspace(0.8, 0.99, num=steps, dtype=float)[::-1]
    # seq_id = [0.8, 0.8, 0.8, 0.95][::-1]
    #+   --
    #step 1 cluster dataset
    cmd = "mmseqs easy-linclust " + file_dir + clu_dir + str(1) + " " + tmp + str(1)  + " --similarity-type 2 -k 8" + " --min-seq-id " + str(seq_id[0])

    proc_out = subprocess.run(cmd, shell=True)

    # do again with representatives of clusters for 'steps' iterations
    for i in range(steps):
        cmd = "mmseqs easy-linclust " + clu_dir + str(i)+"_rep_seq.fasta" + " " + clu_dir + str(i+1) + " " + tmp + str(i+1) + " --min-seq-id " + str(seq_id[i]) + " --similarity-type 2 -k 13"
        subprocess.run(cmd, shell=True)

    return None


# split and backtrack clusters
def split(file, out_dir, steps):

    topclu = pd.read_csv("data/cluster/clu" + str(steps) +"_cluster.tsv", sep='\t', names=['rep', 'mem'])
    reps = np.unique(topclu.rep)

    x=60
    y=30
    z=10

    l=len(reps)

    train, test, val = list(reps[:int(x*l/100)]), list(reps[int(x*l/100):int(y*l/100)+int(x*l/100)]), list(reps[int(y*l/100)+int(x*l/100):])

    for group in [train, test, val]:
        for i in range(steps, 0, -1):
            clu = pd.read_csv("data/cluster/clu" + str(i) +"_cluster.tsv", sep='\t', names=['rep', 'mem'])
            members = []
            for elem in group:
                members.append(clu['mem'].values[clu['rep']==elem])

            for elem in members:
                for e in elem:
                    group.append(e)


    finaltrain, finaltest, finalval = clean_and_save(train, test, val)

    split_fasta(file, finaltrain, finaltest, finalval)

    return None


def clean_and_save(train, test, val):

    train = list(set(train))
    finaltrain = pd.DataFrame([elem[:4] for elem in train])
    finaltrain.to_csv("data/finalclusters/trainlist.csv")

    test = list(set(test))
    finaltest = pd.DataFrame([elem[:4] for elem in test])
    finaltest.to_csv("data/finalclusters/testlist.csv")

    val = list(set(val))
    finalval = pd.DataFrame([elem[:4] for elem in val])
    finalval.to_csv("data/finalclusters/vallist.csv")

    return finaltrain[0].tolist(), finaltest[0].tolist(), finalval[0].tolist()


def split_fasta(file, train, test, val):
    """
    splits the original fasta file into train, test, val
    according to the splits defined before

    ------------

    file: dir to original fastafile
    outdir: where should splits be saved to
    train, test, val: separated pdb ids
    """
    print(type(train), train[0])

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

    with open("data/fasta/trainfasta.fasta", "w") as handle:
        SeqIO.write(train_fasta, handle, "fasta")
    with open("data/fasta/testfasta.fasta", "w") as handle:
        SeqIO.write(test_fasta, handle, "fasta")
    with open("data/fasta/valfasta.fasta", "w") as handle:
        SeqIO.write(val_fasta, handle, "fasta")

    return None

def main():
    """
    please pass :
        1) directory of fasta file to be split
        2) #clustering steps for mmseqs2
    """

    file = load_data(file_dir)
    #clustering(file, 'data/cluster/clu', 'data/cluster/tmp', steps=steps)
    split('data/fasta/properfasta.fasta', 'data/cluster/clu', steps=steps)



if __name__ == "__main__":
    print("starting")
    main()
