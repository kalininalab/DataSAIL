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


    # this is just a test
def resample(file, change, length, steps, train, test, val):
    # if the length of train samples vs test / val is not according to the percentage
    # specified above, we resample the clusters - eg add one cluster from test to train
    x = 60; y=30; z=10

    xf = int(x*length/100)
    yf = int(y*length/100)
    zf = int(z*length/100)

    if change == 1:
        if (len(train) < xf-10):
            train, test, val = splittop(steps)
            train.append(test[-1])
            test = test.pop(-1)
            split(file, "data/cluster/clu", steps, train, test, val)

        elif (len(train) > xf+10):
            train, test, val = splittop(steps)
            test.append(train[-1])
            train = train.pop(-1)
            split(file, "data/cluster/clu", steps, train, test, val)


    elif change == 2:
        if (len(test) < yf-10):
            train, test, val = splittop(steps)
            test.append(train[-1])
            train = train.pop(-1)
            split(file, "data/cluster/clu", steps, train, test, val)

        elif (len(test) > yf+10):
            train, test, val = splittop(steps)
            train.append(test[-1])
            test = test.pop(-1)
            split(file, "data/cluster/clu", steps, train, test, val)


    elif change == 3:
        if (len(val) < zf-10):
            train, test, val = splittop(steps)
            val.append(test[-1])
            test = test.pop(-1)
            split(file, "data/cluster/clu", steps, train, test, val)

        elif (len(val) > zf+10):
            train, test, val = splittop(steps)
            test.append(val[-1])
            val = val.pop(-1)
            split(file, "data/cluster/clu", steps, train, test, val)

    return None



def splittop(steps):
    #split only the topcluster and give lists of reps to split()
    topclu = pd.read_csv("data/cluster/clu" + str(steps) + "_cluster.tsv", sep='\t', names=['rep', 'mem'])
    reps = np.unique(topclu.rep)

    x=60
    y=30
    z=10

    l=len(reps)

    train, test, val = list(reps[:int(x*l/100)]), list(reps[int(x*l/100):int(y*l/100)+int(x*l/100)]), list(reps[int(y*l/100)+int(x*l/100):])

    return train, test, val


# split and backtrack clusters
def split(file, out_dir, steps, train, test, val):

    for group in [train, test, val]:
        for i in range(steps, 0, -1):
            clu = pd.read_csv("data/cluster/clu" + str(i) +"_cluster.tsv", sep='\t', names=['rep', 'mem'])
            members = []
            for elem in group:
                members.append(clu['mem'].values[clu['rep']==elem])

            for elem in members:
                for e in elem:
                    group.append(e)


    return train, test, val


# test for correct proportion size of datasets
def proportion_test(file, train, test, val):

    change = 0

    ids = []
    for seq in SeqIO.parse(file, "fasta"):
        ids.append(seq.id)

    filelen = len(ids)

    x = 60; y=30; z=10

    xf = int(x*filelen/100)
    yf = int(y*filelen/100)
    zf = int(z*filelen/100)

    if (len(train) < xf-10) | (len(train) > xf+10):
        change = 1
    elif (len(test) < yf-10) | (len(test) > yf+10):
        change = 2
    elif (len(val) < zf-10) | (len(val) > zf+10):
        change = 3

    return change, filelen


def clean_and_save(file, change, length, steps, train, test, val):

    train = list(set(train))
    test = list(set(test))
    val = list(set(val))

    # test if sets need to be resampled
    if  change > 0:
        resample(file, change, length, steps, train, test, val)
        # if so, which one?

    finaltrain = pd.DataFrame([elem[:4] for elem in train])
    finaltrain.to_csv("data/finalclusters/trainlist.csv")

    finaltest = pd.DataFrame([elem[:4] for elem in test])
    finaltest.to_csv("data/finalclusters/testlist.csv")

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


###################

def main():
    """
    please pass :
        1) directory of fasta file to be split
        2) #clustering steps for mmseqs2
    """

    file = load_data(file_dir)
    clustering(file, 'data/cluster/clu', 'data/cluster/tmp', steps=steps)
    trainset, testset, valset = splittop(steps)
    train, test, val = split(file_dir, 'data/cluster/clu', steps, trainset, testset, valset)

    #check for correct proportions
    change, filelength = proportion_test(file_dir, list(set(train)), list(set(test)), list(set(val)))

    ftrain, ftest, fval = clean_and_save(file_dir, change, filelength, steps, train, test, val)
    split_fasta(file_dir, ftrain, ftest, fval)



if __name__ == "__main__":
    print("starting")
    main()
