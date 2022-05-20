import numpy as np
import pandas as pd
import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import subprocess
import sys
import random



file_dir = sys.argv[1]
steps = int(sys.argv[2])
out_dir = sys.argv[3]

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
# if not os.path.isdir("data/finalclusters"):
#     os.makedirs("data/finalclusters")



# use mmseqs to cluster x times, cluster directory, tmp
def clustering(file_dir, clu_dir, tmp, steps):
    """
    file_dir : directory of file to be clustered
    out_dir : directory of results
    tmp : tmp directory needed for mmseqs
    steps : number of cluster iterations

    ---

    clustering hierarchically - result are multiple cluster files in mmseqs style
        -> .tsv, .fasta

    """
    # lowering the sequence identity for clustering with each step
    seq_id = np.linspace(0.6, 0.99, num=steps, dtype=float)[::-1]

    #step 1 cluster dataset
    cmd = "mmseqs easy-linclust " + file_dir + " " + clu_dir + str(1) + " " + tmp + str(1)  + " --similarity-type 2 --cov-mode 0 -c 1.0" + " --min-seq-id " + str(seq_id[0])

    proc_out = subprocess.run(cmd, shell=True)

    # do again with representatives of clusters for 'steps' iterations
    for i in range(steps):
        cmd = "mmseqs easy-linclust " + clu_dir + str(i)+ "_rep_seq.fasta" + " " + clu_dir + str(i+1) + " " + tmp + str(i+1) + " --cov-mode 0 -c 1.0"+ " --min-seq-id " + str(seq_id[i]) + " --similarity-type 2 -k 13"
        subprocess.run(cmd, shell=True)

    return None


def resample(file, out_dir, change, length, steps, train, test, val):
    """
    file: original file to be clustered
    out_dir: directory where to save files
    change: which set is unbalanced
    length: length of original file
    steps: clustering steps done
    train, test, val: prev. split sets

    --if the length of train samples vs test / val is not according to the percentage
    --specified above, we resample the clusters - eg. add one cluster from test to train
    """

    x = 60; y=30; z=10

    xf = int(x*length/100)
    yf = int(y*length/100)
    zf = int(z*length/100)

    if change == 1:
        if (len(train) < xf-int(15*length/100)):
            train, test, val = splittop(out_dir, steps)
            train.append(test[-1])
            test = test.pop(-1)
            train, test, val = split(file, out_dir, steps, train, test, val)

        elif (len(train) > xf+int(15*length/100)):
            train, test, val = splittop(out_dir, steps)
            test.append(train[-1])
            train = train.pop(-1)
            train, test, val = split(file, out_dir, steps, train, test, val)


    elif change == 2:
        if (len(test) < yf-int(15*length/100)):
            train, test, val = splittop(out_dir, steps)
            test.append(train[-1])
            train = train.pop(-1)
            train, test, val = split(file, out_dir, steps, train, test, val)

        elif (len(test) > yf+int(15*length/100)):
            train, test, val = splittop(out_dir, steps)
            train.append(test[-1])
            test = test.pop(-1)
            train, test, val = split(file, out_dir, steps, train, test, val)


    elif change == 3:
        if (len(val) < zf-int(15*length/100)):
            train, test, val = splittop(out_dir, steps)
            val.append(test[-1])
            test = test.pop(-1)
            train, test, val = split(file, out_dir, steps, train, test, val)

        elif (len(val) > zf+int(15*length/100)):
            train, test, val = splittop(out_dir, steps)
            test.append(val[-1])
            val = val.pop(-1)
            train, test, val = split(file, out_dir, steps, train, test, val)

    return train, test, val



def splittop(out_dir, steps):
    #split only the topcluster and give lists of reps to split()
    topclu = pd.read_csv(out_dir + str(steps) + "_cluster.tsv", sep='\t', names=['rep', 'mem'])
    reps = np.unique(topclu.rep)

    x=60 # train
    y=30 # test
    z=10 # val

    l=len(reps)

    train, test, val = list(reps[:int(x*l/100)]), list(reps[int(x*l/100):int(y*l/100)+int(x*l/100)]), list(reps[int(y*l/100)+int(x*l/100):])

    return train, test, val



def split(file, out_dir, steps, train, test, val):
    #  backtrack members of the respective sets throughout the lower clusters

    for group in [train, test, val]:
        for i in range(steps, 0, -1):
            clu = pd.read_csv(out_dir + str(i) +"_cluster.tsv", sep='\t', names=['rep', 'mem'])
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

    if (len(train) < xf-int(15*filelen/100)) | (len(train) > xf+int(15*filelen/100)):
        change = 1
    elif (len(test) < yf-int(15*filelen/100)) | (len(test) > yf+int(15*filelen/100)):
        change = 2
    elif (len(val) < zf-int(15*filelen/100)) | (len(val) > zf+int(15*filelen/100)):
        change = 3

    return change, filelen


def clean_and_save(file, out_dir, steps, train, test, val):

    train = list(set(train))
    test = list(set(test))
    val = list(set(val))

    finaltrain = pd.DataFrame([elem[:4] for elem in train])
    finaltrain.to_csv(out_dir+"/trainlist.csv")

    finaltest = pd.DataFrame([elem[:4] for elem in test])
    finaltest.to_csv(out_dir+"/testlist.csv")

    finalval = pd.DataFrame([elem[:4] for elem in val])
    finalval.to_csv(out_dir+"/vallist.csv")

    return finaltrain[0].tolist(), finaltest[0].tolist(), finalval[0].tolist()


def split_fasta(file, out_dir, train, test, val):
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

    with open(out_dir+"/trainfasta.fasta", "w") as handle:
        SeqIO.write(train_fasta, handle, "fasta")
    with open(out_dir+"/testfasta.fasta", "w") as handle:
        SeqIO.write(test_fasta, handle, "fasta")
    with open(out_dir+"/valfasta.fasta", "w") as handle:
        SeqIO.write(val_fasta, handle, "fasta")

    return None


###################

def main():
    """
    please pass :
        1) directory of fasta file to be split
        2) #clustering steps for mmseqs2
        3) directory to save files
    """

    clustering(file_dir, out_dir, out_dir+'_tmp', steps)
    trainset, testset, valset = splittop(out_dir, steps)
    train, test, val = split(file_dir, out_dir, steps, trainset, testset, valset)

    #check for correct proportions

    change, filelength = proportion_test(file_dir, list(set(train)), list(set(test)), list(set(val)))

    while change > 0:
        print("got here - need to resample", change)
        train, test, val = resample(file_dir, out_dir, change, filelength, steps, train, test, val)
        change, filelength = proportion_test(file_dir, list(set(train)), list(set(test)), list(set(val)))

    ftrain, ftest, fval = clean_and_save(file_dir, out_dir, steps, train, test, val)
    split_fasta(file_dir, out_dir, ftrain, ftest, fval)



if __name__ == "__main__":
    print("starting")
    main()
