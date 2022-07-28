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

from datastructures import Environment


def call_mmseqs_clustering(mmseqs2_path, fasta_file, output_path, tmp_folder, seq_id_threshold, silenced = True):

    cmds = [mmseqs2_path, 'easy-linclust', fasta_file, output_path, tmp_folder, '--similarity-type', '2', '--cov-mode', '0', '-c', '1.0', '--min-seq-id', str(seq_id_threshold)]

    if silenced:
        FNULL = open(os.devnull, 'w')
        p = subprocess.Popen(cmds, stdout=FNULL)
    else:
        p = subprocess.Popen(cmds)
    p.wait()

# use mmseqs to cluster x times, cluster directory, tmp
def clustering(env):
    """
    file_dir : directory of file to be clustered
    clu_dir : directory where clusters are saved
    tmp : tmp directory needed for mmseqs
    steps : number of cluster iterations

    ---

    clustering hierarchically - result are multiple cluster files in mmseqs style
        -> .tsv, .fasta

    """
    # lowering the sequence identity for clustering with each step
    seq_id = np.linspace(0.8, 0.99, num=env.steps, dtype=float)[::-1]

    #step 1 cluster dataset
    call_mmseqs_clustering('mmseqs', env.input_file, f'{env.out_dir}/1', f'{env.tmp_folder}', seq_id[0])

    # do again with representatives of clusters for 'steps' iterations
    for i in range(env.steps):
        call_mmseqs_clustering('mmseqs', f'{env.out_dir}/{str(i+1)}_rep_seq.fasta', f'{env.out_dir}/{str(i+1)}', f'{env.tmp_folder}', seq_id[i])

    return None


def resample(env, change, length, train, test, val):
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

    x = env.tr_size; y=env.te_size; z=100-env.tr_size-env.te_size

    xf = int(x*length/100)
    yf = int(y*length/100)
    zf = int(z*length/100)

    if change == 1:
        if (len(train) < xf-int(10*length/100)):
            train, test, val = splittop(env)
            train.append(test[-1])
            test = test.pop(-1)
            train, test, val = split(env, train, test, val)

        elif (len(train) > xf+int(10*length/100)):
            train, test, val = splittop(env)
            test.append(train[-1])
            train = train.pop(-1)
            train, test, val = split(env, train, test, val)


    elif change == 2:
        if (len(test) < yf-int(10*length/100)):
            train, test, val = splittop(env)
            test.append(train[-1])
            train = train.pop(-1)
            train, test, val = split(env, train, test, val)

        elif (len(test) > yf+int(10*length/100)):
            train, test, val = splittop(env)
            train.append(test[-1])
            test = test.pop(-1)
            train, test, val = split(env, train, test, val)


    elif change == 3:
        if (len(val) < zf-int(10*length/100)):
            train, test, val = splittop(env)
            val.append(test[-1])
            test = test.pop(-1)
            train, test, val = split(env, train, test, val)

        elif (len(val) > zf+int(10*length/100)):
            train, test, val = splittop(env)
            test.append(val[-1])
            val = val.pop(-1)
            train, test, val = split(env, train, test, val)

    return train, test, val



def splittop(env):
    #split only the topcluster and give lists of reps to split()
    topclu = pd.read_csv(f'{env.out_dir}/{env.steps}_cluster.tsv', sep='\t', names=['rep', 'mem'])
    reps = np.unique(topclu.rep)

    x=env.tr_size # train
    y=env.te_size # test
    z=100-env.tr_size-env.te_size # val

    l=len(reps)

    train, test, val = list(reps[:int(x*l/100)]), list(reps[int(x*l/100):int(y*l/100)+int(x*l/100)]), list(reps[int(y*l/100)+int(x*l/100):])

    return train, test, val



def split(env, train, test, val):
    #  backtrack members of the respective sets throughout the lower clusters

    for group in [train, test, val]:
        for i in range(env.steps, 0, -1):
            clu = pd.read_csv(f'{env.out_dir}/{env.steps}_cluster.tsv', sep='\t', names=['rep', 'mem'])
            members = []

            for elem in group:
                members.append(clu['mem'].values[clu['rep']==elem])

            for elem in members:
                for e in elem:
                    group.append(e)


    return train, test, val


# test for correct proportion size of datasets
def proportion_test(env, train, test, val):

    change = 0

    ids = []
    for seq in SeqIO.parse(env.input_file, "fasta"):
        ids.append(seq.id)

    filelen = len(ids)

    x = env.tr_size; y=env.te_size; z=100-env.tr_size-env.te_size

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


def insane(train, test, val):
    #checks for insanity - if any sequences are in multiple sets, they are deleted from both
    doubles = []

    for elem in train:
        if elem in test:
            doubles.append(elem)
        elif elem in val:
            doubles.append(elem)

    for elem in test:
        if elem in val:
            doubles.append(elem)

    for d in doubles:
        train = [id for id in train if id!=d]
        test = [id for id in test if id!=d]
        val = [id for id in val if id!=d]

    #return sets without any doubles
    return train, test, val


def clean_and_save(env, train, test, val):

    ntrain = list(set(train))
    notest = list(set(test))
    nval = list(set(val))

    train, test, val = insane(ntrain, ntest, nval)

    finaltrain = pd.DataFrame([elem[:4] for elem in train])
    finaltrain.to_csv(env.out_dir+"/trainlist.csv")

    finaltest = pd.DataFrame([elem[:4] for elem in test])
    finaltest.to_csv(env.out_dir+"/testlist.csv")

    finalval = pd.DataFrame([elem[:4] for elem in val])
    finalval.to_csv(env.out_dir+"/vallist.csv")

    return finaltrain[0].tolist(), finaltest[0].tolist(), finalval[0].tolist()


def split_fasta(env, train, test, val):
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

    for seq in SeqIO.parse(env.input_file, "fasta"):
        rec = SeqRecord(seq.seq, id=seq.id, description=seq.description)
        if seq.id[:4] in train:
            train_fasta.append(rec)
        if seq.id[:4] in test:
            test_fasta.append(rec)
        if seq.id[:4] in val:
            val_fasta.append(rec)

    with open(env.out_dir+"/trainfasta.fasta", "w") as handle:
        SeqIO.write(train_fasta, handle, "fasta")
    with open(env.out_dir+"/testfasta.fasta", "w") as handle:
        SeqIO.write(test_fasta, handle, "fasta")
    with open(env.out_dir+"/valfasta.fasta", "w") as handle:
        SeqIO.write(val_fasta, handle, "fasta")

    return None


def cleanup():
    DIR = os.getcwd()
    deleteItem=False
    for filename in os.listdir(DIR):
        if 'tmp' in filename:
            deleteItem=True
        elif '.fasta' in filename:
            deleteItem=True
            break
        if deleteItem:
            shutil.rmtree(filename)

    return None

###################

def main():

    parser = argparse.ArgumentParser(
        prog = 'SCALA',
        description = "this tool helps providing the most challenging dataset split for a machine learning model in order to prevent information leakage and improve generalizability",
        epilog = "enjoy :)"
    )
    # parser.add_argument("-h", help="please give directory to input dataset and output directory - other settings are optional")
    parser.add_argument("-i", help="directory to input file (FASTA/FASTQ)", required=True, dest='input', action='store')
    parser.add_argument("-s", help="steps to be clustered - default = 4", default=4, dest='steps', action='store', type=int)
    parser.add_argument("-o", help="directory to save the results in", required=True, dest='output', action='store')
    parser.add_argument("-f", help="optional fasta file output (y/n) - default False", default='n', dest='fasta', action='store', type=str)
    parser.add_argument("-tr", help="size of training set - default ~60%", default=60, dest='tr_size', action='store', type=int)
    parser.add_argument("-te", help="size of test set - default ~30%", default=30, dest='te_size', action='store', type=int)
    args = parser.parse_args()

    env = Environment(args.input, args.steps, args.output, args.fasta, args.tr_size, args.te_size)

    clustering(env)
    trainset, testset, valset = splittop(env)
    train, test, val = split(env, trainset, testset, valset)

    #check for correct proportions

    change, filelength = proportion_test(env, list(set(train)), list(set(test)), list(set(val)))

    while change > 0:
        print("got here - need to resample", change)
        train, test, val = resample(env, change, filelength, train, test, val)
        change, filelength = proportion_test(env, list(set(train)), list(set(test)), list(set(val)))

    ftrain, ftest, fval = clean_and_save(env, train, test, val)

    if fasta_store == 'y':
        split_fasta(env, ftrain, ftest, fval)

    cleanup()

if __name__ == "__main__":
    print("starting")
    main()
