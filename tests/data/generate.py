import os.path
import random

import numpy as np


def generate(train_frac=0.8, size=100, folder="perf_80_20"):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    train_frac = (train_frac - (train_frac - train_frac ** 2) ** 0.5) / (2 * train_frac - 1)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    train_size = int(train_frac * size)
    test_size = size - train_size
    with open(folder + "/inter.tsv", "w") as inter, open(folder + "/lig.tsv", "w") as lig, \
            open(folder + "/prot.tsv", "w") as prot, open(folder + "/prot.fasta", "w") as prot_fasta:
        print("Drug_ID\tTarget_ID\tY", file=inter)
        print("Drug_ID\tSMILES", file=lig)
        print("Target_ID\taaseq", file=prot)
        for d in range(train_size):
            print(f"D{d + 1:05}\tCCC", file=lig)
            for p in range(train_size):
                if d == 0:
                    print(f"P{p + 1:05}\tAAT", file=prot)
                    print(f">P{p + 1:05}\nAAT", file=prot_fasta)
                print(f"D{d + 1:05}\tP{p + 1:05}\t{int(random.random() + 0.5)}", file=inter)
        for d in range(train_size, size):
            print(f"D{d + 1:05}\tCCC", file=lig)
            for p in range(train_size, size):
                if d == train_size:
                    print(f"P{p + 1:05}\tAAT", file=prot)
                    print(f">P{p + 1:05}\nAAT", file=prot_fasta)
                print(f"D{d + 1:05}\tP{p + 1:05}\t{int(random.random() + 0.5)}", file=inter)

        prot_sims, drug_sims = np.zeros((size, size)), np.zeros((size, size))
        prot_sims[:train_size, :train_size] = np.random.normal(0.8, 0.2, (train_size, train_size))
        prot_sims[train_size:, train_size:] = np.random.normal(0.8, 0.2, (test_size, test_size))
        prot_sims[train_size:, :train_size] = np.random.normal(0.2, 0.2, (test_size, train_size))
        prot_sims[:train_size, train_size:] = np.random.normal(0.2, 0.2, (train_size, test_size))
        drug_sims[:train_size, :train_size] = np.random.normal(0.8, 0.2, (train_size, train_size))
        drug_sims[train_size:, train_size:] = np.random.normal(0.8, 0.2, (test_size, test_size))
        drug_sims[train_size:, :train_size] = np.random.normal(0.2, 0.2, (test_size, train_size))
        drug_sims[:train_size, train_size:] = np.random.normal(0.2, 0.2, (train_size, test_size))
        drug_sims = np.minimum(np.maximum(drug_sims, np.zeros_like(drug_sims)), np.ones_like(drug_sims))
        prot_sims = np.minimum(np.maximum(prot_sims, np.zeros_like(prot_sims)), np.ones_like(prot_sims))
        with open(folder + "/lig_sim.tsv", "w") as lig_sim, open(folder + "/prot_sim.tsv", "w") as prot_sim:
            for i in range(size + 1):
                for j in range(size + 1):
                    if i == j == 0:
                        print("X", end="", file=lig_sim)
                        print("X", end="", file=prot_sim)
                    elif i == 0:
                        print(f"\tD{j:05}", file=lig_sim, end="")
                        print(f"\tP{j:05}", file=prot_sim, end="")
                    elif j == 0:
                        print(f"D{i:05}", file=lig_sim, end="")
                        print(f"P{i:05}", file=prot_sim, end="")
                    else:
                        print(f"\t{drug_sims[i - 1, j - 1]:.5f}", end="", file=lig_sim)
                        print(f"\t{prot_sims[i - 1, j - 1]:.5f}", end="", file=prot_sim)
                print(file=lig_sim)
                print(file=prot_sim)
