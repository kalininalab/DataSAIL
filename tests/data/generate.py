import random


def generate(train_frac=0.8, size=100, folder="perf_80_20"):

    with open(folder + "/inter.tsv", "w") as inter, open(folder + "/lig.tsv", "w") as lig, open(folder + "/prot.tsv", "w") as prot:
        for d in range(int(train_frac * size)):
            print(f"D{d + 1:05}\tCCC", file=lig)
            for p in range(int(train_frac * size)):
                if d == 0:
                    print(f"P{p + 1:05}\tAAT", file=prot)
                print(f"D{d + 1:05}\tP{p + 1:05}\t{int(random.random() + 0.5)}", file=inter)
        for d in range(int(train_frac * size), size):
            print(f"D{d + 1:05}\tCCC", file=lig)
            for p in range(int(train_frac * size), size):
                if d == int(train_frac * size):
                    print(f"P{p + 1:05}\tAAT", file=prot)
                print(f"D{d + 1:05}\tP{p + 1:05}\t{int(random.random() + 0.5)}", file=inter)


generate(0.6, 10, "perf_8_2")
