import pickle

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def write():
    with open("drugs.tsv", "r") as f:
        data = {}
        for line in f.readlines()[1:]:
            k, v = line.strip().split("\t")
            data[k] = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(v), 2, nBits=1024))

    with open("drugs.pkl", "wb") as f:
        pickle.dump(data, f)

    with h5py.File("drugs.h5", "w") as f:
        for k, v in data.items():
            f[k] = v


def read():
    print("========================\nPickle\n========================")
    with open("drugs.pkl", "rb") as f:
        data = pickle.load(f)
        print(data)

    print("========================\nHDF5\n========================")
    with h5py.File("drugs.h5") as file:
        data = {k: np.array(file[k]) for k in file.keys()}
        print(data)


write()
read()
