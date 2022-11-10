import os

import numpy as np

from scala.ilp_split.read_data import read_data


def ilp_main(args):
    data = read_data(args)
    output = {"inter": None, "drug": None, "prot": None}

    if args.technique == "R":
        output["inter"] = list(sample_categorical(data["interactions"], args.splits))
    if args.technique == "ICD":
        pass
    if args.technique == "ICP":
        pass
    if args.technique == "IC":
        pass
    if args.technique == "CCD":
        pass
    if args.technique == "CCP":
        pass
    if args.technique == "CC":
        pass

    if output["inter"] is not None:
        with open(os.path.join(args.output, "inter.tsv"), "w") as stream:
            for s, split in enumerate(output["inter"]):
                for drug, prot in split:
                    print(drug, prot, args.names[s], sep=args.sep, file=stream)
    if output["drug"] is not None:
        with open(os.path.join(args.output, "drug.tsv"), "w") as stream:
            for s, split in enumerate(output["drug"]):
                for drug in split:
                    print(drug, args.names[s], sep=args.sep, file=stream)
    if output["prot"] is not None:
        with open(os.path.join(args.output, "prot.tsv"), "w") as stream:
            for s, split in enumerate(output["prot"]):
                for prot in split:
                    print(prot, args.names[s], sep=args.sep, file=stream)


def sample_categorical(data, splits):
    np.random.shuffle(data)
    for i in range(len(splits) - 1):
        yield data[int(sum(splits[:i]) * len(data)):int(sum(splits[:(i + 1)]) * len(data))]
    yield data[int(sum(splits[:-1]) * len(data)):]
