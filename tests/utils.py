import os
import shutil


def read_tsv(filepath):
    assert os.path.exists(filepath)
    with open(filepath, "r") as d:
        mols = [line.strip().split("\t") for line in d.readlines()]
    os.remove(filepath)
    return mols


def check_folder(output_root, epsilon, e_weight, f_weight, e_filename, f_filename):
    e_map, f_map = None, None
    if e_weight is not None:
        with open(e_weight, "r") as in_data:
            e_map = dict((k, float(v)) for k, v in [tuple(line.strip().split("\t")[:2]) for line in in_data.readlines()])
    if f_weight is not None:
        with open(f_weight, "r") as in_data:
            f_map = dict((k, float(v)) for k, v in [tuple(line.strip().split("\t")[:2]) for line in in_data.readlines()])

    split_data = []
    if os.path.isfile(os.path.join(output_root, "inter.tsv")):
        split_data.append(("I", read_tsv(os.path.join(output_root, "inter.tsv"))))
    if e_filename is not None and os.path.isfile(os.path.join(output_root, e_filename)):
        split_data.append(("E", read_tsv(os.path.join(output_root, e_filename))))
    if f_filename is not None and os.path.isfile(os.path.join(output_root, f_filename)):
        split_data.append(("F", read_tsv(os.path.join(output_root, f_filename))))

    assert len(split_data) > 0

    for n, data in split_data:
        splits = list(zip(*data))
        if n == "E" and e_map is not None:
            trains = sum(e_map[e] for e, s in data if s == "train")
            tests = sum(e_map[e] for e, s in data if s == "test")
        elif n == "F" and f_map is not None:
            trains = sum(f_map[f] for f, s in data if s == "train")
            tests = sum(f_map[f] for f, s in data if s == "test")
        else:
            trains, tests = splits[-1].count("train"), splits[-1].count("test")
        train_frac, test_frac = trains / (trains + tests), tests / (trains + tests)
        assert 0.7 * (1 - epsilon) <= train_frac <= 0.7 * (1 + epsilon)
        assert 0.3 * (1 - epsilon) <= test_frac <= 0.3 * (1 + epsilon)
        if n == "I":
            break

    shutil.rmtree(output_root)
