import os
import shutil


def read_tsv(filepath):
    assert os.path.exists(filepath)
    with open(filepath, "r") as d:
        mols = [line.strip().split("\t") for line in d.readlines()]
    os.remove(filepath)
    return mols


def check_folder(output_root, limit):
    split_data = []
    if os.path.exists(os.path.join(output_root, "inter.tsv")):
        split_data.append(read_tsv(os.path.join(output_root, "inter.tsv")))
    if os.path.exists(os.path.join(output_root, "proteins.tsv")):
        split_data.append(read_tsv(os.path.join(output_root, "proteins.tsv")))
    if os.path.exists(os.path.join(output_root, "drugs.tsv")):
        split_data.append(read_tsv(os.path.join(output_root, "drugs.tsv")))

    assert len(split_data) > 0

    for data in split_data:
        splits = list(zip(*data))[-1]
        trains, tests = splits.count("train"), splits.count("test")
        train_frac, test_frac = trains / (trains + tests), tests / (trains + tests)
        assert 0.7 * (1 - limit) <= train_frac <= 0.7 * (1 + limit)
        assert 0.3 * (1 - limit) <= test_frac <= 0.3 * (1 + limit)

    shutil.rmtree(output_root)
