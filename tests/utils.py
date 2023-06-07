import os
import shutil
from typing import List

from datasail.sail import sail


def read_tsv(filepath):
    assert os.path.exists(filepath)
    with open(filepath, "r") as d:
        mols = [line.strip().split("\t") for line in d.readlines()]
    os.remove(filepath)
    return mols


def run_sail(
        inter=None, output: str = "", max_sec: int = 100, max_sol: int = 1000, verbosity: str = "I",
        splits: List[float] = None, names: List[str] = None, epsilon: float = 0.05, runs: int = 1, solver: str = "MOSEK",
        techniques: List[str] = None, vectorized: bool = True, cache: bool = False, cache_dir: str = None,
        e_type: str = None, e_data=None, e_weights=None, e_sim=None, e_dist=None, e_args: str = "",
        e_max_sim: float = 1.0, e_max_dist: float = 1.0, f_type: str = None, f_data=None, f_weights=None, f_sim=None,
        f_dist=None, f_args: str = "", f_max_sim: float = 1.0, f_max_dist: float = 1.0, threads: int = 1,
):
    sail(
        inter=inter, output=output, max_sec=max_sec, max_sol=max_sol, verbosity=verbosity, techniques=techniques,
        vectorized=vectorized, splits=splits, names=names, epsilon=epsilon, runs=runs, e_type=e_type, e_data=e_data,
        e_weights=e_weights, e_sim=e_sim, e_dist=e_dist, e_max_sim=e_max_sim, e_max_dist=e_max_dist,
        e_args=e_args, f_type=f_type, f_data=f_data, f_weights=f_weights, f_sim=f_sim, f_dist=f_dist,
        f_max_sim=f_max_sim, f_max_dist=f_max_dist, f_args=f_args, cache=cache, cache_dir=cache_dir,
        solver=solver, threads=threads,
    )


def check_folder(output_root, epsilon, e_weight, f_weight, e_filename, f_filename):
    e_map, f_map = None, None
    if e_weight is not None:
        with open(e_weight, "r") as in_data:
            e_map = dict((k, float(v)) for k, v in [tuple(line.strip().split("\t")[:2]) for line in in_data.readlines()[1:]])
    if f_weight is not None:
        with open(f_weight, "r") as in_data:
            f_map = dict((k, float(v)) for k, v in [tuple(line.strip().split("\t")[:2]) for line in in_data.readlines()[1:]])

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
