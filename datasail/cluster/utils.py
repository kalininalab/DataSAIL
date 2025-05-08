import argparse
from pydoc import locate
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt
import yaml

from datasail.reader.utils import DataSet
from datasail.constants import LOGGER, YAML_FILE_NAMES


def cluster_param_binary_search(
        dataset: DataSet,
        init_args: tuple,
        min_args: tuple,
        max_args: tuple,
        user_args: str,
        threads: int,
        trial: Callable,
        args2str: Callable,
        gen_args: Callable,
        log_dir: Path,
) -> tuple[list[str], dict[str, str], np.ndarray]:
    """
    Perform binary search on the parameter space for clustering algorithms. So far, this is used to find optimal number
    of clusters for CD-HIT and MMseqs2.

    Args:
        dataset: The dataset to cluster on.
        init_args: initial arguments for optimization.
        min_args: The lower bound for the arguments.
        max_args: The upper bound for the arguments.
        user_args: Additional arguments that the user may have provided.
        threads: Number of threads to be used by the clustering algorithm.
        trial: Callable method running the actual clustering algorithm.
        args2str: Convert arguments to string to include them in filenames.
        gen_args: A callable function that generates a new argument configuration for the binary search. Has to be
            callable with two old parameter configurations.
        log_dir: Directory to store the logs.

    Returns:
        Return the cluster names, the mapping from names to cluster names, and a similarity or distance matrix
    """
    def args2log(x: tuple) -> Optional[Path]:
        """
        Compute the name of the log file based on the provided arguments.

        Args:
            x: Arguments used in the run we want to store the results for

        Returns:
            Path to the file to write the execution log to
        """
        user_str = ""
        if user_args != "":
            user_str = f"_{user_args.replace('-', '').replace(' ', '_')}"
        return None if not log_dir else log_dir / f"{dataset.get_name()}_{trial.__name__[:-6]}_" \
                                                  f"{args2str(x).replace('-', '').replace(' ', '_')}{user_str}.log"

    # cluster with the initial arguments
    cluster_names, cluster_map, cluster_sim = trial(
        dataset,
        args2str(init_args),
        user_args,
        threads,
        args2log(init_args)
    )
    num_clusters = len(cluster_names)
    LOGGER.info(f"First round of clustering found {num_clusters} clusters for {len(dataset.names)} samples.")

    # there are too few clusters, rerun with maximal arguments which has to result in every sample becomes a cluster
    if num_clusters <= 10:
        min_args = init_args
        min_clusters = num_clusters
        min_cluster_names, min_cluster_map, min_cluster_sim = cluster_names, cluster_map, cluster_sim
        max_cluster_names, max_cluster_map, max_cluster_sim = dataset.names, dict(
            (n, n) for n in dataset.names), np.zeros((len(dataset.names), len(dataset.names)))
        max_clusters = len(max_cluster_names)
        LOGGER.info(f"Second round of clustering found {max_clusters} clusters for {len(dataset.names)} samples.")

    # if the number of clusters ranges in a good window, return the result
    elif 10 < num_clusters <= 100:
        return cluster_names, cluster_map, cluster_sim

    # too many clusters have been found, rerun the clustering with minimal arguments to find the lower bound of clusters
    else:
        max_args = init_args
        max_clusters = num_clusters
        max_cluster_names, max_cluster_map, max_cluster_sim = cluster_names, cluster_map, cluster_sim
        min_cluster_names, min_cluster_map, min_cluster_sim = \
            trial(dataset, args2str(min_args), user_args, threads, args2log(min_args))
        min_clusters = len(min_cluster_names)
        LOGGER.info(f"Second round of clustering found {min_clusters} clusters for {len(dataset.names)} samples.")

    # if the minimal number of clusters is in the target window, return them
    if 10 < min_clusters <= 100:
        return min_cluster_names, min_cluster_map, min_cluster_sim

    # if the maximal number of clusters is in the target window, return them
    if 10 < max_clusters <= 100:
        return max_cluster_names, max_cluster_map, max_cluster_sim

    # if the maximal number of clusters is still less than the lower bound of the window, report and warn
    if max_clusters < 10:
        LOGGER.warning(f"{trial.__name__[:-6]} cannot optimally cluster the data. The maximal number of clusters is "
                       f"{max_clusters}.")
        return max_cluster_names, max_cluster_map, max_cluster_sim

    # if the minimal number of clusters is still more than the upper bound of the window, report and warn
    if 100 < min_clusters:
        LOGGER.warning(f"{trial.__name__[:-6]} cannot optimally cluster the data. The minimal number of clusters is "
                       f"{min_clusters}.")
        return min_cluster_names, min_cluster_map, min_cluster_sim

    # for 8 rounds, apply binary search on the variable parameter space and try to hit the target window
    iteration_count = 0
    while True:
        iteration_count += 1
        args = gen_args(min_args, max_args)
        cluster_names, cluster_map, cluster_sim = trial(dataset, args2str(args), user_args, threads, args2log(args))
        num_clusters = len(cluster_names)
        LOGGER.info(f"Next round of clustering ({iteration_count + 2}.) "
                    f"found {num_clusters} clusters for {len(dataset.names)} samples.")
        if num_clusters <= 10 and iteration_count < 8:
            min_args = args
        elif 10 < num_clusters <= 100 or iteration_count >= 8:
            return cluster_names, cluster_map, cluster_sim
        else:
            max_args = args


def heatmap(matrix: np.ndarray, output_file: Path) -> None:
    """
    Create a heatmap from a numpy array and two lists of labels.

    Args:
        matrix: A 2D numpy array of shape (M, N).
        output_file: Filename to store the matrix in.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(matrix)
    ax.figure.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(output_file)
    plt.clf()


class MultiYAMLParser(argparse.ArgumentParser):
    def __init__(self, algo_name):
        """
        Initialize the argument parser for DataSAIL. This is a wrapper around the standard argparse.ArgumentParser.

        Args:
            algo_name: Name of the algorithm to parse arguments for.
        """
        super().__init__()
        self.fos_map = {}
        if algo_name is not None:
            self.add_yaml_arguments(YAML_FILE_NAMES[algo_name])

    def parse_args(self, args: Optional[Sequence[str]] = ...) -> argparse.Namespace:
        """
        Parse the arguments provided by the user. This prepends some preprocessing to the arguments before sending them
        to the actual parsing.

        Args:
            args: Arguments provided by the user.

        Returns:
            Namespace of the parsed arguments.
        """
        if isinstance(args, str):
            if " " in args:
                args = args.split(" ")
            elif len(args) > 0:
                args = [args]
        elif args is None:
            args = ""
        return super().parse_args(args)

    def add_yaml_arguments(self, yaml_filepath: Path) -> None:
        """
        Add arguments to the parser based on a YAML file.

        Args:
            yaml_filepath: Path to the YAML file to read the arguments from.
        """
        with open(Path(__file__).parent.resolve() / yaml_filepath, "r") as data:
            data = yaml.safe_load(data)
        for name, values in data.items():
            kwargs = {"dest": name.replace("-", "_"), "type": locate(values["type"])}
            if kwargs["type"] == bool:
                if not values["default"]:
                    kwargs.update({"action": "store_true", "default": False})
                else:
                    kwargs.update({"action": "store_false", "default": True})
                del kwargs["type"]
            else:
                if values["cardinality"] != 0:
                    kwargs["nargs"] = values["cardinality"]
                if "default" in values:
                    kwargs["default"] = values["default"]
            self.fos_map[name.replace("-", "_")] = values.get("fos", 0)
            super().add_argument(
                *values["calls"],
                **kwargs,
            )

    def get_user_arguments(self, args: argparse.Namespace, ds_args: list[str], fos: Literal[0, 1] = 0) -> str:
        """
        Get the arguments that the user provided to the program that differ from default values.

        Args:
            args: Arguments provided by the user.
            ds_args: Arguments that are optimized by DataSAIL and extracted differently.
            fos:

        Returns:
            String representation of the arguments that the user provided for the program to be passed to subprograms.
        """
        cleaned_args = namespace_diff(args, self.parse_args([]))  # the non-standard arguments
        action_map = {action.dest: action.option_strings[0] for action in self._actions}
        fos = {fos, 2}

        for key in ds_args:
            if key in cleaned_args:
                del cleaned_args[key]

        return " ".join([f"{action_map[key]} {value}" for key, value in cleaned_args.items() if self.fos_map[key] in fos])


def namespace_diff(a: argparse.Namespace, b: argparse.Namespace) -> dict:
    """
    Get the difference between two namespaces.

    Args:
        a: First namespace to compare.
        b: Second namespace to compare.

    Returns:
        Dictionary of all attributes that are different between the two namespaces.
    """
    output = {}
    if a is None:
        return output
    for key, value in vars(a).items():
        if not hasattr(b, key) or getattr(b, key) != value:
            output[key] = value
    return output
