import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Callable, Generator, Literal, Optional, Union

import yaml

from datasail.parsers import parse_datasail_args
from datasail.version import __version__
from datasail.routine import datasail_main
from datasail.constants import CDHIT, CDHIT_EST, DATA_INPUT, DIAMOND, DIST_ALGOS, FOLDSEEK, FORMATTER, INSTALLED, KW_ARGS, KW_CACHE, KW_CACHE_DIR, KW_CC, KW_CLI, KW_CLUSTERS, KW_DATA, KW_DELTA, \
        KW_DIST, KW_EPSILON, KW_INTER, KW_LINKAGE, KW_LOGDIR, KW_MAX_SEC, KW_NAMES, KW_OUTDIR, KW_OVERFLOW, KW_RUNS, \
        KW_SIM, KW_SOLVER, KW_SPLITS, KW_STRAT, KW_TECHNIQUES, KW_THREADS, KW_VERBOSE, KW_WEIGHTS, LOGGER, MASH, MATRIX_INPUT, MMSEQS, SIM_ALGOS, SOLVER_SCIP, TMALIGN, VERB_MAP, ALGOS


def error(msg: str, cli: bool) -> None:
    """
    Print an error message with an individual error code to the commandline. Afterward, the program is ended.

    Args:
        msg: Error message
        error_code: Code of the error to identify it
        cli: boolean flag indicating that this program has been started from commandline
    """
    LOGGER.error(msg)
    if cli:
        exit(1)
    else:
        raise ValueError(msg)


def default_pathable_arg(kwargs: dict, key: str) -> Optional[Union[dict, Path]]:
    """
    Set a default path argument if not given.

    Args:
        kwargs: Arguments in kwargs-format
        key: Key of the argument to be set
    """
    if key not in kwargs:
        return None
    if isinstance(kwargs[key], str) and kwargs[key].lower() not in ALGOS:
        return Path(kwargs[key])
    return kwargs[key]


def validate_general_args(**kwargs) -> dict[str, object]:
    """
    Validate the arguments given to the program.

    Notes:
        next error code: 26

    Args:
        **kwargs: Arguments in kwargs-format

    Returns:
        The kwargs in case something has been adjusted, e.g. splits normalization or naming
    """
    # set default arguments
    kwargs[KW_OUTDIR] = Path(kwargs[KW_OUTDIR]) if KW_OUTDIR in kwargs else None
    kwargs[KW_VERBOSE] = kwargs.get(KW_VERBOSE, "I")
    kwargs[KW_MAX_SEC] = kwargs.get(KW_MAX_SEC, 1000)
    kwargs[KW_RUNS] = kwargs.get(KW_RUNS, 1)
    kwargs[KW_INTER] = default_pathable_arg(kwargs, KW_INTER)
    kwargs[KW_DELTA] = kwargs.get(KW_DELTA, 0.05)
    kwargs[KW_EPSILON] = kwargs.get(KW_EPSILON, 0.05)
    kwargs[KW_THREADS] = kwargs.get(KW_THREADS, 0)
    kwargs[KW_CACHE] = kwargs.get(KW_CACHE, False)
    kwargs[KW_CACHE_DIR] = kwargs.get(KW_CACHE_DIR, None)
    kwargs[KW_LINKAGE] = kwargs.get(KW_LINKAGE, "average")
    kwargs[KW_OVERFLOW] = kwargs.get(KW_OVERFLOW, "break")
    kwargs[KW_SOLVER] = kwargs.get(KW_SOLVER, SOLVER_SCIP)

    # create output directory
    if kwargs[KW_OUTDIR] is not None:
        kwargs[KW_OUTDIR] = Path(kwargs[KW_OUTDIR])
        if kwargs[KW_OUTDIR].is_dir():
            kwargs[KW_OUTDIR].mkdir(parents=True, exist_ok=True)
            LOGGER.warning("Output directory does not exist, DataSAIL creates it automatically")

    LOGGER.setLevel(VERB_MAP[kwargs[KW_VERBOSE]])
    LOGGER.handlers[0].setLevel(level=VERB_MAP[kwargs[KW_VERBOSE]])

    if kwargs[KW_OUTDIR] is not None:
        kwargs[KW_LOGDIR] = kwargs[KW_OUTDIR] / "logs"
        kwargs[KW_LOGDIR].mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(kwargs[KW_LOGDIR] / "general.log")
        file_handler.setLevel(level=VERB_MAP[kwargs[KW_VERBOSE]])
        file_handler.setFormatter(FORMATTER)
        LOGGER.addHandler(file_handler)
    else:
        kwargs[KW_LOGDIR] = None

    LOGGER.info("Validating arguments")

    # check splits to be more than 1 and their fractions sum up to 1 and check the names
    if kwargs[KW_TECHNIQUES] is None or (isinstance(kwargs[KW_TECHNIQUES], list) and len(kwargs[KW_TECHNIQUES]) == 0):
        error("No technique(s) specified to be used in the DataSAIL run. Please check the input again.", kwargs[KW_CLI])
    if kwargs[KW_SPLITS] is None or len(kwargs[KW_SPLITS]) < 2:
        error("Less then two splits requested. This is no useful input, please check the input again.", kwargs[KW_CLI])
    if kwargs[KW_NAMES] is None:
        LOGGER.warning("No names for the splits provided. The results splits will be called Split001, Split002, ...")
        kwargs[KW_NAMES] = [f"Split{x+1:03d}" for x in range(len(kwargs[KW_SPLITS]))]
    elif len(kwargs[KW_SPLITS]) != len(kwargs[KW_NAMES]):
        error("Different number of splits and names. You have to give the same number of splits and names for them.", kwargs[KW_CLI])
    elif len(kwargs[KW_NAMES]) != len(set(kwargs[KW_NAMES])):
        error("At least two splits will have the same name. Please check the naming of the splits again to have unique names", kwargs[KW_CLI])
    kwargs[KW_SPLITS] = [x / sum(kwargs[KW_SPLITS]) for x in kwargs[KW_SPLITS]]

    # check search termination criteria
    if kwargs[KW_MAX_SEC] < 1:
        error("The maximal search time must be a positive integer.", kwargs[KW_CLI])
    if kwargs[KW_THREADS] < 0:
        error("The number of threads to use has to be a non-negative integer.", kwargs[KW_CLI])
    if kwargs[KW_THREADS] == 0:
        kwargs[KW_THREADS] = os.cpu_count()
    else:
        kwargs[KW_THREADS] = min(kwargs[KW_THREADS], os.cpu_count())

    # check the interaction file
    if kwargs[KW_INTER] is not None and isinstance(kwargs[KW_INTER], Path) and not kwargs[KW_INTER].is_file():
        error("The interaction filepath is not valid.", kwargs[KW_CLI])

    # check the epsilon value
    if 1 < kwargs[KW_DELTA] or kwargs[KW_DELTA] < 0:
        error("Delta has to be a real value between 0 and 1.", kwargs[KW_CLI])

    # check the epsilon value
    if 1 < kwargs[KW_EPSILON] or kwargs[KW_EPSILON] < 0:
        error("Epsilon has to be a real value between 0 and 1.", kwargs[KW_CLI])

    # check number of runs to be a positive integer
    if kwargs[KW_RUNS] < 1:
        error("The number of runs cannot be lower than 1.", kwargs[KW_CLI])

    # check the input regarding the caching
    if kwargs[KW_CACHE] and kwargs[KW_CACHE_DIR] is not None:
        kwargs[KW_CACHE_DIR] = Path(kwargs[KW_CACHE_DIR])
        if not kwargs[KW_CACHE_DIR].is_dir():
            LOGGER.warning("Cache directory does not exist, DataSAIL creates it automatically.")
        kwargs[KW_CACHE_DIR].mkdir(parents=True, exist_ok=True)

    if kwargs[KW_LINKAGE] not in ["average", "single", "complete"]:
        error("The linkage method has to be one of 'mean', 'single', or 'complete'.", kwargs[KW_CLI])

    return kwargs


def validate_data_args(counter: int, **kwargs) -> dict[str, object]:
    """
    Validate the data-related arguments given to the program.
    
    Args:
        counter: Counter of the data to be validated, i.e., first, second, third, ... dimension
        **kwargs: Arguments in kwargs-format
    """
    kwargs[KW_DATA] = default_pathable_arg(kwargs, KW_DATA)
    kwargs[KW_WEIGHTS] = default_pathable_arg(kwargs, KW_WEIGHTS)
    kwargs[KW_STRAT] = default_pathable_arg(kwargs, KW_STRAT)
    kwargs[KW_SIM] = default_pathable_arg(kwargs, KW_SIM)
    kwargs[KW_DIST] = default_pathable_arg(kwargs, KW_DIST)
    kwargs[KW_CLUSTERS] = kwargs.get(KW_CLUSTERS, 50)
    kwargs[KW_ARGS] = kwargs.get(KW_ARGS, "")

    if kwargs[KW_DATA] is not None and isinstance(kwargs[KW_DATA], Path) and not kwargs[KW_DATA].exists():
        error(f"The filepath to data {counter} is invalid.", kwargs[KW_CLI])
    if kwargs[KW_WEIGHTS] is not None and isinstance(kwargs[KW_WEIGHTS], Path) and not kwargs[KW_WEIGHTS].is_file():
        error(f"The filepath to the weights of data {counter} is invalid.", kwargs[KW_CLI])
    if kwargs[KW_STRAT] is not None and isinstance(kwargs[KW_STRAT], Path) and not kwargs[KW_STRAT].is_file():
        error(f"The filepath to the stratification of data {counter} is invalid.", kwargs[KW_CLI])
    if kwargs[KW_SIM] is not None and isinstance(kwargs[KW_SIM], str) and kwargs[KW_SIM].lower() not in SIM_ALGOS:
        kwargs[KW_SIM] = Path(kwargs[KW_SIM])
        if not kwargs[KW_SIM].is_file():
            error(f"The similarity metric for data {counter} seems to be a file-input but the filepath is invalid.", kwargs[KW_CLI])
    if kwargs[KW_DIST] is not None and isinstance(kwargs[KW_DIST], str) and \
            kwargs[KW_DIST].lower() not in DIST_ALGOS:
        if not kwargs[KW_DIST].is_file():
            error(f"The distance metric for data {counter} seems to be a file-input but the filepath is invalid.", kwargs[KW_CLI])
    if kwargs[KW_CLUSTERS] < 1:
        error(f"The number of clusters to find in data {counter} has to be a positive integer.", kwargs[KW_CLI])
    return kwargs


def create_config(**kwargs) -> dict[str, object]:
    """
    Create a configuration dictionary from the given arguments.

    Args:
        **kwargs: Arguments in kwargs-format
    Returns:
        Configuration dictionary
    """
    LOGGER.critical("You are not using a config file as input. This is deprecated and will be removed in DataSAIL v2. Please update your workflow to use a config file. A description on how to do this is given in the documentation <INSERT-LINK>.")
    tech_map = {"R": "R", "I1e": "I1", "I1f": "I2", "C1e": "S1", "C1f": "S2", "I2": "I1-2", "C2": "S1-2"}
    e_args = {key[2:]: value for key, value in kwargs.items() if key.startswith("e_")}
    f_args = {key[2:]: value for key, value in kwargs.items() if key.startswith("f_")}
    config = {key: value for key, value in kwargs.items() if not key.startswith(("e_", "f_"))}
    if isinstance(kwargs["techniques"], str):
        config["techniques"] = [tech_map[kwargs["techniques"]]]
    elif callable(kwargs["techniques"]) or isinstance(kwargs["techniques"], Generator):
        config["techniques"] = [tech_map[t] for t in kwargs["techniques"]()]
    elif kwargs["techniques"] is None:
        config["techniques"] = None
    else:
        config["techniques"] = [tech_map[t] for t in kwargs["techniques"]]
    config["switched"] = False
    if kwargs["f_type"] is None:
        config["data"] = [e_args]
    elif kwargs["e_type"] is None:
        config["data"] = [f_args]
        config["techniques"] = [t.replace("2", "1") for t in config["techniques"]]
        config["switched"] = True
    else:
        config["data"] = [e_args, f_args]
    return config


def datasail(
        techniques: Optional[Union[str, list[str], Callable[..., list[str]], Generator[str, None, None]]] = None,
        splits: Optional[list[float]] = None,
        names: Optional[list[str]] = None,
        inter: Optional[Union[str, Path, list[tuple[str, str]], Callable[..., list[str]], Generator[str, None, None]]] = None,
        max_sec: int = 100,
        verbose: str = "W",
        delta: float = 0.05,
        epsilon: float = 0.05,
        runs: int = 1,
        solver: str = SOLVER_SCIP,
        cache: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        linkage: Literal["average", "single", "complete"] = "average",
        overflow: Literal["assign", "break"] = "assign",
        e_type: Optional[str] = None,
        e_data: DATA_INPUT = None,
        e_weights: DATA_INPUT = None,
        e_strat: DATA_INPUT = None,
        e_sim: MATRIX_INPUT = None,
        e_dist: MATRIX_INPUT = None,
        e_args: str = "",
        e_clusters: int = 50,
        f_type: Optional[str] = None,
        f_data: DATA_INPUT = None,
        f_weights: DATA_INPUT = None,
        f_strat: DATA_INPUT = None,
        f_sim: MATRIX_INPUT = None,
        f_dist: MATRIX_INPUT = None,
        f_args: str = "",
        f_clusters: int = 50,
        threads: int = 1,
        config: Optional[dict[str, object]] = None
    ) -> Optional[tuple[dict, dict, dict]]:
    """
    Entry point for the Python Package. Invocation routine of DataSAIL.

    Args:
        config: Dictionary with the configuration of DataSAIL

    Returns:

    """
    if config is None:
        config = create_config(**{"techniques": techniques, "splits": splits, "names": names, "inter": inter, "max_sec": max_sec, 
                                  "verbose": verbose, "delta": delta, "epsilon": epsilon, "runs": runs, "solver":solver, "cache": cache,
                                  "cache_dir": cache_dir, "linkage": linkage, "overflow": overflow, "e_type": e_type, "e_data": e_data, 
                                  "e_weights": e_weights, "e_strat": e_strat, "e_sim": e_sim, "e_dist": e_dist, "e_args": e_args, 
                                  "e_clusters": e_clusters, "f_type": f_type, "f_data": f_data, "f_weights": f_weights, "f_strat": f_strat, 
                                  "f_sim": f_sim, "f_dist": f_dist, "f_args": f_args, "f_clusters": f_clusters, "threads": threads})
    if KW_CLI not in config:
        config[KW_CLI] = False
    checked_args = validate_general_args(**config)
    if isinstance(config["data"], dict):
        config["data"] = [config["data"]]
    checked_args["data"] = []
    for c, data_config in enumerate(config["data"]):
        checked_args["data"].append(validate_data_args(c + 1, cli=False, **data_config))
    return datasail_main(**checked_args)


def sail(args=None, **kwargs) -> None:
    """
    Entry point for the CLI tool. Invocation routine of DataSAIL. Here, the arguments are validated and the main
    routine is invoked.
    """
    if kwargs is None or len(kwargs) == 0:
        kwargs = parse_datasail_args(args or sys.argv[1:])

    if kwargs["list_cluster"] or kwargs[KW_CC]:
        if kwargs[KW_CC]:
            LOGGER.critical("The argument --cc is deprecated and will be removed in DataSAIL v2. Please use -lc or --list-cluster instead.")
        print("Available clustering algorithms:", "\tECFP", sep="\n")
        for algo, name in [(CDHIT, "CD-HIT"), (CDHIT_EST, "CD-HIT-EST"), (DIAMOND, "DIAMOND"), (MMSEQS, "MMseqs, MMseqs2"), 
                           (MASH, "MASH"), (FOLDSEEK, "FoldSeek"), (TMALIGN, "TMalign")]:
            if INSTALLED[algo]:
                print("\t", name)
        exit(0)

    if kwargs.get("config", None) is not None:
        with open(kwargs["config"], "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        print("Blub")
        config = create_config(**kwargs)
    config[KW_CLI] = True
    
    datasail(config=config)
