import argparse
import logging
import os.path
import sys
from typing import Dict, List, Tuple

from datasail.parsers import parse_cdhit_args, parse_mash_args, parse_mmseqs_args, DIST_ALGOS, SIM_ALGOS, \
    parse_datasail_args
from datasail.run import bqp_main

verb_map = {
    "C": logging.CRITICAL,
    "F": logging.FATAL,
    "E": logging.ERROR,
    "W": logging.WARNING,
    "I": logging.INFO,
    "D": logging.DEBUG,
}


def error(msg: str, error_code: int) -> None:
    """
    Print an error message with an individual error code to the commandline. Afterwards, the program is stopped.

    Args:
        msg: Error message
        error_code: Code of the error to identify it
    """
    logging.error(msg)
    exit(error_code)


def validate_args(**kwargs) -> Dict[str, object]:
    """
    Validate the arguments given to the program.

    Args:
        **kwargs: Arguments in kwargs-format

    Returns:
        The kwargs in case something has been adjusted, e.g. splits have to transformed into sum=1-vector or naming
    """

    # create output directory
    output_created = False
    if kwargs["output"] is not None and not os.path.isdir(kwargs["output"]):
        output_created = True
        os.makedirs(kwargs["output"], exist_ok=True)

    kwargs["logdir"] = os.path.abspath(os.path.join(kwargs["output"], "logs"))

    os.makedirs(kwargs["logdir"], exist_ok=True)

    formatter = logging.Formatter('%(asctime)s %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level=verb_map[kwargs["verbosity"]])
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(os.path.join(kwargs["output"], "logs", "general.log"))
    file_handler.setLevel(level=verb_map[kwargs["verbosity"]])
    file_handler.setFormatter(formatter)

    logging.basicConfig(level=verb_map[kwargs["verbosity"]], handlers=[stdout_handler, file_handler])

    if output_created:
        logging.warning("Output directory does not exist, DataSAIL creates it automatically")

    logging.info("Validating arguments")

    # check splits to be more than 1 and their fractions sum up to 1 and check the names
    if len(kwargs["splits"]) < 2:
        error("Less then two splits required. This is no useful input, please check the input again.", error_code=1)
    if kwargs["names"] is None:
        kwargs["names"] = [f"Split{x:03d}" for x in range(len(kwargs["splits"]))]
    elif len(kwargs["names"]) != len(kwargs["names"]):
        error("Different number of splits and names. You have to give the same number of splits and names for them.",
              error_code=2)
    kwargs["splits"] = [x / sum(kwargs["splits"]) for x in kwargs["splits"]]

    # convert vectorized from the input question to the flag used in the code
    kwargs["vectorized"] = not kwargs["vectorized"]

    # check search termination criteria
    if kwargs["max_sec"] < 1:
        error("The maximal search time must be a positive integer.", error_code=3)
    if kwargs["max_sol"] < 1:
        error("The maximal number of solutions to look at has to be a positive integer.", error_code=4)

    # check the interaction file
    if kwargs["inter"] is not None and not os.path.isfile(kwargs["inter"]):
        error("The interaction filepath is not valid.", error_code=5)

    # check the epsilon value
    if 1 < kwargs["epsilon"] < 0:
        error("The epsilon value has to be a real value between 0 and 1.", error_code=6)

    # check the input regarding the caching
    if kwargs["cache"] and not os.path.isdir(kwargs["cache_dir"]):
        logging.warning("Cache directory does not exist, DataSAIL creates it automatically")
        os.makedirs(kwargs["cache_dir"], exist_ok=True)

    # syntactically parse the input data for the E-dataset
    if kwargs["e_data"] is not None and not os.path.exists(kwargs["e_data"]):
        error("The filepath to the E-data is invalid.", error_code=7)
    if kwargs["e_weights"] is not None and not os.path.isfile(kwargs["e_weights"]):
        error("The filepath to the weights of the E-data is invalid.", error_code=8)
    if kwargs["e_sim"] is not None and kwargs["e_sim"].lower() not in SIM_ALGOS and not os.path.isfile(kwargs["e_sim"]):
        error(
            f"The similarity metric for the E-data seems to be a file-input but the filepath is invalid.", error_code=9
        )
    if kwargs["e_dist"] is not None and kwargs["e_dist"].lower() not in DIST_ALGOS and not os.path.isfile(
            kwargs["e_dist"]):
        error(
            f"The distance metric for the E-data seems to be a file-input but the filepath is invalid.", error_code=10
        )
    if kwargs["e_sim"] is not None and kwargs["e_sim"].lower() == "cdhit":
        validate_cdhit_args(kwargs["e_args"])
    if kwargs["e_sim"] is not None and kwargs["e_sim"].lower() == "mmseqs":
        validate_mmseqs_args(kwargs["e_args"])
    if kwargs["e_dist"] is not None and kwargs["e_dist"].lower() == "mash":
        validate_mash_args(kwargs["e_args"])
    if 1 < kwargs["e_max_sim"] < 0:
        error("The maximal similarity value for the E-data has to be a real value in [0,1].", error_code=11)
    if 1 < kwargs["e_max_dist"] < 0:
        error("The maximal distance value for the E-data has to be a real value in [0,1].", error_code=12)

    # syntactically parse the input data for the F-dataset
    if kwargs["f_data"] is not None and not os.path.exists(kwargs["f_data"]):
        error("The filepath to the F-data is invalid.", error_code=13)
    if kwargs["f_weights"] is not None and not os.path.isfile(kwargs["f_weights"]):
        error("The filepath to the weights of the F-data is invalid.", error_code=14)
    if kwargs["f_sim"] is not None and kwargs["f_sim"].lower() not in SIM_ALGOS and not os.path.isfile(kwargs["f_sim"]):
        error(
            f"The similarity metric for the F-data seems to be a file-input but the filepath is invalid.", error_code=15
        )
    if kwargs["f_dist"] is not None and kwargs["f_dist"].lower() not in DIST_ALGOS and not os.path.isfile(
            kwargs["f_dist"]):
        error(
            f"The distance metric for the F-data seems to be a file-input but the filepath is invalid.", error_code=16
        )
    if kwargs["f_sim"] is not None and kwargs["f_sim"] == "CDHIT":
        validate_cdhit_args(kwargs["f_args"])
    if kwargs["f_sim"] is not None and kwargs["f_sim"].lower() == "mmseqs":
        validate_mmseqs_args(kwargs["e_args"])
    if kwargs["f_dist"] is not None and kwargs["f_dist"] == "MASH":
        validate_mash_args(kwargs["f_args"])
    if 1 < kwargs["f_max_sim"] < 0:
        error("The maximal similarity value for the F-data has to be a real value in [0,1].", error_code=17)
    if 1 < kwargs["f_max_dist"] < 0:
        error("The maximal distance value for the F-data has to be a real value in [0,1].", error_code=18)

    return kwargs


def validate_cdhit_args(cdhit_args):
    parsed = parse_cdhit_args(cdhit_args)
    if not ((parsed["n"] == 2 and 0.4 <= parsed["c"] <= 0.5) or
            (parsed["n"] == 3 and 0.5 <= parsed["c"] <= 0.6) or
            (parsed["n"] == 4 and 0.6 <= parsed["c"] <= 0.7) or
            (parsed["n"] == 5 and 0.7 <= parsed["c"] <= 1.0)):
        error("There are restrictions on the values for n and c in CD-HIT:\n"
              "n == 5 <=> c in [0.7, 1.0]\n"
              "n == 4 <=> c in [0.6, 0.7]\n"
              "n == 3 <=> c in [0.5, 0.6]\n"
              "n == 2 <=> c in [0.4, 0.5]", error_code=19)


def validate_mash_args(mash_args):
    parsed = parse_mash_args(mash_args)


def validate_mmseqs_args(mmseqs_args):
    parsed = parse_mmseqs_args(mmseqs_args)
    if 1 < parsed["seq_id"] < 0:
        error("The minimum sequence identity for mmseqs has to be a value between 0 and 1.", error_code=21)


def datasail(
        techniques: List[str],
        inter=None,
        max_sec: int = 100,
        max_sol: int = 1000,
        verbose: str = "W",
        splits: List = [0.7, 0.2, 0.1],
        names: List[str] = ["train", "val", "test"],
        epsilon: float = 0.05,
        solver: str = "MOSEK",
        vectorized: bool = True,
        cache: bool = False,
        cache_dir: str = None,
        e_type=None,
        e_data=None,
        e_weights=None,
        e_sim=None,
        e_dist=None,
        e_args="",
        e_max_sim: float = 1.0,
        e_max_dist: float = 1.0,
        f_type=None,
        f_data=None,
        f_weights=None,
        f_sim=None,
        f_dist=None,
        f_args="",
        f_max_sim: float = 1.0,
        f_max_dist: float = 1.0,
) -> Tuple[Dict, Dict, Dict]:
    kwargs = validate_args(
        output=None, techniques=techniques, inter=inter, max_sec=max_sec, max_sol=max_sol, verbosity=verbose,
        splits=splits, names=names, epsilon=epsilon, solver=solver, vectorized=not vectorized, cache=cache,
        cache_dir=cache_dir, e_type=e_type, e_data=e_data, e_weights=e_weights, e_sim=e_sim, e_dist=e_dist,
        e_args=e_args, e_max_sim=e_max_sim, e_max_dist=e_max_dist, f_type=f_type, f_data=f_data, f_weights=f_weights,
        f_sim=f_sim, f_dist=f_dist, f_args=f_args, f_max_sim=f_max_sim, f_max_dist=f_max_dist
    )
    return bqp_main(**kwargs)


def sail(**kwargs) -> None:
    """
    Invocation routine of DataSAIL. Here, the arguments are validated and the main routine is invoked.

    Args:
        **kwargs: Arguments to DataSAIL in kwargs-format.

    """
    kwargs = validate_args(**kwargs)
    bqp_main(**kwargs)


if __name__ == '__main__':
    sail(**parse_datasail_args(sys.argv))
