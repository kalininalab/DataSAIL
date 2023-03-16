import argparse
import logging
import os.path
from typing import Dict

from datasail.run import bqp_main

verb_map = {
    "C": logging.CRITICAL,
    "F": logging.FATAL,
    "E": logging.ERROR,
    "W": logging.WARNING,
    "I": logging.INFO,
    "D": logging.DEBUG,
}

SIM_ALGOS = [
    "wlk", "mmseqs", "foldseek", "cdhit", "ecfp",
]

DIST_ALGOS = [
    "mash",
]


def parse_args() -> Dict[str, object]:
    """
    Define the argument parser for DataSAIL.

    Returns:
        Parser arguments to the program in kwargs-format.
    """
    parser = argparse.ArgumentParser(
        prog="DataSAIL - Data Splitting Against Information Leaking",
        description="DataSAIL is a tool computing with splits of any type of dataset to challenge your AI model. "
                    "The splits computed by DataSAIL try to minimize the amount of leaked information between two "
                    "splits based on what the user requested. Splits can be done based on sample ids but also based on "
                    "clusters within the dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        dest="output",
        help="Output directory to store the splits in.",
    )
    parser.add_argument(
        "-i",
        "--inter",
        type=str,
        default=None,
        dest="inter",
        help="Path to TSV file of interactions between two entities. The first entry in each line has to match an "
             "entry from the e-entity, the second matches one of the f-entity."
    )
    parser.add_argument(
        "--to-sec",
        default=100,
        dest="max_sec",
        type=int,
        help="Maximal time to spend optimizing the objective in seconds. This does not include preparatory work such "
             "as parsing data and cluster the input."
    )
    parser.add_argument(
        "--to-sol",
        default=1000,
        dest="max_sol",
        type=int,
        help="Maximal number of solutions to compute until end of search (in case no optimum was found). This argument "
             "is ignored so far."
    )
    parser.add_argument(
        "--verbose",
        default="W",
        type=str,
        choices=["C", "F", "E", "W", "I", "D"],
        dest='verbosity',
        help="Verbosity level of the program. Choices are: [C]ritical, [F]atal, [E]rror, [W]arning, [I]nfo, [D]ebug",
    )
    parser.add_argument(
        "-v",
        "--version",
        action='version',
        version="%(prog)s 0.0.1"
    )
    split = parser.add_argument_group("Splitting Arguments")
    split.add_argument(
        "-t",
        "--techniques",
        type=str,
        required=True,
        choices=["R", "ICS", "ICD", "CCS", "CCD", "ICSe", "ICSf", "CCSe", "CCSf"],
        nargs="+",
        dest="techniques",
        help="Select the mode to split the data. Choices: R: Random split, ICS: identity-based cold-single split, "
             "ICD: identity-based cold-double split, CCS: similarity-based cold-single split, "
             "CCD: similarity-based cold-double split"
    )
    split.add_argument(
        "-s",
        "--splits",
        default=[0.7, 0.2, 0.1],
        nargs="+",
        type=float,
        dest="splits",
        help="Sizes of the individual splits the program shall produce.",
    )
    split.add_argument(
        "--names",
        default=None,
        dest="names",
        nargs="+",
        type=str,
        help="Names of the splits in order of the -s argument. If left empty, splits will be called Split1, Split2, ..."
    )
    split.add_argument(
        "--epsilon",
        default=0.05,
        type=float,
        dest="epsilon",
        help="Multiplicative factor by how much the limits of the splits can be exceeded.",
    )
    split.add_argument(
        "--solver",
        default="MOSEK",
        type=str,
        choices=["MOSEK", "SCIP"],
        dest="solver",
        help="Solver to use to solve the BDQCP. Choices are SCIP (free of charge) and MOSEK (licensed and only "
             "applicable if a valid mosek license is stored)."
    )
    split.add_argument(
        "--scalar",
        default=False,
        action='store_true',
        dest="vectorized",
        help="Flag indicating to run the program in scalar for instead of vectorized formulation [default]."
    )
    split.add_argument(
        "--cache",
        default=False,
        action='store_true',
        dest="cache",
        help="Store clustering matrices in cache."
    )
    split.add_argument(
        "--cache-dir",
        default=None,
        dest="cache_dir",
        help="Destination of the cache folder. Default is the OS-default cache dir."
    )
    e_ent = parser.add_argument_group("First Input Arguments")
    e_ent.add_argument(
        "--e-type",
        type=str,
        dest="e_type",
        choices=["P", "M", "G", "O"],
        default=None,
        help="Type of the first data batch to the program. Choices are: [P]rotein, [M]olecule, [G]enome, [O]ther",
    )
    e_ent.add_argument(
        "--e-data",
        type=str,
        dest="e_data",
        default=None,
        help="First input to the program. This can either be the filepath a directory containing only data files.",
    )
    e_ent.add_argument(
        "--e-weights",
        type=str,
        dest="e_weights",
        default=None,
        help="Custom weights of the first bunch of samples. The file has to have TSV format where every line is of the "
             "form [e_id >tab< weight]. The e_id has to match an entity id from the first input argument.",
    )
    e_ent.add_argument(
        "--e-sim",
        type=str,
        dest="e_sim",
        default=None,
        help="Provide the name of a method to determine similarity between samples of the first input dataset. This "
             f"can either be {', '.join('[' + x + ']' for x in SIM_ALGOS)}, or a filepath to a file storing the "
             f"pairwise similarities in TSV.",
    )
    e_ent.add_argument(
        "--e-dist",
        type=str,
        dest="e_dist",
        default=None,
        help="Provide the name of a method to determine distance between samples of the first input dataset. This can "
             f"be {', '.join('[' + x + ']' for x in SIM_ALGOS)}, or a filepath to a file storing the pairwise "
             "distances in TSV."
    )
    e_ent.add_argument(
        "--e-args",
        type=str,
        dest="e_args",
        default="",
        help="Additional arguments for the clustering algorithm used in --e-dist or --e-sim."
    )
    e_ent.add_argument(
        "--e-max-sim",
        type=float,
        dest="e_max_sim",
        default=1.0,
        help="Maximum similarity of two samples from the first data in two split."
    )
    e_ent.add_argument(
        "--e-max-dist",
        type=float,
        dest="e_max_dist",
        default=1.0,
        help="Maximal distance of two samples from the second data in the same split."
    )
    f_ent = parser.add_argument_group("Second Input Arguments")
    f_ent.add_argument(
        "--f-type",
        type=str,
        dest="f_type",
        default=None,
        help="Type of the second data batch to the program. Choices are: [P]rotein, [M]olecule, [G]enome, [O]ther",
    )
    f_ent.add_argument(
        "--f-data",
        type=str,
        dest="f_data",
        default=None,
        help="Second input to the program. This can either be the filepath a directory containing only data files.",
    )
    f_ent.add_argument(
        "--f-weights",
        type=str,
        dest="f_weights",
        default=None,
        help="Custom weights of the second bunch of samples. The file has to have TSV format where every line is of "
             "the form [f_id >tab< weight]. The f_id has to match an entity id from the second input argument group.",
    )
    f_ent.add_argument(
        "--f-sim",
        type=str,
        dest="f_sim",
        default=None,
        help="Provide the name of a method to determine similarity between samples of the second input dataset. This "
             "can either be [WLK], [mmseqs], [FoldSeek], [CDHIT], [ECFP], or a filepath to a file storing the pairwise "
             "similarities in TSV.",
    )
    f_ent.add_argument(
        "--f-dist",
        type=str,
        dest="f_dist",
        default=None,
        help="Provide the name of a method to determine distance between samples of the second input dataset. This can "
             "be [MASH] or a filepath to a file storing the pairwise distances in TSV."
    )
    f_ent.add_argument(
        "--f-args",
        type=str,
        dest="f_args",
        default="",
        help="Additional arguments for the clustering algorithm used in --f-dist or --f-sim."
    )
    f_ent.add_argument(
        "--f-max-sim",
        type=float,
        dest="f_max_sim",
        default=1.0,
        help="Maximum similarity of two samples from the second dataset in two split."
    )
    f_ent.add_argument(
        "--f-max-dist",
        type=float,
        dest="f_max_dist",
        default=1.0,
        help="Maximal distance of two samples from the second dataset in the same split."
    )
    return vars(parser.parse_args())


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
    logging.basicConfig(level=verb_map[kwargs["verbosity"]])
    logging.info("Validating arguments")

    # create output directory
    if not os.path.isdir(kwargs["output"]):
        logging.warning("Output directory does not exist, DataSAIL creates it automatically")
        os.makedirs(kwargs["output"], exist_ok=True)

    # check splits to be more than 1 and their fractions sum up to 1 and check the names
    if len(kwargs["splits"]) < 2:
        error("Less then two splits required. This is no useful input, please check the input again.", error_code=1)
    if kwargs["names"] is None:
        kwargs["names"] = [f"Split{x:03s}" for x in range(len(kwargs["splits"]))]
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
    if kwargs["e_dist"] is not None and kwargs["e_dist"].lower() not in DIST_ALGOS and not os.path.isfile(kwargs["e_dist"]):
        error(
            f"The distance metric for the E-data seems to be a file-input but the filepath is invalid.", error_code=10
        )
    if kwargs["e_sim"] == "CDHIT":
        validate_cdhit_args(kwargs["e_args"])
    if kwargs["e_dist"] == "MASH":
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
    if kwargs["f_dist"] is not None and kwargs["f_dist"].lower() not in DIST_ALGOS and not os.path.isfile(kwargs["f_dist"]):
        error(
            f"The distance metric for the F-data seems to be a file-input but the filepath is invalid.", error_code=16
        )
    if kwargs["f_sim"] == "CDHIT":
        validate_cdhit_args(kwargs["f_args"])
    if kwargs["f_dist"] == "MASH":
        validate_mash_args(kwargs["f_args"])
    if 1 < kwargs["f_max_sim"] < 0:
        error("The maximal similarity value for the F-data has to be a real value in [0,1].", error_code=17)
    if 1 < kwargs["f_max_dist"] < 0:
        error("The maximal distance value for the F-data has to be a real value in [0,1].", error_code=18)

    return kwargs


def validate_cdhit_args(cdhit_args):
    cdhit_parser = argparse.ArgumentParser()
    cdhit_parser.add_argument("-c", type=float, default=0.9)
    cdhit.parser.add_argument("-n", type=int, default=5, choices=[2, 3, 4, 5])
    parsed = cdhit_parser.parse_args(cdhit_args)
    if not ((parsed["n"] == 2 and 0.4 <= parsed["c"] <= 0.5) or \
            (parser["n"] == 3 and 0.5 <= parsed["c"] <= 0.6) or \
            (parsed["n"] == 4 and 0.6 <= parsed["c"] <= 0.7) or \
            (parsed["n"] == 5 and 0.7 <= parsed["c"] <= 1.0)):
        error("There are restrictions on the values for n and c in CD-HIT:\n"
                "n == 5 <=> c in [0.7, 1.0]\n"
                "n == 4 <=> c in [0.6, 0.7]\n"
                "n == 3 <=> c in [0.5, 0.6]\n"
                "n == 2 <=> c in [0.4, 0.5]", error_code=19)


def validate_mash_args(mash_args):
    mash_parser = argparse.ArgumentParser()
    mash_parser.add_argument("-k", type=int, default=21)
    mash_parser.add_argument("-s", type=int, default=10000)
    mash_parser.parse_args(mash_args)


def sail(**kwargs) -> None:
    """
    Invocation routine of DataSAIL. Here, the arguments are validated and the main routine is invoked.

    Args:
        **kwargs: Arguments to DataSAIL in kwargs-format.
    """
    kwargs = validate_args(**kwargs)
    bqp_main(**kwargs)


if __name__ == '__main__':
    sail(**parse_args())
