import argparse
from typing import Dict

SIM_ALGOS = [
    "wlk", "mmseqs", "foldseek", "cdhit", "ecfp",
]

DIST_ALGOS = [
    "mash",
]


def parse_datasail_args(args) -> Dict[str, object]:
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
        "--threads",
        default=0,
        dest="threads",
        type=int,
        help="Number of threads to use throughout the computation. This number of threads is also forwarded to "
             "clustering programs used internally. If 0, all available CPUs will be used."
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
        version=f"%(prog)s v0.0.7"
    )
    split = parser.add_argument_group("Splitting Arguments")
    split.add_argument(
        "-t",
        "--techniques",
        type=str,
        required=True,
        choices=["R", "ICSe", "ICSf", "ICD", "CCD", "CCSe", "CCSf"],
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
        help="Flag indicating to run the program in scalar for instead of vectorized formulation."
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
             f"be {', '.join('[' + x + ']' for x in DIST_ALGOS)}, or a filepath to a file storing the pairwise "
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
    return vars(parser.parse_args(args))


def parse_cdhit_args(cdhit_args):
    """
    Check if the provided arguments for CD-HIT are valid.

    Args:
        cdhit_args: String of the additional arguments for CD-HIT

    Returns:
        The arguments as keys of a dictionary matching them to their provided values
    """
    cdhit_parser = argparse.ArgumentParser()
    cdhit_parser.add_argument("-c", type=float, default=0.9)
    cdhit_parser.add_argument("-n", type=int, default=5, choices=[2, 3, 4, 5])
    return vars(cdhit_parser.parse_args(cdhit_args))


def parse_mash_args(mash_args):
    """
    Check if the provided arguments for MASH are valid.

    Args:
        mash_args: String of the additional arguments for MASH

    Returns:
        The arguments as keys of a dictionary matching them to their provided values
    """
    mash_parser = argparse.ArgumentParser()
    mash_parser.add_argument("-k", type=int, default=21)
    mash_parser.add_argument("-s", type=int, default=10000)
    return vars(mash_parser.parse_args(mash_args))


def parse_mmseqs_args(mmseqs_args):
    """
    Check if the provided arguments for MMseq2 are valid.

    Args:
        mmseqs_args: String of the additional arguments for MMseqs2

    Returns:
        The arguments as keys of a dictionary matching them to their provided values
    """
    mmseqs_parser = argparse.ArgumentParser()
    mmseqs_parser.add_argument("--min-seq-id", type=float, default=0, dest="seq_id")
    return vars(mmseqs_parser.parse_args(mmseqs_args))
