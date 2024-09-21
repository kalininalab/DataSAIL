import argparse
from pathlib import Path
from pydoc import locate
from typing import Dict, List, Sequence, Literal

import yaml

from datasail.argparse_patch import insert_patch
from datasail.settings import *
from datasail.version import __version__


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
        "--cc",
        default=False,
        action='store_true',
        dest=KW_CC,
        help="List available clustering algorithms."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        dest=KW_OUTDIR,
        help="Output directory to store the splits in.",
    )
    parser.add_argument(
        "-i",
        "--inter",
        type=Path,
        default=None,
        dest=KW_INTER,
        help="Path to TSV file of interactions between two entities. The first entry in each line has to match an "
             "entry from the e-entity, the second matches one of the f-entity."
    )
    parser.add_argument(
        "--to-sec",
        default=100,
        dest=KW_MAX_SEC,
        type=int,
        help="Maximal time to spend optimizing the objective in seconds. This does not include preparatory work such "
             "as parsing data and cluster the input."
    )
    parser.add_argument(
        "--to-sol",
        default=1000,
        dest=KW_MAX_SOL,
        type=int,
        help="Maximal number of solutions to compute until end of search (in case no optimum was found). This argument "
             "is ignored so far."
    )
    parser.add_argument(
        "--threads",
        default=0,
        dest=KW_THREADS,
        type=int,
        help="Number of threads to use throughout the computation. This number of threads is also forwarded to "
             "clustering programs used internally. If 0, all available CPUs will be used."
    )
    parser.add_argument(
        "--verbose",
        default="W",
        type=str,
        choices=["C", "F", "E", "W", "I", "D"],
        dest=KW_VERBOSE,
        help="Verbosity level of the program. Choices are: [C]ritical, [F]atal, [E]rror, [W]arning, [I]nfo, [D]ebug",
    )
    parser.add_argument(
        "-v",
        "--version",
        action='version',
        version=f"%(prog)s v{__version__}"
    )
    split = parser.add_argument_group("Splitting Arguments")
    split.add_argument(
        "-t",
        "--techniques",
        type=str,
        required=True,
        choices=[TEC_R, TEC_I1 + MODE_E, TEC_I1 + MODE_F, TEC_I2, TEC_C1 + MODE_E, TEC_C1 + MODE_F, TEC_C2],
        nargs="+",
        dest=KW_TECHNIQUES,
        help=f"Select the mode to split the data. Choices: {TEC_R}: Random split, "
             f"{TEC_I1}: identity-based one-dimensional split, {TEC_I2}: identity-based two-dimensional split, "
             f"{TEC_C1}: cluster-based one-dimensional split, {TEC_C2}: cluster-based two_dimensional split"
    )
    split.add_argument(
        "-s",
        "--splits",
        default=[0.7, 0.2, 0.1],
        nargs="+",
        type=float,
        dest=KW_SPLITS,
        help="Sizes of the individual splits the program shall produce.",
    )
    split.add_argument(
        "-n",
        "--names",
        default=None,
        dest=KW_NAMES,
        nargs="+",
        type=str,
        help="Names of the splits in order of the -s argument. If left empty, splits will be called Split1, Split2, ..."
    )
    split.add_argument(
        "-d",
        "--delta",
        default=0.3,
        type=float,
        dest=KW_DELTA,
        help="Relative error for stratification. This is only used if stratification is provided."
    )
    split.add_argument(
        "-e",
        "--epsilon",
        default=0.3,
        type=float,
        dest=KW_EPSILON,
        help="Relative error how much the limits of the splits can be exceeded.",
    )
    split.add_argument(
        "-r",
        "--runs",
        default=1,
        type=int,
        dest=KW_RUNS,
        help="Specify a number of runs to perform per split. This may introduce some variance in the splits."
    )
    split.add_argument(
        "--solver",
        default=SOLVER_SCIP,
        type=str,
        choices=[SOLVER_SCIP, SOLVER_CPLEX, SOLVER_GUROBI, SOLVER_MOSEK, SOLVER_XPRESS],
        dest=KW_SOLVER,
        help="Solver to use to solve the BLP. Free options is SCIP. CPLEX, GUROBI, MOSEK, and XPRESS are also "
             "supported, but commercial and need to be installed separately. Check the docu for more information."
    )
    split.add_argument(
        "--cache",
        default=False,
        action='store_true',
        dest=KW_CACHE,
        help="Store clustering matrices in cache."
    )
    split.add_argument(
        "--cache-dir",
        default=None,
        type=Path,
        dest=KW_CACHE_DIR,
        help="Destination of the cache folder. Default is the OS-default cache dir."
    )
    e_ent = parser.add_argument_group("First Input Arguments")
    e_ent.add_argument(
        "--e-type",
        type=str,
        dest=KW_E_TYPE,
        choices=[P_TYPE, M_TYPE, G_TYPE, O_TYPE],
        default=None,
        help="Type of the first data batch to the program. Choices are: [P]rotein, [M]olecule, [G]enome, [O]ther",
    )
    e_ent.add_argument(
        "--e-data",
        type=Path,
        dest=KW_E_DATA,
        default=None,
        help="First input to the program. This can either be the filepath a directory containing only data files.",
    )
    e_ent.add_argument(
        "--e-weights",
        type=Path,
        dest=KW_E_WEIGHTS,
        default=None,
        help="Custom weights of the first bunch of samples. The file has to have TSV format where every line is of the "
             "form [e_id >tab< weight]. The e_id has to match an entity id from the first input argument.",
    )
    e_ent.add_argument(
        "--e-sim",
        type=str,
        dest=KW_E_SIM,
        default=None,
        help="Provide the name of a method to determine similarity between samples of the first input dataset. This "
             f"can either be {', '.join('[' + x + ']' for x in SIM_ALGOS)}, or a filepath to a file storing the "
             f"pairwise similarities in TSV.",
    )
    e_ent.add_argument(
        "--e-dist",
        type=str,
        dest=KW_E_DIST,
        default=None,
        help="Provide the name of a method to determine distance between samples of the first input dataset. This can "
             f"be {', '.join('[' + x + ']' for x in DIST_ALGOS)}, or a filepath to a file storing the pairwise "
             "distances in TSV."
    )
    e_ent.add_argument(
        "--e-args",
        type=str,
        dest=KW_E_ARGS,
        default="",
        help="Additional arguments for the clustering algorithm used in --e-dist or --e-sim."
    )
    e_ent.add_argument(
        "--e-strat",
        type=str,
        dest=KW_E_STRAT,
        default=None,
        help="Provide the filename of a CSV file specifying the classes for the samples of the E-Dataset."
    )
    e_ent.add_argument(
        "--e-num-classes",
        type=int,
        dest=KW_E_CLUSTERS,
        default=50,
        help="Number of classes to use for clustering the e-data."
    )
    f_ent = parser.add_argument_group("Second Input Arguments")
    f_ent.add_argument(
        "--f-type",
        type=str,
        dest=KW_F_TYPE,
        default=None,
        help="Type of the second data batch to the program. Choices are: [P]rotein, [M]olecule, [G]enome, [O]ther",
    )
    f_ent.add_argument(
        "--f-data",
        type=Path,
        dest=KW_F_DATA,
        default=None,
        help="Second input to the program. This can either be the filepath a directory containing only data files.",
    )
    f_ent.add_argument(
        "--f-weights",
        type=Path,
        dest=KW_F_WEIGHTS,
        default=None,
        help="Custom weights of the second bunch of samples. The file has to have TSV format where every line is of "
             "the form [f_id >tab< weight]. The f_id has to match an entity id from the second input argument group.",
    )
    f_ent.add_argument(
        "--f-sim",
        type=str,
        dest=KW_F_SIM,
        default=None,
        help="Provide the name of a method to determine similarity between samples of the second input dataset. This "
             "can either be [WLK], [mmseqs], [FoldSeek], [CDHIT], [ECFP], or a filepath to a file storing the pairwise "
             "similarities in TSV.",
    )
    f_ent.add_argument(
        "--f-dist",
        type=str,
        dest=KW_F_DIST,
        default=None,
        help="Provide the name of a method to determine distance between samples of the second input dataset. This can "
             "be [MASH] or a filepath to a file storing the pairwise distances in TSV."
    )
    f_ent.add_argument(
        "--f-args",
        type=str,
        dest=KW_F_ARGS,
        default="",
        help="Additional arguments for the clustering algorithm used in --f-dist or --f-sim."
    )
    e_ent.add_argument(
        "--f-strat",
        type=str,
        dest=KW_F_STRAT,
        default=None,
        help="Provide the filename of a CSV file specifying the classes for the samples of the F-Dataset."
    )
    e_ent.add_argument(
        "--f-num-classes",
        type=int,
        dest=KW_F_CLUSTERS,
        default=50,
        help="Number of classes to use for clustering the f-data."
    )
    args = insert_patch(args)
    return vars(parser.parse_args(args))


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

    def get_user_arguments(self, args: argparse.Namespace, ds_args: List[str], fos: Literal[0, 1] = 0) -> str:
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
