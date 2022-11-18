import argparse
import logging
import os

from scala.sat_split.args import ilp_args, validate_args as ilp_validate
from scala.sat_split.sat import ilp_main
from scala.tree_split.args import tree_args, validate_args as tree_validate
from scala.tree_split.tree import tree_main


verb_map = {
    "C": logging.CRITICAL,
    "F": logging.FATAL,
    "E": logging.ERROR,
    "W": logging.WARNING,
    "I": logging.INFO,
    "D": logging.DEBUG,
}


def parse_args():
    parser = argparse.ArgumentParser(
        prog='SCALA',
        description="This tool helps providing the most challenging dataset split for a machine learning model in "
                    "order to prevent information leakage and improve generalizability",
    )
    req = parser.add_argument_group("General Required Arguments")
    req.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        dest='input',
        help="filepath of input file (FASTA/FASTQ) or directory of protein structure (PDB/mmCIF)",
    )
    req.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        dest='output',
        help="directory to save the results in",
    )
    req.add_argument(
        "-m",
        "--method",
        required=True,
        type=str,
        choices=["tree", "ilp"],
        dest="method",
        help="Which method to use to split the data."
    )
    opt = parser.add_argument_group(description="General Optional Arguments")
    opt.add_argument(
        "-w",
        "--weights",
        default=None,
        type=str,
        dest='weight_file',
        help="Path to TSV file with weight of protein sequences from -i argument.",
    )
    opt.add_argument(
        "-v",
        "--verbosity",
        default="W",
        type=str,
        choices=["C", "F", "E", "W", "I", "D"],
        dest='verbosity',
        help="Verbosity level of the program",
    )
    ilp_args(parser)
    tree_args(parser)
    return parser.parse_args()


def validate_args(args):
    logging.basicConfig(level=verb_map[args.verbosity])
    logging.info("Validating arguments")
    if not os.path.exists(args.input):
        logging.error("The protein file does not exist.")
        exit(5)
    if args.weight_file is not None and not os.path.exists(args.weight_file):
        logging.error(f"Weight file {args.weight_file} for protein weighting does not exist.")
        exit(6)

    os.makedirs(args.output, exist_ok=True)


def main():
    args = parse_args()

    validate_args(args)

    if args.method == "tree":
        tree_validate(args)
        os.makedirs(args.output, exist_ok=True)
        tree_main(args)
    elif args.method == "ilp":
        ilp_validate(args)
        os.makedirs(args.output, exist_ok=True)
        ilp_main(args)


if __name__ == "__main__":
    main()
