import logging
import os
from argparse import ArgumentParser


def ilp_args(parser: ArgumentParser):
    ilp = parser.add_argument_group(description="Argument for ILP Solver")
    ilp.add_argument(
        "-s",
        "--splits",
        default=[0.7, 0.2, 0.1],
        nargs="+",
        type=float,
        dest="splits",
        help="Sizes of the individual splits the program shall produce.",
    )
    ilp.add_argument(
        "-l",
        "--limit",
        default=0.05,
        type=float,
        dest="limit",
        help="Multiplicative factor by how much the limits of the splits can be exceeded.",
    )
    ilp.add_argument(
        "-d",
        "--drugs",
        default=None,
        type=str,
        dest="drugs",
        help="Path to TSV file of drugs and their SMILES string.",
    )
    ilp.add_argument(
        "--drug-weights",
        default=None,
        type=str,
        dest="drug_weights",
        help="Path to TSV file with the weights for the drugs."
    )
    ilp.add_argument(
        "--inter",
        default=None,
        type=str,
        dest="inter",
        help="Path to TSV file of drug-protein interactions.",
    )
    ilp.add_argument(
        "--inter-weights",
        action='store_true',
        default=False,
        help="If True, compute the weights of proteins and drugs based on occurrences in interaction file. "
             "This overwrites the -w and --drug-weights arguments."
    )
    ilp.add_argument(
        "-t",
        "--technique",
        type=str,
        choices=["R", "ICP", "ICD", "IC", "CR", "SCP", "SCD", "SC"],
        default="R",
        dest="technique",
        help="Select the mode to split the data. R: random split, ICP: identity-based cold-protein split, "
             "ICD: identity-based cold-drug split, IC: identity-based cold-drug-target split, "
             "SCP: similarity-based cold-protein split, SCD: similarity-based cold-drug split, "
             "SC: similarity-based cold-drug-protein split"
    )
    ilp.add_argument(
        "--prot-sim",
        type=str,
        default=None,
        dest="prot_sim",
        help="Provide the name of a method to determine similarity between proteins. This can either be >WLK<, "
             ">mmseqs<, or a filepath to a file storing the clusters in TSV."
    )


def validate_args(args):
    if sum(args.splits) != 1:
        logging.warning("Sum of splits ratios is not 1. This might lead to unexpected behaviour.")
    if len(args.splits) < 2:
        logging.error("Zero or one split given. This is meaningless as no work to do.")
        exit(4)
    if args.limit < 0:
        logging.warning("Factor for tolerating split limit is negative. Split limits are reduced by this.")
    if args.drug_weights is not None and args.drugs is None:
        logging.warning("Weights for drugs are given but no drugs are given. Weights will be ignored.")
    if args.inter_weights and args.inter is None:
        logging.warning("Flag to weight by interactions is True but no interactions are given. Weights will be ignored.")
    if args.inter is not None and args.drugs is None:
        logging.error("Interactions are given but no drugs are provided.")
        exit(1)
    if args.technique in {"IC", "CR", "SC"} and (args.inter is None or args.drugs is None):
        logging.error(f"The technique {args.technique} requires drug and interaction data.")
        exit(2)
    if args.technique in {"ICD", "IC", "SCD", "SC"} and args.drugs is None:
        logging.error(f"The technique {args.technique} required drug data.")
        exit(3)
    if args.prot_sim not in {"WLK", "mmseqs"} and not os.path.exists(args.prot_sim):
        logging.error(f"The protein similarity argument {args.prot_sim} is no valid mode and no file.")
