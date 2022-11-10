import logging
import os
from argparse import ArgumentParser


def ilp_args(parser: ArgumentParser):
    ilp = parser.add_argument_group(description="Argument for ILP Solver")
    ilp.add_argument(
        "--inter",
        required=True,
        type=str,
        dest="inter",
        help="Path to TSV file of drug-protein interactions.",
    )
    ilp.add_argument(
        "-d",
        "--drugs",
        required=True,
        type=str,
        dest="drugs",
        help="Path to TSV file of drugs and their SMILES string.",
    )
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
        "--drug-weights",
        default=None,
        type=str,
        dest="drug_weights",
        help="Path to TSV file with the weights for the drugs."
    )
    ilp.add_argument(
        "-t",
        "--technique",
        type=str,
        choices=["R", "ICP", "ICD", "IC", "CCP", "CCD", "CC"],
        default="R",
        dest="technique",
        help="Select the mode to split the data. R: random split, ICP: identity-based cold-protein split, "
             "ICD: identity-based cold-drug split, IC: identity-based cold-drug-target split, "
             "CCP: similarity-based cold-protein split, SCD: similarity-based cold-drug split, "
             "SC: similarity-based cold-drug-protein split."
    )
    ilp.add_argument(
        "--prot-sim",
        type=str,
        default=None,
        dest="prot_sim",
        help="Provide the name of a method to determine similarity between proteins. This can either be >WLK<, "
             ">mmseqs<, or a filepath to a file storing the clusters in TSV."
    )
    ilp.add_argument(
        "--drug-sim",
        type=str,
        default=None,
        dest="drug_sim",
        help="Provide the name of a method to determine similarity between proteins. This can either be >WLK< or a "
             "filepath to a file storing the clusters in TSV."
    )
    ilp.add_argument(
        "--header",
        action='store_true',
        default=False,
        dest="header",
        help="If true, every given CSV file has a header row."
    )
    ilp.add_argument(
        "--sep",
        default=",",
        dest="sep",
        type=str,
        help="Separator for the data files named CSV in the help."
    )
    ilp.add_argument(
        "--names",
        default=None,
        dest="names",
        nargs="+",
        type=str,
        help="Names of the splits in order of the -s argument."
    )


def validate_args(args):
    if abs(sum(args.splits) - 1) > 0.00001:
        logging.error("Sum of splits ratios is not 1.")
        exit(7)
    if len(args.splits) < 2:
        logging.error("Zero or one split given. This is meaningless as no work to do.")
        exit(4)
    if args.names is not None and len(args.names) != len(args.splits):
        logging.error("Different number of splits and names of the splits.")
        exit(13)
    if args.limit < 0:
        logging.warning("Factor for tolerating split limit is negative. Split limits are reduced by this.")
    if not os.path.exists(args.inter):
        logging.error("The interactions file does not exist.")
        exit(1)
    if not os.path.exists(args.drugs):
        logging.error("The drugs file does not exist.")
        exit(12)
    if args.drug_weights is not None and args.drugs is None:
        logging.warning("Weights for drugs are given but no drugs are given. Weights will be ignored.")
    if args.technique in {"R", "IC", "CC"} and (args.inter is None or args.drugs is None):
        logging.error(f"The technique {args.technique} requires drug and interaction data.")
        exit(2)
    if args.technique in {"ICD", "IC", "CCD", "CC"} and args.drugs is None:
        logging.error(f"The technique {args.technique} required drug data.")
        exit(3)
    if args.technique in {"CCD", "CC"} and args.drug_sim is None:
        logging.error(f"The technique {args.technique} requires drug clustering method.")
        exit(11)
    if args.technique in {"CCP", "CC"} and args.protein_sim is None:
        logging.error(f"The technique {args.technique} requires protein clustering method.")
        exit(10)
    if args.prot_sim not in {"WLK", "mmseqs"} and args.prot_sim is not None and not os.path.exists(args.prot_sim):
        logging.error(f"The protein similarity argument {args.prot_sim} is no valid mode and no file.")
        exit(8)
    if args.drug_sim not in {"WLK"} and args.drug_sim is not None and not os.path.exists(args.drug_sim):
        logging.error(f"The drug similarity argument {args.drug_sim} is no valid mode and no file.")
        exit(9)

    if args.sep == "\\t":
        args.sep = "\t"
    if args.names is None:
        args.names = [f"Split_{i}" for i in range(len(args.splits))]
