import argparse

from tree.tree import scala


def parse_args():
    parser = argparse.ArgumentParser(
        prog='SCALA',
        description="This tool helps providing the most challenging dataset split for a machine learning model in "
                    "order to prevent information leakage and improve generalizability",
    )
    parser.add_argument(
        "-i",
        required=True,
        type=str,
        dest='input',
        help="directory to input file (FASTA/FASTQ)",
    )
    parser.add_argument(
        "-o",
        required=True,
        type=str,
        dest='output',
        help="directory to save the results in",
    )
    parser.add_argument(
        "-tr",
        default=60,
        type=int,
        dest='tr_size',
        help="size of training set",
    )
    parser.add_argument(
        "-te",
        default=30,
        type=int,
        dest='te_size',
        help="size of test set"
    )
    parser.add_argument(
        "-st",
        default=1.0,
        type=float,
        dest='seq_id_threshold',
        help="sequence identity threshold for undistinguishable sequences - range: [0,1]",
    )
    parser.add_argument(
        "-v",
        default=1,
        type=int,
        dest='verbosity',
        help="verbosity level - range: [0,5]",
    )
    parser.add_argument(
        "-w",
        default=None,
        type=str,
        dest='weight_file',
        help="filepath of weight file tsv in TSV-format: >Sequence ID< >tab< >weight value<",
    )
    parser.add_argument(
        "-lw",
        default=False,
        action='store_true',
        dest='length_weighting',
        help="sequence length weighting",
    )
    parser.add_argument(
        "--tree",
        default=False,
        action='store_true',
        dest='tree_file',
        help="print tree file - default: False",
    )
    return parser.parse_args()


if __name__ == "__main__":
    scala(parse_args())
