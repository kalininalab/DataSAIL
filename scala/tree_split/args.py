from argparse import ArgumentParser


def tree_args(parser: ArgumentParser):
    tree = parser.add_argument_group(description="Optional arguments for tree method")
    tree.add_argument(
        "-tr",
        default=60,
        type=int,
        dest='tr_size',
        help="size of training set",
    )
    tree.add_argument(
        "-te",
        default=30,
        type=int,
        dest='te_size',
        help="size of test set"
    )
    tree.add_argument(
        "-st",
        default=1.0,
        type=float,
        dest='seq_id_threshold',
        help="sequence identity threshold for undistinguishable sequences - range: [0,1]",
    )
    tree.add_argument(
        "-lw",
        default=False,
        action='store_true',
        dest='length_weighting',
        help="sequence length weighting",
    )
    tree.add_argument(
        "--tree",
        default=False,
        action='store_true',
        dest='tree_file',
        help="print tree file - default: False",
    )


def validate_args(args):
    pass
