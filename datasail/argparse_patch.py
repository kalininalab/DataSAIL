from typing import List

from datasail.settings import KW_E_ARGS, KW_F_ARGS


def insert_patch(args: List[str]) -> List[str]:
    """
    Insert a patch into the arguments to make sure that the arguments are not parsed as flags.
    Compare: https://stackoverflow.com/questions/76955386/parsing-arguments-in-argparse-into-other-arguments

    Args:
        args: arguments as list of strings as extracted from sys.argv

    Returns:

    """
    idx = {"E": None, "F": None}

    # identify the indices of e_args and f_args in the list of arguments
    for i, elem in enumerate(args):
        if elem == "--" + KW_E_ARGS.replace("_", "-"):
            idx["E"] = i
        elif elem == "--" + KW_F_ARGS.replace("_", "-"):
            idx["F"] = i

    # insert a patch to make sure that the values of e_args and f_args are not parsed as flags
    if idx["E"] is not None:
        args[idx["E"] + 1] = "X" + args[idx["E"] + 1]
    if idx["F"] is not None:
        args[idx["F"] + 1] = "X" + args[idx["F"] + 1]

    return args


def remove_patch(**kwargs):
    """
    Remove the patch that was inserted by insert_patch().

    Args:
        **kwargs: Parsed commandline arguments to DataSAIL.

    Returns:
        Parsed commandline arguments after fixing the patch.
    """
    if kwargs.get(KW_E_ARGS, None) is not None:
        kwargs[KW_E_ARGS] = kwargs[KW_E_ARGS][1:]
    if kwargs.get(KW_F_ARGS, None) is not None:
        kwargs[KW_F_ARGS] = kwargs[KW_F_ARGS][1:]
    return kwargs
