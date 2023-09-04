from datasail.settings import KW_E_ARGS, KW_F_ARGS


def insert_patch(args):
    idx = {"E": None, "F": None}
    for i, elem in enumerate(args):
        if elem == "--" + KW_E_ARGS.replace("_", "-"):
            idx["E"] = i
        elif elem == "--" + KW_F_ARGS.replace("_", "-"):
            idx["F"] = i
    if idx["E"] is not None:
        args[idx["E"] + 1] = "X" + args[idx["E"] + 1]
    if idx["F"] is not None:
        args[idx["F"] + 1] = "X" + args[idx["F"] + 1]

    return args


def remove_patch(**kwargs):
    if kwargs.get(KW_E_ARGS, None) is not None:
        kwargs[KW_E_ARGS] = kwargs[KW_E_ARGS][1:]
    if kwargs.get(KW_F_ARGS, None) is not None:
        kwargs[KW_F_ARGS] = kwargs[KW_F_ARGS][1:]
    return kwargs
