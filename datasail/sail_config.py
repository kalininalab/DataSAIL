import logging
import os
from pathlib import Path
from datasail.routine import datasail_main
from datasail.settings import DIST_ALGOS, FORMATTER, KW_CACHE, KW_CACHE_DIR, KW_CLI, KW_CLUSTERS, KW_DATA, KW_DELTA, \
        KW_DIST, KW_EPSILON, KW_INTER, KW_LINKAGE, KW_LOGDIR, KW_MAX_SEC, KW_MAX_SOL, KW_NAMES, KW_OUTDIR, KW_RUNS, \
        KW_SIM, KW_SPLITS, KW_STRAT, KW_THREADS, KW_VERBOSE, KW_WEIGHTS, LOGGER, SIM_ALGOS, VERB_MAP


def error(msg: str, cli: bool) -> None:
    """
    Print an error message with an individual error code to the commandline. Afterward, the program is ended.

    Args:
        msg: Error message
        error_code: Code of the error to identify it
        cli: boolean flag indicating that this program has been started from commandline
    """
    LOGGER.error(msg)
    if cli:
        exit(1)
    else:
        raise ValueError(msg)


def validate_general_args(**kwargs) -> dict[str, object]:
    """
    Validate the arguments given to the program.

    Notes:
        next error code: 26

    Args:
        **kwargs: Arguments in kwargs-format

    Returns:
        The kwargs in case something has been adjusted, e.g. splits normalization or naming
    """
    # create output directory
    output_created = False
    if kwargs[KW_OUTDIR] is not None and not kwargs[KW_OUTDIR].is_dir():
        output_created = True
        kwargs[KW_OUTDIR].mkdir(parents=True, exist_ok=True)

    LOGGER.setLevel(VERB_MAP[kwargs[KW_VERBOSE]])
    LOGGER.handlers[0].setLevel(level=VERB_MAP[kwargs[KW_VERBOSE]])

    if kwargs[KW_OUTDIR] is not None:
        kwargs[KW_LOGDIR] = kwargs[KW_OUTDIR] / "logs"
        kwargs[KW_LOGDIR].mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(kwargs[KW_LOGDIR] / "general.log")
        file_handler.setLevel(level=VERB_MAP[kwargs[KW_VERBOSE]])
        file_handler.setFormatter(FORMATTER)
        LOGGER.addHandler(file_handler)
    else:
        kwargs[KW_LOGDIR] = None

    if output_created:
        LOGGER.warning("Output directory does not exist, DataSAIL creates it automatically")

    LOGGER.info("Validating arguments")

    # check splits to be more than 1 and their fractions sum up to 1 and check the names
    if len(kwargs[KW_SPLITS]) < 2:
        error("Less then two splits required. This is no useful input, please check the input again.", kwargs[KW_CLI])
    if kwargs[KW_NAMES] is None:
        LOGGER.warning("No names for the splits provided. The results splits will be called Split001, Split002, ...")
        kwargs[KW_NAMES] = [f"Split{x+1:03d}" for x in range(len(kwargs[KW_SPLITS]))]
    elif len(kwargs[KW_SPLITS]) != len(kwargs[KW_NAMES]):
        error("Different number of splits and names. You have to give the same number of splits and names for them.", kwargs[KW_CLI])
    elif len(kwargs[KW_NAMES]) != len(set(kwargs[KW_NAMES])):
        error("At least two splits will have the same name. Please check the naming of the splits again to have unique names", kwargs[KW_CLI])
    kwargs[KW_SPLITS] = [x / sum(kwargs[KW_SPLITS]) for x in kwargs[KW_SPLITS]]

    # check search termination criteria
    if kwargs[KW_MAX_SEC] < 1:
        error("The maximal search time must be a positive integer.", kwargs[KW_CLI])
    if kwargs[KW_MAX_SOL] < 1:
        error("The maximal number of solutions to look at has to be a positive integer.", kwargs[KW_CLI])
    if kwargs[KW_THREADS] < 0:
        error("The number of threads to use has to be a non-negative integer.", kwargs[KW_CLI])
    if kwargs[KW_THREADS] == 0:
        kwargs[KW_THREADS] = os.cpu_count()
    else:
        kwargs[KW_THREADS] = min(kwargs[KW_THREADS], os.cpu_count())

    # check the interaction file
    if kwargs[KW_INTER] is not None and isinstance(kwargs[KW_INTER], Path) and not kwargs[KW_INTER].is_file():
        error("The interaction filepath is not valid.", kwargs[KW_CLI])

    # check the epsilon value
    if 1 < kwargs[KW_DELTA] or kwargs[KW_DELTA] < 0:
        error("Delta has to be a real value between 0 and 1.", kwargs[KW_CLI])

    # check the epsilon value
    if 1 < kwargs[KW_EPSILON] or kwargs[KW_EPSILON] < 0:
        error("Epsilon has to be a real value between 0 and 1.", kwargs[KW_CLI])

    # check number of runs to be a positive integer
    if kwargs[KW_RUNS] < 1:
        error("The number of runs cannot be lower than 1.", kwargs[KW_CLI])

    # check the input regarding the caching
    if kwargs[KW_CACHE] and kwargs[KW_CACHE_DIR] is not None:
        kwargs[KW_CACHE_DIR] = Path(kwargs[KW_CACHE_DIR])
        if not kwargs[KW_CACHE_DIR].is_dir():
            LOGGER.warning("Cache directory does not exist, DataSAIL creates it automatically.")
        kwargs[KW_CACHE_DIR].mkdir(parents=True, exist_ok=True)

    if kwargs[KW_LINKAGE] not in ["average", "single", "complete"]:
        error("The linkage method has to be one of 'mean', 'single', or 'complete'.", kwargs[KW_CLI])

    return kwargs


def validate_data_args(counter: int, **kwargs) -> dict[str, object]:
    # syntactically parse the input data for the F-dataset
    if kwargs[KW_DATA] is not None and isinstance(kwargs[KW_DATA], Path) and not kwargs[KW_DATA].exists():
        error(f"The filepath to data {counter} is invalid.", kwargs[KW_CLI])
    if kwargs[KW_WEIGHTS] is not None and isinstance(kwargs[KW_WEIGHTS], Path) and not kwargs[KW_WEIGHTS].is_file():
        error(f"The filepath to the weights of data {counter} is invalid.", kwargs[KW_CLI])
    if kwargs[KW_STRAT] is not None and isinstance(kwargs[KW_STRAT], Path) and not kwargs[KW_STRAT].is_file():
        error(f"The filepath to the stratification of data {counter} is invalid.", kwargs[KW_CLI])
    if kwargs[KW_SIM] is not None and isinstance(kwargs[KW_SIM], str) and kwargs[KW_SIM].lower() not in SIM_ALGOS:
        kwargs[KW_SIM] = Path(kwargs[KW_SIM])
        if not kwargs[KW_SIM].is_file():
            error(f"The similarity metric for data {counter} seems to be a file-input but the filepath is invalid.", kwargs[KW_CLI])
    if kwargs[KW_DIST] is not None and isinstance(kwargs[KW_DIST], str) and \
            kwargs[KW_DIST].lower() not in DIST_ALGOS:
        if not kwargs[KW_DIST].is_file():
            error(f"The distance metric for data {counter} seems to be a file-input but the filepath is invalid.", kwargs[KW_CLI])
    if kwargs[KW_CLUSTERS] < 1:
        error(f"The number of clusters to find in data {counter} has to be a positive integer.", kwargs[KW_CLI])


def datasail(config: dict[str, object]):
    checked_args = validate_general_args(cli=False, **config)
    if isinstance(config["data"], dict):
        config["data"] = [config["data"]]
    checked_args["data"] = []
    for c, data_config in enumerate(config["data"]):
        checked_args["data"].append(validate_data_args(c + 1, cli=False, **data_config))
    datasail_main(checked_args)
