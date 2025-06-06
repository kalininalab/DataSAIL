import time
from pathlib import Path
from typing import Any, Callable, Generator, Optional

from datasail.reader.read_genomes import read_genome_data
from datasail.reader.read_molecules import read_molecule_data
from datasail.reader.read_other import read_other_data
from datasail.reader.read_proteins import read_protein_data
from datasail.reader.utils import read_csv
from datasail.dataset import DataSet
from datasail.cluster.clustering import cluster
from datasail.solver.solve import run_solver
from datasail.report import report
from datasail.constants import G_TYPE, KW_ARGS, KW_CLUSTERS, KW_DATA, KW_DIST, KW_INTER, KW_SIM, KW_STRAT, KW_TYPE, KW_WEIGHTS, M_TYPE, O_TYPE, P_TYPE, \
    KW_DATA, LOGGER, KW_INTER, KW_TECHNIQUES, KW_EPSILON, KW_RUNS, KW_SPLITS, KW_NAMES, \
    KW_MAX_SEC, KW_MAX_SOL, KW_SOLVER, KW_LOGDIR, NOT_ASSIGNED, KW_OUTDIR, DIM_2, SRC_CL, KW_DELTA


def datasail_main(**kwargs) -> Optional[tuple[dict, dict, dict]]:
    """
    Main routine of DataSAIL. Here the parsed input is aggregated into structures and then split and saved.

    Args:
        **kwargs: Parsed commandline arguments to DataSAIL.
    """
    start = time.time()
    LOGGER.info("Read data")

    # read e-entities and f-entities
    inter = read_inter(**kwargs)
    datasets = read_data(inter, kwargs[KW_DATA])

    # if required, cluster the input otherwise define the cluster-maps to be None
    clusters = list(filter(lambda x: x[0].startswith(SRC_CL), kwargs[KW_TECHNIQUES]))
    clusterings = [len(clusters) != 0 and any(d + 1 in set(c[1:].split(".")) for c in clusters) for d in len(datasets)]

    for i, (clustering, dataset) in enumerate(zip(clusterings, datasets)):
        if clustering:
            LOGGER.info(f"Clustering data {i + 1}.")
            dataset = cluster(dataset, **kwargs)

    if inter is not None:
        LOGGER.debug("Rename interactions based on id_maps")
        new_inter = []
        for interaction in inter:
            build = []
            for dataset, entity in zip(datasets, interaction):
                if dataset.type is not None:
                    build.append(entity)
                else:
                    new_inter.append(dataset.id_map[entity])
            new_inter.append(build)
    else:
        new_inter = None

    LOGGER.info("Split data")

    # split the data into dictionaries mapping interactions, e-entities, and f-entities into the splits
    inter_split_map, dataset_name_split_maps = run_solver(
        techniques=kwargs[KW_TECHNIQUES],
        inter=new_inter,
        delta=kwargs[KW_DELTA],
        epsilon=kwargs[KW_EPSILON],
        runs=kwargs[KW_RUNS],
        splits=kwargs[KW_SPLITS],
        split_names=kwargs[KW_NAMES],
        max_sec=kwargs[KW_MAX_SEC],
        max_sol=kwargs[KW_MAX_SOL],
        solver=kwargs[KW_SOLVER],
        log_dir=kwargs[KW_LOGDIR],
        datasets=datasets,
    )

    LOGGER.info("Store results")

    # infer interaction assignment from entity assignment if necessary and possible
    output_inter_split_map = dict()
    if new_inter is not None:
        for technique in kwargs[KW_TECHNIQUES]:
            output_inter_split_map[technique] = []
            for run in range(kwargs[KW_RUNS]):
                output_inter_split_map[technique].append(dict())
                for e, f in inter:
                    if technique.endswith(DIM_2) or technique == "R":
                        output_inter_split_map[technique][-1][(e, f)] = inter_split_map[technique][run].get(
                            (e_dataset.id_map.get(e, ""), f_dataset.id_map.get(f, "")), NOT_ASSIGNED)
                    elif technique in e_name_split_map:
                        output_inter_split_map[technique][-1][(e, f)] = e_name_split_map[technique][run].get(
                            e_dataset.id_map.get(e, ""), NOT_ASSIGNED)
                    elif technique in f_name_split_map:
                        output_inter_split_map[technique][-1][(e, f)] = f_name_split_map[technique][run].get(
                            f_dataset.id_map.get(f, ""), NOT_ASSIGNED)
                    else:
                        raise ValueError()

    LOGGER.info("BQP splitting finished and results stored.")
    LOGGER.info(f"Total runtime: {time.time() - start:.5f}s")

    if kwargs[KW_OUTDIR] is not None:
        report(
            e_dataset=datasets,
            techniques=kwargs[KW_TECHNIQUES],
            e_name_split_map=e_name_split_map,
            f_name_split_map=f_name_split_map,
            e_cluster_split_map=e_cluster_split_map,
            f_cluster_split_map=f_cluster_split_map,
            inter_split_map=output_inter_split_map,
            runs=kwargs[KW_RUNS],
            output_dir=kwargs[KW_OUTDIR],
            split_names=kwargs[KW_NAMES],
        )
    else:
        full_e_name_split_map = fill_split_maps(e_dataset, e_name_split_map)
        full_f_name_split_map = fill_split_maps(f_dataset, f_name_split_map)
        return full_e_name_split_map, full_f_name_split_map, output_inter_split_map


def read_inter(**kwargs: Any) -> Optional[list[tuple]]:
    """
    Read the interactions from the input. The code will always read as many interactions 
    from each input line as there are entities in the dataset.

    Args:
        kwargs: The whole, parsed configuration given to DataSAIL
    """
    # TODO: Semantic checks of arguments
    num_entities = len(kwargs[KW_DATA]) if isinstance(kwargs[KW_DATA], list) else 1
    if kwargs.get(KW_INTER, None) is None:
        return None
    elif isinstance(kwargs[KW_INTER], Path):
        if kwargs[KW_INTER].is_file():
            if kwargs[KW_INTER].suffix[1:] == "tsv":
                return list(tuple(x) for x in read_csv(kwargs[KW_INTER], "\t", num_entities))
            elif kwargs[KW_INTER].suffix[1:] == "csv":
                return list(tuple(x) for x in read_csv(kwargs[KW_INTER], ",", num_entities))
            else:
                raise ValueError()
        else:
            raise ValueError()
    elif isinstance(kwargs[KW_INTER], (list, tuple, Generator)):
        return [x[:num_entities] for x in kwargs[KW_INTER]]
    elif isinstance(kwargs[KW_INTER], Callable):
        return [x[:num_entities] for x in kwargs[KW_INTER]()]
    else:
        raise ValueError(f"Unknown type {type(kwargs[KW_INTER])} found for ")


def read_data(inter: list[tuple], **kwargs) -> list[DataSet]:
    """
    Read data from the input arguments.

    Args:
        **kwargs: Arguments from commandline

    Returns:
        A list with all datasets storing the information on the input entities
    """
    return [
        read_data_type(data_kwargs[KW_TYPE])(
            data_kwargs[KW_DATA], data_kwargs[KW_WEIGHTS], data_kwargs[KW_STRAT], data_kwargs[KW_SIM], 
            data_kwargs[KW_DIST], inter, i, data_kwargs[KW_CLUSTERS], data_kwargs[KW_ARGS],
        ) for i, data_kwargs in enumerate(kwargs[KW_DATA])
    ]


def read_data_type(data_type: chr) -> Callable:
    """
    Convert single-letter representation of the type of data to handle to the full name.

    Args:
        data_type: Single letter representation of the type of data

    Returns:
        full name of the type of data
    """
    if data_type == P_TYPE:
        return read_protein_data
    elif data_type == M_TYPE:
        return read_molecule_data
    elif data_type == G_TYPE:
        return read_genome_data
    elif data_type == O_TYPE:
        return read_other_data
    else:
        raise ValueError(f"Unknown type of data. Found {data_type}, expected one of {P_TYPE}, {M_TYPE}, {G_TYPE}, or {O_TYPE}.")


def fill_split_maps(dataset: DataSet, name_split_map: dict) -> dict:
    """
    Convert structure of name split map.

    Args:
        dataset: dataset to work on
        name_split_map: Mapping of names to splits

    Returns:
        Converted mapping
    """
    if dataset.type is not None:
        full_name_split_map = dict()
        for technique, runs in name_split_map.items():
            full_name_split_map[technique] = []
            for r, run in enumerate(runs):
                full_name_split_map[technique].append(dict())
                for name, rep in dataset.id_map.items():
                    full_name_split_map[technique][-1][name] = name_split_map[technique][r][rep]
        return full_name_split_map
    else:
        return {}
