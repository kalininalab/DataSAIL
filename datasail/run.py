import time
from typing import Dict, Tuple

from datasail.argparse_patch import remove_patch
from datasail.cluster.clustering import cluster
from datasail.reader.read import read_data, check_duplicates
from datasail.report import report
from datasail.settings import LOGGER, KW_TECHNIQUES, KW_VECTORIZED, KW_EPSILON, KW_RUNS, KW_SPLITS, KW_NAMES, \
    KW_MAX_SEC, KW_MAX_SOL, KW_SOLVER, KW_LOGDIR, NOT_ASSIGNED, KW_OUTDIR, MODE_E, MODE_F
from datasail.solver.solve import run_solver, insert


def datasail_main(**kwargs) -> Tuple[Dict, Dict, Dict]:
    """
    Main routine of DataSAIL. Here the parsed input is aggregated into structures and then split and saved.

    Args:
        **kwargs: Parsed commandline arguments to DataSAIL.
    """
    kwargs = remove_patch(**kwargs)
    start = time.time()
    LOGGER.info("Read data")

    kwargs = check_duplicates(**kwargs)

    # read e-entities and f-entities
    e_dataset, f_dataset, inter, old_inter = read_data(**kwargs)

    # if required, cluster the input otherwise define the cluster-maps to be None
    clusters = list(filter(lambda x: x[0] == "C", kwargs[KW_TECHNIQUES]))
    cluster_e = len(clusters) != 0 and any(c[-1] in {"D", MODE_E} for c in clusters)
    cluster_f = len(clusters) != 0 and any(c[-1] in {"D", MODE_F} for c in clusters)

    if cluster_e:
        LOGGER.info("Cluster first set of entities.")
        e_dataset = cluster(e_dataset, **kwargs)
    if cluster_f:
        LOGGER.info("Cluster second set of entities.")
        f_dataset = cluster(f_dataset, **kwargs)

    if inter is not None:
        if e_dataset.type is not None and f_dataset.type is not None:
            inter = list(filter(lambda x: x[0] in e_dataset.names and x[1] in f_dataset.names, inter))
        elif e_dataset.type is not None:
            inter = list(filter(lambda x: x[0] in e_dataset.names, inter))
        elif f_dataset.type is not None:
            inter = list(filter(lambda x: x[1] in f_dataset.names, inter))
        else:
            raise ValueError()

    LOGGER.info("Split data")
    # split the data into dictionaries mapping interactions, e-entities, and f-entities into the splits
    inter_split_map, e_name_split_map, f_name_split_map, e_cluster_split_map, f_cluster_split_map = run_solver(
        techniques=kwargs[KW_TECHNIQUES],
        vectorized=kwargs[KW_VECTORIZED],
        e_dataset=e_dataset,
        f_dataset=f_dataset,
        inter=inter,
        epsilon=kwargs[KW_EPSILON],
        runs=kwargs[KW_RUNS],
        splits=kwargs[KW_SPLITS],
        split_names=kwargs[KW_NAMES],
        max_sec=kwargs[KW_MAX_SEC],
        max_sol=kwargs[KW_MAX_SOL],
        solver=kwargs[KW_SOLVER],
        log_dir=kwargs[KW_LOGDIR],
    )

    LOGGER.info("Store results")

    # infer interaction assignment from entity assignment if necessary and possible
    if old_inter is not None:
        for technique in kwargs[KW_TECHNIQUES]:
            if len(inter_split_map.get(technique, [])) < kwargs[KW_RUNS]:
                for run in range(kwargs[KW_RUNS]):
                    # How to deal with duplicates in ?CD-splits when interactions are already assigned in the splitting process
                    # TODO: Detect the duplicates in old_inter and assign them based on an id_map

                    if e_name_split_map.get(technique, None) is not None:
                        insert(
                            inter_split_map,
                            technique,
                            {(e, f): e_name_split_map[technique][run].get(e, NOT_ASSIGNED) for e, f in old_inter}
                        )
                    if f_name_split_map.get(technique, None) is not None:
                        insert(
                            inter_split_map,
                            technique,
                            {(e, f): f_name_split_map[technique][run].get(f, NOT_ASSIGNED) for e, f in old_inter},
                        )

    LOGGER.info("BQP splitting finished and results stored.")
    LOGGER.info(f"Total runtime: {time.time() - start:.5f}s")

    if kwargs[KW_OUTDIR] is not None:
        report(
            techniques=kwargs[KW_TECHNIQUES],
            e_dataset=e_dataset,
            f_dataset=f_dataset,
            e_name_split_map=e_name_split_map,
            f_name_split_map=f_name_split_map,
            e_cluster_split_map=e_cluster_split_map,
            f_cluster_split_map=f_cluster_split_map,
            inter_split_map=inter_split_map,
            runs=kwargs[KW_RUNS],
            output_dir=kwargs[KW_OUTDIR],
            split_names=kwargs[KW_NAMES],
        )
    else:
        full_e_name_split_map = fill_split_maps(e_dataset, e_name_split_map)
        full_f_name_split_map = fill_split_maps(f_dataset, f_name_split_map)
        return full_e_name_split_map, full_f_name_split_map, inter_split_map


def fill_split_maps(dataset, name_split_map):
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
