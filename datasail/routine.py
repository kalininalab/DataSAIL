import random
import time
import pickle
from typing import Dict, Tuple, Optional

import numpy as np

from datasail.argparse_patch import remove_patch
from datasail.cluster.clustering import cluster
from datasail.reader.read import read_data
from datasail.reader.utils import DataSet
from datasail.report import report
from datasail.settings import DIM_1, KW_CLI, LOGGER, KW_INTER, KW_TECHNIQUES, KW_EPSILON, KW_RUNS, KW_SPLITS, KW_NAMES, \
    KW_MAX_SEC, KW_SOLVER, KW_LOGDIR, NOT_ASSIGNED, KW_OUTDIR, MODE_E, MODE_F, DIM_2, SRC_CL, KW_DELTA, \
    KW_E_CLUSTERS, KW_F_CLUSTERS, KW_CC, CDHIT, INSTALLED, FOLDSEEK, TMALIGN, CDHIT_EST, DIAMOND, MMSEQS, MASH, TEC_R, TEC_I1, TEC_C1, TEC_I2, TEC_C2, MODE_E, MODE_F, KW_LINKAGE, KW_OVERFLOW
from datasail.solver.overflow import check_dataset
from datasail.solver.solve import run_solver, random_inter_split


def list_cluster_algos():
    """
    List all available clustering algorithms.
    """

    print("Available clustering algorithms:", "\tECFP", sep="\n")
    for algo, name in [(CDHIT, "CD-HIT"), (CDHIT_EST, "CD-HIT-EST"), (DIAMOND, "DIAMOND"), (MMSEQS, "MMseqs, MMseqs2"),
                       (MASH, "MASH"), (FOLDSEEK, "FoldSeek"), (TMALIGN, "TMalign")]:
        if INSTALLED[algo]:
            print("\t", name, sep="")


def tech2oneD(tech: str) -> tuple[str, str]:
    if tech == TEC_I2:
        return TEC_I1 + MODE_E, TEC_I1 + MODE_F
    elif tech == TEC_C2:
        return TEC_C1 + MODE_E, TEC_C1 + MODE_F
    else:
        raise ValueError(f"Technique {tech} is not a two-dimensional technique.")


def datasail_main(**kwargs) -> Optional[Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]]:
    """
    Main routine of DataSAIL. Here the parsed input is aggregated into structures and then split and saved.

    Args:
        **kwargs: Parsed commandline arguments to DataSAIL.
    """
    kwargs = remove_patch(**kwargs)
    if kwargs.get(KW_CC, False):
        list_cluster_algos()
        return None

    # seed the stuff
    random.seed(42)
    np.random.seed(42)

    start = time.time()
    LOGGER.info("Read data")

    # read e-entities and f-entities
    e_dataset, f_dataset, inter = read_data(**kwargs)

    # if required, cluster the input otherwise define the cluster-maps to be None
    clusters = list(filter(lambda x: x[0].startswith(SRC_CL), kwargs[KW_TECHNIQUES]))
    cluster_e = len(clusters) != 0 and any(c[-1] in {DIM_2, MODE_E} for c in clusters)
    cluster_f = len(clusters) != 0 and any(c[-1] in {DIM_2, MODE_F} for c in clusters)

    if cluster_e:
        LOGGER.info("Cluster first set of entities.")
        e_dataset = cluster(e_dataset, **kwargs)
    if cluster_f:
        LOGGER.info("Cluster second set of entities.")
        f_dataset = cluster(f_dataset, **kwargs)

    if inter is not None:
        if e_dataset.type is not None and f_dataset.type is not None:
            new_inter = [(e_dataset.id_map[x[0]], f_dataset.id_map[x[1]])
                         for x in filter(lambda x: x[0] in e_dataset.id_map and x[1] in f_dataset.id_map, inter)]
        elif e_dataset.type is not None:
            new_inter = [(e_dataset.id_map[x[0]], x[1]) for x in filter(lambda x: x[0] in e_dataset.id_map, inter)]
        elif f_dataset.type is not None:
            new_inter = [(x[0], f_dataset.id_map[x[1]]) for x in filter(lambda x: x[1] in f_dataset.id_map, inter)]
        else:
            raise ValueError()
    else:
        new_inter = None
    
    e_dataset, pre_e_name_split_map, pre_e_cluster_split_map, e_split_ratios, e_split_names = check_dataset(
        e_dataset,
        kwargs[KW_SPLITS],
        kwargs[KW_NAMES],
        kwargs[KW_OVERFLOW],
        kwargs[KW_LINKAGE],
        (TEC_I1 + MODE_E) if any(x in kwargs[KW_TECHNIQUES] for x in [TEC_I1 + MODE_E, TEC_I2]) else None, 
        TEC_I2 in kwargs[KW_TECHNIQUES],
        (TEC_C1 + MODE_E) if any(x in kwargs[KW_TECHNIQUES] for x in [TEC_C1 + MODE_E, TEC_C2]) else None,
        TEC_C2 in kwargs[KW_TECHNIQUES],
    )
    f_dataset, pre_f_name_split_map, pre_f_cluster_split_map, f_split_ratios, f_split_names = check_dataset(
        f_dataset,
        kwargs[KW_SPLITS],
        kwargs[KW_NAMES],
        kwargs[KW_OVERFLOW],
        kwargs[KW_LINKAGE],
        (TEC_I1 + MODE_F) if any(x in kwargs[KW_TECHNIQUES] for x in [TEC_I1 + MODE_F, TEC_I2]) else None,
        TEC_I2 in kwargs[KW_TECHNIQUES],
        (TEC_C1 + MODE_F) if any(x in kwargs[KW_TECHNIQUES] for x in [TEC_C1 + MODE_F, TEC_C2]) else None,
        TEC_C2 in kwargs[KW_TECHNIQUES],
    )
    split_ratios = e_split_ratios | f_split_ratios
    split_names = e_split_names | f_split_names

    LOGGER.info("Split data")

    # split the data into dictionaries mapping interactions, e-entities, and f-entities into the splits
    inter_split_map = {}
    if TEC_R in kwargs[KW_TECHNIQUES]:
        inter_split_map[TEC_R] = random_inter_split(kwargs[KW_RUNS], inter, kwargs[KW_SPLITS], kwargs[KW_NAMES])
    
    e_name_split_map, f_name_split_map, e_cluster_split_map, f_cluster_split_map = run_solver(
        techniques=kwargs[KW_TECHNIQUES],
        e_dataset=e_dataset,
        f_dataset=f_dataset,
        delta=kwargs[KW_DELTA],
        epsilon=kwargs[KW_EPSILON],
        runs=kwargs[KW_RUNS],
        split_ratios=split_ratios,
        split_names=split_names,
        max_sec=kwargs[KW_MAX_SEC],
        solver=kwargs[KW_SOLVER],
        log_dir=kwargs[KW_LOGDIR],
    )
    # integrate pre_maps into the split maps
    for run in range(kwargs[KW_RUNS]):
        for map_, pre_map in [(e_name_split_map, pre_e_name_split_map),
                                (f_name_split_map, pre_f_name_split_map),
                                (e_cluster_split_map, pre_e_cluster_split_map),
                                (f_cluster_split_map, pre_f_cluster_split_map)]:    
            for technique in kwargs[KW_TECHNIQUES]:
                if technique == "R":
                    continue
                if technique[1] == DIM_1:
                    if technique not in pre_map:
                        continue
                    if technique not in map_:
                        map_[technique] = []
                    if run >= len(map_[technique]):
                        map_[technique].append({})
                    map_[technique][run].update(pre_map[technique])
                else:
                    for one_d_tech in tech2oneD(technique):
                        if one_d_tech not in pre_map:
                            continue
                        if technique not in map_:
                            map_[technique] = []
                        if run >= len(map_[technique]):
                            map_[technique].append({})
                        map_[technique][run].update(pre_map[one_d_tech])

    if all(len(e_run) == 0 for e_techs in e_name_split_map.values() for e_run in e_techs) and \
            all(len(f_run) == 0 for f_techs in f_name_split_map.values() for f_run in f_techs) and \
            "R" not in inter_split_map:
        LOGGER.error("No assignments could be made for any technique! Please check your input data and values for cluster-numbers, delta, and epsilon.")
        return None, None, None
    LOGGER.info("Store results")

    # infer interaction assignment from entity assignment if necessary and possible
    if new_inter is not None:
        for technique in kwargs[KW_TECHNIQUES]:
            if technique == TEC_R:
                continue
            inter_split_map[technique] = []
            for run in range(kwargs[KW_RUNS]):
                inter_split_map[technique].append({})
                for e, f in inter:
                    try:
                        if technique.endswith(DIM_2):
                            e_assi = e_name_split_map[technique][run].get(e_dataset.id_map.get(e, ""), NOT_ASSIGNED)
                            f_assi = f_name_split_map[technique][run].get(f_dataset.id_map.get(f, ""), NOT_ASSIGNED)
                            inter_split_map[technique][-1][(e, f)] = e_assi if e_assi == f_assi else NOT_ASSIGNED
                        elif technique in e_name_split_map:
                            inter_split_map[technique][-1][(e, f)] = e_name_split_map[technique][run].get(e_dataset.id_map.get(e, ""), NOT_ASSIGNED)
                        elif technique in f_name_split_map:
                            inter_split_map[technique][-1][(e, f)] = f_name_split_map[technique][run].get(f_dataset.id_map.get(f, ""), NOT_ASSIGNED)
                        else:
                            raise ValueError()
                    except:
                        pass

    LOGGER.info("ILP finished and results stored.")
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
    if not kwargs[KW_CLI]:
        full_e_name_split_map = fill_split_maps(e_dataset, e_name_split_map)
        full_f_name_split_map = fill_split_maps(f_dataset, f_name_split_map)
        return full_e_name_split_map, full_f_name_split_map, inter_split_map


def fill_split_maps(dataset: DataSet, name_split_map: Dict) -> Dict:
    """
    Convert structure of name split map.

    Args:
        dataset: dataset to work on
        name_split_map: Mapping of names to splits

    Returns:
        Converted mapping
    """
    if dataset.type is not None:
        full_name_split_map = {}
        for technique, runs in name_split_map.items():
            full_name_split_map[technique] = []
            for r in range(len(runs)):
                full_name_split_map[technique].append({})
                for name, rep in dataset.id_map.items():
                    full_name_split_map[technique][-1][name] = name_split_map[technique][r][rep]
        return full_name_split_map
    else:
        return {}
