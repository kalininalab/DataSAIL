import time
import pickle
from typing import Dict, Tuple, Optional

from datasail.argparse_patch import remove_patch
from datasail.cluster.clustering import cluster
from datasail.reader.read import read_data
from datasail.reader.utils import DataSet
from datasail.report import report
from datasail.settings import LOGGER, KW_INTER, KW_TECHNIQUES, KW_EPSILON, KW_RUNS, KW_SPLITS, KW_NAMES, \
    KW_MAX_SEC, KW_MAX_SOL, KW_SOLVER, KW_LOGDIR, NOT_ASSIGNED, KW_OUTDIR, MODE_E, MODE_F, DIM_2, SRC_CL, KW_DELTA, \
    KW_E_CLUSTERS, KW_F_CLUSTERS, KW_CC, CDHIT, INSTALLED, FOLDSEEK, TMALIGN, CDHIT_EST, DIAMOND, MMSEQS, MASH
from datasail.solver.solve import run_solver


def list_cluster_algos():
    """
    List all available clustering algorithms.
    """

    print("Available clustering algorithms:", "\tECFP", sep="\n")
    for algo, name in [(CDHIT, "CD-HIT"), (CDHIT_EST, "CD-HIT-EST"), (DIAMOND, "DIAMOND"), (MMSEQS, "MMseqs, MMseqs2"),
                       (MASH, "MASH"), (FOLDSEEK, "FoldSeek"), (TMALIGN, "TMalign")]:
        if INSTALLED[algo]:
            print("\t", name, sep="")


def datasail_main(**kwargs) -> Optional[Tuple[Dict, Dict, Dict]]:
    """
    Main routine of DataSAIL. Here the parsed input is aggregated into structures and then split and saved.

    Args:
        **kwargs: Parsed commandline arguments to DataSAIL.
    """
    kwargs = remove_patch(**kwargs)
    if kwargs.get(KW_CC, False):
        list_cluster_algos()
        return None

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

    #split = str(kwargs[KW_INTER]).split("/")[-2]
    #with open(f"/scratch/SCRATCH_SAS/roman/DataSAIL/PLINDER/{split}.pkl", "wb") as f:
    #    pickle.dump((e_dataset, f_dataset), f)
    #with open(f"/scratch/SCRATCH_SAS/roman/DataSAIL/PLINDER/{split}.pkl", "rb") as f:
    #    e_dataset, f_dataset = pickle.load(f)
    #f_dataset.id_map = f_dataset_tmp.id_map

    #print("E_ID_Map is None:", e_dataset.id_map is None)
    #print("F_ID_Map is None:", f_dataset.id_map is None)
    #print("Nones in inter  :", sum([x is None for x in inter]))

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

    LOGGER.info("Split data")

    # split the data into dictionaries mapping interactions, e-entities, and f-entities into the splits
    inter_split_map, e_name_split_map, f_name_split_map, e_cluster_split_map, f_cluster_split_map = run_solver(
        techniques=kwargs[KW_TECHNIQUES],
        e_dataset=e_dataset,
        f_dataset=f_dataset,
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
            techniques=kwargs[KW_TECHNIQUES],
            e_dataset=e_dataset,
            f_dataset=f_dataset,
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
