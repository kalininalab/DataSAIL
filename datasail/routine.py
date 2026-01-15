import copy
from pathlib import Path
import random
import time
from typing import Any, Callable, Optional, Union

import numpy as np

from datasail.reader.read_genomes import read_genome_data
from datasail.reader.read_molecules import read_molecule_data
from datasail.reader.read_other import read_other_data
from datasail.reader.read_proteins import read_protein_data
from datasail.reader.utils import Technique, read_csv
from datasail.dataset import DataSet
from datasail.cluster.clustering import cluster
from datasail.solver.solve import run_solver, random_inter_split
from datasail.report import report
from datasail.solver.overflow import check_dataset
from datasail.constants import CDHIT, CDHIT_EST, DIAMOND, DIM_1, FOLDSEEK, G_TYPE, INSTALLED, KW_ARGS, KW_CC, KW_CLUSTERS, KW_DATA, KW_DIST, KW_INTER, KW_LINKAGE, \
    KW_OVERFLOW, KW_SIM, KW_STRAT, KW_TYPE, KW_WEIGHTS, M_TYPE, MASH, MMSEQS, MODE_E, MODE_F, O_TYPE, P_TYPE, KW_DATA, LOGGER, \
    KW_INTER, KW_TECHNIQUES, KW_EPSILON, KW_RUNS, KW_SPLITS, KW_NAMES, KW_MAX_SEC, KW_SOLVER, KW_LOGDIR, \
    NOT_ASSIGNED, KW_OUTDIR, DIM_2, SRC_CL, KW_DELTA, TEC_C1, TEC_C2, TEC_I1, TEC_I2, TEC_R, TMALIGN


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


def datasail_main(**kwargs) -> Optional[tuple[dict, dict, dict]]:
    """
    Main routine of DataSAIL. Here the parsed input is aggregated into structures and then split and saved.

    Args:
        **kwargs: Parsed commandline arguments to DataSAIL.
    """
    # seed the stuff
    random.seed(42)
    np.random.seed(42)

    start = time.time()
    LOGGER.info("Read data")

    # read e-entities and f-entities
    inter = read_inter(**kwargs)
    datasets = read_data(inter, kwargs[KW_DATA])
    techniques = Technique.read_techniques(kwargs[KW_TECHNIQUES])

    # if required, cluster the input otherwise define the cluster-maps to be None
    clusters = list(sorted(set([dim.dim for technique in techniques for dim in technique if dim.clustering])))

    for c_idx in clusters:
        LOGGER.info(f"Clustering data {c_idx}.")
        datasets[c_idx - 1] = cluster(datasets[c_idx - 1], **kwargs)

    if inter is not None:
        LOGGER.debug("Rename interactions based on id_maps")
        new_inter = []
        for interaction in inter:
            build = []
            for dataset, entity in zip(datasets, interaction):
                if dataset.type is None:
                    build.append(entity)
                else:
                    build.append(dataset.id_map[entity])
            new_inter.append(build)
    else:
        new_inter = None
    
    name_split_maps, cluster_split_maps, inter_split_map = {}, {}, {}
    for tech in techniques:
        LOGGER.info("Split data")

        # split the data into dictionaries mapping interactions, e-entities, and f-entities into the splits
        if tech.is_random():  # random split
            LOGGER.info("Perform random splitting")
            inter_split_map[Technique(TEC_R)] = random_inter_split(kwargs[KW_RUNS], new_inter, kwargs[KW_SPLITS], kwargs[KW_NAMES])
            continue
        
        tmp_name_split_map, tmp_cluster_split_map = run_solver(
            technique=tech,
            datasets=datasets,
            delta=kwargs[KW_DELTA],
            epsilon=kwargs[KW_EPSILON],
            runs=kwargs[KW_RUNS],
            splits=kwargs[KW_SPLITS],
            names=kwargs[KW_NAMES],
            overflow=kwargs[KW_OVERFLOW],
            linkage=kwargs[KW_LINKAGE],
            max_sec=kwargs[KW_MAX_SEC],
            solver=kwargs[KW_SOLVER],
            log_dir=kwargs[KW_LOGDIR],
        )
        name_split_maps.update(tmp_name_split_map)
        cluster_split_maps.update(tmp_cluster_split_map)
    
        # # integrate pre_maps into the split maps
        # for run in range(kwargs[KW_RUNS]):
        #     for map_, pre_map in [(e_name_split_map, pre_e_name_split_map),
        #                             (f_name_split_map, pre_f_name_split_map),
        #                             (e_cluster_split_map, pre_e_cluster_split_map),
        #                             (f_cluster_split_map, pre_f_cluster_split_map)]:    
        #         for technique in kwargs[KW_TECHNIQUES]:
        #             if technique == "R":
        #                 continue
        #             if technique[1] == DIM_1:
        #                 if technique not in pre_map:
        #                     continue
        #                 if technique not in map_:
        #                     map_[technique] = []
        #                 if run >= len(map_[technique]):
        #                     map_[technique].append({})
        #                 map_[technique][run].update(pre_map[technique])
        #             else:
        #                 for one_d_tech in tech2oneD(technique):
        #                     if one_d_tech not in pre_map:
        #                         continue
        #                     if technique not in map_:
        #                         map_[technique] = []
        #                     if run >= len(map_[technique]):
        #                         map_[technique].append({})
        #                     map_[technique][run].update(pre_map[one_d_tech])

    LOGGER.info("Store results")

    # infer interaction assignment from entity assignment if necessary and possible
    # if new_inter is not None:
    #     for technique in techniques:
    #         if technique.is_random():
    #             continue
    #         inter_split_map[technique] = []
    #         for run in range(kwargs[KW_RUNS]):
    #             inter_split_map[technique].append({})
    #             for interaction in inter: # type: ignore
    #                 try:
    #                     # if technique.endswith(DIM_2):
    #                     #     e_assi = e_name_split_map[technique][run].get(e_dataset.id_map.get(e, ""), NOT_ASSIGNED)
    #                     #     f_assi = f_name_split_map[technique][run].get(f_dataset.id_map.get(f, ""), NOT_ASSIGNED)
    #                     #     inter_split_map[technique][-1][(e, f)] = e_assi if e_assi == f_assi else NOT_ASSIGNED
    #                     # elif technique in e_name_split_map:
    #                     #     inter_split_map[technique][-1][(e, f)] = e_name_split_map[technique][run].get(e_dataset.id_map.get(e, ""), NOT_ASSIGNED)
    #                     # elif technique in f_name_split_map:
    #                     #     inter_split_map[technique][-1][(e, f)] = f_name_split_map[technique][run].get(f_dataset.id_map.get(f, ""), NOT_ASSIGNED)
    #                     # else:
    #                     #     raise ValueError()
    #                     pass
    #                 except:
    #                     pass

    LOGGER.info("ILP finished and results stored.")
    LOGGER.info(f"Total runtime: {time.time() - start:.5f}s")

    # with open("tmp_output.pkl", "wb") as f:
    #     import pickle
    #     pickle.dump((name_split_maps, cluster_split_maps, techniques, inter), f)
    #     exit(0)
    e_name_split_map, f_name_split_map, e_cluster_split_map, f_cluster_split_map, inter_split_map, v1_techniques = convert_v1_output(techniques, name_split_maps, cluster_split_maps, inter_split_map, new_inter)  # new_inter or inter
    if kwargs["switched"]:
        if "I1e" in e_name_split_map:
            f_name_split_map["I1f"] = e_name_split_map.pop("I1e")
            if "I1e" in inter_split_map:
                inter_split_map["I1f"] = inter_split_map.pop("I1e")
            v1_techniques = [tec if tec != "I1e" else "I1f" for tec in v1_techniques]
        if "C1e" in e_name_split_map:
            f_name_split_map["C1f"] = e_name_split_map.pop("C1e")
            f_cluster_split_map["C1f"] = e_cluster_split_map.pop("C1e")
            if "C1e" in inter_split_map:
                inter_split_map["C1f"] = inter_split_map.pop("C1e")
            v1_techniques = [tec if tec != "C1e" else "C1f" for tec in v1_techniques]
        datasets = [DataSet(), datasets[0]]
    
    if kwargs[KW_OUTDIR] is not None:
        report(
            techniques=v1_techniques,
            e_dataset=datasets[0],
            f_dataset=datasets[1] if len(datasets) > 1 else DataSet(),
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
        full_e_name_split_map = fill_split_maps(datasets[0], e_name_split_map) if datasets[0] is not None else {}
        full_f_name_split_map = fill_split_maps(datasets[1], f_name_split_map) if len(datasets) > 1 else {}
        return full_e_name_split_map, full_f_name_split_map, inter_split_map


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
    elif isinstance(kwargs[KW_INTER], list):
        return kwargs[KW_INTER]
    else:
        raise ValueError(f"Unknown type {type(kwargs[KW_INTER])} found for ")


def read_data(inter: Optional[list[tuple]], data_args: Union[list, dict]) -> list[DataSet]:
    """
    Read data from the input arguments.

    Args:
        **kwargs: Arguments from commandline

    Returns:
        A list with all datasets storing the information on the input entities
    """
    if isinstance(data_args, dict):
        data_args = [data_args]
    return [
        read_data_type(data_kwargs[KW_TYPE])(
            data_kwargs[KW_DATA], data_kwargs[KW_WEIGHTS], data_kwargs[KW_STRAT], data_kwargs[KW_SIM], 
            data_kwargs[KW_DIST], inter, i, data_kwargs[KW_CLUSTERS], data_kwargs[KW_ARGS],
        ) for i, data_kwargs in enumerate(data_args)
    ]


def read_data_type(data_type: str) -> Callable:
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


def convert_v1_output(techniques, nsm, csm, ism, inter):
    inv_tech_map = {"R": "R", "I1": "I1e", "I2": "I1f", "S1": "C1e", "S2": "C1f", "I1-I2": "I2", "I1-2": "I2", "S1-S2": "C2", "S1-2": "C2"}
    tech_map = {
        str(x): x for x in techniques
    }
    v1_techniques = [inv_tech_map[tech] for tech in tech_map.keys()]
    e_name_split_map, f_name_split_map, e_cluster_split_map, f_cluster_split_map, inter_split_map = {}, {}, {}, {}, {}

    if "R" in tech_map:
        inter_split_map["R"] = ism[tech_map["R"]]
    
    if "I1" in tech_map and 1 in nsm[tech_map["I1"]]:
        e_name_split_map["I1e"] = nsm[tech_map["I1"]][1]
    if "S1" in tech_map and 1 in nsm[tech_map["S1"]]:
        e_name_split_map["C1e"] = nsm[tech_map["S1"]][1]
    if "S1" in tech_map and 1 in csm[tech_map["S1"]]:
        e_cluster_split_map["C1e"] = csm[tech_map["S1"]][1]
    if "I1-I2" in tech_map and 1 in nsm[tech_map["I1-I2"]]:
        e_name_split_map["I2"] = nsm[tech_map["I1-I2"]][1]
    if "S1-S2" in tech_map and 1 in nsm[tech_map["S1-S2"]]:
        e_name_split_map["C2"] = nsm[tech_map["S1-S2"]][1]
    if "S1-S2" in tech_map and 1 in csm[tech_map["S1-S2"]]:
        e_cluster_split_map["C2"] = csm[tech_map["S1-S2"]][1]

    if "I2" in tech_map and 2 in nsm[tech_map["I2"]]:
        f_name_split_map["I1f"] = nsm[tech_map["I2"]][2]
    if "S2" in tech_map and 2 in nsm[tech_map["S2"]]:
        f_name_split_map["C1f"] = nsm[tech_map["S2"]][2]
    if "S2" in tech_map and 2 in csm[tech_map["S2"]]:
        f_cluster_split_map["C1f"] = csm[tech_map["S2"]][2]
    if "I1-I2" in tech_map and 2 in nsm[tech_map["I1-I2"]]:
        f_name_split_map["I2"] = nsm[tech_map["I1-I2"]][2]
    if "S1-S2" in tech_map and 2 in nsm[tech_map["S1-S2"]]:
        f_name_split_map["C2"] = nsm[tech_map["S1-S2"]][2]
    if "S1-S2" in tech_map and 2 in csm[tech_map["S1-S2"]]:
        f_cluster_split_map["C2"] = csm[tech_map["S1-S2"]][2]

    if inter is not None:
        for tech, name_map in [("I1e", e_name_split_map), ("C1e", e_name_split_map), ("I1f", f_name_split_map), ("C1f", f_name_split_map)]:
            if tech in name_map:
                if tech not in inter_split_map:
                    inter_split_map[tech] = []
                for run in range(len(name_map[tech])):
                    inter_split_map[tech].append({
                        (e, f): name_map[tech][run][e if tech[-1] == "e" else f] for e, f in inter
                    })

        if "I2" in e_name_split_map:
            inter_split_map["I2"] = []
            for run in range(1):
                inter_split_map["I2"].append({})
                for e, f in inter:
                    e_split = e_name_split_map["I2"][run][e]
                    f_split = f_name_split_map["I2"][run][f]
                    if e_split != f_split:
                        inter_split_map["I2"][run][(e, f)] = NOT_ASSIGNED
                    else:
                        inter_split_map["I2"][run][(e, f)] = e_split

        if "C2" in e_name_split_map:
            inter_split_map["C2"] = []
            for run in range(1):
                inter_split_map["C2"].append({})
                for e, f in inter:
                    e_split = e_name_split_map["C2"][run][e]
                    f_split = f_name_split_map["C2"][run][f]
                    if e_split != f_split:
                        inter_split_map["C2"][run][(e, f)] = NOT_ASSIGNED
                    else:
                        inter_split_map["C2"][run][(e, f)] = e_split
    
    return e_name_split_map, f_name_split_map, e_cluster_split_map, f_cluster_split_map, inter_split_map, v1_techniques


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
