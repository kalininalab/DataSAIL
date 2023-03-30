import logging
import time
from typing import Dict, Tuple

from datasail.cluster.clustering import cluster
from datasail.reader.read import read_data
from datasail.report import report
from datasail.solver.solve import run_solver


def bqp_main(**kwargs) -> Tuple[Dict, Dict, Dict]:
    """
    Main routine of DataSAIL. Here the parsed input is aggregated into structures and then split and saved.

    Args:
        **kwargs: Parsed commandline arguments to DataSAIL.
    """
    start = time.time()
    logging.info("Read data")

    # read e-entities and f-entities in
    e_dataset, f_dataset, inter = read_data(**kwargs)

    # if required, cluster the input otherwise define the cluster-maps to be None
    clusters = list(filter(lambda x: x[0] == "C", kwargs["techniques"]))
    cluster_e = len(clusters) != 0 and any(c[-1] in {"D", "e"} for c in clusters)
    cluster_f = len(clusters) != 0 and any(c[-1] in {"D", "f"} for c in clusters)
    print(cluster_e)
    if cluster_e:
        e_dataset = cluster(e_dataset, **kwargs)
    if cluster_f:
        f_dataset = cluster(f_dataset, **kwargs)

    logging.info("Split data")
    # split the data into dictionaries mapping interactions, e-entities, and f-entities into the splits
    inter_split_map, e_name_split_map, f_name_split_map, e_cluster_split_map, f_cluster_split_map = run_solver(
        techniques=kwargs["techniques"],
        vectorized=kwargs["vectorized"],
        e_dataset=e_dataset,
        f_dataset=f_dataset,
        inter=inter,
        epsilon=kwargs["epsilon"],
        splits=kwargs["splits"],
        names=kwargs["names"],
        max_sec=kwargs["max_sec"],
        max_sol=kwargs["max_sol"],
        solver=kwargs["solver"],
    )

    logging.info("Store results")

    # infer interaction assignment from entity assignment if necessary and possible
    if inter is not None:
        for technique in kwargs["techniques"]:
            t = technique[:3]
            if inter_split_map.get(technique, None) is None:
                if e_name_split_map.get(t, None) is not None and f_name_split_map.get(t, None) is None:
                    inter_split_map[technique] = [(e, f, e_name_split_map[t][e]) for e, f in inter]
                elif e_name_split_map.get(t, None) is None and f_name_split_map.get(t, None) is not None:
                    inter_split_map[technique] = [(e, f, f_name_split_map[t][f]) for e, f in inter]
                elif e_name_split_map.get(t, None) is not None and f_name_split_map.get(t, None) is not None:
                    inter_split_map[technique] = [
                        (e, f, e_name_split_map[t][e]) for e, f in inter if e_name_split_map[t][e] == f_name_split_map[t][f]
                    ]

    logging.info("BQP splitting finished and results stored.")
    logging.info(f"Total runtime: {time.time() - start:.5f}s")

    if kwargs["output"] is not None:
        report(
            kwargs["techniques"],
            e_dataset,
            f_dataset,
            e_name_split_map,
            f_name_split_map,
            e_cluster_split_map,
            f_cluster_split_map,
            inter_split_map,
            kwargs["output"],
            kwargs["names"],
        )
    else:
        return e_name_split_map, f_name_split_map, inter_split_map
