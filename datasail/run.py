import logging
import time

from datasail.cluster.clustering import cluster
from datasail.reader.read import read_data
from datasail.report import report
from datasail.solver.solve import run_solver


def bqp_main(**kwargs) -> None:
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
    if any("C" == technique[0] for technique in kwargs["techniques"]):
        e_dataset = cluster(e_dataset)
        f_dataset = cluster(f_dataset)

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

    logging.info("BQP splitting finished and results stored.")
    logging.info(f"Total runtime: {time.time() - start:.5f}s")
