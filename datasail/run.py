import time
from typing import Dict, Tuple

from datasail.cluster.clustering import cluster
from datasail.reader.read import read_data, check_duplicates
from datasail.report import report
from datasail.settings import LOGGER
from datasail.solver.solve import run_solver, insert


def datasail_main(**kwargs) -> Tuple[Dict, Dict, Dict]:
    """
    Main routine of DataSAIL. Here the parsed input is aggregated into structures and then split and saved.

    Args:
        **kwargs: Parsed commandline arguments to DataSAIL.
    """
    start = time.time()
    LOGGER.info("Read data")

    kwargs = check_duplicates(**kwargs)

    # read e-entities and f-entities
    e_dataset, f_dataset, inter, old_inter = read_data(**kwargs)

    # if required, cluster the input otherwise define the cluster-maps to be None
    clusters = list(filter(lambda x: x[0] == "C", kwargs["techniques"]))
    cluster_e = len(clusters) != 0 and any(c[-1] in {"D", "e"} for c in clusters)
    cluster_f = len(clusters) != 0 and any(c[-1] in {"D", "f"} for c in clusters)

    if cluster_e:
        LOGGER.info("Cluster first set of entities.")
        e_dataset = cluster(e_dataset, **kwargs)
    if cluster_f:
        LOGGER.info("Cluster second set of entities.")
        f_dataset = cluster(f_dataset, **kwargs)

    LOGGER.info("Split data")
    # split the data into dictionaries mapping interactions, e-entities, and f-entities into the splits
    inter_split_map, e_name_split_map, f_name_split_map, e_cluster_split_map, f_cluster_split_map = run_solver(
        techniques=kwargs["techniques"],
        vectorized=kwargs["vectorized"],
        e_dataset=e_dataset,
        f_dataset=f_dataset,
        inter=inter,
        epsilon=kwargs["epsilon"],
        runs=kwargs["runs"],
        splits=kwargs["splits"],
        split_names=kwargs["names"],
        max_sec=kwargs["max_sec"],
        max_sol=kwargs["max_sol"],
        solver=kwargs["solver"],
        log_dir=kwargs["logdir"],
    )

    LOGGER.info("Store results")

    # infer interaction assignment from entity assignment if necessary and possible
    if old_inter is not None:
        for technique in kwargs["techniques"]:
            for run in range(kwargs["runs"]):
                t = technique[:3]
                # How to deal with duplicates in ?CD-splits when interactions are already assigned in the splitting process
                # TODO: Detect the duplicates in old_inter and assign them based on an id_map
                if len(inter_split_map.get(technique, [])) < kwargs["runs"]:
                    if e_name_split_map.get(t, None) is not None:
                        insert(
                            inter_split_map,
                            technique,
                            [(e, f, e_name_split_map[t][run].get(e, "not selected")) for e, f in old_inter]
                        )
                    if f_name_split_map.get(t, None) is not None:
                        insert(
                            inter_split_map,
                            technique,
                            [(e, f, f_name_split_map[t][run].get(f, "not selected")) for e, f in old_inter],
                        )

    LOGGER.info("BQP splitting finished and results stored.")
    LOGGER.info(f"Total runtime: {time.time() - start:.5f}s")

    if kwargs["output"] is not None:
        report(
            techniques=kwargs["techniques"],
            e_dataset=e_dataset,
            f_dataset=f_dataset,
            e_name_split_map=e_name_split_map,
            f_name_split_map=f_name_split_map,
            e_cluster_split_map=e_cluster_split_map,
            f_cluster_split_map=f_cluster_split_map,
            inter_split_map=inter_split_map,
            runs=kwargs["runs"],
            output_dir=kwargs["output"],
            split_names=kwargs["names"],
        )
    else:
        return e_name_split_map, f_name_split_map, inter_split_map

