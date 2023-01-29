import logging
from typing import List, Set, Tuple, Optional, Dict, Union

import cvxpy
import numpy as np


def solve_icd_bqp(
        drug_clusters: List[Union[str, int]],
        drug_similarities: Optional[np.ndarray],
        drug_distances: Optional[np.ndarray],
        drug_threshold: float,
        prot_clusters: List[Union[str, int]],
        prot_similarities: Optional[np.ndarray],
        prot_distances: Optional[np.ndarray],
        prot_threshold: float,
        inter: np.ndarray,
        limit: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
) -> Optional[Tuple[List[Tuple[str, str, str]], Dict[str, str], Dict[str, str]]]:
    logging.info("Define optimization problem")

    alpha = 0.1
    inter_count = np.sum(inter)
    min_lim = [split * inter_count * (1 - limit) for split in splits]
    max_lim = [split * inter_count * (1 - limit) for split in splits]

    x_d = cvxpy.Variable((len(drug_clusters), len(splits)), boolean=True)
    x_p = cvxpy.Variable((len(prot_clusters), len(splits)), boolean=True)
    x_e = cvxpy.Variable((len(drug_clusters), len(prot_clusters), len(splits)), boolean=True)

    constraints = [
        cvxpy.sum(x_d, axis=1) == np.ones((len(drug_clusters))),
        cvxpy.sum(x_p, axis=1) == np.ones((len(prot_clusters))),
        cvxpy.sum(x_e, axis=2) <= np.ones((len(drug_clusters), len(prot_clusters))),
        min_lim <= cvxpy.sum(cvxpy.sum(cvxpy.multiply(inter, x_e), axis=0), axis=1),
        cvxpy.sum(cvxpy.sum(cvxpy.multiply(inter, x_e), axis=0), axis=1) <= max_lim,
        None,
        None,
        x_d >= cvxpy.sum(x_e, axis=1),
        x_p >= cvxpy.sum(x_e, axis=0),
    ]
    if drug_similarities is not None:
        constraints += [
            cvxpy.square(cvxpy.hstack([x_d for _ in range(len(drug_clusters))]) - cvxpy.vstack([x_d for _ in range(len(drug_clusters))])) * drug_similarities <= drug_threshold,
        ]
    else:
        constraints += [

        ]

    if prot_similarities is not None:
        constraints += [

        ]
    else:
        constraints += [

        ]

    return None
