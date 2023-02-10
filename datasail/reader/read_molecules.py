import os

import numpy as np

from datasail.reader.utils import read_csv, read_similarity_file, count_inter, ParseInfo


def read_molecule_data(data, weights, sim, dist, max_sim, max_dist, inter, index) -> ParseInfo:
    if data.lower().endswith(".tsv"):
        molecules = dict(read_csv(data, False, "\t"))
    elif os.path.isdir(data):
        pass  # What data-formats could be read here?
    else:
        raise ValueError()

    # parse molecular weights
    if weights is not None:
        molecule_weights = dict((x, float(v)) for x, v in read_csv(weights, False, "\t"))
    elif inter is not None:
        molecule_weights = dict(count_inter(inter, index))
    else:
        molecule_weights = dict((d, 1) for d in list(molecules.keys()))

    # parse molecular similarity
    molecule_similarity, molecule_distance = None, None
    if sim is None and dist is None:
        molecule_similarity = np.ones((len(molecules), len(molecules)))
        molecule_names = list(molecules.keys())
        molecule_threshold = 1
    elif sim is not None and os.path.isfile(sim):
        molecule_names, molecule_similarity = read_similarity_file(sim)
        molecule_threshold = max_sim
    elif dist is not None and os.path.isfile(dist):
        molecule_names, molecule_distance = read_similarity_file(dist)
        molecule_threshold = max_dist
    else:
        if sim is not None:
            molecule_similarity = sim
            molecule_threshold = max_sim
        else:
            molecule_distance = dist
            molecule_threshold = max_dist
        molecule_names = list(molecules.keys())

    return molecule_names, molecules, molecule_weights, molecule_similarity, molecule_distance, molecule_threshold
