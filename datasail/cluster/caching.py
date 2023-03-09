import os.path
import pickle
from typing import Optional, List, Dict, Tuple
from pip._internal.utils.appdirs import user_cache_dir

import numpy as np

from datasail.reader.utils import DataSet


def load_from_cache(dataset: DataSet, **kwargs) -> Optional[Tuple[List[str], Dict[str, str], np.ndarray]]:
    if kwargs["cache"]:
        name = f"{hex(hash(dataset))[2:34]}.pkl"
        cache_dir = kwargs.get("cache_dir", user_cache_dir("DataSAIL"))
        if os.path.isfile(os.path.join(cache_dir, name)):
            return pickle.load(open(os.path.join(cache_dir, name), "rb"))


def store_to_cache(dataset: DataSet, names, mapping, matrix, **kwargs) -> None:
    if kwargs["cache"]:
        name = f"{hex(hash(dataset))[2:34]}.pkl"
        cache_dir = kwargs.get("cache_dir", user_cache_dir("DataSAIL"))
        os.makedirs(cache_dir, exist_ok=True)
        pickle.dump((names, mapping, matrix), open(os.path.join(cache_dir, name), "wb"))
