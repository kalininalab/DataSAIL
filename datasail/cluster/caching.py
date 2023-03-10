import os.path
import pickle
from typing import Optional
from pip._internal.utils.appdirs import user_cache_dir

from datasail.reader.utils import DataSet


def load_from_cache(dataset: DataSet, **kwargs) -> Optional[DataSet]:
    """
    Load a dataset from cache.

    Args:
        dataset: dataset that probably has been stored in cache.
        **kwargs: Further arguments to the program regarding caching.

    Returns:

    """
    if kwargs.get("cache", False):
        name = f"{hex(hash(dataset))[2:34]}.pkl"
        cache_dir = kwargs.get("cache_dir", user_cache_dir("DataSAIL"))
        if os.path.isfile(os.path.join(cache_dir, name)):
            return pickle.load(open(os.path.join(cache_dir, name), "rb"))


def store_to_cache(dataset: DataSet, **kwargs) -> None:
    """
    Store a clustered dataset to the cache for later reloading.

    Args:
        dataset: Dataset to store
        **kwargs: Further arguments to the program regarding caching.
    """
    if kwargs.get("cache", False):
        name = f"{hex(hash(dataset))[2:34]}.pkl"
        cache_dir = kwargs.get("cache_dir", user_cache_dir("DataSAIL"))
        os.makedirs(cache_dir, exist_ok=True)
        pickle.dump(dataset, open(os.path.join(cache_dir, name), "wb"))
