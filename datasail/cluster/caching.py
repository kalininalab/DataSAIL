import pickle
from pathlib import Path
from typing import Optional
from pip._internal.utils.appdirs import user_cache_dir

from datasail.reader.utils import DataSet
from datasail.settings import KW_CACHE_DIR


def load_from_cache(dataset: DataSet, **kwargs) -> Optional[DataSet]:
    """
    Load a dataset from cache.

    Args:
        dataset: dataset that probably has been stored in cache.
        **kwargs: Further arguments to the program regarding caching.

    Returns:
        The dataset if it could be loaded from cache, else none
    """
    if kwargs.get("cache", False):
        name = f"{hex(hash(dataset))[2:34]}.pkl"
        cache_dir = kwargs.get(KW_CACHE_DIR, Path(user_cache_dir("DataSAIL")))
        if (cache_file := (cache_dir / name)).is_file():
            return pickle.load(open(cache_file, "rb"))


def store_to_cache(dataset: DataSet, **kwargs) -> None:
    """
    Store a clustered dataset to the cache for later reloading.

    Args:
        dataset: Dataset to store
        **kwargs: Further arguments to the program regarding caching.
    """
    if kwargs.get("cache", False):
        name = f"{hex(hash(dataset))[2:34]}.pkl"
        cache_dir = kwargs.get(KW_CACHE_DIR, Path(user_cache_dir("DataSAIL")))
        cache_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(dataset, open(cache_dir / name, "wb"))
