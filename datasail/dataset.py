from argparse import Namespace
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np

from datasail.constants import UNK_LOCATION, format2ending


def permute(names, similarity=None, distance=None) -> tuple[list[str], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Permute the order of the data the names list and the according distance or similarity matrix.

    Args:
        names: List of names of samples in the dataset
        similarity: Similarity matrix of datapoints in the dataset
        distance: Distance matrix of datapoints in the dataset

    Returns:
        Permuted names, similarity and distance matrix
    """
    permutation = np.random.permutation(len(names))
    names = [names[x] for x in permutation]
    if isinstance(similarity, np.ndarray):
        similarity = similarity[permutation, :]
        similarity = similarity[:, permutation]
    if isinstance(distance, np.ndarray):
        distance = distance[permutation, :]
        distance = distance[:, permutation]
    return names, similarity, distance


@dataclass
class DataSet:
    type: Optional[str] = None  # Type of data, e.g., Protein, Molecule, Genomic, or Other
    format: Optional[str] = None  # Format of the data, e.g., FASTA, PDB, ...
    num_clusters: int = 50  # Number of clusters to compute
    location: Optional[Path] = None  # Path to where the data is stored in memory, if it's stored in memeory
    id_map: Optional[dict[str, str]] = None  # Mapping of input data point names to "names" (used for duplicate removal, or similar)
    names: Optional[list[str]] = None  # Names of individual, unique datapoints
    data: Optional[dict[str, Union[str, np.ndarray]]] = None  # Mapping of "names" to their data, i.e. FASTA/SMILES sequence or PDF file location
    weights: Optional[dict[str, float]] = None  # Mapping of "names" to their weights
    classes: Optional[dict[Any, int]] = None  # Mapping of input data classes to numerical representations theirof
    class_oh: Optional[np.ndarray] = None  # OneHot encodings of the "classes", i.e., basically an identity matrix of dimension max(classes)^2
    stratification: Optional[dict[str, Any]] = None  # Mapping of "names" to the classes to be considered in this dataset
    args: Optional[Namespace] = None  # Custom arguments to the cluster algorithm
    cluster_names: Optional[list[str]] = None  # Names of clusters
    cluster_map: Optional[dict[str, str]] = None  # Mapping of "names" to "cluster_names"
    cluster_weights: Optional[dict[str, float]] = None  # Mapping of "clusters_names" to their weights
    cluster_stratification: Optional[dict[str, np.ndarray]] = None  # Mapping of "cluster_names" to the number of classes present in each cluster
    similarity: Optional[Union[np.ndarray, str]] = None  # Name of similarity algorithm or a pairwise similarities matrix of data points in order of "names"
    cluster_similarity: Optional[np.ndarray] = None  # Pairwise similarity matrix of custers in order of "cluster_names"
    distance: Optional[Union[np.ndarray, str]] = None  # Name of distance algorithm or a pairwise distance matrix of data points in order of "names"
    cluster_distance: Optional[np.ndarray] = None  # Pairwise distance matrix of custers in order of "cluster_names"

    def __hash__(self) -> int:
        """
        Compute the hash value for this dataset to be used in caching. Therefore, the hash is computed on properties
        that do not change during clustering.

        Returns:
            The cluster-insensitive hash-value of the instance.
        """
        hash_val = 0
        for field in filter(lambda f: "cluster" not in f.name, fields(DataSet)):
            obj = getattr(self, field.name)
            if obj is None:
                hv = 0
            elif isinstance(obj, dict):
                hv = hash(tuple(obj.items()))
            elif isinstance(obj, list):
                hv = hash(tuple(obj))
            elif isinstance(obj, np.ndarray):
                hv = 0
            elif isinstance(obj, Namespace):
                hv = hash(tuple(obj.__dict__.items()))
            else:
                hv = hash(obj)
            hash_val ^= hv
        return hash_val

    def __eq__(self, other: Any) -> bool:
        """
        Determine equality of two DataSets based on their hash value.

        Args:
            other: Other  object to compare to

        Returns:
            True if other object is a DataSet and contains the same information as this one.
        """
        return isinstance(other, DataSet) and hash(self) == hash(other)

    def get_name(self) -> str:
        """
        Compute the name of the dataset as the name of the file or the folder storing the data.

        Returns:
            Name of the dataset
        """
        if self.location is None or self.location == UNK_LOCATION:
            return "unknown"
        if isinstance(self.location, Path):
            if self.location.is_file():
                return self.location.stem
            return self.location.name
        return str(self.location)

    def get_location_path(self) -> Path:
        """
        Get the location of the dataset.

        Returns:
            The location of the dataset
        """
        if self.location is None or self.location == UNK_LOCATION or not self.location.exists():
            return Path("unknown." + format2ending(self.format))
        return self.location

    def strat2oh(self, name: Optional[str] = None, classes: Optional[Union[str, set[str]]] = None) -> Optional[np.ndarray]:
        """
        Convert the stratification to a one-hot encoding.

        Args:
            name: Name of the sample to get the onehot encoding for
            class_: Class to get the onehot encoding for

        Returns:
            A one-hot encoding of the stratification
        """
        if classes is None:
            if name is None:
                raise ValueError("Either name or class must be provided.")
            classes = self.stratification[name]
        if not isinstance(classes, Iterable):
            classes = [classes]
        if self.classes is not None:
            return self.class_oh[[self.classes[class_] for class_ in classes]].sum(axis=0)
        return None

    def shuffle(self) -> None:
        """
        Shuffle this dataset randomly to introduce variance in the solution space.
        """
        if self.type is None:
            return

        self.names, self.similarity, self.distance = permute(self.names, self.similarity, self.distance)

        if self.cluster_names is not None:
            self.cluster_names, self.cluster_similarity, self.cluster_distance = \
                permute(self.cluster_names, self.cluster_similarity, self.cluster_distance)
