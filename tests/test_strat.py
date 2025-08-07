from typing import Any, Dict

import pytest
import numpy as np

from datasail.reader.utils import convert_stratification
from datasail.reader.utils import DataSet


@pytest.fixture
def dataset():
    ds = DataSet()
    ds.names = [f"S{i}" for i in range(10)]
    return ds


def test_none_stratification(dataset):
    dataset.stratification = None
    result = convert_stratification(dataset)
    assert isinstance(result, dict)
    assert all(isinstance(v, list) for v in result.values())
    assert all(len(v) == 1 for v in result.values())


def test_multiclass_stratification(dataset):
    dataset.stratification = {name: [i % 3] for i, name in enumerate(dataset.names)}
    result = convert_stratification(dataset)
    assert isinstance(result, dict)
    assert all(isinstance(v, (list, np.ndarray)) for v in result.values())
    assert all(len(v) == 3 for v in result.values())
    assert all(sum(v) == 1 for v in result.values())


def test_multiclass_stratification_str(dataset):
    dataset.stratification = {name: str([i % 3]) for i, name in enumerate(dataset.names)}
    result = convert_stratification(dataset)
    assert isinstance(result, dict)
    assert all(isinstance(v, (list, np.ndarray)) for v in result.values())
    assert all(len(v) == 3 for v in result.values())
    assert all(sum(v) == 1 for v in result.values())


def test_multilabel_stratification(dataset):
    dataset.stratification = {name: [int(x) for x in bin(i)[2:].zfill(4)] for i, name in enumerate(dataset.names)}
    result = convert_stratification(dataset)
    assert isinstance(result, dict)
    assert all(isinstance(v, (list, np.ndarray)) for v in result.values())
    assert all(len(v) == 4 for v in result.values())


def test_multilabel_multiclass_stratification(dataset):
    dataset.stratification = {name: [int(x) for x in bin(i)[2:].zfill(4)] + [str([i % 3]), i % 3] for i, name in enumerate(dataset.names)}
    result = convert_stratification(dataset)
    print(result)
    assert isinstance(result, dict)
    assert all(isinstance(v, (list, np.ndarray)) for v in result.values())
    assert all(len(v) == 10 for v in result.values())
