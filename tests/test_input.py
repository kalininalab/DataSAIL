import copy
from pathlib import Path

import numpy as np

from datasail.reader.read import read_data
from datasail.settings import KW_INTER, KW_E_TYPE, KW_E_DATA, KW_E_WEIGHTS, DEFAULT_KWARGS, KW_E_STRAT


def test_simple_input():
    base = Path("data") / "pipeline"
    kwargs = copy.deepcopy(DEFAULT_KWARGS)
    kwargs.update({
        KW_INTER: None,
        KW_E_TYPE: "M",
        KW_E_DATA: base / "drugs.tsv",
    })
    e_dataset, f_dataset, inter = read_data(**kwargs)

    assert e_dataset.type == "M"
    assert set(e_dataset.names) == set(e_dataset.stratification.keys())
    for name in e_dataset.names:
        assert np.array_equal(e_dataset.stratification[name], np.array([], dtype=int))
    assert f_dataset.type is None
    assert inter is None


def test_stratification_input():
    base = Path("data") / "pipeline"
    kwargs = copy.deepcopy(DEFAULT_KWARGS)
    kwargs.update({
        KW_INTER: None,
        KW_E_TYPE: "M",
        KW_E_DATA: base / "drugs.tsv",
        KW_E_STRAT: base / "drug_strat.tsv",
    })
    e_dataset, f_dataset, inter = read_data(**kwargs)

    assert e_dataset.type == "M"
    assert set(e_dataset.names) == set(e_dataset.stratification.keys())
    for name,  in e_dataset.names:
        assert np.array_equal(e_dataset.stratification[name], np.array([], dtype=int))

    assert f_dataset.type is None
    assert inter is None