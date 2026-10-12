from pathlib import Path

import pandas as pd
import pytest

from datasail.reader.read_molecules import read_molecule_data
from datasail.reader.utils import DataSet, read_data, read_data_input
from datasail.settings import FORM_SMILES, M_TYPE, UNK_LOCATION


@pytest.mark.parametrize("ending", [".csv", ".tsv"])
def test_read_mol_file(ending):
    dataset = read_molecule_data(data=Path("data") / "pipeline" / ("drugs" + ending))

    assert dataset is not None
    
    assert dataset.location is not None
    assert dataset.location.exists()
    assert dataset.location.is_file()
    assert dataset.location == Path("data") / "pipeline" / ("drugs" + ending)

    assert dataset.format == FORM_SMILES
    assert dataset.type == M_TYPE
    
    assert dataset.names is not None
    assert set(dataset.names) == set([f"D{i:05d}" for i in range(1, 49)])
    assert len(dataset.data) == 48
    assert all(name in dataset.data for name in dataset.names)

    df = pd.read_csv(Path("data") / "pipeline" / "drugs.csv")
    data = df[["Drug_ID", "SMILES"]]
    for _, row in data.iterrows():
        assert row["SMILES"] == dataset.data[row["Drug_ID"]]
