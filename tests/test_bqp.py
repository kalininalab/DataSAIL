import shutil
from pathlib import Path
from typing import Tuple

import pytest

from tests.utils import check_folder, run_sail


@pytest.mark.full
@pytest.mark.parametrize("root_dir", [Path("data") / "perf_7_3", Path("data") / "perf_70_30"])
@pytest.mark.parametrize("mode", [("R", "random"), ("I1e", "id_cold_single"), ("I2", "id_cold_double")])
def test_perf_bin_2(root_dir: Path, mode: Tuple[str, str]):
    base = root_dir / mode[1]

    run_sail(
        output=base,
        inter=root_dir / "inter.tsv",
        techniques=[mode[0]],
        splits=[0.7, 0.3],
        names=["train", "test"],
        epsilon=0.05,
        e_type="M",
        e_data=root_dir / "lig.tsv",
        f_type="P",
        f_data=root_dir / "prot.tsv",
        solver="SCIP",
        max_sec=500,
    )

    check_folder(base / mode[0], 0.5 if mode[0] == "I2" else 0.05, None, None, "Molecule_lig_splits.tsv", "Protein_prot_splits.tsv")

    shutil.rmtree(base / mode[0], ignore_errors=True)
