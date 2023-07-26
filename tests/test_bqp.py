import shutil

import pytest

from tests.utils import check_folder, run_sail


@pytest.mark.parametrize("root_dir", ["data/perf_7_3", "data/perf_70_30"])
@pytest.mark.parametrize("mode", [("R", "random"), ("ICSe", "id_cold_single"), ("ICD", "id_cold_double")])
@pytest.mark.parametrize("vectorized", [True, False])
def test_perf_bin_2(root_dir, mode, vectorized):
    shutil.rmtree(f"{root_dir}/{mode[1]}/", ignore_errors=True)

    run_sail(
        output=f"{root_dir}/{mode[1]}/",
        inter=f"{root_dir}/inter.tsv",
        techniques=[mode[0]],
        scalar=vectorized,
        splits=[0.7, 0.3],
        names=["train", "test"],
        epsilon=0.05,
        e_type="M",
        e_data=f"{root_dir}/lig.tsv",
        f_type="P",
        f_data=f"{root_dir}/prot.tsv",
        solver="SCIP",
    )

    check_folder(f"{root_dir}/{mode[1]}/{mode[0]}", 0.25, None, None, "Molecule_lig_splits.tsv", "Protein_prot_splits.tsv")
