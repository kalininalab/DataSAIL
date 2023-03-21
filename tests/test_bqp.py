import pytest

from datasail.sail import sail
from tests.utils import check_folder


def run_identity_splitting(root_dir, out_folder, mode, vectorized):
    sail(
        inter=f"{root_dir}/inter.tsv",
        output=f"{root_dir}/{out_folder}/",
        max_sec=10,
        max_sol=10,
        verbosity="I",
        techniques=[mode],
        vectorized=vectorized,
        splits=[0.7, 0.3],
        names=["train", "test"],
        epsilon=0.05,
        e_type="M",
        e_data=f"{root_dir}/lig.tsv",
        e_weights=None,
        e_sim=None,
        e_dist=None,
        e_max_sim=1,
        e_max_dist=1,
        e_args="",
        f_type="P",
        f_data=f"{root_dir}/prot.tsv",
        f_weights=None,
        f_sim=None,
        f_dist=None,
        f_max_sim=1,
        f_max_dist=1,
        f_args="",
        cache=False,
        cache_dir=None,
        solver="SCIP",
    )


@pytest.mark.parametrize("root_dir", ["data/perf_7_3", "data/perf_70_30"])
@pytest.mark.parametrize("mode", [("R", "random"), ("ICS", "id_cold_single"), ("ICD", "id_cold_double")])
@pytest.mark.parametrize("vectorized", [True, False])
def test_perf_bin_2(root_dir, mode, vectorized):
    run_identity_splitting(root_dir, mode[1], mode[0], vectorized)

    check_folder(f"{root_dir}/{mode[1]}/{mode[0]}", 0.25, None, None, "Molecule_drug_splits.tsv", "Protein_prot_splits.tsv")
