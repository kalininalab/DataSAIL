import pytest

from scala.salsa import salsa
from tests.utils import check_folder


def run_identity_splitting(root_dir, out_folder, mode):
    salsa(
        inter=f"{root_dir}/inter.tsv",
        output=f"{root_dir}/{out_folder}/",
        max_sec=10,
        max_sol=10,
        verbosity="I",
        technique=mode,
        splits=[0.7, 0.3],
        names=["train", "test"],
        limit=0.05,
        protein_data=f"{root_dir}/prot.tsv",
        protein_weights=None,
        protein_sim=None,
        protein_dist=None,
        protein_max_sim=1,
        protein_max_dist=1,
        ligand_data=f"{root_dir}/lig.tsv",
        ligand_weights=None,
        ligand_sim=None,
        ligand_dist=None,
        ligand_max_sim=1,
        ligand_max_dist=1,
    )


@pytest.mark.parametrize("root_dir", ["data/perf_7_3", "data/perf_70_30"])
@pytest.mark.parametrize("mode", [("R", "random"), ("ICD", "id_cold_drug"), ("ICP", "id_cold_protein")])
def test_perf_bin_2(root_dir, mode):
    run_identity_splitting(root_dir, mode[1], mode[0])

    check_folder(f"{root_dir}/{mode[1]}", 0.05)
