"""Resource settings reach the existing Foldseek algorithm unchanged."""
import importlib

import pytest
from datasail.parsers import parse_datasail_args
from datasail.reader.utils import DataSet
from datasail.settings import DEFAULT_KWARGS


@pytest.mark.parametrize("size", ["1", "64K", "256M", "1G"])
def test_public_sort_memory_and_cli_validation(tmp_path, monkeypatch, size):
    sail = importlib.import_module("datasail.sail")
    seen = []
    monkeypatch.setattr(sail, "datasail_main", lambda **kw: seen.append(kw))
    sail.datasail(techniques=["R"], splits=[8, 1, 1], tmp_dir=tmp_path / "scratch", foldseek_sort_memory=size)
    assert seen[0]["foldseek_sort_memory"] == size
    assert parse_datasail_args(["--foldseek-sort-memory", size])["foldseek_sort_memory"] == size
    assert DEFAULT_KWARGS["foldseek_sort_memory"] == "256M"

@pytest.mark.parametrize("size", [0, "0", "-1G", "1%", "1.5G", "1g", True])
def test_bad_sort_memory_rejected_before_work(tmp_path, size):
    from datasail.sail import datasail
    with pytest.raises(ValueError, match="sort memory"):
        datasail(tmp_dir=tmp_path / "scratch", foldseek_sort_memory=size)
    assert not (tmp_path / "scratch").exists()

def test_similarity_dispatch_forwards_buffer(tmp_path, monkeypatch):
    from datasail.cluster import clustering
    seen = []
    monkeypatch.setattr(clustering, "run_foldseek", lambda *a, **kw: seen.append(kw))
    monkeypatch.setattr(clustering, "finish_clustering", lambda d: d)
    d = DataSet(type="P", similarity="foldseek", names=["P"], data={"P": tmp_path / "P.pdb"})
    clustering.similarity_clustering(d, threads=4, tmp_dir=tmp_path, foldseek_sort_memory="1G")
    assert seen == [{"sort_memory": "1G"}]


@pytest.mark.skipif(__import__("shutil").which("foldseek") is None, reason="native Foldseek unavailable")
def test_native_public_api_buffers_and_scratch_preserve_exact_matrix(tmp_path, monkeypatch):
    import inspect
    from pathlib import Path

    from datasail.cluster import clustering
    from datasail.sail import datasail

    source = Path(__file__).parent / "raw_data/cyclophilins/8cu5_a.pdb"
    inputs = tmp_path / "inputs with spaces"
    inputs.mkdir()
    text = source.read_text()
    atoms = [line for line in text.splitlines() if line.startswith("ATOM")]
    (inputs / "5abc_LIG_A_101.pdb").write_text(text)
    (inputs / "6def_LIG_B_202.pdb").write_text("\n".join(atoms + [line[:21] + "B" + line[22:] for line in atoms]) + "\nEND\n")
    matrices = []
    original = clustering.run_foldseek
    def observed(dataset, *args, **kwargs):
        original(dataset, *args, **kwargs)
        matrix = dataset.cluster_similarity
        matrices.append((list(dataset.cluster_names), matrix.shape, matrix.dtype.str, matrix.tobytes()))
    monkeypatch.setattr(clustering, "run_foldseek", observed)
    options = dict(techniques=["C1f"], splits=[1, 1], names=["train", "test"], f_type="P",
                   f_data={p.stem: p for p in inputs.iterdir()}, f_sim="foldseek", threads=1,
                   epsilon=1., delta=1., max_sec=1)
    datasail(**options, tmp_dir=tmp_path / "ordinary scratch", foldseek_sort_memory="64K")
    persistent = {"checkpoint_dir": tmp_path / "persistent"} if "checkpoint_dir" in inspect.signature(datasail).parameters else {}
    datasail(**options, **persistent, tmp_dir=tmp_path / "fresh scratch", foldseek_sort_memory="1M")
    assert matrices[0] == matrices[1]
    if persistent:
        datasail(**options, **persistent, tmp_dir=tmp_path / "new retry scratch", foldseek_sort_memory="2M")
        assert len(matrices) == 2
        assert (tmp_path / "persistent/f/matrix.json").is_file()
    assert not (tmp_path / "ordinary scratch").exists()
    assert not (tmp_path / "fresh scratch").exists()
