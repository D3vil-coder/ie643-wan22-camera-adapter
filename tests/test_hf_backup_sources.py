import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import hf_backup  # noqa: E402


def make(tmp_path):
    root = tmp_path / "v2"
    cache = root / "cache_1280x704_f121_v2"
    (cache / "clips").mkdir(parents=True)
    (cache / "references").mkdir()
    (cache / "clips" / "a.safetensors").write_bytes(b"x")
    (cache / "references" / "a.pt").write_bytes(b"x")
    (root / "sources" / "train2017").mkdir(parents=True)
    (root / "sources" / "train2017" / "1.jpg").write_bytes(b"j")
    (root / "_work").mkdir()
    (root / "_work" / "annotations_trainval2017.zip").write_bytes(b"z")
    (root / "sources_v2.json").write_text("{}")
    (root / "hf_backup_ledger_cache_v2.json").write_text("{}")
    (root / "sources" / "x.tmp").write_bytes(b"t")
    return root, cache


def test_sources_exclude_cache_dir_and_ledger(tmp_path):
    root, cache = make(tmp_path)
    b = hf_backup.CacheBackup(None, cache, "u/r", "cache_v2", sources_root=root)
    rels = {rel for rel, _, _ in b.candidates() if rel.startswith("sources_root/")}
    assert rels == {"sources_root/sources_v2.json", "sources_root/sources/train2017/1.jpg",
                    "sources_root/_work/annotations_trainval2017.zip"}


def test_source_allowed():
    assert not hf_backup.source_allowed("cache_1280x704_f121_v2/clips/a.safetensors")
    assert not hf_backup.source_allowed("hf_backup_ledger_cache_v2.json")
    assert hf_backup.source_allowed("sources/val2017/2.jpg")
