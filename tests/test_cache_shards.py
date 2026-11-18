import hashlib
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hf_backup  # noqa: E402
import restore_from_hf as rf  # noqa: E402


def sha(b):
    return hashlib.sha256(b).hexdigest()


def make_cache(root, n=6, size=3000):
    cache = root / "cache"
    rows = []
    for i in range(n):
        lat, ref = f"clips/c{i}.safetensors", f"references/c{i}.safetensors"
        lb, rb = bytes([i]) * size, bytes([i + 100]) * 700
        for rel, b in ((lat, lb), (ref, rb), (f"references/c{i}.png", b"png%d" % i)):
            (cache / rel).parent.mkdir(parents=True, exist_ok=True)
            (cache / rel).write_bytes(b)
        rows.append({"id": f"c{i}", "latent_file": lat, "sha256": sha(lb), "reference_file": ref,
                     "reference_sha256": sha(rb)})
    (cache / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    src = root / "v2"
    src.mkdir()
    (src / "sources_v2.json").write_text('{"a": 1}')
    return cache, src


class Remote:
    def __init__(self):
        self.files = {}

    def upload(self, path, name):
        self.files[name] = Path(path).read_bytes()

    def shas(self):
        return {k: sha(v) for k, v in self.files.items() if k.startswith("shard_")}


def pack(cache, src, tmp, remote, shard_bytes=9000):
    return hf_backup.pack_cache(cache, tmp / "work", shard_bytes, remote.upload, remote.shas(), src)


def test_round_trip_manifest_and_resume(tmp_path):
    cache, src = make_cache(tmp_path)
    r = Remote()
    man = pack(cache, src, tmp_path, r)
    assert len(man["shards"]) > 2 and "shards.json" in r.files
    assert json.loads(r.files["shards.json"]) == man
    names = [m["name"] for s in man["shards"] for m in s["members"]]
    assert names == sorted(names) and "sources_root/sources_v2.json" in names and "index.jsonl" in names
    for s in man["shards"]:  # shard + member shas match the uploaded bytes
        data = r.files[s["name"]]
        assert sha(data) == s["sha256"] and len(data) == s["size"]
    # deterministic: a second pack gives identical shards, and nothing is re-uploaded
    first = dict(r.files)
    uploads = []
    orig = r.upload
    r.upload = lambda p, n: (uploads.append(n), orig(p, n))
    man2 = pack(cache, src, tmp_path, r)
    assert man2 == man and uploads == ["shards.json"]
    assert {k: v for k, v in r.files.items() if k.startswith("shard_")} == {k: v for k, v in first.items() if k.startswith("shard_")}
    # resume: one shard missing remotely -> only that one is uploaded
    gone = man["shards"][1]["name"]
    del r.files[gone]
    uploads.clear()
    pack(cache, src, tmp_path, r)
    assert uploads == [gone, "shards.json"]

    # restore into a fresh place
    dest, sroot = tmp_path / "new" / "cache", tmp_path / "new" / "v2"

    def fetch_fn(rp, staging):
        staging.mkdir(parents=True, exist_ok=True)
        p = staging / Path(rp).name
        p.write_bytes(r.files[Path(rp).name])
        return p
    n = rf.restore_cache_shards(None, "r", "cache_v2_shards", dest, 2, sroot, fetch_fn=fetch_fn)
    assert n == 6
    for i in range(6):
        assert (dest / f"clips/c{i}.safetensors").read_bytes() == (cache / f"clips/c{i}.safetensors").read_bytes()
    assert (sroot / "sources_v2.json").read_text() == '{"a": 1}'
    # second restore downloads nothing
    calls = []
    rf.restore_cache_shards(None, "r", "cache_v2_shards", dest, 2, sroot,
                            fetch_fn=lambda rp, st: (calls.append(rp), fetch_fn(rp, st))[1])
    assert calls == ["cache_v2_shards/shards.json"]


def test_corrupt_shard_rejected(tmp_path):
    cache, src = make_cache(tmp_path)
    r = Remote()
    man = pack(cache, src, tmp_path, r)
    name = man["shards"][0]["name"]
    b = bytearray(r.files[name])
    b[600] ^= 0xFF
    r.files[name] = bytes(b)

    def fetch_fn(rp, staging):
        staging.mkdir(parents=True, exist_ok=True)
        p = staging / Path(rp).name
        p.write_bytes(r.files[Path(rp).name])
        return p
    with pytest.raises(SystemExit):
        rf.restore_cache_shards(None, "r", "x", tmp_path / "d" / "cache", 2, tmp_path / "d" / "v2", fetch_fn=fetch_fn)


def test_index_sha_mismatch_and_unsafe_member(tmp_path):
    cache, src = make_cache(tmp_path)
    (cache / "clips/c0.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="index row"):
        pack(cache, src, tmp_path, Remote())
    bad = tmp_path / "bad.tar"
    with tarfile.open(bad, "w") as tf:
        import io
        ti = tarfile.TarInfo("../evil")
        ti.size = 1
        tf.addfile(ti, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        rf.extract_shard(bad, [{"name": "../evil", "size": 1, "sha256": sha(b"x")}], tmp_path / "o", tmp_path / "s")
