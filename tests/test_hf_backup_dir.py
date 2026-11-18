import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hf_backup  # noqa: E402
import molab_recover  # noqa: E402


def w(p, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def make(tmp):
    root = tmp / "eval"
    done = root / "models" / "v2" / "pair1"
    w(done / "run.json"), w(done / "video.mp4"), w(done / "tracks.npz"), w(done / "reveal_metrics.json")
    part = root / "models" / "v2" / "pair2"  # video written, run.json not yet
    w(part / "video.mp4"), w(part / "request.json")
    part2 = root / "models" / "v2" / "pair3"  # run.json but no video
    w(part2 / "run.json")
    w(root / "reveal_eval.json"), w(root / "reveal_eval.md"), w(root / "x.json.tmp"), w(root / "big.bin")
    w(root / "pairs" / "p1" / "pair.json")
    return root


def test_candidates_complete_only(tmp_path):
    root = make(tmp_path)
    rels = {r for r, _ in hf_backup.dir_candidates(root)}
    assert rels == {"models/v2/pair1/run.json", "models/v2/pair1/video.mp4", "models/v2/pair1/reveal_metrics.json",
                    "reveal_eval.json", "reveal_eval.md", "pairs/p1/pair.json"}


def test_settle_delays_fresh_aggregates(tmp_path):
    root = make(tmp_path)
    rels = {r for r, _ in hf_backup.dir_candidates(root, settle_seconds=3600)}
    assert "reveal_eval.json" not in rels and "models/v2/pair1/run.json" in rels


class FakeApi:
    def __init__(self):
        self.commits = []


def make_backup(tmp, root, monkeypatch, remote=None):
    calls = []
    monkeypatch.setattr(hf_backup, "commit", lambda api, repo, rt, ops, msg: calls.append([o.path_in_repo for o in ops]))
    monkeypatch.setattr(hf_backup, "remote_tree", lambda *a, **k: dict(remote or {}))
    b = hf_backup.DirBackup(FakeApi(), root, "u/r", "dataset", "eval/t", tmp / "ledger.json",
                            batch_files=1, min_commit_interval=0, settle_seconds=0)
    return b, calls


def test_ledger_prevents_reupload_and_detects_change(tmp_path, monkeypatch):
    root = make(tmp_path)
    b, calls = make_backup(tmp_path, root, monkeypatch)
    assert b.pass_once() is True
    assert sorted(calls[0]) == sorted("eval/t/" + r for r, _ in hf_backup.dir_candidates(root))
    assert b.pass_once() is False and len(calls) == 1
    # persisted ledger survives a restart
    b2, calls2 = make_backup(tmp_path, root, monkeypatch)
    assert b2.pass_once() is False and not calls2
    # an aggregate changes -> only it is re-uploaded; a partial run becomes complete -> uploaded
    (root / "reveal_eval.json").write_bytes(b"changed!")
    w(root / "models" / "v2" / "pair2" / "run.json")
    assert b2.pass_once() is True
    assert sorted(calls2[0]) == ["eval/t/models/v2/pair2/request.json", "eval/t/models/v2/pair2/run.json",
                                 "eval/t/models/v2/pair2/video.mp4", "eval/t/reveal_eval.json"]
    assert json.loads((tmp_path / "ledger.json").read_text())


def test_remote_already_has_file_is_not_uploaded(tmp_path, monkeypatch):
    root = tmp_path / "e"
    w(root / "a.json", b"hello")
    sha = hf_backup.sha256_file(root / "a.json")
    b, calls = make_backup(tmp_path, root, monkeypatch, remote={"a.json": (5, sha, None)})
    assert b.pass_once() is False and not calls
    assert "a.json" in b.ledger


def test_batching_waits_for_threshold(tmp_path, monkeypatch):
    root = tmp_path / "e"
    w(root / "a.json")
    b, calls = make_backup(tmp_path, root, monkeypatch)
    b.batch_files, b.batch_seconds = 5, 1e9
    assert b.pass_once() is False and not calls
    assert b.pass_once(flush=True) is True and len(calls) == 1


def test_recover_arg_parsers():
    assert molab_recover.parse_name_dir("v2_72:artifacts/x") == ("v2_72", "artifacts/x")
    assert molab_recover.parse_steps("77, 250") == [77, 250]


def test_exclusive_lock(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setenv("HF_BACKUP_LOCK_DIR", str(tmp_path / "locks"))
    assert hf_backup.acquire_lock("u/r", "cache_v2") is not None
    assert hf_backup.lock_path("u/r", "cache_v2") != hf_backup.lock_path("u/r", "runs/a")
    code = ("import sys;sys.path.insert(0,%r);import hf_backup;print(hf_backup.acquire_lock('u/r','cache_v2'),"
            "hf_backup.acquire_lock('u/r','other') is not None)" % str(Path(hf_backup.__file__).parent))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**__import__('os').environ, "HF_BACKUP_LOCK_DIR": str(tmp_path / "locks")}).stdout.split()
    assert out == ["None", "True"]
