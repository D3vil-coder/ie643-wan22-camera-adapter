import importlib.util
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "restore_from_hf", Path(__file__).resolve().parents[1] / "scripts" / "restore_from_hf.py")
rfh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rfh)


class _Resp:
    def __init__(self, code, headers=None):
        self.status_code, self.headers = code, headers or {}


class _Err(Exception):
    def __init__(self, resp):
        self.response = resp


def test_retry_after_honored():
    d = rfh.backoff_delay(0, _Err(_Resp(429, {"Retry-After": "90"})), rng=lambda: 0.5)
    assert d == 90.5


def test_exponential_with_jitter_and_cap():
    lo = rfh.backoff_delay(2, None, rng=lambda: 0.0)
    hi = rfh.backoff_delay(2, None, rng=lambda: 1.0)
    assert lo == 10.0 and hi == 20.0
    assert rfh.backoff_delay(20, None, rng=lambda: 1.0) == 120.0


def test_fetch_retries_on_429_then_succeeds(tmp_path):
    calls = []

    def fake(*a, **k):
        calls.append(1)
        if len(calls) < 3:
            raise _Err(_Resp(429, {"Retry-After": "1"}))
        return str(tmp_path / "x")

    api = mock.Mock(token="t")
    slept = []
    with mock.patch("huggingface_hub.hf_hub_download", fake):
        out = rfh.fetch(api, "r", "dataset", "p", tmp_path, sleep=slept.append)
    assert out == tmp_path / "x" and len(calls) == 3 and len(slept) == 2


class EntryNotFoundError(Exception):
    pass


def test_tree_missing_folder_is_empty_and_other_errors_propagate():
    import pytest

    class Api:
        def __init__(self, exc):
            self.exc = exc

        def list_repo_tree(self, *a, **k):
            raise self.exc

    assert rfh.tree(Api(EntryNotFoundError("404 Client Error")), "r", "dataset", "eval/x") == {}
    with pytest.raises(RuntimeError):
        rfh.tree(Api(RuntimeError("boom")), "r", "dataset", "eval/x")


def test_restore_checkpoints_fetches_run_config_but_not_latest(tmp_path, monkeypatch):
    import json

    adapter = tmp_path / "remote_adapter.safetensors"
    adapter.write_bytes(b"w")
    man = {"adapter_file": "adapter_000007.safetensors", "adapter_sha256": rfh.sha256_file(adapter)}
    remote = {"checkpoints/ckpt_000007.json": (1, None), "config.json": (1, None), "latest.json": (1, None),
              "progress.json": (1, None)}
    monkeypatch.setattr(rfh, "tree", lambda *a, **k: remote)

    def fake_fetch(api, repo, repo_type, remote_path, staging, **k):
        staging.mkdir(parents=True, exist_ok=True)
        out = staging / remote_path.replace("/", "_")
        if remote_path.endswith("ckpt_000007.json"):
            out.write_text(json.dumps(man))
        elif remote_path.endswith("adapter_000007.safetensors"):
            out.write_bytes(adapter.read_bytes())
        else:
            out.write_text("{}")
        return out

    monkeypatch.setattr(rfh, "fetch", fake_fetch)
    run = tmp_path / "run"
    assert rfh.restore_checkpoints(None, "repo", "v2_full", run, [7]) == [7]
    assert (run / "config.json").is_file() and (run / "progress.json").is_file()
    assert not (run / "latest.json").exists()
    assert (run / "checkpoints" / "adapter_000007.safetensors").is_file()
