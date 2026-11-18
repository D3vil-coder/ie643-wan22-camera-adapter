#!/usr/bin/env python
"""Back up a training run's checkpoints and the v2 latent cache to private Hugging Face repos.

Built for an ephemeral sandbox (Molab): runs as a detached daemon next to the trainer.
Stdlib + huggingface_hub only (no torch, no repo imports), so it can live outside the repo
(e.g. /home/marimo/work/tools_staging/hf_backup.py).  Token: ``$HF_HOME/token`` or ``HF_TOKEN``.

  python hf_backup.py watch-run   RUN_DIR   --path-in-repo runs/<name> [--repo USER/ie643-camera-adapter-v2]
  python hf_backup.py once-run    RUN_DIR   --path-in-repo runs/<name>       # single pass
  python hf_backup.py watch-cache CACHE_DIR --path-in-repo cache_v2 [--repo USER/ie643-camera-v2-cache]
                                  [--sources-root data/v2]                  # + sources_v2.json, sources/**
  python hf_backup.py once-cache  CACHE_DIR --path-in-repo cache_v2 [--sources-root data/v2]
  python hf_backup.py watch-dir   DIR --repo-type dataset --path-in-repo eval/<name> [--repo USER/ie643-camera-v2-cache]
  python hf_backup.py once-dir    DIR --repo-type dataset --path-in-repo eval/<name>

Directory backup (evaluation outputs): a *run folder* (a directory holding ``run.json``) is uploaded only
once COMPLETE, i.e. ``run.json`` (written last by flow_infer) and ``video.mp4`` both exist; ``tracks.npz`` and
``*.tmp``/``*.partial`` are never uploaded.  Other files (aggregate json/jsonl/md/png/html/js/csv/txt, e.g.
``reveal_eval.json``, ``validation.json``, ``selection.json``, ``pairs/*/pair.json``) are uploaded when older
than ``--settle-seconds``.  A local ledger (path -> size:mtime:sha256) prevents re-uploads; changed files are
re-uploaded.  Batched like the cache (``--batch-files`` / ``--batch-seconds``).

(``watch-run RUN_DIR ... --once`` is the same as ``once-run``.)

Run backup (layout of the remote ``<path-in-repo>/``): ``latest.json``, ``config.json``,
``progress.json[l]``, ``launches.jsonl``, ``run_status.json``, ``checkpoints/{adapter,resume,ckpt}_NNNNNN.*``.
A pass uploads every complete (sha256-verified against its manifest) checkpoint that the remote
lacks, plus changed small files, and deletes remote checkpoints outside the retention policy
(newest ``--keep-last`` + every ``--keep-every``-th step) in ONE atomic commit.  ``*.tmp`` is never
uploaded.  Commits are rate limited (``--min-commit-interval``) and retried with backoff.

Cache backup: uploads clip files (``clips/*.safetensors``) and reference files (``references/*``)
named by the index rows, after checking sha256 against the row, in batches (>= ``--batch-files``
pending, or anything pending after ``--batch-seconds``).  The remote ``index.jsonl`` is a merge of the
local ``index*.jsonl`` filtered to rows whose files are on the remote.  Idempotent: the remote tree
(LFS sha256) plus a local ledger tell what is already uploaded.
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

_MANIFEST_RE = re.compile(r"^ckpt_(\d{6,})\.json$")
RUN_EXTRAS = ("config.json", "progress.json", "progress.jsonl", "launches.jsonl", "run_status.json")
# Sources are uploaded by allow-list only: the cache dir often lives inside sources-root (data/v2) and
# must never be uploaded a second time under sources_root/ (incident 2026-11-16: 7 GB duplicate).
SOURCES_ALLOW = ("sources_v2.json", "sources/*", "_work/annotations*.zip")
CACHE_EXTRAS = ("config_snapshot.json", "rejected.jsonl", "progress.json")


def log(msg):
    """Timestamped print (flushed, so daemon logs are live)."""
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def sha256_file(path) -> str:
    """Streaming sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_blob_id(data: bytes) -> str:
    """Git blob id (sha1 of ``blob <len>`` + data) of a byte string; lets small non-LFS remote files be compared without downloading."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# ---------------------------------------------------------------------------- HF helpers
def get_api():
    """``HfApi`` using ``HF_TOKEN`` / ``HUGGINGFACE_TOKEN`` (or the token in ``$HF_HOME``)."""
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None
    return HfApi(token=token)


def default_repo(api, repo_type):
    """Default repo id for the logged-in user: ``<user>/ie643-camera-adapter-v2`` (model) or ``<user>/ie643-camera-v2-cache`` (dataset)."""
    user = api.whoami()["name"]
    return f"{user}/ie643-camera-adapter-v2" if repo_type == "model" else f"{user}/ie643-camera-v2-cache"


def retry(fn, what, attempts=8, base=5.0):
    """Retry on any network / rate-limit / server error with exponential backoff."""
    for i in range(attempts):
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - classify loosely, a daemon must not die
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403, 404) and "rate" not in str(exc).lower():
                raise
            wait = min(base * (2 ** i), 900)
            if status == 429:
                wait = max(wait, 120)
            log(f"[retry] {what}: {type(exc).__name__}: {str(exc)[:200]} (attempt {i + 1}/{attempts}, sleep {wait:.0f}s)")
            time.sleep(wait)
    raise RuntimeError(f"{what}: gave up after {attempts} attempts")


def remote_tree(api, repo, repo_type, prefix):
    """{relative path under prefix: (size, sha256-or-None, blob_id)} for the remote prefix."""
    out = {}
    try:
        items = list(api.list_repo_tree(repo, path_in_repo=prefix, recursive=True, repo_type=repo_type))
    except Exception as exc:  # missing folder
        if type(exc).__name__ in ("EntryNotFoundError", "RepositoryNotFoundError") or "404" in str(exc):
            return out
        raise
    for it in items:
        if type(it).__name__ != "RepoFile":
            continue
        rel = it.path[len(prefix) + 1:] if prefix else it.path
        lfs = getattr(it, "lfs", None)
        sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
        out[rel] = (it.size, sha, getattr(it, "blob_id", None))
    return out


def source_allowed(rel: str) -> bool:
    """True if ``rel`` (posix, relative to sources-root) matches the sources allow-list."""
    from fnmatch import fnmatchcase
    rel = rel.replace("\\", "/")
    if rel.endswith((".tmp", ".partial")) or "__pycache__" in rel.split("/"):
        return False
    return any(fnmatchcase(rel, pat) for pat in SOURCES_ALLOW)


def entry_matches_bytes(entry, data: bytes) -> bool:
    """True if a remote tree entry ``(size, sha256, blob_id)`` equals the given bytes."""
    if entry is None:
        return False
    size, sha, blob = entry
    if sha:
        return sha == hashlib.sha256(data).hexdigest()
    return blob == git_blob_id(data)


def entry_matches_file(entry, path: Path, known_sha=None) -> bool:
    """True if a remote tree entry equals a local file (size, then sha256 or blob id)."""
    if entry is None:
        return False
    size, sha, blob = entry
    if size != path.stat().st_size:
        return False
    if sha:
        return sha == (known_sha or sha256_file(path))
    return blob == git_blob_id(path.read_bytes())


def commit(api, repo, repo_type, operations, message):
    """``create_commit`` with retry/backoff."""
    retry(lambda: api.create_commit(repo_id=repo, repo_type=repo_type, operations=operations,
                                    commit_message=message), "create_commit")


# ---------------------------------------------------------------------------- run backup
def local_valid_manifests(run_dir: Path):
    """{step: manifest} for local manifests whose two files exist with the right size-agnostic names."""
    out = {}
    cdir = run_dir / "checkpoints"
    if not cdir.is_dir():
        return out
    for p in cdir.iterdir():
        m = _MANIFEST_RE.match(p.name)
        if not m:
            continue
        try:
            man = json.loads(p.read_text(encoding="utf8"))
            if (cdir / man["adapter_file"]).is_file() and (cdir / man["resume_file"]).is_file():
                out[int(man["step"])] = man
        except Exception:  # half-written / unreadable manifest: ignore this pass
            continue
    return out


def verify_checkpoint(run_dir: Path, man) -> bool:
    """True if the adapter and resume files of a checkpoint match the sha256 values in its manifest (never upload a corrupt one)."""
    cdir = run_dir / "checkpoints"
    for f, k in ((man["adapter_file"], "adapter_sha256"), (man["resume_file"], "resume_sha256")):
        p = cdir / f
        if not p.is_file() or sha256_file(p) != man[k]:
            return False
    return True


def names(step):
    """File names ``(adapter, resume, manifest)`` of checkpoint ``step``."""
    return (f"adapter_{step:06d}.safetensors", f"resume_{step:06d}.pt", f"ckpt_{step:06d}.json")


def retention(steps, keep_last, keep_every):
    """Steps to keep: the newest ``keep_last`` plus every ``keep_every``-th (same policy as ``CheckpointManager.prune``)."""
    steps = sorted(steps)
    keep = set(steps[-keep_last:])
    if keep_every > 0:
        keep |= {s for s in steps if s % keep_every == 0}
    return keep


class RunBackup:
    """Backs up one training run folder to ``<prefix>/`` of a model repo: verified checkpoints (adapter, resume, manifest),
    ``latest.json`` and small extras (config, progress, launches, run_status), pruning remote checkpoints by the retention policy in
    one atomic commit per pass.
    """

    def __init__(self, api, run_dir, repo, prefix, keep_last=3, keep_every=250,
                 extras_interval=600.0, min_commit_interval=60.0):
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        self.Add, self.Delete = CommitOperationAdd, CommitOperationDelete
        self.api, self.run_dir, self.repo, self.prefix = api, Path(run_dir), repo, prefix.strip("/")
        self.keep_last, self.keep_every = keep_last, keep_every
        self.extras_interval, self.min_commit_interval = extras_interval, min_commit_interval
        self.last_commit = 0.0
        self.last_extras = 0.0
        self.bad_steps = {}  # step -> adapter sha that failed verification (do not re-hash every pass)

    def rp(self, rel):
        """Path inside the repo: ``<prefix>/<rel>``."""
        return f"{self.prefix}/{rel}"

    def pass_once(self, final=False):
        """One backup pass. Returns True if a commit was made."""
        local = local_valid_manifests(self.run_dir)
        if not local:
            return False
        remote = retry(lambda: remote_tree(self.api, self.repo, "model", self.prefix), "list remote")
        latest_local = max(local)
        # which local checkpoints does the remote lack (or hold with different content)?
        need = []
        for step, man in sorted(local.items()):
            a, r, m = names(step)
            ok = (
                f"checkpoints/{m}" in remote
                and (remote.get(f"checkpoints/{a}") or (0, None, None))[1] == man["adapter_sha256"]
                and (remote.get(f"checkpoints/{r}") or (0, None, None))[1] == man["resume_sha256"]
            )
            if not ok:
                need.append(step)
        # retention over (remote U local) steps up to the newest local one
        remote_steps = {int(m.group(1)) for rel in remote
                        if (m := _MANIFEST_RE.match(rel.removeprefix("checkpoints/")))}
        universe = {s for s in remote_steps | set(local) if s <= latest_local}
        keep = retention(universe, self.keep_last, self.keep_every) | {latest_local}
        if remote_steps and max(remote_steps) > latest_local:
            log(f"[warn] remote has newer step {max(remote_steps)} than local {latest_local}; "
                f"not deleting anything newer than local latest")
        upload_steps = [s for s in need if s in keep]
        delete_steps = sorted(s for s in remote_steps if s <= latest_local and s not in keep)

        ops, uploaded = [], []
        for step in upload_steps:
            man = local[step]
            if self.bad_steps.get(step) == man["adapter_sha256"]:
                continue
            if not verify_checkpoint(self.run_dir, man):
                log(f"[warn] step {step}: sha256 mismatch against manifest, not uploading")
                self.bad_steps[step] = man["adapter_sha256"]
                continue
            a, r, m = names(step)
            c = self.run_dir / "checkpoints"
            for f in (a, r, m):
                ops.append(self.Add(path_in_repo=self.rp(f"checkpoints/{f}"), path_or_fileobj=str(c / f)))
            uploaded.append(step)

        # latest.json: point at the newest checkpoint that is (or will be) on the remote and verified
        remote_have = {s for s in remote_steps if s in keep or s == latest_local} | set(uploaded)
        ptr_step = max((s for s in remote_have if s in local), default=None)
        extras_due = final or (time.monotonic() - self.last_extras) >= self.extras_interval or bool(uploaded)
        if ptr_step is not None and (uploaded or extras_due):
            man = local[ptr_step]
            a, r, m = names(ptr_step)
            local_latest = self.run_dir / "latest.json"
            ptr = None
            try:
                ptr = json.loads(local_latest.read_text(encoding="utf8"))
            except Exception:
                pass
            if not ptr or int(ptr.get("step", -1)) != ptr_step:
                ptr = {"step": ptr_step, "manifest": m, "adapter_file": a, "adapter_sha256": man["adapter_sha256"],
                       "resume_file": r, "resume_sha256": man["resume_sha256"], "reason": man.get("reason", "backup")}
            data = json.dumps(ptr, indent=2).encode()
            if not entry_matches_bytes(remote.get("latest.json"), data):
                ops.append(self.Add(path_in_repo=self.rp("latest.json"), path_or_fileobj=data))
        if extras_due:
            for name in RUN_EXTRAS:
                p = self.run_dir / name
                if p.is_file():
                    data = p.read_bytes()  # snapshot: a consistent copy even if the trainer appends
                    if not entry_matches_bytes(remote.get(name), data):
                        ops.append(self.Add(path_in_repo=self.rp(name), path_or_fileobj=data))
        for step in delete_steps:
            for f in names(step):
                if f"checkpoints/{f}" in remote:
                    ops.append(self.Delete(path_in_repo=self.rp(f"checkpoints/{f}")))
        if not ops:
            return False
        wait = self.min_commit_interval - (time.monotonic() - self.last_commit)
        if wait > 0:
            time.sleep(wait)
        msg = (f"checkpoint {','.join(map(str, uploaded))}" if uploaded else "update run metadata")
        if delete_steps:
            msg += f"; prune {','.join(map(str, delete_steps))}"
        t0 = time.monotonic()
        commit(self.api, self.repo, "model", ops, msg)
        self.last_commit = time.monotonic()
        if extras_due:
            self.last_extras = self.last_commit
        log(f"[run] committed '{msg}' ({len(ops)} ops, {self.last_commit - t0:.0f}s)")
        return True


# ---------------------------------------------------------------------------- cache backup
def read_index_rows(cache: Path):
    """Rows of every ``index*.jsonl`` in a cache folder keyed by clip id."""
    rows = {}
    for idx in sorted(cache.glob("index*.jsonl")):
        with open(idx, encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partially appended last line
                rows[row["id"]] = row
    return rows


class CacheBackup:
    """Backs up the v2 latent cache (clips, references, merged ``index.jsonl``, extras and the sources manifest) to a dataset repo in
    batches; a local ledger and the remote LFS hashes make every pass idempotent.
    """

    def __init__(self, api, cache_dir, repo, prefix, sources_root=None, ledger=None,
                 batch_files=50, batch_seconds=600.0, max_batch=120, min_commit_interval=60.0):
        from huggingface_hub import CommitOperationAdd
        self.Add = CommitOperationAdd
        self.api, self.cache, self.repo, self.prefix = api, Path(cache_dir), repo, prefix.strip("/")
        self.sources_root = Path(sources_root) if sources_root else None
        self.ledger_path = Path(ledger) if ledger else None
        self.batch_files, self.batch_seconds, self.max_batch = batch_files, batch_seconds, max_batch
        self.min_commit_interval = min_commit_interval
        self.ledger = {}  # rel path (under prefix) -> sha256 (uploaded)
        if self.ledger_path and self.ledger_path.is_file():
            try:
                self.ledger = json.loads(self.ledger_path.read_text())
            except Exception:
                self.ledger = {}
        self.last_commit = 0.0
        self.pending_since = None
        self.synced_remote = False

    def rp(self, rel):
        """Path inside the repo: ``<prefix>/<rel>``."""
        return f"{self.prefix}/{rel}"

    def save_ledger(self):
        """Persist the local upload ledger (path -> sha256) atomically."""
        if self.ledger_path:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.ledger_path.with_name(self.ledger_path.name + ".tmp")
            tmp.write_text(json.dumps(self.ledger))
            os.replace(tmp, self.ledger_path)

    def refresh_remote(self):
        """Seed the ledger from the remote tree so a fresh sandbox does not re-upload what is already there."""
        remote = retry(lambda: remote_tree(self.api, self.repo, "dataset", self.prefix), "list remote")
        for rel, (size, sha, blob) in remote.items():
            self.ledger.setdefault(rel, sha or f"blob:{blob}")
        self.synced_remote = True
        return remote

    def candidates(self):
        """[(rel, local_path, expected_sha or None)] of data files that should exist remotely."""
        out, seen = [], set()
        for row in read_index_rows(self.cache).values():
            for fkey, skey in (("latent_file", "sha256"), ("reference_file", "reference_sha256")):
                rel = row[fkey]
                if rel in seen:
                    continue
                seen.add(rel)
                out.append((rel, self.cache / rel, row.get(skey)))
            png = Path(row["reference_file"]).with_suffix(".png").as_posix()
            if png not in seen and (self.cache / png).is_file():
                seen.add(png)
                out.append((png, self.cache / png, None))
        if self.sources_root and self.sources_root.is_dir():
            cache_res = self.cache.resolve()
            for p in sorted(self.sources_root.rglob("*")):
                if not p.is_file():
                    continue
                if cache_res == p.resolve() or cache_res in p.resolve().parents:
                    continue  # never upload the watched cache dir via sources
                sub = p.relative_to(self.sources_root).as_posix()
                if source_allowed(sub):
                    out.append(("sources_root/" + sub, p, None))
        return out

    def pass_once(self, flush=False):
        """One pass: find files not yet on the remote, upload them in batches (``flush`` ignores the batch thresholds), then update the
        remote index.  Returns True if anything was committed.
        """
        if not self.synced_remote:
            self.refresh_remote()
        pending, bad = [], 0
        for rel, path, sha in self.candidates():
            if not path.is_file() or path.name.endswith((".partial", ".tmp")):
                continue
            if rel in self.ledger and (sha is None or self.ledger[rel] == sha):
                continue
            pending.append((rel, path, sha))
        now = time.monotonic()
        if pending and self.pending_since is None:
            self.pending_since = now
        if not pending:
            self.pending_since = None
        due = flush or len(pending) >= self.batch_files or (
            pending and now - (self.pending_since or now) >= self.batch_seconds)
        committed = False
        if pending and due:
            while pending:
                batch, pending = pending[: self.max_batch], pending[self.max_batch:]
                ops, shas = [], {}
                for rel, path, sha in batch:
                    actual = sha256_file(path)
                    if sha is not None and actual != sha:
                        log(f"[warn] {rel}: sha256 does not match index row, skipped")
                        bad += 1
                        continue
                    shas[rel] = actual
                    ops.append(self.Add(path_in_repo=self.rp(rel), path_or_fileobj=str(path)))
                if not ops:
                    continue
                self._commit(ops, f"add {len(ops)} cache files", shas)
                committed = True
            self.pending_since = None
        # metadata: filtered merged index + small extras, whenever something changed
        meta_ops = self.meta_ops()
        if meta_ops:
            self._commit(meta_ops, "update cache index/metadata", {})
            committed = True
        if not committed and pending:
            log(f"[cache] {len(pending)} files pending (batch threshold {self.batch_files})")
        return committed

    def meta_ops(self):
        """Commit operations for the remote ``index.jsonl`` (rows whose files are on the remote) and the small extras."""
        rows = read_index_rows(self.cache)
        keep = [r for r in rows.values()
                if r["latent_file"] in self.ledger and r["reference_file"] in self.ledger]
        data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in sorted(keep, key=lambda r: r["id"])).encode()
        remote = retry(lambda: remote_tree(self.api, self.repo, "dataset", self.prefix), "list remote meta")
        ops = []
        if keep and not entry_matches_bytes(remote.get("index.jsonl"), data):
            ops.append(self.Add(path_in_repo=self.rp("index.jsonl"), path_or_fileobj=data))
        for name in CACHE_EXTRAS:
            p = self.cache / name
            if p.is_file():
                b = p.read_bytes()
                if not entry_matches_bytes(remote.get(name), b):
                    ops.append(self.Add(path_in_repo=self.rp(name), path_or_fileobj=b))
        if self.sources_root:
            p = self.sources_root / "sources_v2.json"
            if p.is_file():
                b = p.read_bytes()
                if not entry_matches_bytes(remote.get("sources_root/sources_v2.json"), b):
                    ops.append(self.Add(path_in_repo=self.rp("sources_root/sources_v2.json"), path_or_fileobj=b))
        return ops

    def _commit(self, ops, message, shas):
        wait = self.min_commit_interval - (time.monotonic() - self.last_commit)
        if wait > 0:
            time.sleep(wait)
        commit(self.api, self.repo, "dataset", ops, message)
        self.last_commit = time.monotonic()
        self.ledger.update(shas)
        self.save_ledger()
        log(f"[cache] committed '{message}' ({len(ops)} ops)")


# ---------------------------------------------------------------------------- dir backup
DIR_EXTS = (".json", ".jsonl", ".md", ".png", ".jpg", ".html", ".js", ".csv", ".txt", ".log")
RUN_FOLDER_SKIP = ("tracks.npz",)
TMP_SUFFIXES = (".tmp", ".partial")


def is_run_folder(d: Path) -> bool:
    """True for a generation run folder (has ``run.json`` or ``video.mp4``)."""
    return (d / "run.json").is_file() or (d / "video.mp4").is_file()


def run_folder_complete(d: Path) -> bool:
    """A run folder is complete when run.json (final marker) and video.mp4 both exist."""
    return (d / "run.json").is_file() and (d / "video.mp4").is_file()


def dir_candidates(root: Path, settle_seconds=0.0, now=None):
    """[(rel posix, path)] of files that are safe to upload now (completed run folders + settled aggregates)."""
    now = time.time() if now is None else now
    out = []
    root = Path(root)

    def walk(d: Path, in_run: bool, run_ok: bool):
        """Recursive walk yielding uploadable files; run folders only once complete, temp files never."""
        for p in sorted(d.iterdir()):
            if p.name.endswith(TMP_SUFFIXES) or p.name == "__pycache__" or p.name.startswith(".hf_"):
                continue
            if p.is_dir():
                if is_run_folder(p):
                    walk(p, True, run_folder_complete(p))
                else:
                    walk(p, in_run, run_ok)
            elif p.is_file():
                if in_run:
                    if run_ok and p.name not in RUN_FOLDER_SKIP:
                        out.append((p.relative_to(root).as_posix(), p))
                elif p.suffix.lower() in DIR_EXTS and now - p.stat().st_mtime >= settle_seconds:
                    out.append((p.relative_to(root).as_posix(), p))

    if root.is_dir():
        walk(root, False, False)
    return out


class DirBackup:
    """Backs up an evaluation output directory tree (run folders once complete, aggregate json/md/png/html files once settled) to a
    repo, with a local ledger keyed by ``size:mtime:sha256`` so changed files are re-uploaded and unchanged ones are not.
    """

    def __init__(self, api, root, repo, repo_type, prefix, ledger=None, batch_files=20, batch_seconds=300.0,
                 max_batch=100, min_commit_interval=60.0, settle_seconds=10.0):
        from huggingface_hub import CommitOperationAdd
        self.Add = CommitOperationAdd
        self.api, self.root, self.repo, self.repo_type = api, Path(root), repo, repo_type
        self.prefix = prefix.strip("/")
        self.ledger_path = Path(ledger) if ledger else None
        self.batch_files, self.batch_seconds, self.max_batch = batch_files, batch_seconds, max_batch
        self.min_commit_interval, self.settle_seconds = min_commit_interval, settle_seconds
        self.ledger = {}
        if self.ledger_path and self.ledger_path.is_file():
            try:
                self.ledger = json.loads(self.ledger_path.read_text())
            except Exception:
                self.ledger = {}
        self.last_commit = 0.0
        self.pending_since = None

    def rp(self, rel):
        """Path inside the repo: ``<prefix>/<rel>``."""
        return f"{self.prefix}/{rel}"

    def save_ledger(self):
        """Persist the local upload ledger atomically."""
        if self.ledger_path:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.ledger_path.with_name(self.ledger_path.name + ".tmp")
            tmp.write_text(json.dumps(self.ledger))
            os.replace(tmp, self.ledger_path)

    @staticmethod
    def stamp(path: Path):
        """Cheap change detector of a file: ``size:mtime_ns``."""
        st = path.stat()
        return f"{st.st_size}:{st.st_mtime_ns}"

    def pending(self):
        """Files that are new or changed since the ledger entry."""
        out = []
        for rel, path in dir_candidates(self.root, self.settle_seconds):
            try:
                led = self.ledger.get(rel)
                if led and led.rsplit(":", 1)[0] == self.stamp(path):
                    continue
                out.append((rel, path, self.stamp(path)))
            except OSError:
                continue
        return out

    def pass_once(self, flush=False):
        """One pass: batch up pending files (``flush`` ignores thresholds), upload them in rate-limited commits; True if committed."""
        pend = self.pending()
        now = time.monotonic()
        if pend and self.pending_since is None:
            self.pending_since = now
        if not pend:
            self.pending_since = None
            return False
        due = flush or len(pend) >= self.batch_files or now - self.pending_since >= self.batch_seconds
        if not due:
            log(f"[dir] {len(pend)} files pending (batch threshold {self.batch_files})")
            return False
        remote = None
        committed = False
        while pend:
            batch, pend = pend[: self.max_batch], pend[self.max_batch:]
            ops, led = [], {}
            for rel, path, stamp in batch:
                sha = sha256_file(path)
                if self.stamp(path) != stamp:
                    continue  # changed while hashing: next pass
                old = self.ledger.get(rel)
                if old and old.rsplit(":", 1)[1] == sha:
                    led[rel] = f"{stamp}:{sha}"  # touched, same content
                    continue
                if remote is None:
                    remote = retry(lambda: remote_tree(self.api, self.repo, self.repo_type, self.prefix), "list remote")
                ent = remote.get(rel)
                if ent and (ent[1] == sha or (ent[1] is None and ent[2] == git_blob_id(path.read_bytes()))):
                    led[rel] = f"{stamp}:{sha}"
                    continue
                ops.append(self.Add(path_in_repo=self.rp(rel), path_or_fileobj=str(path)))
                led[rel] = f"{stamp}:{sha}"
            if ops:
                wait = self.min_commit_interval - (time.monotonic() - self.last_commit)
                if wait > 0:
                    time.sleep(wait)
                commit(self.api, self.repo, self.repo_type, ops, f"add {len(ops)} eval files")
                self.last_commit = time.monotonic()
                committed = True
                log(f"[dir] committed {len(ops)} files")
            self.ledger.update(led)
            self.save_ledger()
        self.pending_since = None
        return committed


# ---------------------------------------------------------------------------- sharded cache pack
SHARD_FORMAT = 1
SOURCES_MEMBER = "sources_root/sources_v2.json"


def collect_pack_members(cache: Path, sources_root=None):
    """{member name: ('file', path, expected_sha|None) | ('bytes', data, None)} for clips, references (+png),
    a filtered index.jsonl (rows whose clip and reference both exist), small extras and the sources manifest."""
    cache = Path(cache)
    members, keep = {}, []
    for row in sorted(read_index_rows(cache).values(), key=lambda r: r["id"]):
        lat, ref = row["latent_file"], row["reference_file"]
        if not ((cache / lat).is_file() and (cache / ref).is_file()):
            continue
        keep.append(row)
        members[lat] = ("file", cache / lat, row.get("sha256"))
        members[ref] = ("file", cache / ref, row.get("reference_sha256"))
        png = Path(ref).with_suffix(".png").as_posix()
        if png not in members and (cache / png).is_file():
            members[png] = ("file", cache / png, None)
    members["index.jsonl"] = ("bytes", "".join(json.dumps(r, sort_keys=True) + "\n" for r in keep).encode(), None)
    for name in CACHE_EXTRAS:
        if (cache / name).is_file():
            members[name] = ("file", cache / name, None)
    if sources_root and (Path(sources_root) / "sources_v2.json").is_file():
        members[SOURCES_MEMBER] = ("file", Path(sources_root) / "sources_v2.json", None)
    return members


def member_size(m):
    """Size in bytes of a pack member (inline bytes or a file)."""
    return len(m[1]) if m[0] == "bytes" else Path(m[1]).stat().st_size


def plan_shards(members, shard_bytes):
    """Deterministic greedy partition of the name-sorted members into shards of ~shard_bytes (tar overhead counted)."""
    shards, cur, used = [], [], 0
    for name in sorted(members):
        cost = ((member_size(members[name]) + 511) // 512) * 512 + 1024
        if cur and used + cost > shard_bytes:
            shards.append(cur)
            cur, used = [], 0
        cur.append(name)
        used += cost
    if cur:
        shards.append(cur)
    return shards


class _HashingReader:
    def __init__(self, f):
        self.f, self.h = f, hashlib.sha256()

    def read(self, n=-1):
        """Read and feed the bytes to the running sha256 (hash computed while uploading)."""
        b = self.f.read(n)
        self.h.update(b)
        return b


def write_shard(path: Path, names, members):
    """Write an uncompressed, reproducible tar (fixed mtime/owner/mode, GNU format). Returns (shard_sha256,
    [{name,size,sha256}]); each member's sha256 is computed while streaming and checked against the index sha."""
    import io
    import tarfile
    entries = []
    tmp = Path(str(path) + ".tmp")
    with tarfile.open(tmp, "w", format=tarfile.GNU_FORMAT) as tf:
        for name in names:
            kind, src, want = members[name]
            ti = tarfile.TarInfo(name)
            ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = 0, 0o644, 0, 0, "", ""
            if kind == "bytes":
                ti.size = len(src)
                tf.addfile(ti, io.BytesIO(src))
                sha = hashlib.sha256(src).hexdigest()
            else:
                ti.size = Path(src).stat().st_size
                with open(src, "rb") as f:
                    hr = _HashingReader(f)
                    tf.addfile(ti, hr)
                sha = hr.h.hexdigest()
            if want is not None and sha != want:
                raise RuntimeError(f"{name}: sha256 {sha} does not match the index row ({want})")
            entries.append({"name": name, "size": ti.size, "sha256": sha})
    os.replace(tmp, path)
    return sha256_file(path), entries


def pack_cache(cache_dir, work_dir, shard_bytes, upload, remote=None, sources_root=None, keep_local=False):
    """Pack the cache into tar shards and upload the new ones, then shards.json (last, so a manifest on the remote
    means the set is complete). ``upload(local_path, name)``; ``remote``: {shard name: sha256} already uploaded.
    Resumable: a shard whose freshly built sha256 equals the remote one is not uploaded again."""
    remote = remote or {}
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    members = collect_pack_members(cache_dir, sources_root)
    plan = plan_shards(members, shard_bytes)
    log(f"[pack] {len(members)} members -> {len(plan)} shards (~{shard_bytes / 1e9:.2f} GB each)")
    shards, uploaded, skipped = [], 0, 0
    for i, names in enumerate(plan):
        name = f"shard_{i:04d}.tar"
        path = work / name
        sha, entries = write_shard(path, names, members)
        size = path.stat().st_size
        if remote.get(name) == sha:
            skipped += 1
            log(f"[pack] {name} already uploaded (sha matches), skipped")
        else:
            upload(path, name)
            uploaded += 1
            log(f"[pack] {name} uploaded ({size / 1e6:.0f} MB, {len(entries)} members)")
        shards.append({"name": name, "size": size, "sha256": sha, "members": entries})
        if not keep_local:
            path.unlink()
    manifest = {"format": SHARD_FORMAT, "shard_bytes": shard_bytes, "n_members": len(members), "shards": shards}
    mpath = work / "shards.json"
    mpath.write_text(json.dumps(manifest, sort_keys=True, indent=1), encoding="utf8")
    upload(mpath, "shards.json")
    log(f"[pack] shards.json uploaded: {uploaded} shards uploaded, {skipped} skipped")
    return manifest


def pack_cache_hf(api, cache_dir, repo, prefix, shard_bytes, work_dir, sources_root=None):
    """Pack the cache into ~``shard_bytes`` shards plus ``shards.json`` and upload only the shards whose sha256 differs from the
    remote (one commit per shard: far fewer commits than per-file upload; ``restore_from_hf.py --prefer-shards`` reads them).
    """
    prefix = prefix.strip("/")
    remote = {rel: sha for rel, (size, sha, blob) in remote_tree(api, repo, "dataset", prefix).items() if sha}

    def upload(path, name):
        """Upload one packed file with retries."""
        retry(lambda: api.upload_file(path_or_fileobj=str(path), path_in_repo=f"{prefix}/{name}", repo_id=repo,
                                      repo_type="dataset", commit_message=f"cache shard {name}"), f"upload {name}")
    return pack_cache(cache_dir, work_dir, shard_bytes, upload, remote, sources_root)


# ---------------------------------------------------------------------------- single-instance lock
_HELD_LOCKS = []


def lock_path(repo, path_in_repo):
    """Lock file path for a (repo, path-in-repo) pair (``$HF_BACKUP_LOCK_DIR`` or the temp dir)."""
    import tempfile
    d = Path(os.environ.get("HF_BACKUP_LOCK_DIR") or Path(tempfile.gettempdir()) / "hf_backup_locks")
    key = hashlib.sha1(f"{repo}|{path_in_repo.strip('/')}".encode()).hexdigest()[:16]
    return d / f"{key}.lock"


def acquire_lock(repo, path_in_repo):
    """Exclusive, non-blocking OS lock per (repo, path-in-repo), held until the process exits (released by the OS
    even on a crash). Returns the lock file path, or None when another process holds it."""
    p = lock_path(repo, path_in_repo)
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    f.seek(0)
    f.truncate()
    f.write(f"{os.getpid()} {repo}:{path_in_repo}" + chr(10))
    f.flush()
    _HELD_LOCKS.append(f)
    return p


def require_lock(repo, path_in_repo):
    """Exit with status 3 if another hf_backup process already serves this (repo, path); prevents duplicate uploaders."""
    if acquire_lock(repo, path_in_repo) is None:
        log(f"[lock] another hf_backup process already serves {repo}:{path_in_repo} ({lock_path(repo, path_in_repo)}); exiting")
        raise SystemExit(3)


# ---------------------------------------------------------------------------- CLI
def run_loop(step, once, interval, label):
    """Run ``step`` once, or poll it every ``interval`` seconds forever; a failed pass is logged and retried, never fatal."""
    if once:
        step(True)
        return
    log(f"[{label}] watching, poll {interval}s")
    while True:
        try:
            step(False)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            log(f"[{label}] pass failed: {type(exc).__name__}: {str(exc)[:300]} (continuing)")
        time.sleep(interval)


def main(argv=None):
    """CLI entry: sub-commands ``watch-run/once-run``, ``watch-cache/once-cache``, ``watch-dir/once-dir``, ``pack-cache`` (see module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("watch-run", "once-run"):
        p = sub.add_parser(name)
        p.add_argument("run_dir")
        p.add_argument("--repo")
        p.add_argument("--path-in-repo", required=True)
        p.add_argument("--poll", type=float, default=60.0)
        p.add_argument("--keep-last", type=int, default=3)
        p.add_argument("--keep-every", type=int, default=250)
        p.add_argument("--extras-interval", type=float, default=600.0)
        p.add_argument("--min-commit-interval", type=float, default=60.0)
        p.add_argument("--once", action="store_true")
    for name in ("watch-cache", "once-cache"):
        p = sub.add_parser(name)
        p.add_argument("cache_dir")
        p.add_argument("--repo")
        p.add_argument("--path-in-repo", required=True)
        p.add_argument("--sources-root", help="e.g. data/v2: uploads sources_v2.json, sources/**, _work/annotations*.zip under sources_root/ (never the cache dir)")
        p.add_argument("--ledger", help="local ledger JSON (default <cache_dir>/../hf_backup_ledger_<path>.json)")
        p.add_argument("--poll", type=float, default=120.0)
        p.add_argument("--batch-files", type=int, default=50)
        p.add_argument("--batch-seconds", type=float, default=600.0)
        p.add_argument("--min-commit-interval", type=float, default=60.0)
        p.add_argument("--once", action="store_true")
    for name in ("watch-dir", "once-dir"):
        p = sub.add_parser(name)
        p.add_argument("dir")
        p.add_argument("--repo")
        p.add_argument("--repo-type", choices=("dataset", "model"), default="dataset")
        p.add_argument("--path-in-repo", required=True)
        p.add_argument("--ledger", help="local ledger JSON (default <dir>/../hf_backup_ledger_<path>.json)")
        p.add_argument("--poll", type=float, default=120.0)
        p.add_argument("--batch-files", type=int, default=20)
        p.add_argument("--batch-seconds", type=float, default=300.0)
        p.add_argument("--settle-seconds", type=float, default=10.0)
        p.add_argument("--min-commit-interval", type=float, default=60.0)
        p.add_argument("--once", action="store_true")
    p = sub.add_parser("pack-cache", help="tar the cache into ~1 GB uncompressed shards + shards.json and upload them")
    p.add_argument("cache_dir")
    p.add_argument("--repo")
    p.add_argument("--path-in-repo", default="cache_v2_shards")
    p.add_argument("--shard-gb", type=float, default=1.0)
    p.add_argument("--sources-root", help="e.g. data/v2: includes sources_v2.json as member sources_root/sources_v2.json")
    p.add_argument("--work-dir", help="temp dir for shards (default <cache_dir>/../.hf_pack_<path>)")
    a = ap.parse_args(argv)
    api = get_api()
    if a.cmd == "pack-cache":
        repo = a.repo or default_repo(api, "dataset")
        require_lock(repo, a.path_in_repo)
        retry(lambda: api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True), "create repo")
        work = a.work_dir or str(Path(a.cache_dir).resolve().parent / (".hf_pack_" + a.path_in_repo.strip("/").replace("/", "_")))
        pack_cache_hf(api, a.cache_dir, repo, a.path_in_repo, int(a.shard_gb * 1e9), work, a.sources_root)
        return 0
    once = a.cmd.startswith("once") or a.once
    mci = min(a.min_commit_interval, 5.0) if once else a.min_commit_interval
    if a.cmd.endswith("dir"):
        repo = a.repo or default_repo(api, a.repo_type)
        require_lock(repo, a.path_in_repo)
        retry(lambda: api.create_repo(repo, repo_type=a.repo_type, private=True, exist_ok=True), "create repo")
        ledger = a.ledger or str(Path(a.dir).resolve().parent / f"hf_backup_ledger_{a.path_in_repo.strip('/').replace('/', '_')}.json")
        b = DirBackup(api, a.dir, repo, a.repo_type, a.path_in_repo, ledger, a.batch_files, a.batch_seconds,
                      min_commit_interval=mci, settle_seconds=a.settle_seconds)
        log(f"[dir] {a.dir} -> {repo}:{a.path_in_repo}")
        run_loop(lambda final: b.pass_once(flush=final), once, a.poll, "dir")
    elif a.cmd.endswith("run"):
        repo = a.repo or default_repo(api, "model")
        require_lock(repo, a.path_in_repo)
        retry(lambda: api.create_repo(repo, repo_type="model", private=True, exist_ok=True), "create repo")
        b = RunBackup(api, a.run_dir, repo, a.path_in_repo, a.keep_last, a.keep_every, a.extras_interval, mci)
        log(f"[run] {a.run_dir} -> {repo}:{a.path_in_repo}")
        run_loop(lambda final: b.pass_once(final=final), once, a.poll, "run")
    else:
        repo = a.repo or default_repo(api, "dataset")
        require_lock(repo, a.path_in_repo)
        retry(lambda: api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True), "create repo")
        ledger = a.ledger or str(Path(a.cache_dir).resolve().parent / f"hf_backup_ledger_{a.path_in_repo.strip('/').replace('/', '_')}.json")
        b = CacheBackup(api, a.cache_dir, repo, a.path_in_repo, a.sources_root, ledger,
                        a.batch_files, a.batch_seconds, min_commit_interval=mci)
        log(f"[cache] {a.cache_dir} -> {repo}:{a.path_in_repo}")
        run_loop(lambda final: b.pass_once(flush=final), once, a.poll, "cache")
    return 0


if __name__ == "__main__":
    sys.exit(main())
