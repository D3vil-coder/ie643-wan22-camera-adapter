#!/usr/bin/env python
"""Restore a training run / the v2 latent cache / the sources from the private HF backup repos.

Run on a fresh sandbox after the token is installed ($HF_HOME/token or HF_TOKEN).  Needs only
huggingface_hub (no torch, no fpcamera imports).  Idempotent: files that already exist locally with
the right sha256 are skipped; downloads go to a staging dir and are verified before being moved in place.

  python restore_from_hf.py --run NAME --run-dir artifacts/v2/NAME [--step N] [--model-repo USER/...]
  python restore_from_hf.py --cache --cache-dir data/v2/cache_1280x704_f121_v2 [--cache-path cache_v2] [--workers 8]
  python restore_from_hf.py --sources --sources-root data/v2          # sources_v2.json + sources/**

Run restore: tries the checkpoint named by the remote ``latest.json`` first, then every remote manifest
newest-first (or exactly ``--step``); the first one whose adapter + resume files verify against the
manifest sha256 is installed under ``<run-dir>/checkpoints`` and ``latest.json`` is written last, so
``train_v2.py`` with the same arguments auto-resumes.  config/progress/launches files are restored when
absent locally; ``run_status.json`` is deliberately not restored.
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import argparse
import concurrent.futures as cf
import fnmatch
import hashlib
import json
import os
import random
from pathlib import Path
import re
import shutil
import sys
import time

_MANIFEST_RE = re.compile(r"^ckpt_(\d{6,})\.json$")
RUN_EXTRAS = ("config.json", "progress.json", "progress.jsonl", "launches.jsonl")


def log(msg):
    """Timestamped print."""
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sha256_file(path) -> str:
    """Streaming sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def get_api():
    """``HfApi`` using ``HF_TOKEN`` / ``HUGGINGFACE_TOKEN`` from the environment (or the cached login)."""
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None)


def tree(api, repo, repo_type, prefix):
    """``{path: (size, lfs_sha256)}`` of a remote folder (empty if missing)."""
    out = {}
    try:
        items = list(api.list_repo_tree(repo, path_in_repo=prefix, recursive=True, repo_type=repo_type))
    except Exception as exc:  # noqa: BLE001 - a missing folder means "nothing there yet" (fresh eval/run start)
        if type(exc).__name__ in ("EntryNotFoundError", "RepositoryNotFoundError") or "404" in str(exc):
            return out
        raise
    for it in items:
        if type(it).__name__ == "RepoFile":
            lfs = getattr(it, "lfs", None)
            sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
            out[it.path[len(prefix) + 1:]] = (it.size, sha)
    return out


def backoff_delay(attempt, exc=None, base=5.0, cap=120.0, rng=random.random):
    """Exponential backoff with jitter; honors a ``Retry-After`` header (seconds) on HTTP 429."""
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        ra = (getattr(resp, "headers", None) or {}).get("Retry-After")
        try:
            if ra is not None:
                return min(float(ra) + rng(), 600.0)
        except (TypeError, ValueError):
            pass
        base = max(base, 30.0)  # rate limit window is minutes; wait longer than for generic errors
    return min(base * 2 ** attempt, cap) * (0.5 + rng() / 2)


def fetch(api, repo, repo_type, remote_path, staging: Path, attempts=8, sleep=time.sleep) -> Path:
    """Download one remote file to the staging folder with retries; returns the local path."""
    from huggingface_hub import hf_hub_download
    last = None
    for i in range(attempts):
        try:
            return Path(hf_hub_download(repo, remote_path, repo_type=repo_type, local_dir=str(staging),
                                        token=api.token))
        except Exception as exc:  # noqa: BLE001
            last = exc
            d = backoff_delay(i, exc)
            log(f"[retry] {remote_path}: {type(exc).__name__}: {str(exc)[:160]} (sleep {d:.0f}s)")
            sleep(d)
    raise RuntimeError(f"download failed: {remote_path}: {last!r}")


def bulk_download(api, repo, repo_type, prefix, rels, staging: Path, workers, attempts=8):
    """One ``snapshot_download`` (single listing call + parallel LFS downloads) for ``rels`` under prefix.

    Files land in ``staging/<prefix>/<rel>``; returns the number of attempts used.  Retries the whole
    call with backoff (already-downloaded files are skipped by huggingface_hub).
    """
    from huggingface_hub import snapshot_download
    patterns = [f"{prefix}/{r}" for r in rels]
    for i in range(attempts):
        try:
            snapshot_download(repo, repo_type=repo_type, local_dir=str(staging), allow_patterns=patterns,
                              max_workers=workers, token=api.token)
            return i + 1
        except Exception as exc:  # noqa: BLE001
            d = backoff_delay(i, exc)
            log(f"[retry] snapshot_download: {type(exc).__name__}: {str(exc)[:160]} (sleep {d:.0f}s)")
            time.sleep(d)
    raise RuntimeError("snapshot_download failed")


def place(src: Path, dst: Path):
    """Atomically move a downloaded file to its destination."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.move(str(src), str(tmp))
    os.replace(tmp, dst)


# ------------------------------------------------------------------------------------ run
def restore_run(api, repo, name, run_dir: Path, step=None, path_in_repo=None):
    """Restore a training run (latest or ``step`` checkpoint + config/progress/ledgers) into ``run_dir`` so training or
    inference can continue; verifies sha256 against the manifest.
    """
    prefix = path_in_repo or f"runs/{name}"
    remote = tree(api, repo, "model", prefix)
    if not remote:
        raise SystemExit(f"nothing under {repo}:{prefix}")
    steps = sorted({int(m.group(1)) for rel in remote if (m := _MANIFEST_RE.match(rel.removeprefix("checkpoints/")))},
                   reverse=True)
    order = [step] if step is not None else []
    if step is None:
        staging0 = run_dir / ".hf_restore_tmp"
        if "latest.json" in remote:
            try:
                p = fetch(api, repo, "model", f"{prefix}/latest.json", staging0)
                order.append(int(json.loads(p.read_text())["step"]))
            except Exception as exc:
                log(f"[warn] remote latest.json unusable: {exc!r}")
        order += [s for s in steps if s not in order]
    staging = run_dir / ".hf_restore_tmp"
    cdir = run_dir / "checkpoints"
    cdir.mkdir(parents=True, exist_ok=True)
    chosen, man = None, None
    for s in order:
        mname = f"ckpt_{s:06d}.json"
        if f"checkpoints/{mname}" not in remote:
            log(f"[skip] step {s}: no manifest on remote")
            continue
        try:
            man = json.loads(fetch(api, repo, "model", f"{prefix}/checkpoints/{mname}", staging).read_text())
            got = []
            for key, shakey in (("adapter_file", "adapter_sha256"), ("resume_file", "resume_sha256")):
                fname = man[key]
                dst = cdir / fname
                if dst.is_file() and sha256_file(dst) == man[shakey]:
                    log(f"[have] {fname}")
                    continue
                log(f"[get] {fname}")
                p = fetch(api, repo, "model", f"{prefix}/checkpoints/{fname}", staging)
                if sha256_file(p) != man[shakey]:
                    raise ValueError(f"sha256 mismatch for {fname}")
                got.append((p, dst))
            for p, dst in got:
                place(p, dst)
            tmpm = cdir / (mname + ".tmp")
            tmpm.write_text(json.dumps(man, indent=2))
            os.replace(tmpm, cdir / mname)
            chosen = s
            break
        except Exception as exc:  # noqa: BLE001 - corrupt/partial remote step: fall back to the previous one
            log(f"[skip] step {s}: {type(exc).__name__}: {exc}")
    if chosen is None:
        raise SystemExit("no remote checkpoint verified")
    for extra in RUN_EXTRAS:
        if extra in remote and not (run_dir / extra).exists():
            place(fetch(api, repo, "model", f"{prefix}/{extra}", staging), run_dir / extra)
    ptr = {"step": chosen, "manifest": f"ckpt_{chosen:06d}.json", "adapter_file": man["adapter_file"],
           "adapter_sha256": man["adapter_sha256"], "resume_file": man["resume_file"],
           "resume_sha256": man["resume_sha256"], "reason": man.get("reason", "restored")}
    tmp = run_dir / "latest.json.tmp"
    tmp.write_text(json.dumps(ptr, indent=2))
    os.replace(tmp, run_dir / "latest.json")  # last, as in checkpointing.py
    shutil.rmtree(staging, ignore_errors=True)
    log(f"[run] restored step {chosen} into {run_dir}")
    return chosen


# ------------------------------------------------------------------------------- cache / sources
def read_rows(path: Path):
    """Read the rows of an ``index*.jsonl``."""
    rows = []
    for line in path.read_text(encoding="utf8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def sync_files(api, repo, prefix, remote, rels, dest_root: Path, workers, expected=None):
    """Download ``rels`` (relative to prefix) into dest_root, verifying sha256 (remote LFS or expected)."""
    expected = expected or {}
    staging = dest_root.parent / f".hf_restore_{dest_root.name}"
    stats = {"have": 0, "got": 0, "bad": 0}
    need = []
    for rel in rels:
        dst = dest_root / (rel.removeprefix("sources_root/") if rel.startswith("sources_root/") else rel)
        size, lfs_sha = remote[rel]
        want = expected.get(rel) or lfs_sha
        if not (dst.is_file() and dst.stat().st_size == size and (want is None or sha256_file(dst) == want)):
            need.append(rel)
    if need:
        log(f"[bulk] snapshot_download of {len(need)} files")
        try:
            bulk_download(api, repo, "dataset", prefix, need, staging, workers)
        except Exception as exc:  # noqa: BLE001 - fall back to per-file path with backoff
            log(f"[warn] bulk download failed ({exc!r}); falling back to per-file")

    def one(rel):
        """Download one file unless it is already present with the expected size/hash."""
        dst = dest_root / rel.removeprefix("sources_root/") if rel.startswith("sources_root/") else dest_root / rel
        size, lfs_sha = remote[rel]
        want = expected.get(rel) or lfs_sha
        if dst.is_file() and dst.stat().st_size == size and (want is None or sha256_file(dst) == want):
            return "have"
        p = staging / prefix / rel
        if not p.is_file():
            p = fetch(api, repo, "dataset", f"{prefix}/{rel}", staging)
        if want is not None and sha256_file(p) != want:
            p.unlink()
            return "bad"
        place(p, dst)
        return "got"

    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(one, r): r for r in rels}
        for i, f in enumerate(cf.as_completed(futs), 1):
            try:
                stats[f.result()] += 1
            except Exception as exc:  # noqa: BLE001
                stats["bad"] += 1
                log(f"[bad] {futs[f]}: {exc!r}")
            if i % 100 == 0:
                log(f"[progress] {i}/{len(futs)} {stats}")
    shutil.rmtree(staging, ignore_errors=True)
    return stats


def restore_cache(api, repo, prefix, cache_dir: Path, workers):
    """Restore the v2 latent cache file by file (resumable; verifies every sha256)."""
    remote = tree(api, repo, "dataset", prefix)
    if "index.jsonl" not in remote:
        raise SystemExit(f"no index.jsonl under {repo}:{prefix}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    staging = cache_dir.parent / f".hf_restore_{cache_dir.name}"
    idx_path = fetch(api, repo, "dataset", f"{prefix}/index.jsonl", staging)
    rows = read_rows(idx_path)
    expected, rels = {}, []
    for r in rows:
        for fkey, skey in (("latent_file", "sha256"), ("reference_file", "reference_sha256")):
            if r[fkey] in remote and r[fkey] not in expected:
                expected[r[fkey]] = r.get(skey)
                rels.append(r[fkey])
    rels += [k for k in remote if k.endswith(".png") and k.startswith("references/")]
    rels += [k for k in ("config_snapshot.json", "rejected.jsonl", "progress.json") if k in remote]
    log(f"[cache] {len(rows)} rows, {len(rels)} files")
    stats = sync_files(api, repo, prefix, remote, rels, cache_dir, workers, expected)
    # final verification pass + local index containing only verified rows
    good = []
    for r in rows:
        f, ref = cache_dir / r["latent_file"], cache_dir / r["reference_file"]
        if f.is_file() and ref.is_file() and sha256_file(f) == r["sha256"] and sha256_file(ref) == r["reference_sha256"]:
            good.append(r)
    tmp = cache_dir / "index.jsonl.partial"
    tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in good), encoding="utf8")
    os.replace(tmp, cache_dir / "index.jsonl")
    shutil.rmtree(staging, ignore_errors=True)
    log(f"[cache] {stats}; verified rows {len(good)}/{len(rows)}")
    if len(good) != len(rows):
        raise SystemExit(f"{len(rows) - len(good)} rows failed verification")
    return len(good)


# ------------------------------------------------------------------------------------ sharded cache
def shards_available(api, repo, prefix):
    """True if the remote cache is stored as packed shards (``shards.json``)."""
    try:
        return "shards.json" in tree(api, repo, "dataset", prefix)
    except Exception:  # noqa: BLE001 - missing folder / transient: fall back to per-file
        return False


def _safe_member_path(root: Path, name: str) -> Path:
    p = (root / name).resolve()
    if root.resolve() != p and root.resolve() not in p.parents:
        raise ValueError(f"unsafe tar member path: {name}")
    return p


def extract_shard(tar_path: Path, members, dest_root: Path, sources_root: Path):
    """Extract regular files only, verifying each member's sha256 against ``members`` ([{name,size,sha256}]) while
    streaming; a verified file is placed atomically (``sources_root/...`` goes under ``sources_root``)."""
    import tarfile
    want = {m["name"]: m for m in members}
    seen = set()
    with tarfile.open(tar_path, "r") as tf:
        for ti in tf:
            m = want.get(ti.name)
            if m is None or not ti.isreg():
                raise ValueError(f"unexpected tar member {ti.name!r} in {tar_path.name}")
            if ti.name.startswith("sources_root/"):
                dst = _safe_member_path(sources_root, ti.name[len("sources_root/"):])
            else:
                dst = _safe_member_path(dest_root, ti.name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".tmp")
            h = hashlib.sha256()
            n = 0
            src = tf.extractfile(ti)
            with open(tmp, "wb") as out:
                for b in iter(lambda: src.read(8 << 20), b""):
                    h.update(b)
                    n += len(b)
                    out.write(b)
            if h.hexdigest() != m["sha256"] or n != m["size"]:
                tmp.unlink()
                raise ValueError(f"member {ti.name}: sha256/size mismatch")
            os.replace(tmp, dst)
            seen.add(ti.name)
    missing = set(want) - seen
    if missing:
        raise ValueError(f"{tar_path.name}: {len(missing)} members missing from the tar")
    return len(seen)


def shard_done(shard, dest_root: Path, sources_root: Path):
    """True if every member of a shard is already present locally with the expected size and hash."""
    for m in shard["members"]:
        p = (sources_root / m["name"][len("sources_root/"):]) if m["name"].startswith("sources_root/") else dest_root / m["name"]
        if not (p.is_file() and p.stat().st_size == m["size"] and sha256_file(p) == m["sha256"]):
            return False
    return True


def restore_cache_shards(api, repo, prefix, cache_dir: Path, workers, sources_root: Path = None, fetch_fn=None):
    """Download ``<prefix>/shards.json`` and each shard (parallel, 429 backoff), verify the shard sha256, extract and
    verify every member sha256; shards whose members are already present and valid are skipped. Finally checks the
    extracted index.jsonl rows against their files. ``fetch_fn(remote_path, staging)`` is injectable for tests."""
    fetch_fn = fetch_fn or (lambda rp, st: fetch(api, repo, "dataset", rp, st))
    cache_dir.mkdir(parents=True, exist_ok=True)
    sources_root = sources_root or cache_dir.parent
    staging = cache_dir.parent / f".hf_restore_shards_{cache_dir.name}"
    man = json.loads(Path(fetch_fn(f"{prefix}/shards.json", staging)).read_text(encoding="utf8"))
    shards = man["shards"]
    log(f"[shards] {len(shards)} shards, {man.get('n_members')} members")

    def one(sh):
        """Download and unpack one shard unless all its members are present."""
        if shard_done(sh, cache_dir, sources_root):
            return "have"
        p = Path(fetch_fn(f"{prefix}/{sh['name']}", staging))
        if p.stat().st_size != sh["size"] or sha256_file(p) != sh["sha256"]:
            p.unlink()
            raise ValueError(f"{sh['name']}: size/sha256 mismatch after download")
        extract_shard(p, sh["members"], cache_dir, sources_root)
        p.unlink()
        return "got"

    stats = {"have": 0, "got": 0, "bad": 0}
    with cf.ThreadPoolExecutor(max(1, min(workers, 4))) as ex:
        futs = {ex.submit(one, s): s["name"] for s in shards}
        for f in cf.as_completed(futs):
            try:
                stats[f.result()] += 1
            except Exception as exc:  # noqa: BLE001
                stats["bad"] += 1
                log(f"[bad] {futs[f]}: {exc!r}")
    shutil.rmtree(staging, ignore_errors=True)
    if stats["bad"]:
        raise SystemExit(f"{stats['bad']} shards failed: {stats}")
    rows = read_rows(cache_dir / "index.jsonl")
    for r in rows:
        if not ((cache_dir / r["latent_file"]).is_file() and (cache_dir / r["reference_file"]).is_file()):
            raise SystemExit(f"row {r.get('id')}: files missing after shard restore")
    log(f"[shards] {stats}; {len(rows)} index rows, all members sha256-verified")
    return len(rows)


def restore_sources(api, repo, prefix, sources_root: Path, workers):
    """Restore the source photographs (``sources_v2.json`` + ``sources/``) used to render the cache."""
    allow = ("sources_v2.json", "sources/*", "_work/annotations*.zip")  # same list as tools/hf_backup.py
    remote = {k: v for k, v in tree(api, repo, "dataset", prefix).items()
              if k.startswith("sources_root/") and any(fnmatch.fnmatchcase(k[len("sources_root/"):], g) for g in allow)}
    if not remote:
        raise SystemExit(f"no sources under {repo}:{prefix}")
    sources_root.mkdir(parents=True, exist_ok=True)
    stats = sync_files(api, repo, prefix, remote, list(remote), sources_root, workers)
    log(f"[sources] {stats}")
    if stats["bad"]:
        raise SystemExit("some source files failed")
    return stats


def restore_checkpoints(api, repo, name, run_dir: Path, steps, path_in_repo=None):
    """Download the adapter (+ manifest) of specific historical steps for validation/eval.

    Does NOT touch ``latest.json`` or the resume files; sha256-verified; missing steps are reported.
    Returns the list of steps that are now present locally.
    """
    prefix = path_in_repo or f"runs/{name}"
    remote = tree(api, repo, "model", prefix)
    cdir = run_dir / "checkpoints"
    cdir.mkdir(parents=True, exist_ok=True)
    staging = run_dir / ".hf_restore_tmp"
    have = []
    for s in steps:
        mname = f"ckpt_{s:06d}.json"
        if f"checkpoints/{mname}" not in remote:
            log(f"[miss] step {s}: not on the remote (pruned by retention?)")
            continue
        man = json.loads(fetch(api, repo, "model", f"{prefix}/checkpoints/{mname}", staging).read_text())
        fname = man["adapter_file"]
        dst = cdir / fname
        if not (dst.is_file() and sha256_file(dst) == man["adapter_sha256"]):
            p = fetch(api, repo, "model", f"{prefix}/checkpoints/{fname}", staging)
            if sha256_file(p) != man["adapter_sha256"]:
                log(f"[bad] step {s}: sha256 mismatch")
                continue
            place(p, dst)
        if not (cdir / mname).exists():
            (cdir / mname).write_text(json.dumps(man, indent=2))
        log(f"[ckpt] step {s}: {fname}")
        have.append(s)
    # config.json carries the training fill_mode that inference reads next to the checkpoints; without it a v2
    # adapter would silently run with the v1 zero fill. Progress/launch logs are cheap too (latest.json stays untouched).
    for extra in RUN_EXTRAS:
        if extra in remote and not (run_dir / extra).exists():
            place(fetch(api, repo, "model", f"{prefix}/{extra}", staging), run_dir / extra)
            log(f"[extra] {extra}")
    shutil.rmtree(staging, ignore_errors=True)
    return have


def restore_eval(api, repo, name, dest: Path, workers, prefix=None):
    """Restore ``eval/<name>`` (backed up by ``hf_backup.py watch-dir``) into ``dest``; same-size files are kept."""
    prefix = prefix or f"eval/{name}"
    remote = tree(api, repo, "dataset", prefix)
    dest.mkdir(parents=True, exist_ok=True)
    if not remote:
        log(f"[eval] nothing under {repo}:{prefix} (fresh start)")
        return {"have": 0, "got": 0, "bad": 0}
    stats = sync_files(api, repo, prefix, remote, list(remote), dest, workers)
    log(f"[eval] {name}: {stats}")
    if stats["bad"]:
        raise SystemExit("some eval files failed")
    return stats


def parse_name_dir(text):
    """Parse ``NAME:DIR`` (argparse type)."""
    if ":" not in text:
        raise argparse.ArgumentTypeError(f"expected NAME:DIR, got {text!r}")
    name, d = text.split(":", 1)
    return name, d


def main(argv=None):
    """CLI entry: ``--run NAME`` | ``--cache`` | ``--sources`` | ``--dir NAME:DIR`` (see module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run")
    ap.add_argument("--run-dir")
    ap.add_argument("--step", type=int)
    ap.add_argument("--run-path", help="remote folder (default runs/<name>)")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--cache-dir")
    ap.add_argument("--cache-path", default="cache_v2")
    ap.add_argument("--cache-shards", action="store_true",
                    help="restore the cache from tar shards (<--shards-path>/shards.json); needs --cache-dir")
    ap.add_argument("--prefer-shards", action="store_true",
                    help="with --cache: use shards when shards.json exists on HF, else per-file")
    ap.add_argument("--shards-path", default="cache_v2_shards")
    ap.add_argument("--sources", action="store_true")
    ap.add_argument("--sources-root", default="data/v2")
    ap.add_argument("--eval", type=parse_name_dir, action="append", default=[], metavar="NAME:DIR",
                    help="restore dataset-repo eval/NAME into DIR (repeatable; watch-dir backups)")
    ap.add_argument("--checkpoints", help="with --run: comma-separated historical steps; adapter-only, latest.json untouched")
    ap.add_argument("--model-repo")
    ap.add_argument("--dataset-repo")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    if not (a.run or a.cache or a.cache_shards or a.sources or a.eval):
        ap.error("nothing to do: pass --run, --cache and/or --sources")
    api = get_api()
    user = api.whoami()["name"]
    log(f"hf user: {user}")
    if a.cache_shards or a.cache:
        if not a.cache_dir:
            ap.error("--cache/--cache-shards needs --cache-dir")
        drepo = a.dataset_repo or f"{user}/ie643-camera-v2-cache"
        if a.cache_shards or (a.prefer_shards and shards_available(api, drepo, a.shards_path)):
            restore_cache_shards(api, drepo, a.shards_path, Path(a.cache_dir), a.workers, Path(a.sources_root))
        else:
            restore_cache(api, drepo, a.cache_path, Path(a.cache_dir), a.workers)
    if a.sources:
        restore_sources(api, a.dataset_repo or f"{user}/ie643-camera-v2-cache", a.cache_path, Path(a.sources_root), a.workers)
    for name, d in a.eval:
        restore_eval(api, a.dataset_repo or f"{user}/ie643-camera-v2-cache", name, Path(d), a.workers)
    if a.run and a.checkpoints:
        if not a.run_dir:
            ap.error("--run needs --run-dir")
        steps = [int(x) for x in a.checkpoints.split(",") if x.strip()]
        got = restore_checkpoints(api, a.model_repo or f"{user}/ie643-camera-adapter-v2", a.run, Path(a.run_dir), steps, a.run_path)
        if set(got) != set(steps):
            raise SystemExit(f"missing steps: {sorted(set(steps) - set(got))}")
        return 0
    if a.run:
        if not a.run_dir:
            ap.error("--run needs --run-dir")
        restore_run(api, a.model_repo or f"{user}/ie643-camera-adapter-v2", a.run, Path(a.run_dir), a.step, a.run_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
