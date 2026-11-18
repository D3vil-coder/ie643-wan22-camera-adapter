#!/usr/bin/env python
"""Laptop helper: pull milestone checkpoints (every 250th update + the final/latest) from the HF backup.

  python tools/pull_milestones.py RUN_NAME [--out-dir DIR] [--keep 4]
                                  [--every 250] [--with-resume] [--repo USER/ie643-camera-adapter-v2]

Output folder (``--out-dir``): default ``$FPCAMERA_CHECKPOINTS_DIR``, else ``<repo root>/artifacts/checkpoints/v2``
(git-ignored); the run lands in ``<out-dir>/RUN_NAME``.
Token: env HF_TOKEN, else a token variable read at runtime from the KEY=value file ``$FPCAMERA_KEYS_ENV`` (default
``<repo root>/keys.env``, git-ignored): HF_DOWNLOAD_TOKEN, then HF_FULL_ACCESS_TOKEN, then HF_WRITE_TOKEN; the value is never
printed.  Failing both, a token cached by ``huggingface-cli login`` is used.  See tools/local_config.py.
Downloads adapter_NNNNNN.safetensors + ckpt_NNNNNN.json (+ resume_*.pt with --with-resume), checks sha256
against the manifest, then keeps only the newest ``--keep`` steps locally (the old keep-last-3-4 convention).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_config import CHECKPOINTS_DIR_VAR, KEYS_ENV_VAR, KeysEnvNotFound, checkpoints_dir, keys_env_path  # noqa: E402

TOKEN_VARS = ("HF_TOKEN", "HF_DOWNLOAD_TOKEN", "HF_FULL_ACCESS_TOKEN", "HF_WRITE_TOKEN")
_MAN = re.compile(r"^ckpt_(\d{6,})\.json$")


def read_token():
    """``HF_TOKEN`` from the environment, else a download-capable token read from the local keys file (never printed).

    The keys file is ``$FPCAMERA_KEYS_ENV`` or ``<repo root>/keys.env``. When neither exists this returns None (a token cached
    by ``huggingface-cli login`` may still work), except that an ``$FPCAMERA_KEYS_ENV`` pointing at a missing file is an error.
    """
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    vals = {}
    try:
        for line in open(keys_env_path(), encoding="utf8"):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip().strip("\"'")
    except KeysEnvNotFound as exc:
        if os.environ.get(KEYS_ENV_VAR):
            sys.exit(str(exc))
    except OSError:
        pass
    for k in TOKEN_VARS:
        if vals.get(k):
            return vals[k]
    return None


def sha256_file(p):
    """Streaming sha256 of a file."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    """CLI entry: download the milestone checkpoints of a run, verify them against the manifests, prune old local ones."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help=f"parent of the run folder (default: ${CHECKPOINTS_DIR_VAR}, else <repo root>/artifacts/checkpoints/v2)")
    ap.add_argument("--repo")
    ap.add_argument("--path-in-repo")
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--keep", type=int, default=4)
    ap.add_argument("--with-resume", action="store_true")
    a = ap.parse_args(argv)
    try:
        from huggingface_hub import HfApi, get_token, hf_hub_download
    except ImportError:
        sys.exit("pip install huggingface_hub")
    token = read_token()
    if token is None and not get_token():
        sys.exit(f"no Hugging Face token: set HF_TOKEN, or put HF_DOWNLOAD_TOKEN=... in a KEY=value file and point "
                 f"{KEYS_ENV_VAR} at it (default <repo root>/keys.env), or run `huggingface-cli login`")
    api = HfApi(token=token)
    repo = a.repo or f"{api.whoami()['name']}/ie643-camera-adapter-v2"
    prefix = a.path_in_repo or f"runs/{a.run}"
    files = {it.path[len(prefix) + 1:] for it in api.list_repo_tree(repo, path_in_repo=prefix, recursive=True)
             if type(it).__name__ == "RepoFile"}
    steps = sorted(int(m.group(1)) for rel in files if (m := _MAN.match(rel.removeprefix("checkpoints/"))))
    if not steps:
        sys.exit(f"no checkpoints under {repo}:{prefix}")
    final = steps[-1]
    wanted = sorted({s for s in steps if a.every > 0 and s % a.every == 0} | {final})
    wanted = wanted[-a.keep:]  # only what would survive pruning anyway
    out = (a.out_dir if a.out_dir is not None else checkpoints_dir()) / a.run
    out.mkdir(parents=True, exist_ok=True)
    stage = out / ".hf_stage"
    for s in wanted:
        man = json.loads(Path(hf_hub_download(repo, f"{prefix}/checkpoints/ckpt_{s:06d}.json",
                                              local_dir=str(stage), token=token)).read_text())
        todo = [("adapter_file", "adapter_sha256")] + ([("resume_file", "resume_sha256")] if a.with_resume else [])
        for k, sk in todo:
            dst = out / man[k]
            if dst.is_file() and sha256_file(dst) == man[sk]:
                print(f"have {dst.name}")
                continue
            p = hf_hub_download(repo, f"{prefix}/checkpoints/{man[k]}", local_dir=str(stage), token=token)
            if sha256_file(p) != man[sk]:
                sys.exit(f"sha256 mismatch for {man[k]}")
            os.replace(p, dst)
            print(f"got  {dst.name}  sha256 ok")
        (out / f"ckpt_{s:06d}.json").write_text(json.dumps(man, indent=2))
    for extra in ("config.json", "progress.json", "latest.json"):
        if extra in files:
            os.replace(hf_hub_download(repo, f"{prefix}/{extra}", local_dir=str(stage), token=token), out / extra)
    import shutil
    shutil.rmtree(stage, ignore_errors=True)
    # keep only the newest --keep steps locally
    local = sorted(int(m.group(1)) for p in out.iterdir() if (m := _MAN.match(p.name)))
    for s in local[:-a.keep] if a.keep > 0 else []:
        for n in (f"adapter_{s:06d}.safetensors", f"resume_{s:06d}.pt", f"ckpt_{s:06d}.json"):
            try:
                (out / n).unlink()
                print(f"pruned {n}")
            except FileNotFoundError:
                pass
    print(f"milestones in {out}: {sorted(set(local[-a.keep:]))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
