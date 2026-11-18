#!/usr/bin/env python
"""Publish the final v2 camera adapter to a PUBLIC Hugging Face model repo (runs locally).

  python tools/publish_public_hf.py --dry-run          # stage + safety scan + list files and sizes, no upload
  python tools/publish_public_hf.py                    # create/reuse the repo, upload in ONE commit, verify anonymously
  python tools/publish_public_hf.py --verify-only      # anonymous download + import check of what is on the hub

Only an explicit allow-list is ever staged (nothing else may go public):

  README.md                                     model card, rendered from tools/model_card_v2.md
  adapter/adapter_NNNNNN.safetensors            from --checkpoint-dir (sha256 checked against ckpt_NNNNNN.json)
  adapter/ckpt_NNNNNN.json                      manifest: flow_infer.resolve_checkpoint_meta needs it next to the weights
  adapter/config.json                           the run config, so guide fill_mode resolves to "replicate"
  configs/strength_calibration_v2.json          configs/strength_calibration_v2.json if present, else the identity file
  configs/inference_v2.json                     (its `calibration` entry is pointed at the file above)
  code/fpcamera_src.zip                         src/fpcamera/** (no __pycache__), requirements*.txt, VERSION (git commit)

The flat ``adapter/`` folder is a valid v2 run layout (run dir = the weights' folder), so after a download
``generate(checkpoint=<dir>/adapter/adapter_NNNNNN.safetensors)`` finds config.json and resolves fill_mode=replicate.

Never staged: the latent cache / training data, eval videos, photos, keys.env, other checkpoints.  Every staged text
file (and every member of the zip) is scanned for tokens, molab URLs, keys.env, local user paths and e-mail addresses.

Checkpoint folder (``--checkpoint-dir``): default ``v2_full`` inside ``$FPCAMERA_CHECKPOINTS_DIR`` (else ``<repo root>/artifacts/checkpoints/v2``).
Token: HF_WRITE_TOKEN from the KEY=value file ``$FPCAMERA_KEYS_ENV`` (default ``<repo root>/keys.env``; same loader as
tools/molab_recover.py, see tools/local_config.py); it is never printed.  The publish refuses a
source tree with uncommitted changes (so VERSION always names the last commit that touched the bundle) unless --allow-dirty.
Re-running is idempotent: the zip is byte-deterministic, unchanged files are skipped by the hub, and an upload that changes
nothing creates no commit.  Stale files under adapter/, configs/ and code/ are removed in the same commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_config import KEYS_ENV_VAR, checkpoints_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_NAME = "v2_full"
DEFAULT_STEP = 415
EXPECTED_SHA_PREFIX = "83df1e3f"
REPO_NAME = "ie643-wan22-camera-adapter-v2"
WAN_REVISION = "b8fff7315c768468a5333511427288870b2e9635"
DELETE_PATTERNS = ["adapter/*", "configs/*", "code/*"]
CARD_TEMPLATE = ROOT / "tools" / "model_card_v2.md"
ZIP_NAME = "code/fpcamera_src.zip"
ZIP_DATE = (2026, 1, 1, 0, 0, 0)

# Things that must never appear in a public text file or in a zip member.
FORBIDDEN = {
    "hf token": re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    "molab url": re.compile(r"molab\.run|sb-[a-z0-9]{6,}\.sb\.", re.I),
    "keys.env": re.compile(r"keys\.env", re.I),
    "local user path": re.compile(r"[A-Za-z]:[\\/]+Users[\\/]|/Users/[A-Za-z]|/home/[a-z]+/Downloads|\bDownloads[\\/]"),
    "e-mail": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}"),
}
TEXT_SUFFIXES = {".md", ".json", ".py", ".txt", ".yaml", ".yml", ""}


def log(msg):
    """Print (flushed)."""
    print(msg, flush=True)


def sha256_file(path) -> str:
    """Streaming sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get_token():
    """HF_WRITE_TOKEN from the local keys file (tools/molab_recover.read_hf_token: ``$FPCAMERA_KEYS_ENV`` or
    ``<repo root>/keys.env``); env HF_TOKEN only as a fallback."""
    try:
        from molab_recover import read_hf_token

        return read_hf_token()
    except (OSError, SystemExit) as exc:
        token = os.environ.get("HF_TOKEN")
        if token:
            return token
        raise SystemExit(f"no HF write token: HF_TOKEN is unset and the keys file gave none ({exc}). Set HF_TOKEN, or put "
                         f"HF_WRITE_TOKEN=... in a KEY=value file and point {KEYS_ENV_VAR} at it") from None


def default_checkpoint_dir() -> Path:
    """Default ``--checkpoint-dir``: the ``v2_full`` run folder inside ``$FPCAMERA_CHECKPOINTS_DIR`` (see tools/local_config.py)."""
    return checkpoints_dir() / DEFAULT_RUN_NAME


def git(*args, cwd=ROOT) -> str:
    """Run a git command in the repository and return its stdout."""
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


BUNDLE_PATHS = ("src/fpcamera", "requirements.txt", "requirements-ui.txt")


def source_state():
    """(commit, dirty) for the files that go into the code bundle.

    ``commit`` is the last commit that touched the bundle's files (not HEAD), so unrelated commits do not change
    VERSION and re-publishing stays a no-op until the code actually changes.
    """
    commit = git("log", "-1", "--format=%H", "--", *BUNDLE_PATHS).strip() or git("rev-parse", "HEAD").strip()
    dirty = bool(git("status", "--porcelain", "--", *BUNDLE_PATHS).strip())
    return commit, dirty


# ---------------------------------------------------------------------------- pure helpers (unit tested)
def expected_paths(step: int):
    """The exact file set the public model repo must contain for checkpoint ``step`` (anything else is a leak)."""
    return {
        "README.md",
        f"adapter/adapter_{step:06d}.safetensors",
        f"adapter/ckpt_{step:06d}.json",
        "adapter/config.json",
        "configs/strength_calibration_v2.json",
        "configs/inference_v2.json",
        ZIP_NAME,
    }


def scan_text(name: str, text: str, extra_secrets=()):
    """Return a list of '<name>: <what>' findings (never the matched secret text itself)."""
    found = []
    for label, rx in FORBIDDEN.items():
        if rx.search(text):
            found.append(f"{name}: {label}")
    for secret in extra_secrets:
        if secret and secret in text:
            found.append(f"{name}: literal secret value")
    return found


def render_card(template: str, values: dict) -> str:
    """Fill the ``{{key}}`` placeholders of the model-card template."""
    out = template
    for key, value in values.items():
        out = out.replace("{{" + key + "}}", str(value))
    left = re.findall(r"\{\{[A-Z0-9_]+\}\}", out)
    if left:
        raise ValueError(f"unresolved model-card placeholders: {sorted(set(left))}")
    return out


def code_files(src_root: Path, repo_root: Path):
    """[(arcname, path)] sorted: src/fpcamera/** without __pycache__/*.pyc, requirements*.txt."""
    files = []
    for path in sorted((src_root / "fpcamera").rglob("*")):
        if path.is_dir() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        files.append((path.relative_to(src_root.parent).as_posix(), path))
    for name in ("requirements.txt", "requirements-ui.txt"):
        files.append((name, repo_root / name))
    return files


def build_zip(dest: Path, files, version_text: str):
    """Byte-deterministic zip (sorted, fixed timestamps and modes) so re-publishing the same commit changes nothing."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted([(arc, path.read_bytes()) for arc, path in files] + [("VERSION", version_text.encode())])
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for arc, data in entries:
            info = zipfile.ZipInfo(arc, ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
    return [arc for arc, _ in entries]


def read_manifest_sha(ckpt_dir: Path, step: int) -> str:
    """adapter sha256 recorded in the checkpoint manifest (after checking the manifest describes that adapter)."""
    manifest = json.loads((ckpt_dir / f"ckpt_{step:06d}.json").read_text(encoding="utf8"))
    if manifest.get("step") != step or manifest.get("adapter_file") != f"adapter_{step:06d}.safetensors":
        raise SystemExit(f"manifest ckpt_{step:06d}.json does not describe adapter_{step:06d}.safetensors")
    return manifest["adapter_sha256"]


def safetensors_param_count(path: Path) -> int:
    """Parameter count from the safetensors header only."""
    import math
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return sum(math.prod(v["shape"]) for k, v in header.items() if k != "__metadata__")


def stage(stage_dir: Path, *, ckpt_dir: Path, step: int, repo_id: str, repo_root: Path = ROOT,
          expect_sha_prefix: str | None = EXPECTED_SHA_PREFIX, secrets=(), allow_dirty=True):
    """Build the staging folder. Returns (files, commit, dirty): files = [(relpath, size, sha256)]."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    name = f"adapter_{step:06d}.safetensors"
    src_adapter = ckpt_dir / name
    digest = sha256_file(src_adapter)
    want = read_manifest_sha(ckpt_dir, step)
    if digest != want:
        raise SystemExit(f"{name}: sha256 {digest[:12]} does not match manifest {want[:12]}")
    if expect_sha_prefix and not digest.startswith(expect_sha_prefix):
        raise SystemExit(f"{name}: sha256 {digest[:12]} does not start with the expected {expect_sha_prefix}")
    for required in (f"ckpt_{step:06d}.json", "config.json"):
        if not (ckpt_dir / required).is_file():
            raise SystemExit(f"missing {ckpt_dir / required}")

    commit, dirty = source_state()
    if dirty and not allow_dirty:
        raise SystemExit("src/fpcamera or requirements*.txt have uncommitted changes: commit first (or --allow-dirty)")
    version = commit + ("-dirty" if dirty else "")

    (stage_dir / "adapter").mkdir(exist_ok=True)
    (stage_dir / "configs").mkdir(exist_ok=True)
    shutil.copyfile(src_adapter, stage_dir / "adapter" / name)
    for fname in (f"ckpt_{step:06d}.json", "config.json"):
        shutil.copyfile(ckpt_dir / fname, stage_dir / "adapter" / fname)

    calib = repo_root / "configs" / "strength_calibration_v2.json"
    calib_is_identity = not calib.is_file()
    if calib_is_identity:
        calib = repo_root / "configs" / "strength_calibration_v2.identity.json"
    shutil.copyfile(calib, stage_dir / "configs" / "strength_calibration_v2.json")

    inference = json.loads((repo_root / "configs" / "inference_v2.json").read_text(encoding="utf8"))
    inference["calibration"] = "configs/strength_calibration_v2.json"  # the path this repo actually has
    (stage_dir / "configs" / "inference_v2.json").write_text(json.dumps(inference, indent=2) + "\n", encoding="utf8")

    build_zip(stage_dir / ZIP_NAME, code_files(repo_root / "src", repo_root), version + "\n")

    card = render_card(
        CARD_TEMPLATE.read_text(encoding="utf8"),
        {
            "REPO_ID": repo_id,
            "STEP": step,
            "STEP6": f"{step:06d}",
            "ADAPTER_SHA256": digest,
            "ADAPTER_MB": f"{src_adapter.stat().st_size / 1e6:.1f}",
            "PARAMS": f"{safetensors_param_count(src_adapter) / 1e6:.2f}M",
            "WAN_REVISION": WAN_REVISION,
            "CODE_COMMIT": version,
            "CALIBRATION_NOTE": "identity map, not yet calibrated" if calib_is_identity else "fitted for this checkpoint",
        },
    )
    (stage_dir / "README.md").write_text(card, encoding="utf8")

    files = []
    for path in sorted(p for p in stage_dir.rglob("*") if p.is_file()):
        files.append((path.relative_to(stage_dir).as_posix(), path.stat().st_size, sha256_file(path)))
    problems = audit_stage(stage_dir, step, secrets)
    if problems:
        raise SystemExit("staging audit failed:\n  " + "\n  ".join(problems))
    return files, commit, dirty


def audit_stage(stage_dir: Path, step: int, secrets=()):
    """Exact allow-list + content scan of text files and zip members. Returns a list of problems."""
    problems = []
    staged = {p.relative_to(stage_dir).as_posix() for p in stage_dir.rglob("*") if p.is_file()}
    extra, missing = staged - expected_paths(step), expected_paths(step) - staged
    if extra:
        problems.append(f"files outside the allow-list: {sorted(extra)}")
    if missing:
        problems.append(f"missing files: {sorted(missing)}")
    for rel in sorted(staged):
        path = stage_dir / rel
        if rel.endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                for member in zf.namelist():
                    if "__pycache__" in member or member.endswith(".pyc"):
                        problems.append(f"{rel}: contains {member}")
                    if Path(member).suffix in TEXT_SUFFIXES:
                        problems += scan_text(f"{rel}!{member}", zf.read(member).decode("utf8", "replace"), secrets)
        elif path.suffix in TEXT_SUFFIXES - {""}:
            problems += scan_text(rel, path.read_text(encoding="utf8", errors="replace"), secrets)
    return problems


# ---------------------------------------------------------------------------- hub
def resolve_repo_id(api, explicit):
    """Explicit repo id, else ``<logged-in user>/<REPO_NAME>``."""
    if explicit:
        return explicit
    return f"{api.whoami()['name']}/{REPO_NAME}"


def ensure_public_repo(api, repo_id, make_public=False):
    """Create the public model repo, or reuse it. A pre-existing PRIVATE repo is only flipped with --make-public."""
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        info = api.model_info(repo_id)
    except RepositoryNotFoundError:
        api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
        log(f"created PUBLIC model repo {repo_id}")
        return
    if info.private:
        if not make_public:
            raise SystemExit(f"{repo_id} exists and is private; re-run with --make-public to publish it")
        api.update_repo_settings(repo_id, private=False, repo_type="model")
        log(f"{repo_id} was private -> now public")
    else:
        log(f"reusing existing public repo {repo_id}")


def upload(api, repo_id, stage_dir, step, commit_message):
    """Upload the staged folder to the model repo in one commit (``delete_patterns`` removes stale files), then check the remote
    file list against the allow-list and abort on any stray file; returns the commit info.
    """
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(stage_dir),
        commit_message=commit_message,
        delete_patterns=DELETE_PATTERNS,
    )
    log(f"upload_folder -> {getattr(info, 'commit_url', info)}")
    remote = set(api.list_repo_files(repo_id, repo_type="model"))
    stray = remote - expected_paths(step) - {".gitattributes"}
    if stray:
        raise SystemExit(f"unexpected files on the hub (not in the allow-list): {sorted(stray)}")
    return info


# ---------------------------------------------------------------------------- anonymous verification
_VERIFY_CODE = r"""
import hashlib, json, sys, zipfile
from pathlib import Path
d = Path(sys.argv[1]); step = int(sys.argv[2])
with zipfile.ZipFile(d / "code" / "fpcamera_src.zip") as zf:
    zf.extractall(d / "fpcamera_code")
sys.path.insert(0, str(d / "fpcamera_code" / "src"))
import fpcamera, fpcamera.flow_infer as fi
from fpcamera.flow_geometry import VideoContract
assert Path(fpcamera.__file__).resolve().is_relative_to((d / "fpcamera_code").resolve()), fpcamera.__file__
ckpt = d / "adapter" / f"adapter_{step:06d}.safetensors"
meta, source = fi.resolve_checkpoint_meta(ckpt)
assert source == "v2_manifest" and meta["run_config_found"] is True, (source, meta.get("run_config_found"))
mode = fi.resolve_guide_fill_mode(None, meta)
assert mode == ("replicate", "checkpoint", None), mode
h = hashlib.sha256(ckpt.read_bytes()).hexdigest()
assert h == meta["checkpoint_sha256"], "adapter digest differs from the manifest"
contract = VideoContract(**meta["identity"]["video_contract"])
path, request, *_ = fi.build_camera_request(
    contract, motion="pan_left", strength=0.5, calibration=d / "configs" / "strength_calibration_v2.json",
    max_amplitudes=dict(yaw_deg=8.0, pitch_deg=8.0, zoom_ratio=1.25), root=d, checkpoint_sha256=h)
assert path.shape == (contract.frames, 3, 3)
print(json.dumps({"sha256": h, "fill": mode[0], "meta_source": source, "steps": meta["completed_updates"],
                  "fpcamera": str(Path(fpcamera.__file__).parent.name), "version": (d / "fpcamera_code" / "VERSION").read_text().strip(),
                  "request_strength": request["strength_control"]["strength_u"]}))
"""


def verify_anonymous(repo_id, step, expect_sha_prefix=EXPECTED_SHA_PREFIX, keep_dir=None):
    """Download with NO token, check the digest, unzip the code and import it (in an isolated interpreter)."""
    from huggingface_hub import HfApi, hf_hub_download

    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    anon = HfApi(token=False)
    info = anon.model_info(repo_id, token=False)
    if info.private:
        raise SystemExit(f"{repo_id} is not public")
    log(f"anonymous model_info ok: {repo_id} private={info.private} sha={info.sha[:10]}")
    tmp = tempfile.TemporaryDirectory(prefix="ie643_verify_") if keep_dir is None else None
    d = Path(keep_dir or tmp.name)
    d.mkdir(parents=True, exist_ok=True)
    try:
        for rel in sorted(expected_paths(step)):
            got = hf_hub_download(repo_id, rel, repo_type="model", local_dir=str(d), token=False)
            log(f"  downloaded {rel} ({Path(got).stat().st_size} bytes)")
        ckpt = d / "adapter" / f"adapter_{step:06d}.safetensors"
        digest = sha256_file(ckpt)
        want = read_manifest_sha(d / "adapter", step)
        if digest != want or (expect_sha_prefix and not digest.startswith(expect_sha_prefix)):
            raise SystemExit(f"downloaded adapter sha256 {digest[:12]} != manifest {want[:12]} / expected {expect_sha_prefix}")
        log(f"  adapter sha256 {digest} == manifest")
        # The downloaded folder is untrusted data: run from an empty neutral cwd with -P (no cwd/script dir on sys.path;
        # -I would also hide the user site-packages), and let the child add only the unzipped code path itself.
        flags = ["-P"] if sys.version_info >= (3, 11) else []
        with tempfile.TemporaryDirectory(prefix="ie643_cwd_") as neutral:
            out = subprocess.run([sys.executable, *flags, "-W", "ignore", "-c", _VERIFY_CODE, str(d), str(step)],
                                 capture_output=True, text=True, cwd=neutral)
        if out.returncode != 0:
            raise SystemExit("import/resolve check failed:\n" + out.stderr[-2000:])
        result = json.loads(out.stdout.strip().splitlines()[-1])
        log(f"  import + resolve ok: {result}")
        return result
    finally:
        if tmp is not None:
            tmp.cleanup()


# ---------------------------------------------------------------------------- CLI
def main(argv=None):
    """CLI entry: stage, audit (file whitelist, hashes, parameter count, no tokens), upload, then anonymous download + import
    check; ``--dry-run`` / ``--verify-only`` available.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="stage and audit, list files and sizes, upload nothing")
    ap.add_argument("--verify-only", action="store_true", help="only run the anonymous download + import check")
    ap.add_argument("--no-verify", action="store_true", help="skip the anonymous check after uploading")
    ap.add_argument("--repo-id", help=f"default: <hf username>/{REPO_NAME}")
    ap.add_argument("--checkpoint-dir", type=Path, default=None,
                    help=f"folder with adapter_NNNNNN.safetensors, ckpt_NNNNNN.json, config.json "
                         f"(default: {DEFAULT_RUN_NAME} inside $FPCAMERA_CHECKPOINTS_DIR, else artifacts/checkpoints/v2)")
    ap.add_argument("--step", type=int, default=DEFAULT_STEP)
    ap.add_argument("--expect-sha-prefix", default=EXPECTED_SHA_PREFIX, help="'' disables the pinned-digest check")
    ap.add_argument("--allow-dirty", action="store_true", help="publish although src/fpcamera has uncommitted changes")
    ap.add_argument("--make-public", action="store_true", help="flip an existing PRIVATE repo of that name to public")
    ap.add_argument("--keep-stage", type=Path, help="stage here and keep the folder (default: a temp dir)")
    ap.add_argument("--message", default=None, help="commit message")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from huggingface_hub import HfApi

    token = None if args.verify_only else get_token()
    api = HfApi(token=token)
    repo_id = args.repo_id or (f"<username>/{REPO_NAME}" if args.verify_only else resolve_repo_id(api, None))
    if args.verify_only:
        if not args.repo_id:
            raise SystemExit("--verify-only needs --repo-id (it uses no token, so it cannot look the username up)")
        verify_anonymous(repo_id, args.step, args.expect_sha_prefix or None)
        return 0

    log(f"repo: {repo_id} (model, public)")
    stage_cm = tempfile.TemporaryDirectory(prefix="ie643_stage_") if args.keep_stage is None else None
    stage_dir = Path(args.keep_stage or stage_cm.name)
    try:
        files, commit, dirty = stage(
            stage_dir, ckpt_dir=args.checkpoint_dir or default_checkpoint_dir(), step=args.step, repo_id=repo_id,
            expect_sha_prefix=args.expect_sha_prefix or None, secrets=(token,),
            allow_dirty=args.dry_run or args.allow_dirty,
        )
        log(f"code bundle source commit: {commit}{' (DIRTY working tree)' if dirty else ''}")
        log(f"{'path':<48} {'bytes':>12}  sha256")
        for rel, size, digest in files:
            log(f"{rel:<48} {size:>12,}  {digest[:16]}")
        log(f"total: {len(files)} files, {sum(f[1] for f in files):,} bytes; audit (allow-list + secret scan) passed")
        with zipfile.ZipFile(stage_dir / ZIP_NAME) as zf:
            members = zf.namelist()
        log(f"{ZIP_NAME}: {len(members)} members ({sum(m.startswith('src/fpcamera/') for m in members)} under src/fpcamera/)")
        if args.dry_run:
            log("dry run: nothing uploaded")
            return 0
        ensure_public_repo(api, repo_id, args.make_public)
        message = args.message or f"v2 camera adapter step {args.step} (code {commit[:10]})"
        upload(api, repo_id, stage_dir, args.step, message)
    finally:
        if stage_cm is not None:
            stage_cm.cleanup()
    log(f"https://huggingface.co/{repo_id}")
    if not args.no_verify:
        verify_anonymous(repo_id, args.step, args.expect_sha_prefix or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
