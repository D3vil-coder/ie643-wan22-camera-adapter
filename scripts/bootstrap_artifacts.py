"""Submission 1 (v1) helper: restore the full local package from a supplied bundle on the GPU host.

Pipeline position: one-time setup of the frozen v1 package (large files that are not in git: checkpoint, cache,
media; see ``ASSETS.md``).  Verifies the bundle with ``verify_frozen_package.py`` (hashes from
``submission_manifest.json``), copies the files into the target checkout, re-verifies the copy and checks that the
pinned Wan base revision is in the local Hugging Face cache (``--download-base`` fetches it).  No token is copied.

    python scripts/bootstrap_artifacts.py --full-package /path/to/full_package [--target-project .] [--download-base]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
REVISION = "b8fff7315c768468a5333511427288870b2e9635"
COPY_PREFIXES = ("artifacts/flow_poc_all/", "data/flow_poc/")
FORBIDDEN_PARTS = {".cache", "credentials", "credential", "tokens", "accounts"}


def sha256(path: Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_gpu() -> None:
    """Exit with install instructions unless torch and a CUDA device are available (this setup is GPU-host only)."""
    try:
        import torch
    except Exception as exc:
        raise SystemExit(
            "Install the verified CUDA runtime on the GPU host before bootstrapping; "
            "this setup is not intended for a laptop."
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device is available. Run this bootstrap on the designated GPU host, not a laptop."
        )


def verify_base(download: bool) -> Path:
    """Return the local snapshot path of the pinned Wan base revision; with ``download`` fetch it from the Hub,
    otherwise exit with instructions if it is missing from the local cache.
    """
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=MODEL,
            revision=REVISION,
            local_files_only=not download,
        )
        return Path(path)
    except Exception as exc:
        if download:
            raise SystemExit(
                "Could not retrieve the pinned Wan revision. Check model access and the approved "
                "Hugging Face authentication on this GPU host; no token was copied by this script."
            ) from None
        raise SystemExit(
            "The pinned Wan revision is not present in the local Hugging Face cache. "
            "Run again on the GPU host with --download-base if model access is available."
        ) from None


def main() -> int:
    """Verify the bundle, copy its files into the target project, re-verify the copy and check the pinned base model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-package",
        type=Path,
        required=True,
        help="Path to the full offline Submission 1 project directory on this GPU host.",
    )
    parser.add_argument(
        "--download-base",
        action="store_true",
        help="Explicitly retrieve the pinned Wan base revision on this GPU host.",
    )
    parser.add_argument(
        "--target-project",
        type=Path,
        help="Project checkout to restore into; defaults to the project beside this script.",
    )
    args = parser.parse_args()
    require_gpu()

    source_root = args.full_package.expanduser().resolve()
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "scripts" and (script_dir.parent / "configs").is_dir():
        default_target = script_dir.parent
    else:
        default_target = script_dir / "project"
    target_root = (
        args.target_project.expanduser().resolve()
        if args.target_project is not None
        else default_target
    )
    source_manifest_path = source_root / "submission_manifest.json"
    if not source_manifest_path.is_file():
        raise SystemExit("The supplied full package has no submission_manifest.json.")
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    verifier = source_root / "scripts/verify_frozen_package.py"
    source_check = subprocess.run(
        [sys.executable, str(verifier)], cwd=source_root, capture_output=True, text=True
    )
    if source_check.returncode:
        raise SystemExit("The supplied full package failed integrity verification.")
    base_path = verify_base(args.download_base)

    wanted = [
        (relative, details)
        for relative, details in manifest["files"].items()
        if relative.startswith(COPY_PREFIXES)
    ]
    if not wanted:
        raise SystemExit(
            "The supplied manifest lists no adapter artifacts or data cache files."
        )
    copied = 0
    for relative, details in sorted(wanted):
        path_parts = Path(relative).parts
        if any(part.lower() in FORBIDDEN_PARTS for part in path_parts):
            raise SystemExit(
                f"Forbidden credential/cache path in source manifest: {relative}"
            )
        source = source_root / Path(relative)
        destination = target_root / Path(relative)
        if not source.is_file() or source.stat().st_size != details["bytes"]:
            raise SystemExit(f"Source artifact missing or size mismatch: {relative}")
        if sha256(source) != details["sha256"]:
            raise SystemExit(f"Source artifact hash mismatch: {relative}")
        if destination.exists():
            if (
                destination.stat().st_size != details["bytes"]
                or sha256(destination) != details["sha256"]
            ):
                raise SystemExit(
                    f"Target already contains a different file; refusing overwrite: {relative}"
                )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        shutil.copy2(source, temporary)
        if sha256(temporary) != details["sha256"]:
            temporary.unlink(missing_ok=True)
            raise SystemExit(f"Copied artifact failed SHA-256 verification: {relative}")
        temporary.replace(destination)
        copied += 1

    target_manifest = target_root / "submission_manifest.json"
    source_readme = source_root / "README.md"
    target_readme = target_root / "README.md"
    if (
        source_root != target_root
        and source_readme.is_file()
        and target_readme.is_file()
    ):
        source_bytes = source_readme.read_bytes()
        target_bytes = target_readme.read_bytes()
        if target_bytes != source_bytes:
            if b"SOURCE-ONLY CHECKOUT NOTICE" not in target_bytes:
                raise SystemExit(
                    "Target README differs from the full package and has no source-only bootstrap notice; refusing overwrite."
                )
            shutil.copy2(source_readme, target_readme)
    if target_manifest.exists():
        if target_manifest.read_bytes() != source_manifest_path.read_bytes():
            raise SystemExit(
                "Target has a different submission manifest; refusing overwrite."
            )
    else:
        shutil.copy2(source_manifest_path, target_manifest)

    target_verifier = target_root / "scripts/verify_frozen_package.py"
    target_check = subprocess.run(
        [sys.executable, str(target_verifier)], cwd=target_root
    )
    if target_check.returncode:
        raise SystemExit(
            "The restored full package failed its frozen inventory verification."
        )
    print(f"Restored {copied} files into {target_root}")
    print(f"Pinned model cache is available at {base_path}")
    print("No authentication files were copied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
