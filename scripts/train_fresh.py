"""Submission 1 (v1) helper: start a new 300-update training run from zero (no checkpoint reuse).

Verifies that the byte-exact executed training sources (``src/fpcamera/provenance/executed_source``) match the hashes
recorded in the frozen step-300 checkpoint's sidecar, that the v1 cache index is unchanged and that the config has no
resume / warm-start state, then trains with ``flow_train_delta`` under ``artifacts/<run-name>``.  v2 training uses
``scripts/train_v2.py`` instead.

    python scripts/train_fresh.py --run-name fresh1 --stop-after-updates 300 [--check-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/train_fresh_300.json"
EXPECTED_SOURCE_NAMES = {
    "flow_train_delta.py",
    "flow_control.py",
    "flow_cache.py",
    "flow_geometry.py",
    "wan.py",
    "control_lora.py",
}


def sha256(path: Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sources(root: Path) -> dict:
    """Check the executed-source snapshot and the cache index against the frozen checkpoint's sidecar; returns the sidecar."""
    sidecar_path = root / "artifacts/flow_poc_all/adapter_000300.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected = sidecar["identity"]["source_hashes"]
    if set(expected) != EXPECTED_SOURCE_NAMES:
        raise RuntimeError(
            "Checkpoint source-hash set does not match the frozen training recipe"
        )
    snapshot = root / "src/fpcamera/provenance/executed_source"
    for name, digest in expected.items():
        path = snapshot / name
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(
                f"Exact executed training source differs from the checkpoint identity: {name}"
            )
    index = root / "data/flow_poc/cache_1280x704_f121_v1/index.jsonl"
    if sha256(index) != sidecar["identity"]["index_sha256"]:
        raise RuntimeError("Cache index does not match the frozen training recipe")
    return sidecar


def main() -> int:
    """CLI entry (see module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name",
        required=True,
        help="New output name (letters, digits, dot, underscore or hyphen).",
    )
    parser.add_argument(
        "--stop-after-updates",
        type=int,
        required=True,
        help="Explicitly set the first run's stop update; the supplied profile is fixed at 300 of 1000.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify source hashes, cache index, and config without CUDA, model loading, or writes.",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.run_name):
        parser.error(
            "--run-name must be 1-64 letters/digits/dots/underscores/hyphens and start with a letter or digit"
        )
    if args.stop_after_updates != 300:
        parser.error(
            "The fresh-run profile requires --stop-after-updates 300; the configured schedule remains 1000 updates."
        )
    root = PROJECT_ROOT.resolve()
    sidecar = verify_sources(root)
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "resume" in config or config.get("initial_adapter"):
        raise RuntimeError(
            "Fresh-run config must not contain resume or warm-start state."
        )
    if int(config.get("updates", 0)) != 1000:
        raise RuntimeError("Fresh-run profile must preserve the 1000-update schedule.")
    if args.check_only:
        print(
            json.dumps(
                {
                    "status": "fresh_source_and_index_verified",
                    "source_files": len(sidecar["identity"]["source_hashes"]),
                    "index_sha256": sidecar["identity"]["index_sha256"],
                    "model_loaded": False,
                    "output_created": False,
                },
                sort_keys=True,
            )
        )
        return 0

    try:
        import torch
    except Exception as exc:
        raise SystemExit(
            "Install the verified CUDA runtime on the GPU host before starting a fresh run."
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device is available. Run this launcher on the designated GPU host."
        )

    run_root = root / "artifacts/fresh_runs" / args.run_name
    output = run_root / "artifacts/flow_poc_all"
    if run_root.exists() or output.exists():
        parser.error(
            f"Fresh-run path already exists; refusing to reuse or overwrite it: {run_root}"
        )
    run_root.parent.mkdir(parents=True, exist_ok=True)
    # Validate that the target host permits the data symlink before reserving the run name.
    try:
        with tempfile.TemporaryDirectory(
            prefix=".symlink_check_", dir=run_root.parent
        ) as temp:
            os.symlink(
                root / "data/flow_poc",
                Path(temp) / "flow_poc",
                target_is_directory=True,
            )
    except OSError as exc:
        raise SystemExit(
            "The GPU host must support local directory symlinks for the data cache; no cache copy is made."
        ) from exc

    run_root.mkdir()
    (run_root / "data").mkdir()
    os.symlink(
        root / "data/flow_poc", run_root / "data/flow_poc", target_is_directory=True
    )
    config["project_root"] = str(run_root)
    index_path = root / config["index"]
    config["index"] = str(index_path.resolve())
    config["output_dir"] = str(output)
    config["stop_after_updates"] = 300
    config.pop("resume", None)
    config.pop("initial_adapter", None)
    (run_root / "fresh_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    sys.path.insert(0, str(root / "src"))
    from fpcamera.provenance.executed_source.flow_train_delta import train

    print(
        json.dumps(
            {
                "run_name": args.run_name,
                "initial_step": 0,
                "stop_after_updates": 300,
                "configured_total_updates": config["updates"],
                "source_hashes": len(sidecar["identity"]["source_hashes"]),
                "output_dir": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    ledger = train(
        config, progress=lambda record: print(json.dumps(record), flush=True)
    )
    print(
        json.dumps(
            {
                "status": ledger.get("status"),
                "completed_updates": ledger.get("completed_updates"),
                "backbone_unchanged": ledger.get("backbone_unchanged"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
