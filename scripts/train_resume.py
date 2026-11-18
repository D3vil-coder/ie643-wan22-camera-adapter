"""Resume the frozen Submission 1 checkpoint using project-relative paths."""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/train_resume_300.json"
    )
    parser.add_argument(
        "--stop-after-updates",
        type=int,
        required=True,
        help="Explicitly choose the stop update; the archived resume state is at update 300.",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Verify the archived training-source identity without loading a model or writing state.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = PROJECT_ROOT
    config["project_root"] = str(root)
    for key in ("index", "output_dir", "resume"):
        config[key] = str((root / config[key]).resolve())
    if not 300 <= args.stop_after_updates <= int(config["updates"]):
        parser.error(
            "--stop-after-updates must be between 300 and the configured total updates"
        )
    config["stop_after_updates"] = args.stop_after_updates
    # Exact resume includes the source hashes in its identity. The readable
    # package copies have formatting changes, so resume through the byte-exact
    # executed source package preserved alongside them.
    snapshot = root / "src/fpcamera/provenance/executed_source"
    sidecar = json.loads((root / "artifacts/flow_poc_all/adapter_000300.json").read_text(encoding="utf-8"))
    expected = sidecar["identity"]["source_hashes"]
    for name, digest in expected.items():
        actual = hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f"Archived training source changed: {name}")
    if args.check_only:
        print(json.dumps({"status": "exact_training_sources_verified", "source_files": len(expected), "model_loaded": False}))
        return 0
    sys.path.insert(0, str(root / "src"))
    from fpcamera.provenance.executed_source.flow_train_delta import train

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
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
