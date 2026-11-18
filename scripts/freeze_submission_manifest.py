"""Create the package inventory after all package files have been finalized."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "submission_manifest.json"


def _is_package_file(path: Path) -> bool:
    return (
        path.is_file()
        and path != MANIFEST
        and path.suffix.lower() != ".pyc"
        and "__pycache__" not in {part.lower() for part in path.parts}
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if MANIFEST.exists():
        raise SystemExit(f"Refusing to overwrite {MANIFEST}")
    files = {}
    for path in sorted(p for p in ROOT.rglob("*") if _is_package_file(p)):
        relative = path.relative_to(ROOT).as_posix()
        if any(
            part.lower()
            in {".cache", "credentials", "credential", "tokens", "accounts"}
            for part in path.parts
        ):
            raise SystemExit(
                f"Credential/cache path is forbidden in package: {relative}"
            )
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

    artifact_dir = ROOT / "artifacts/flow_poc_all"
    latest = json.loads((artifact_dir / "latest.json").read_text(encoding="utf-8"))
    sidecar = json.loads(
        (artifact_dir / "adapter_000300.json").read_text(encoding="utf-8")
    )
    index_path = ROOT / "data/flow_poc/cache_1280x704_f121_v1/index.jsonl"
    index_digest = sha256(index_path)
    if latest["identity"]["index_sha256"] != index_digest:
        raise SystemExit("Cache index SHA-256 differs from checkpoint identity")
    if latest.get("checkpoint_sha256") != sha256(artifact_dir / latest["checkpoint"]):
        raise SystemExit("Checkpoint SHA-256 differs from latest metadata")
    if latest.get("resume_sha256") != sha256(artifact_dir / "resume.pt"):
        raise SystemExit("Resume SHA-256 differs from latest metadata")
    if latest.get("checkpoint_sha256") != sidecar.get("checkpoint_sha256"):
        raise SystemExit("Checkpoint sidecar digest mismatch")
    if latest.get("resume_sha256") != sidecar.get("resume_sha256"):
        raise SystemExit("Resume sidecar digest mismatch")

    records = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    splits: dict[str, int] = {}
    modes: dict[str, int] = {}
    sources = set()
    for row in records:
        splits[row["split"]] = splits.get(row["split"], 0) + 1
        modes[row["mode"]] = modes.get(row["mode"], 0) + 1
        sources.add(row["source_id"])
    manifest = {
        "format": "submission-1-project-inventory-v1",
        "description": "SHA-256 inventory of this self-contained project package; excludes this manifest itself.",
        "files": files,
        "frozen_training": {
            "checkpoint": f"artifacts/flow_poc_all/{latest['checkpoint']}",
            "checkpoint_sha256": latest["checkpoint_sha256"],
            "resume_sha256": latest["resume_sha256"],
            "completed_updates": latest["completed_updates"],
            "adapter_parameters": latest["adapter_parameters"],
            "backbone_unchanged": latest["backbone_unchanged"],
            "model": latest["identity"]["model"],
            "revision": latest["identity"]["revision"],
            "source_hashes": latest["identity"]["source_hashes"],
        },
        "dataset": {
            "index": "data/flow_poc/cache_1280x704_f121_v1/index.jsonl",
            "index_sha256": index_digest,
            "rows": len(records),
            "sources": len(sources),
            "splits": splits,
            "modes": modes,
        },
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote submission_manifest.json with {len(files)} files; checkpoint and resume digests verified."
    )


if __name__ == "__main__":
    main()
