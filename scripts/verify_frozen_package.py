"""Submission 1 (v1) helper: verify the frozen package against ``submission_manifest.json`` without unpickling anything.

Checks file sizes and sha256 values, the safetensors parameter count (reads only the JSON header) and the provenance
sources.  Prints PASS or exits with ``FAIL: <reason>``.

    python scripts/verify_frozen_package.py [--root .]
"""

from __future__ import annotations

import hashlib
import ast
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "submission_manifest.json"


def sha256(path: Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    """Abort with ``FAIL: <message>``."""
    raise SystemExit(f"FAIL: {message}")


def _is_package_file(path: Path) -> bool:
    return (
        path.is_file()
        and path != MANIFEST
        and path.suffix.lower() != ".pyc"
        and "__pycache__" not in {part.lower() for part in path.parts}
    )


def safetensors_parameter_count(path: Path) -> int:
    """Parameter count read from the safetensors JSON header only (no tensor payload is loaded)."""
    # Read only the format's JSON header; tensor payloads are neither loaded nor executed.
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            fail("checkpoint is too short to be a safetensors file")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length <= 0 or header_length > 100_000_000:
            fail("checkpoint has an invalid safetensors header length")
        header = json.loads(handle.read(header_length))
    return sum(
        math.prod(item["shape"])
        for name, item in header.items()
        if name != "__metadata__"
    )


def main() -> int:
    """Verify the file inventory, sizes and sha256 values, the checkpoint / resume digests against ``latest.json`` and the sidecar,
    the backbone audit, the adapter parameter count and the executed-source hashes.
    """
    if not MANIFEST.is_file():
        fail(
            "submission_manifest.json is missing; run scripts/freeze_submission_manifest.py after finalizing files"
        )
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    listed = data["files"]
    actual = {
        p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if _is_package_file(p)
    }
    if actual != set(listed):
        fail(
            f"file inventory differs (unlisted={sorted(actual - set(listed))[:5]}, missing={sorted(set(listed) - actual)[:5]})"
        )
    for relative, expected in listed.items():
        path = ROOT / Path(relative)
        if path.stat().st_size != expected["bytes"]:
            fail(f"size mismatch: {relative}")
        if sha256(path) != expected["sha256"]:
            fail(f"SHA-256 mismatch: {relative}")

    artifacts = ROOT / "artifacts/flow_poc_all"
    latest = json.loads((artifacts / "latest.json").read_text(encoding="utf-8"))
    sidecar = json.loads(
        (artifacts / "adapter_000300.json").read_text(encoding="utf-8")
    )
    checkpoint = artifacts / latest["checkpoint"]
    resume = artifacts / "resume.pt"
    for label, path in (("checkpoint", checkpoint), ("resume", resume)):
        digest = sha256(path)
        if latest[f"{label}_sha256"] != digest or sidecar[f"{label}_sha256"] != digest:
            fail(f"{label} digest does not match latest and sidecar metadata")
    if not latest.get("backbone_unchanged") or latest.get(
        "backbone_sha256_before"
    ) != latest.get("backbone_sha256_after"):
        fail("backbone_unchanged provenance check failed")
    if latest["adapter_parameters"] != 14261248:
        fail("unexpected adapter parameter count")
    if safetensors_parameter_count(checkpoint) != latest["adapter_parameters"]:
        fail(
            "safetensors tensor shapes do not match the recorded adapter parameter count"
        )
    if latest["identity"]["source_hashes"] != data["frozen_training"]["source_hashes"]:
        fail("frozen source identity differs from inventory summary")
    for filename, digest in latest["identity"]["source_hashes"].items():
        source = ROOT / "src/fpcamera/provenance/executed_source" / filename
        if not source.is_file() or sha256(source) != digest:
            fail(f"preserved executed training source identity mismatch: {filename}")
    preserved = json.loads(
        (ROOT / "src/fpcamera/provenance/executed_source/source_hashes.json").read_text(
            encoding="utf-8"
        )
    )
    for filename, expected in preserved.items():
        source = ROOT / "src/fpcamera/provenance/executed_source" / filename
        current = ROOT / "src/fpcamera" / filename
        if sha256(source) != expected:
            fail(f"preserved executed source inventory mismatch: {filename}")
        if ast.dump(
            ast.parse(source.read_text(encoding="utf-8")), include_attributes=False
        ) != ast.dump(
            ast.parse(current.read_text(encoding="utf-8")), include_attributes=False
        ):
            fail(
                f"packaged source code differs semantically from the executed source: {filename}"
            )

    index_path = ROOT / data["dataset"]["index"]
    if (
        sha256(index_path) != data["dataset"]["index_sha256"]
        or sha256(index_path) != latest["identity"]["index_sha256"]
    ):
        fail("dataset index digest mismatch")
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
    actual_stats = {
        "rows": len(records),
        "sources": len(sources),
        "splits": splits,
        "modes": modes,
    }
    for key, value in actual_stats.items():
        if data["dataset"][key] != value:
            fail(f"dataset {key} mismatch")
    print(
        f"PASS: {len(listed)} package files; adapter={latest['adapter_parameters']:,} parameters; "
        f"checkpoint/resume/source/index digests verified; dataset={actual_stats}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
