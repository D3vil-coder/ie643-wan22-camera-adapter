#!/usr/bin/env python
"""Build (or dry-run) the v2 synthetic camera-motion latent cache.

Examples (from the repo root; see docs/v2/DATA_V2.md):

    PYTHONPATH=src python scripts/build_cache_v2.py --project-root . --dry-run --limit 6 \
        --preview-dir artifacts/preview_v2
    PYTHONPATH=src python scripts/build_cache_v2.py --project-root . --limit 24   # smoke
    PYTHONPATH=src python scripts/build_cache_v2.py --project-root .               # full, resumable
    PYTHONPATH=src python scripts/build_cache_v2.py --project-root . --shard 0/2
    PYTHONPATH=src python scripts/build_cache_v2.py --project-root . --merge-index
    PYTHONPATH=src python scripts/build_cache_v2.py --derive-overscan

--dry-run renders only (no VAE, no torch import) and writes preview strips.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from fpcamera import flow_cache_v2 as fc  # noqa: E402
from fpcamera.flow_geometry import (  # noqa: E402
    V2_MOTIONS,
    VideoContract,
    camera_path_v2,
    required_overscan,
    revealed_fraction,
)


def parse_shard(text):
    """Parse ``--shard i/n`` into ``(i, n)`` with ``0 <= i < n`` (argparse type)."""
    i, n = (int(x) for x in text.split("/"))
    if not 0 <= i < n:
        raise argparse.ArgumentTypeError("--shard must be i/n with 0 <= i < n")
    return i, n


def synthetic_source(width, height, seed):
    """Textured test image (gradients, stripes, shapes, noise) for geometry previews."""
    import cv2

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    img = np.stack(
        [
            127 + 120 * np.sin(xx / 37.0 + seed),
            127 + 120 * np.sin(yy / 29.0 + 2 * seed),
            127 + 120 * np.sin((xx + yy) / 53.0),
        ],
        -1,
    )
    img = np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))  # OpenCV 5 drawing needs CV_8U
    for _ in range(30):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        p = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        if rng.random() < 0.5:
            cv2.circle(img, p, int(rng.integers(10, 60)), c, -1)
        else:
            q = (p[0] + int(rng.integers(20, 120)), p[1] + int(rng.integers(20, 120)))
            cv2.rectangle(img, p, q, c, -1)
    # coordinate ticks so motion direction is obvious
    for x in range(0, width, 80):
        cv2.putText(img, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (255, 0, 0), 2)  # source border
    return np.clip(img, 0, 255).astype(np.uint8)


def make_synthetic_manifest(out_dir, count):
    """Write ``count`` synthetic images of mixed sizes plus a manifest, for ``--synthetic`` dry runs that exercise planning
    and overscan without the real photographs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = [(640, 480), (640, 427), (800, 450), (500, 640), (1000, 400)]
    records = []
    for i in range(count):
        w, h = sizes[i % len(sizes)]
        path = out_dir / f"synthetic_{i:02d}.png"
        Image.fromarray(synthetic_source(w, h, i + 1)).save(path)
        import hashlib

        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(
            {
                "source_id": f"synthetic_{i:02d}",
                "source_image_id": i,
                "target_category": "synthetic",
                "original_image_sha256": sha,
                "source_image": str(path.resolve()),
                "training_caption": "synthetic",
                "split": "train" if i % 6 else "development",
            }
        )
    manifest = out_dir / "sources_synthetic.json"
    manifest.write_text(json.dumps({"schema_version": 2, "records": records}, indent=1))
    return manifest


def derive_overscan(cfg):
    """Dry-run helper: print the overscan each unit-strength path needs for a range of source aspect ratios, the universal
    requirement and the revealed fraction at the last frame.
    """
    contract = fc.contract_of(cfg)
    paths = {
        f"{f}_{d}": camera_path_v2(f, d, 1.0, contract, **fc.camera_kwargs(cfg))[0]
        for f, d in V2_MOTIONS
    }
    print(f"{'aspect':>7} {'max':>7}  " + " ".join(f"{k:>10}" for k in paths))
    worst = 0.0
    for aspect in (0.5, 0.667, 0.75, 1.0, 1.333, 1.5, 1.667, 1.818, 2.0, 2.4, 3.0):
        size = (int(round(1000 * aspect)), 1000)
        per = {k: required_overscan(size, contract, [h]) for k, h in paths.items()}
        worst = max(worst, max(per.values()))
        print(f"{aspect:7.3f} {max(per.values()):7.4f}  " + " ".join(f"{v:10.4f}" for v in per.values()))
    print(f"universal requirement: {worst:.4f}")
    for k, h in paths.items():
        print(f"revealed fraction at last frame, u=1, {k}: {revealed_fraction(h[-1], contract, stride=1):.4f}")


def main():
    """CLI entry: build or dry-run the v2 cache via ``fpcamera.flow_cache_v2.build`` (sharding, deadline, previews,
    index rebuild).
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--config", type=Path, default=fc.DEFAULT_CONFIG)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--shard", type=parse_shard, default=(0, 1), help="i/n over sorted source ids")
    p.add_argument("--limit", type=int, help="max new clips (build) or rendered clips (dry-run)")
    p.add_argument("--dry-run", action="store_true", help="render + coverage only, no VAE/torch")
    p.add_argument("--preview-dir", type=Path)
    p.add_argument("--preview-count", type=int, default=6)
    p.add_argument("--synthetic", type=int, metavar="N", help="use N generated test images (dry-run/tests)")
    p.add_argument("--no-verify", action="store_true", help="skip sha256 re-verification on resume")
    p.add_argument("--force-disk", action="store_true")
    p.add_argument("--force-config", action="store_true")
    p.add_argument("--allow-download", action="store_true", help="allow HF downloads of VAE/T5")
    p.add_argument("--no-lease", action="store_true", help="do not take artifacts/gpu.lock (multi-shard on 1 GPU)")
    p.add_argument("--merge-index", action="store_true", help="rebuild index.jsonl from clip files and exit")
    p.add_argument("--derive-overscan", action="store_true", help="print required overscan table and exit")
    p.add_argument("--project", action="store_true", help="print projected clips/size/time and exit")
    p.add_argument("--s-per-clip", type=float, default=45.0, help="assumed s/clip for --project")
    from fpcamera.session_deadline import add_deadline_args, deadline_from_args

    add_deadline_args(p)
    a = p.parse_args()
    deadline = deadline_from_args(a)

    cfg = fc.load_config(a.config)
    if a.derive_overscan:
        derive_overscan(cfg)
        return
    root = a.project_root.resolve()
    cache_root = a.cache_dir or root / cfg["cache"]["dir"]
    if a.merge_index:
        rows = fc.rebuild_index(cache_root, verify=not a.no_verify)
        print(json.dumps({"event": "index_rebuilt", "rows": len(rows), "index": str(Path(cache_root) / "index.jsonl")}))
        return
    manifest = a.manifest
    if a.synthetic:
        tmp = Path(a.preview_dir or tempfile.mkdtemp(prefix="fpv2_")) / "synthetic_sources"
        manifest = make_synthetic_manifest(tmp, a.synthetic)
    if a.project:
        records = fc.load_sources(manifest or root / cfg["sources"]["manifest"])
        n = len(records) * (cfg["sampling"]["static_clips_per_source"] + cfg["sampling"]["moving_clips_per_source"])
        per = cfg["cache"]["bytes_per_clip_estimate"]
        print(json.dumps({"sources": len(records), "clips": n, "gb": round(n * per / 1e9, 1),
                          "hours_at_s_per_clip": [a.s_per_clip, round(n * a.s_per_clip / 3600, 1)]}))
        return
    if a.dry_run and not a.preview_dir:
        a.preview_dir = Path(cache_root) / "preview_dry_run"
    out = fc.build(
        root, cfg,
        cache_root=cache_root, manifest=manifest, shard=a.shard, limit=a.limit,
        dry_run=a.dry_run, preview_dir=a.preview_dir, preview_count=a.preview_count,
        verify=not a.no_verify, force_disk=a.force_disk, force_config=a.force_config,
        allow_download=a.allow_download, use_lease=not a.no_lease, deadline=deadline,
    )
    print(json.dumps({k: v for k, v in out.items() if k != "stats"}, default=str))


if __name__ == "__main__":
    main()
