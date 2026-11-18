"""Measure newly-revealed-border quality of generated camera videos.

For every run folder (``video.mp4`` + ``request.json`` [+ ``run.json``]) the
revealed region R is recomputed from the recorded first->current homographies
(``request.json["homographies_first_to_current"]``; if absent they are
regenerated from ``run.json`` mode/magnitude with ``flow_geometry.camera_path``).
Metrics at frames 40/80/120 and averaged over all frames with R > 1% are written
to ``<run>/border_quality.json``. With several runs an aggregated CSV and
markdown table are written as well.

    python scripts/measure_border_quality.py artifacts/a artifacts/b \
        --out-csv border_table.csv --out-md border_table.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpcamera import border_metrics as bm  # noqa: E402

DEFAULT_FRAMES = (40, 80, 120)
TABLE_METRICS = ("artifact_score_local", "artifact_score_v2", "artifact_score_v1", "gray_band_score", "colorfulness_ratio", "hf_ratio", "seam_score")


def clean(value):
    """Recursively replace NaN/inf by None so the result is strict JSON."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def load_homographies(run_dir):
    """Return (homographies [T,3,3], contract dict, run record dict, source)."""
    run_dir = Path(run_dir)
    request = json.loads((run_dir / "request.json").read_text(encoding="utf8"))
    run = {}
    if (run_dir / "run.json").is_file():
        run = json.loads((run_dir / "run.json").read_text(encoding="utf8"))
    contract = request.get("contract") or run.get("contract") or {}
    if "homographies_first_to_current" in request:
        h = np.asarray(request["homographies_first_to_current"], dtype=np.float64)
        return h, contract, run, "request.json"
    from fpcamera.flow_geometry import VideoContract, camera_path

    fields = {
        k: contract[k] for k in ("width", "height", "frames", "fps") if k in contract
    }
    h, _ = camera_path(
        run.get("mode", request.get("mode")),
        VideoContract(**fields),
        magnitude=run.get("magnitude", request.get("magnitude", 1.0)),
    )
    return h, contract, run, "regenerated_from_mode"


def read_video_rgb(path):
    """Decode a video as a list of RGB uint8 frames."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames


def measure_run(
    run_dir, frames=DEFAULT_FRAMES, debug_dir=None, write=True, mean_stride=1
):
    """Compute and (optionally) write ``border_quality.json`` for one run folder."""
    run_dir = Path(run_dir)
    h, contract, run, source = load_homographies(run_dir)
    video = read_video_rgb(run_dir / "video.mp4")
    height, width = video[0].shape[:2]
    n = min(len(video), len(h))
    if len(video) != len(h):
        print(f"warning: {run_dir}: {len(video)} video frames vs {len(h)} homographies")
    wanted = sorted({t for t in range(0, n, max(1, mean_stride))} | {i for i in frames if i < n})
    per = {}
    for t in wanted:
        mask = bm.revealed_mask_pixels(h, width, height, frame_indices=[t])[0]
        per[t] = bm.frame_metrics(video[t], mask)
    per_frame = [per[t] for t in range(0, n, max(1, mean_stride))]
    result = {
        "run_dir": str(run_dir),
        "homography_source": source,
        "mode": run.get("mode"),
        "magnitude": run.get("magnitude"),
        "guide_fill_mode": run.get("guide_fill_mode", "zeros"),
        "camera_min_sigma": run.get("camera_min_sigma"),
        "adapter_enabled": run.get("adapter_enabled"),
        "frame_size": [width, height],
        "n_frames": n,
        "mean_stride": mean_stride,
        "at_frame": {str(i): per[i] for i in frames if i < n},
        "mean_over_frames_with_R_gt_1pct": bm.mean_metrics(per_frame, 0.01),
        "metric_notes": {
            "artifact_score_v2": "|ln clip(colorfulness_ratio,0.2,5)| + |ln clip(hf_ratio,0.2,5)| + glow_term (symmetric); per-frame, then averaged",
            "artifact_score_local": "same as v2 but the colorfulness/HF ratios and glow compare R with a LOCAL ring (valid pixels 8-48 px from R, same frame) instead of the whole valid region",
            "artifact_score": "v1 = max(0,1-colorfulness_ratio)+max(0,1-hf_ratio) (ratios clipped to [0,1.5]) + glow_term; per-frame, then averaged",
            "glow_term": "|mean luma in R - mean luma of valid pixels within 24px of the boundary| / 255",
            "gray_band_score": "gray fraction in R minus gray fraction in valid",
            "colorfulness_ratio": "Hasler-Susstrunk R / valid",
            "hf_ratio": "Laplacian variance R / valid (regions eroded 3px)",
            "seam_score": "gradient in 3px band at R/valid boundary / gradient 3-12px inside valid",
        },
    }
    debug_dir = Path(debug_dir) if debug_dir else None
    if debug_dir is not None and (debug_dir / "debug_latents.safetensors").is_file():
        result["latent_stats"] = measure_latents(debug_dir, h, width, height)
    if write:
        (run_dir / "border_quality.json").write_text(
            json.dumps(clean(result), indent=2), encoding="utf8"
        )
    return result


def measure_latents(debug_dir, homographies, width, height):
    """Latent R-vs-valid statistics of every latent tensor in debug_latents.safetensors."""
    from safetensors.numpy import load_file

    tensors = load_file(str(Path(debug_dir) / "debug_latents.safetensors"))
    mask = bm.revealed_mask_latent(homographies, width, height)
    out = {"revealed_fraction_latent": float(mask[1:].mean())}
    for key, value in tensors.items():
        if value.ndim == 4 and value.shape[0] == 48 and value.shape[1:] == mask.shape:
            out[key] = bm.latent_stats(value, mask)
    return out


def _g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def aggregate_rows(results, names=None):
    """One flat row per measured run (mode, fill, min_sigma, metrics at frame 120 and mean)."""
    rows = []
    for i, r in enumerate(results):
        row = {
            "run": (names[i] if names else Path(r["run_dir"]).name),
            "mode": r.get("mode"),
            "fill": r.get("guide_fill_mode"),
            "min_sigma": r.get("camera_min_sigma"),
            "adapter": r.get("adapter_enabled"),
            "R_frac@120": _g(r, "at_frame", "120", "revealed_fraction"),
        }
        for label, src in (
            ("f40", ("at_frame", "40")),
            ("f80", ("at_frame", "80")),
            ("f120", ("at_frame", "120")),
            ("mean", ("mean_over_frames_with_R_gt_1pct",)),
        ):
            for metric in TABLE_METRICS:
                row[f"{metric}@{label}"] = _g(r, *src, metric)
        rows.append(row)
    return rows


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_csv(rows, path):
    """Write the aggregate rows as CSV."""
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})


def markdown_table(rows, labels=("f120", "mean")):
    """Render the aggregate rows as a markdown table for the chosen frame labels."""
    head = ["run", "mode", "fill", "min_sigma", "R@120"]
    for label in labels:
        head += [f"{m}@{label}" for m in TABLE_METRICS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["run"], r["mode"], r["fill"], r["min_sigma"], r["R_frac@120"]]
        for label in labels:
            cells += [r[f"{m}@{label}"] for m in TABLE_METRICS]
        lines.append("| " + " | ".join(_fmt(c) for c in cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    """CLI entry: measure every run folder (writes ``border_quality.json`` in each) and optionally the CSV/markdown tables."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="run folders")
    ap.add_argument("--frames", type=int, nargs="+", default=list(DEFAULT_FRAMES))
    ap.add_argument("--debug-subdir", default=None,
                    help="subfolder of each run (or absolute path) with debug_latents.safetensors")
    ap.add_argument("--mean-stride", type=int, default=1,
                    help="use every k-th frame for the all-frames mean (default 1 = all)")
    ap.add_argument("--force", action="store_true", help="recompute existing border_quality.json")
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args(argv)
    results = []
    for run in args.runs:
        cached = run / "border_quality.json"
        if cached.is_file() and not args.force:
            results.append(json.loads(cached.read_text(encoding="utf8")))
            print(f"[cached] {run}")
            continue
        debug = None
        if args.debug_subdir:
            debug = Path(args.debug_subdir)
            debug = debug if debug.is_absolute() else run / debug
        results.append(clean(measure_run(run, tuple(args.frames), debug_dir=debug, mean_stride=args.mean_stride)))
        print(f"[done]   {run}")
    rows = aggregate_rows(results)
    table = markdown_table(rows)
    print(table)
    if args.out_csv:
        write_csv(rows, args.out_csv)
    if args.out_md:
        args.out_md.write_text(table, encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
