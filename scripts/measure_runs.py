"""Border metrics + tracker gain for many run folders at once (CPU, parallel), as one table.

For every run folder (``video.mp4``, ``request.json``, ``run.json``) it computes, with the repository's own code and
exactly as E1 did: ``border_quality.json`` (``measure_border_quality.measure_run`` with ``mean_stride=4``: mean over
every 4th frame with R > 1 %, plus frames 40/80/120) and ``measure.json`` (``calibrate_strength.measure_run_gain``:
sparse-flow tracker, ``signed_motion_gain_pathwide``). Cached files are reused. Prints a markdown table and can
write JSON / CSV.

    python scripts/measure_runs.py G1/userphoto/full/pan_left/s1.00 G1/userphoto/full/tilt_up/s1.00 --workers 4 --json rows.json
    python scripts/measure_runs.py --root G1 --workers 4 --md table.md        # every run folder below G1
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

COLUMNS = ("local_mean", "v2_mean", "local_f120", "v2_f120", "gray_band", "colorf_RV", "hf_RV", "seam", "R_frac_f120",
           "gain", "cosine", "u", "a_obs")


def find_runs(root):
    """Run folders below ``root`` with ``video.mp4`` and ``request.json``."""
    return sorted(p.parent for p in Path(root).rglob("run.json") if (p.parent / "video.mp4").is_file()
                  and (p.parent / "request.json").is_file())


def measure_one(run_dir):
    """Returns one flat row dict for a run folder (computing and caching the two measurement files)."""
    import calibrate_strength as cs
    import measure_border_quality as mbq

    run_dir = Path(run_dir)
    cached = run_dir / "border_quality.json"
    if cached.is_file():
        bq = json.loads(cached.read_text(encoding="utf8"))
    else:
        bq = mbq.clean(mbq.measure_run(run_dir, mean_stride=4))
    gain = cs.measure_run_gain(run_dir)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf8"))
    mean = bq.get("mean_over_frames_with_R_gt_1pct", {}) or {}
    f120 = (bq.get("at_frame", {}) or {}).get("120", {}) or {}
    u, g = gain.get("strength_u"), gain.get("gain")
    sc = run.get("strength_control") or {}
    return {
        "run_dir": str(run_dir),
        "mode": run.get("mode") or gain.get("motion"),
        "fill": run.get("guide_fill_mode"),
        "checkpoint_sha256": run.get("checkpoint_sha256"),
        "seconds": run.get("seconds"),
        "attn_backend": ((run.get("speed_options") or {}).get("attn_backend") or {}).get("requested"),
        "amplitude_source": sc.get("amplitude_source"),
        "label": sc.get("label"),
        "local_mean": mean.get("artifact_score_local"),
        "v2_mean": mean.get("artifact_score_v2"),
        "local_f120": f120.get("artifact_score_local"),
        "v2_f120": f120.get("artifact_score_v2"),
        "gray_band": mean.get("gray_band_score"),
        "colorf_RV": mean.get("colorfulness_ratio"),
        "hf_RV": mean.get("hf_ratio"),
        "seam": mean.get("seam_score"),
        "R_frac_f120": f120.get("revealed_fraction"),
        "n_R_frames": mean.get("n_frames"),
        "gain": g,
        "cosine": gain.get("cosine_pathwide"),
        "u": u,
        "s": gain.get("strength_s"),
        "a_obs": (g * u) if (g is not None and u is not None) else None,
    }


def fmt(v, nd=3):
    """Format a metric for the table (``n/a`` for missing / non-finite)."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def markdown(rows, names):
    """Render the rows (border metrics + tracker gain) as a markdown table."""
    head = ["run", *COLUMNS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, r in zip(names, rows):
        lines.append("| " + " | ".join([name, *(fmt(r.get(c)) for c in COLUMNS)]) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    """CLI entry: measure many runs in parallel (cached results reused) and print/write the table."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--root", type=Path, default=None, help="measure every run folder below this directory")
    ap.add_argument("--relative-to", type=Path, default=None, help="show run names relative to this directory")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=None)
    args = ap.parse_args(argv)
    runs = list(args.runs) + (find_runs(args.root) if args.root else [])
    if not runs:
        ap.error("no run folders")
    base = args.relative_to or args.root
    if args.workers > 1 and len(runs) > 1:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(measure_one, runs))
    else:
        rows = [measure_one(r) for r in runs]
    names = [str(r.relative_to(base)).replace("\\", "/") if base and r.is_relative_to(base) else str(r) for r in runs]
    md = markdown(rows, names)
    print(md)
    if args.json:
        args.json.write_text(json.dumps([{"name": n, **r} for n, r in zip(names, rows)], indent=2) + "\n", encoding="utf8")
    if args.md:
        args.md.write_text(md, encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
