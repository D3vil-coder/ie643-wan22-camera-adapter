"""Tables and comparison sheets for the F1 flat-image diagnosis / fix (CPU).

Reads, per run folder, ``frame_stats.json`` (``scripts/analyze_frames.py``), optionally ``measure.json`` (tracker gain) and
``border_quality.json`` (``scripts/measure_runs.py``) and writes one markdown table; ``--sheet OUT.png RUN RUN ...`` writes a
contact sheet (one row per run, frames 0 / 20 / 40 / 80 / 120) with the global luma ratio printed on every tile.

    python scripts/summarize_ff.py RUNS_ROOT --md table.md
    python scripts/summarize_ff.py RUNS_ROOT --runs ball/v415_default ball/v415_f1mean --md t.md --sheet sheet.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(run_dir):
    run_dir = Path(run_dir)
    fs = json.loads((run_dir / "frame_stats.json").read_text(encoding="utf8"))
    meas = json.loads((run_dir / "measure.json").read_text(encoding="utf8")) if (run_dir / "measure.json").is_file() else {}
    bq = json.loads((run_dir / "border_quality.json").read_text(encoding="utf8")) if (run_dir / "border_quality.json").is_file() else {}
    run = json.loads((run_dir / "run.json").read_text(encoding="utf8")) if (run_dir / "run.json").is_file() else {}
    return fs, meas, bq, run


def fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return "nan" if not np.isfinite(v) else f"{v:.{nd}f}"
    return str(v)


COLUMNS = ["Y@4", "chroma@4", "Y@20", "Y@40", "Y@80", "Y@120", "valid Y min", "valid Y @120", "chroma shift", "max dY/frame", "mc median", "mc spikes",
           "hf last/first", "hf spikes", "gain", "cosine", "gray-band", "local mean"]


def row(name, run_dir):
    fs, meas, bq, run = load(run_dir)
    s = fs["summary"]
    mean = (bq.get("mean_over_frames_with_R_gt_1pct") or {}) if bq else {}
    gain = meas.get("gain")
    sp = s["spike_frames_mod4"]
    spikes = f"{s['n_spikes_mc']}" + (f" {fs['spikes_mc'][:6]}" if s["n_spikes_mc"] else "")
    vals = [s.get("luma_ratio_4"), s.get("chroma_shift_4"), s["luma_ratio_20"], s["luma_ratio_40"], s["luma_ratio_80"], s["luma_ratio_120"], s["valid_luma_ratio_min"],
            s["valid_luma_ratio_final"], s["chroma_shift_final"], s["max_luma_jump"], s["mc_resid_dm_median"], spikes,
            s["hf_ratio_last_first"], s["n_spikes_hf"], gain if run.get("adapter_enabled", True) else None,
            meas.get("cosine_pathwide"), mean.get("gray_band_score"), mean.get("artifact_score_local")]
    return [name] + [fmt(v, 2 if i in (1, 8) else 3) if not isinstance(v, str) else v for i, v in enumerate(vals)]


def table(names, root):
    head = ["run"] + COLUMNS
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for n in names:
        try:
            lines.append("| " + " | ".join(row(n, Path(root) / n)) + " |")
        except FileNotFoundError:
            lines.append(f"| {n} | " + " | ".join(["missing"] * len(COLUMNS)) + " |")
    return "\n".join(lines) + "\n"


def sheet(names, root, out, frames=(0, 20, 40, 80, 120), tile_w=360):
    import cv2
    from PIL import Image, ImageDraw

    rows = []
    for n in names:
        cap = cv2.VideoCapture(str(Path(root) / n / "video.mp4"))
        fs = json.loads((Path(root) / n / "frame_stats.json").read_text(encoding="utf8"))
        tiles = []
        for i in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, bgr = cap.read()
            if not ok:
                continue
            tiles.append((cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), i, fs["luma_ratio"][i]))
        cap.release()
        rows.append((n, tiles))
    h, w = rows[0][1][0][0].shape[:2]
    th = int(round(tile_w * h / w))
    img = Image.new("RGB", (len(frames) * tile_w, len(rows) * (th + 18)), (24, 24, 24))
    dr = ImageDraw.Draw(img)
    for r, (n, tiles) in enumerate(rows):
        for c, (rgb, i, y) in enumerate(tiles):
            img.paste(Image.fromarray(rgb).resize((tile_w, th), Image.LANCZOS), (c * tile_w, r * (th + 18) + 18))
            dr.text((c * tile_w + 4, r * (th + 18) + 3), f"{n}  f{i}  Y {y:.2f}", fill=(255, 230, 120))
    img.save(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--runs", nargs="*", default=None, help="run names relative to root (default: all with frame_stats.json)")
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--sheet", nargs=1, type=Path, default=None)
    args = ap.parse_args(argv)
    names = args.runs or sorted(str(p.parent.relative_to(args.root)).replace("\\", "/") for p in args.root.rglob("frame_stats.json"))
    md = table(names, args.root)
    print(md)
    if args.md:
        args.md.write_text(md, encoding="utf8")
    if args.sheet:
        sheet(names, args.root, args.sheet[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
