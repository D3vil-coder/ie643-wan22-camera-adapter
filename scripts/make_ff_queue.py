"""Write the F1 flat-image-fix job files for ``scripts/generate_batch.py`` (one JSON list per file, name = priority).

Used on the Molab sandbox (paths relative to the project root; inputs from ``scripts/make_flat_inputs.py`` and the
regression sources copied to ``artifacts/ff_inputs``). Run names are ``artifacts/ff/<image>/<variant>``.

    python scripts/make_ff_queue.py --queue-dir artifacts/ff_queue --groups p1_ball p1_synth
    python scripts/make_ff_queue.py --queue-dir artifacts/ff_queue --list

Image / motion table (physical amplitude, so the calibration file is irrelevant and u is exact):

    ball       pan_left   2.5 deg   (u = 0.3125, the G1 setting that dims to 0.65x luma)
    synthetic  zoom_out   x1.25     (u = 1, the notebook's built-in landscape)
    airplane   tilt_up    4 deg     (COCO val, flat grey sky)
    table      pan_right  2.5 deg   (synthetic gradient wall + table + mug)
    photo      pan_right  8 deg     (user photo, full strength, regression)
    motorcycle pan_right  2.5 deg   (regression, E1 case at matched amplitude)
    heldout    zoom_out   x1.12     (regression, E1 case at matched amplitude)
    coach      pan_left   2.5 deg   (COCO val with people, regression)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IN = "artifacts/ff_inputs"
OUT = "artifacts/ff"
CKPT = {
    "415": "artifacts/v2_full/checkpoints/adapter_000415.safetensors",
    "250": "artifacts/v2_full/checkpoints/adapter_000250.safetensors",
    "77": "artifacts/v2_full/checkpoints/adapter_000077.safetensors",
}
IMAGES = {
    "ball": ("ball.png", {"motion": "pan_left", "amplitude_deg": 2.5}),
    "synthetic": ("synthetic_landscape.png", {"motion": "zoom_out", "zoom_ratio": 1.25}),
    "airplane": ("airplane_sky.png", {"motion": "tilt_up", "amplitude_deg": 4.0}),
    "table": ("table_object.png", {"motion": "pan_right", "amplitude_deg": 2.5}),
    "photo": ("user_photo.jpeg", {"motion": "pan_right", "amplitude_deg": 8.0}),
    "motorcycle": ("motorcycle.png", {"motion": "pan_right", "amplitude_deg": 2.5}),
    "heldout": ("heldout.png", {"motion": "zoom_out", "zoom_ratio": 1.12}),
    "coach": ("coach_person.png", {"motion": "pan_left", "amplitude_deg": 2.5}),
}


def job(image, variant, *, ckpt="415", extra=None, motion_override=None):
    fname, motion = IMAGES[image]
    kw = dict(motion if motion_override is None else motion_override)
    kw.update(extra or {})
    if not kw.get("frozen_baseline"):
        kw["trace_residual"] = True  # diagnostic only (per-step residual statistics in run.json); no effect on the video
    return {"image": f"{IN}/{fname}", "run_dir": f"{OUT}/{image}/{variant}", "checkpoint": CKPT[ckpt], "generate_kwargs": kw}


def groups():
    g = {}
    g["00_p1_ball"] = [
        job("ball", "base_wan", extra={"frozen_baseline": True}),
        job("ball", "v415_s0", motion_override={"motion": "pan_left", "strength": 0.0, "calibration": None}),
        job("ball", "v77", ckpt="77"),
        job("ball", "v250", ckpt="250"),
        job("ball", "v415_seed43", extra={"seed": 43}),
        job("ball", "v415_minsigma08", extra={"camera_min_sigma": 0.8}),
    ]
    g["05_p1_synth"] = [
        job("synthetic", "base_wan", extra={"frozen_baseline": True}),
        job("synthetic", "v415_default"),
    ]
    g["10_f_ball"] = [
        job("ball", "v415_f1mean", extra={"residual_lowfreq": "mean"}),
        job("ball", "v415_f2blur8", extra={"residual_lowfreq": "blur:8"}),
    ]
    g["15_f_synth"] = [
        job("synthetic", "v415_f1mean", extra={"residual_lowfreq": "mean"}),
        job("synthetic", "v415_f2blur8", extra={"residual_lowfreq": "blur:8"}),
    ]
    return g


def fix_groups(tag, extra, prefix="25"):
    """Fix runs for the other flat images (and the regression set): ``extra`` = generate kwargs of the fix."""
    g = {}
    g[f"{prefix}_flat_{tag}"] = [job(i, f"v415_{tag}", extra=extra) for i in ("airplane", "table")]
    g[f"{prefix}b_reg_{tag}"] = [job(i, f"v415_{tag}", extra=extra) for i in ("photo", "motorcycle", "heldout", "coach")]
    return g


def default_groups():
    return {
        "20_flat_default": [job("airplane", "v415_default"), job("table", "v415_default")],
        "21_coach_default": [job("coach", "v415_default")],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--queue-dir", type=Path, required=True)
    ap.add_argument("--groups", nargs="*", default=None, help="group names to write (default: none; see --list)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--defaults", action="store_true", help="write the default-adapter runs of the extra flat images and the coach")
    ap.add_argument("--fix", default=None, help="generate-kwargs JSON of a fix, e.g. '{\"residual_lowfreq\": \"mean\"}' (with --tag, --prefix)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--prefix", default="25")
    ap.add_argument("--only", nargs="*", default=None, help="with --fix: keep only these image names")
    ap.add_argument("--spec-file", type=Path, default=None, help="write the single JSON list from this file as <name>.json (see --name)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)
    g = groups()
    if args.list:
        for name, jobs in g.items():
            print(name, [j["run_dir"] for j in jobs])
        return 0
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    if args.spec_file:
        (args.queue_dir / f"{args.name}.json").write_text(args.spec_file.read_text(encoding="utf8"), encoding="utf8")
        return 0
    if args.defaults:
        for name, jobs in default_groups().items():
            (args.queue_dir / f"{name}.json").write_text(json.dumps(jobs, indent=1) + "\n", encoding="utf8")
            print("wrote", name, len(jobs), "jobs")
    if args.fix:
        for name, jobs in fix_groups(args.tag, json.loads(args.fix), args.prefix).items():
            if args.only:
                jobs = [j for j in jobs if j["run_dir"].split("/")[-2] in args.only]
            (args.queue_dir / f"{name}.json").write_text(json.dumps(jobs, indent=1) + "\n", encoding="utf8")
            print("wrote", name, len(jobs), "jobs")
    for name in args.groups or []:
        (args.queue_dir / f"{name}.json").write_text(json.dumps(g[name], indent=1) + "\n", encoding="utf8")
        print("wrote", name, len(g[name]), "jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
