"""Phase P1 (Molab GPU): diagnose the gray-border artefact of the v1 adapter.

For every image x mode x variant the script runs ``flow_infer.generate`` with the
Wan pipeline loaded ONCE, saves final / x0-hat latents (``<run>/debug``), computes
border-quality metrics on the decoded video and latent R-vs-valid statistics, and
finally writes ``diagnosis_summary.md`` (+ ``.json``) with automatic verdicts for
hypotheses H1 (zero-guide shortcut), H2 (regression target) and H3 (hand-off at
sigma=0.6). Resumable: a finished run (``run.json`` present) is never regenerated.

Variants
  v1_baseline    zeros guide fill, camera_min_sigma 0.6     (exactly v1)
  fill_replicate replicate guide fill, camera_min_sigma 0.6 (tests H1)
  min_sigma_0.3  zeros, camera_min_sigma 0.3                (tests H3)
  min_sigma_0.0  zeros, camera_min_sigma 0.0                (tests H3)
  frozen_base    adapter disabled (frozen Wan I2V)          (control)

Example
  python scripts/diagnose_v1_borders.py --project-root . \
      --images heldout=examples/inputs/heldout.jpg motorcycle=examples/inputs/motorcycle.jpg
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import measure_border_quality as mbq  # noqa: E402
from fpcamera import border_metrics as bm  # noqa: E402

VARIANTS = {
    "v1_baseline": dict(guide_fill_mode="zeros", camera_min_sigma=0.6),
    "fill_replicate": dict(guide_fill_mode="replicate", camera_min_sigma=0.6),
    "min_sigma_0.3": dict(guide_fill_mode="zeros", camera_min_sigma=0.3),
    "min_sigma_0.0": dict(guide_fill_mode="zeros", camera_min_sigma=0.0),
    "frozen_base": dict(
        guide_fill_mode="zeros", camera_min_sigma=None, frozen_baseline=True
    ),
}
DEFAULT_PAIRS = (("heldout", "zoom_out"), ("motorcycle", "pan_right"))
DEFAULT_MODES = ("zoom_out", "pan_right")
DEFAULT_CHECKPOINT = "artifacts/flow_poc_all/adapter_000300.safetensors"

# Verdict thresholds (see docs/experiments/P1_RESULTS.md for the calibration rationale).
# ARTIFACT_PRESENT_MIN was chosen on only two cases (heldout zoom_out, motorcycle pan_right):
# v1_baseline scores about 0.8-1.0 there, the heldout frozen_base control about 0.17.
ARTIFACT_PRESENT_MIN = 0.40  # baseline artifact_score above this => border problem present
ARTIFACT_REDUCTION_MIN = 0.25  # relative reduction of artifact_score that counts (H1, H3)
GRAY_PRESENT_MIN = 0.15  # legacy gray-band gate; now informational only
X0HAT_HF_RATIO_MAX = 0.3  # x0-hat R/valid Laplacian-variance ratio => "smooth"
MIN_R_FRACTION = 0.01
MEAN_STRIDE = 4  # "mean over frames" uses every 4th frame (one per latent frame)


# ------------------------------------------------------------------ cases
def parse_images(items, root):
    """Parse ``name=path`` / ``path`` items into ``{name: Path}``."""
    images = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep:
            path, name = item, Path(item).stem
        p = Path(path)
        images[name] = p if p.is_absolute() else root / p
    return images


def default_images(root):
    """Look for the v1 example inputs under examples/inputs (not tracked in git)."""
    found = {}
    folder = root / "examples" / "inputs"
    for key in ("heldout", "motorcycle"):
        hits = sorted(
            p
            for p in folder.rglob("*")
            if p.is_file()
            and key in p.name.lower()
            and p.suffix.lower() in (".png", ".jpg", ".jpeg")
        ) if folder.is_dir() else []
        if hits:
            found[key] = hits[0]
    return found


def build_cases(args, root):
    """Turn the CLI selection (images x motions x magnitudes, or explicit cases) into the list of diagnosis cases."""
    if args.cases:
        images, pairs, mags = {}, [], {}
        for item in args.cases:
            spec, _, mode = item.rpartition(":")
            mode, _, mag = mode.partition("@")  # optional magnitude: name=path:mode@0.5
            if not spec or not mode:
                sys.exit(f"--cases expects name=path:mode[@magnitude], got {item!r}")
            name, sep, path = spec.partition("=")
            if not sep:
                path, name = spec, Path(spec).stem
            p = Path(path)
            images[name] = p if p.is_absolute() else root / p
            pairs.append((name, mode))
            mags[name] = float(mag) if mag else 1.0
        for name, _ in pairs:
            if not images[name].is_file():
                sys.exit(f"Image not found: {images[name]}")
        return [dict(name=n, mode=m, image=images[n], id=f"{n}__{m}", magnitude=mags[n]) for n, m in pairs]
    if args.images:
        images = parse_images(args.images, root)
    else:
        images = default_images(root)
        if not images:
            sys.exit(
                "No --images given and no examples/inputs/*heldout*|*motorcycle* found. "
                "Pass --images name=path ... (see docs/experiments/P1_DIAGNOSTICS_PROTOCOL.md)."
            )
    if args.modes:
        if len(args.modes) != len(images):
            sys.exit("--modes must pair one-to-one with --images (zip), or use --cases name=path:mode")
        pairs = list(zip(images, args.modes))
    elif args.images:
        pairs = [(n, m) for n, m in DEFAULT_PAIRS if n in images]
        if not pairs:
            sys.exit("--images given without --modes and no default pair matches; use --modes or --cases")
    else:
        pairs = [(n, m) for n, m in DEFAULT_PAIRS if n in images]
    for name, _ in pairs:
        if not images[name].is_file():
            sys.exit(f"Image not found: {images[name]}")
    return [dict(name=n, mode=m, image=images[n], id=f"{n}__{m}") for n, m in pairs]


# ---------------------------------------------------------------- metrics
def x0hat_metrics(debug_dir, homographies):
    """Frame metrics on the decoded x0-hat PNGs (sigma ~0.6), mean over frames."""
    import cv2

    meta_path = Path(debug_dir) / "debug_meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf8"))
    rows = []
    for name in meta.get("files", {}).get("pngs", []):
        if "sigma_0.60" not in name:
            continue
        index = int(name.rsplit("_", 1)[1].split(".")[0])
        bgr = cv2.imread(str(Path(debug_dir) / name))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = bm.revealed_mask_pixels(
            homographies, rgb.shape[1], rgb.shape[0], frame_indices=[index]
        )[0]
        rows.append(bm.frame_metrics(rgb, mask))
    return bm.mean_metrics(rows, MIN_R_FRACTION) if rows else None


def measure_variant(run_dir, mean_stride=MEAN_STRIDE):
    """Border-quality measurement of one generated variant folder (reuses its debug x0-hat captures)."""
    run_dir = Path(run_dir)
    debug = run_dir / "debug"
    result = mbq.measure_run(
        run_dir, debug_dir=debug if debug.is_dir() else None, mean_stride=mean_stride
    )
    result = mbq.clean(result)
    h, *_ = mbq.load_homographies(run_dir)
    x0 = x0hat_metrics(debug, h) if debug.is_dir() else None
    result["x0hat_sigma_0.60_metrics"] = mbq.clean(x0) if x0 else None
    (run_dir / "border_quality.json").write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
    return result


def _stale(bq):
    """True if border_quality.json is missing or predates the artifact_score metric."""
    if not bq.is_file():
        return True
    try:
        mean = json.loads(bq.read_text(encoding="utf8")).get("mean_over_frames_with_R_gt_1pct", {})
    except ValueError:
        return True
    return "artifact_score" not in mean


# ------------------------------------------------------------- generation
def pending_jobs(cases, variants, out_root):
    """(case, variant) jobs whose run folder is not complete yet."""
    jobs = []
    for case in cases:
        for variant in variants:
            run_dir = out_root / case["id"] / variant
            if not (run_dir / "run.json").is_file():
                jobs.append((case, variant, run_dir))
    return jobs


def run_generation(args, jobs, root, checkpoint):
    """Generate every pending variant with one loaded pipeline under the GPU lease; returns ``{job: error}``."""
    from fpcamera.flow_infer import generate, load_inference_pipeline
    from fpcamera.runtime import gpu_lease

    errors = {}
    from contextlib import nullcontext

    with (nullcontext() if getattr(args, "no_lease", False) else gpu_lease(root)):
        print("loading Wan pipeline (once)...", flush=True)
        t0 = time.perf_counter()
        pipe = load_inference_pipeline(
            flow_shift=5.0, log_path=root / "artifacts" / "runtime" / "diagnose_library.log"
        )
        print(f"pipeline loaded in {time.perf_counter() - t0:.0f}s", flush=True)
        for i, (case, variant, run_dir) in enumerate(jobs):
            spec = dict(VARIANTS[variant])
            print(f"[{i + 1}/{len(jobs)}] {case['id']} / {variant}", flush=True)
            t0 = time.perf_counter()
            try:
                generate(
                    root,
                    checkpoint,
                    case["image"],
                    run_dir,
                    mode=case["mode"],
                    magnitude=case.get("magnitude", 1.0),
                    seed=args.seed,
                    steps=args.steps,
                    debug_dir=run_dir / "debug",
                    debug_sigmas=tuple(args.debug_sigmas),
                    debug_decode=not args.no_decode,
                    pipe=pipe,
                    # hash the 5B backbone only for the first and last job
                    verify_backbone=(i == 0 or i == len(jobs) - 1),
                    **spec,
                )
                print(f"    done in {time.perf_counter() - t0:.0f}s", flush=True)
            except Exception:  # keep going: Molab sessions are precious
                key = case["id"] + "/" + variant
                errors[key] = traceback.format_exc()
                print("    FAILED\n" + errors[key], flush=True)
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
    return errors


# ---------------------------------------------------------------- summary
def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _score(res):
    return mbq._g(res, "mean_over_frames_with_R_gt_1pct", "artifact_score")


def motion_summary(run_dir):
    """Tracker results (path_adherence.json from summarize_flow_adherence.py) or None."""
    f = Path(run_dir) / "path_adherence.json"
    if not f.is_file():
        return None
    d = json.loads(f.read_text(encoding="utf8"))
    cov = d["adherence"].get("coverage_of_expected_visible_tracks_per_frame") or []
    cov = [c for c in cov if c is not None and math.isfinite(c)]
    m, a = d["motion"], d["adherence"]
    return dict(
        gain=m.get("signed_motion_gain_pathwide"),
        gain_end=m.get("signed_motion_gain_endpoint"),
        cosine=m.get("motion_cosine_pathwide"),
        ade=a.get("ADE_px_missing_penalty_image_diagonal"),
        fde=a.get("FDE_px_missing_penalty_image_diagonal"),
        coverage=float(np.mean(cov)) if cov else None,
    )


def _latent_ratios(res, key="final_latent"):
    s = mbq._g(res, "latent_stats", key)
    if not s:
        return None, None
    return s.get("std_ratio_R_over_valid"), s.get("mean_abs_ratio_R_over_valid")


def verdicts(results):
    """results: {variant: measured dict} for one case. Returns dict of verdict dicts."""
    base = results.get("v1_baseline")
    out = {}
    if not base:
        return {h: dict(status="NO DATA", why="v1_baseline missing") for h in ("H1", "H2", "H3")}
    n_r = mbq._g(base, "mean_over_frames_with_R_gt_1pct", "n_frames") or 0
    b_score = _score(base)
    no_problem = n_r == 0 or b_score is None or b_score <= ARTIFACT_PRESENT_MIN
    if no_problem:
        why = (
            f"baseline shows no border artifact (R frames={n_r}, "
            f"artifact_score={_fmt(b_score)} <= {ARTIFACT_PRESENT_MIN})"
        )
        return {h: dict(status="NOT APPLICABLE", why=why) for h in ("H1", "H2", "H3")}

    def reduces(variant):
        """Score of a variant, or (None, None) if it was not measured."""
        r = results.get(variant)
        s = _score(r) if r else None
        if s is None:
            return None, None
        return (b_score - s) >= ARTIFACT_REDUCTION_MIN * b_score, s

    # H1: replicate guide fill lowers the artifact score by >= 25% relative.
    std_r, abs_r = _latent_ratios(base)
    red_rep, s_rep = reduces("fill_replicate")
    if red_rep is None:
        h1 = dict(status="NO DATA", why="missing fill_replicate run")
    else:
        h1 = dict(
            status="SUPPORTED" if red_rep else "NOT SUPPORTED",
            why=(
                f"artifact_score {_fmt(b_score)} -> {_fmt(s_rep)} with replicate fill "
                f"(relative reduction {(b_score - s_rep) / b_score:.0%}, need >= {ARTIFACT_REDUCTION_MIN:.0%}). "
                f"Final-latent R/valid std ratio={_fmt(std_r)}, mean|z| ratio={_fmt(abs_r)} (informational; "
                "~1 means no latent collapse). Caveat: v1 was trained with zero fill, so replicate at inference "
                "is a distribution shift; a null result does not refute H1."
            ),
        )
    out["H1"] = h1

    # H2: x0-hat at sigma 0.6 is smooth in R relative to valid.
    x0 = base.get("x0hat_sigma_0.60_metrics")
    x_std, _ = _latent_ratios(base, "x0_hat_sigma_0.60")
    x_hf = x0.get("hf_ratio") if x0 else None
    if x_hf is None and x_std is None:
        out["H2"] = dict(status="NO DATA", why="no x0-hat capture at sigma 0.6")
    else:
        smooth = x_hf is not None and x_hf < X0HAT_HF_RATIO_MAX
        out["H2"] = dict(
            status="INDICATOR PRESENT (not separable from H1)" if smooth else "INDICATOR ABSENT",
            why=(
                f"x0-hat(sigma~0.6) R/valid hf_ratio={_fmt(x_hf)} (< {X0HAT_HF_RATIO_MAX} = smooth); "
                f"latent std ratio={_fmt(x_std)} (informational). Report only: x0-hat is "
                "blurry everywhere; compare with the final-video hf_ratio="
                f"{_fmt(mbq._g(base, 'mean_over_frames_with_R_gt_1pct', 'hf_ratio'))}."
            ),
        )

    # H3: lowering min_sigma lowers the artifact score by >= 25% relative.
    rows, any_red = [], False
    for v in ("min_sigma_0.3", "min_sigma_0.0"):
        red, s = reduces(v)
        if red is None:
            continue
        any_red |= bool(red)
        rows.append(f"{v}: artifact_score {_fmt(b_score)} -> {_fmt(s)} ({'reduced' if red else 'not reduced'})")
    out["H3"] = (
        dict(status="SUPPORTED" if any_red else "NOT SUPPORTED", why="; ".join(rows)
             + f" (reduction means >= {ARTIFACT_REDUCTION_MIN:.0%} relative)")
        if rows
        else dict(status="NO DATA", why="no min_sigma runs")
    )
    return out


def write_summary(cases, variants, out_root, errors):
    """Write the P1 diagnosis summary (markdown + json) comparing the variants."""
    all_results, lines = {}, []
    lines.append("# P1 diagnosis summary (v1 borders)\n")
    lines.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Metrics are means over frames "
                 f"with R fraction > {MIN_R_FRACTION:.0%} unless marked @120.\n")
    cases = [c for c in cases if any((out_root / c["id"] / v / "run.json").is_file() for v in variants)]
    for case in cases:
        results = {}
        for variant in variants:
            run_dir = out_root / case["id"] / variant
            bq = run_dir / "border_quality.json"
            if (run_dir / "run.json").is_file():
                if _stale(bq):
                    try:
                        measure_variant(run_dir)
                    except Exception:
                        errors[f"{case['id']}/{variant}/metrics"] = traceback.format_exc()
                if bq.is_file():
                    results[variant] = json.loads(bq.read_text(encoding="utf8"))
        all_results[case["id"]] = results
        lines.append(f"\n## {case['id']} (mode `{case['mode']}`)\n")
        head = ["variant", "fill", "min_sigma", "R@120", "artifact", "glow", "gray score (secondary)", "gray@120",
                "color ratio", "hf ratio", "seam", "lat std R/V", "lat mean-abs R/V",
                "x0hat std R/V", "x0hat hf R/V"]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
        for variant in variants:
            r = results.get(variant)
            if not r:
                lines.append(f"| {variant} | " + " | ".join(["missing"] * (len(head) - 1)) + " |")
                continue
            m = r["mean_over_frames_with_R_gt_1pct"]
            ls, la = _latent_ratios(r)
            xs, _ = _latent_ratios(r, "x0_hat_sigma_0.60")
            xh = (r.get("x0hat_sigma_0.60_metrics") or {}).get("hf_ratio")
            cells = [variant, r.get("guide_fill_mode"), r.get("camera_min_sigma"),
                     mbq._g(r, "at_frame", "120", "revealed_fraction"),
                     m.get("artifact_score"), m.get("glow_term"),
                     m.get("gray_band_score"), mbq._g(r, "at_frame", "120", "gray_band_score"),
                     m.get("colorfulness_ratio"), m.get("hf_ratio"), m.get("seam_score"),
                     ls, la, xs, xh]
            lines.append("| " + " | ".join(_fmt(c) for c in cells) + " |")
        lines.append("")
        mrows = [(variant, motion_summary(out_root / case["id"] / variant)) for variant in variants]
        if any(m for _, m in mrows):
            lines.append("Motion (tracker; signed path gain = observed/requested displacement, 1.0 = requested amplitude):")
            lines.append("")
            lines.append("| variant | gain (pathwide) | gain (endpoint) | dir cosine | ADE px | FDE px | coverage |")
            lines.append("|---|---|---|---|---|---|---|")
            for variant, m in mrows:
                if m:
                    cells = [variant, m["gain"], m["gain_end"], m["cosine"], m["ade"], m["fde"], m["coverage"]]
                    lines.append("| " + " | ".join(_fmt(c) for c in cells) + " |")
            lines.append("")
        v = verdicts(results)
        for h, name in (("H1", "zero-guide shortcut"), ("H2", "deterministic regression target"),
                        ("H3", "adapter hand-off at sigma=0.6")):
            if h in v:
                lines.append(f"- **{h} ({name}): {v[h]['status']}.** {v[h]['why']}")
        all_results[case["id"]]["_verdicts"] = v
    lines.append("\n## Overall\n")
    for h in ("H1", "H2", "H3"):
        statuses = [all_results[c["id"]]["_verdicts"].get(h, {}).get("status", "NO DATA") for c in cases]
        lines.append(f"- {h}: " + "; ".join(f"{c['id']}={s}" for c, s in zip(cases, statuses)))
    lines.append("\n## Thresholds\n")
    lines.append(
        f"- gray problem present: baseline gray_band_score > {GRAY_PRESENT_MIN}\n"
        f"- gray reduction counts: >= {GRAY_REDUCTION_MIN:.0%} relative drop of gray_band_score\n"
        f"- latent collapse (H1): R/valid ratio of std AND mean|z| of the final latent < {LATENT_RATIO_MAX}\n"
        f"- x0-hat smooth (H2): R/valid Laplacian-variance ratio < {X0HAT_HF_RATIO_MAX} or latent std ratio < {LATENT_RATIO_MAX}\n"
        "- H4 (data scarcity) is not testable here.\n"
        "- `frozen_base` is a control: the frozen model ignores the camera path, so its R "
        "region is only the area that WOULD be revealed; it shows natural statistics there.\n"
    )
    if errors:
        lines.append("\n## Errors\n")
        for k, tb in errors.items():
            lines.append(f"### {k}\n```\n{tb}\n```\n")
    (out_root / "diagnosis_summary.md").write_text("\n".join(lines), encoding="utf8")
    (out_root / "diagnosis_summary.json").write_text(
        json.dumps(mbq.clean(all_results), indent=2), encoding="utf8"
    )
    return all_results


# ------------------------------------------------------------------- main
def main(argv=None):
    """CLI entry: build cases -> generate -> measure -> summarise (see module docstring for the variants)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    ap.add_argument("--images", nargs="+", default=None, help="name=path or path")
    ap.add_argument("--modes", nargs="+", default=None, help="one mode per --images entry (zipped)")
    ap.add_argument("--cases", nargs="+", default=None, help="explicit name=path:mode entries")
    ap.add_argument("--output-root", type=Path, default=Path("artifacts/p1_diagnostics"))
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--debug-sigmas", type=float, nargs="+", default=[0.6])
    ap.add_argument("--no-decode", action="store_true", help="skip decoding x0-hat PNGs")
    ap.add_argument("--no-lease", action="store_true",
                    help="do not take artifacts/gpu.lock (use only when another job holds it and GPU memory fits)")
    ap.add_argument("--summarize-only", action="store_true",
                    help="no GPU: (re)compute metrics and summary from existing runs")
    args = ap.parse_args(argv)

    root = args.project_root.resolve()
    out_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    cases = build_cases(args, root)
    out_root.mkdir(parents=True, exist_ok=True)
    errors = {}
    jobs = pending_jobs(cases, args.variants, out_root)
    print(f"{len(cases)} cases x {len(args.variants)} variants; {len(jobs)} runs pending")
    if jobs and not args.summarize_only:
        if not checkpoint.is_file():
            sys.exit(f"Checkpoint not found: {checkpoint}")
        errors = run_generation(args, jobs, root, checkpoint)
    for case in cases:  # metrics for newly finished runs
        for variant in args.variants:
            run_dir = out_root / case["id"] / variant
            if (run_dir / "run.json").is_file() and _stale(run_dir / "border_quality.json"):
                print(f"metrics: {case['id']}/{variant}", flush=True)
                try:
                    measure_variant(run_dir)
                except Exception:
                    errors[f"{case['id']}/{variant}/metrics"] = traceback.format_exc()
    write_summary(cases, args.variants, out_root, errors)
    print(f"wrote {out_root / 'diagnosis_summary.md'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
