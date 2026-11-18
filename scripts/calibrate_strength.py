"""Calibrate the v2 strength slider (Molab GPU for the sweep; CPU for fit/report).

The adapter is asked for a strength ``u`` (u = 1 is the trained maximum: 8 deg or a
1.25x zoom) but realises ``a_obs(u) = g(u) * u`` of it, where ``g`` is the signed motion
gain of the existing tracker (``scripts/measure_flow_video.py`` tracks, then
``summarize_flow_adherence.summarize``: ``signed_motion_gain_pathwide``, observed
displacement projected on the requested displacement). This script

1. sweeps u in {0.25, 0.5, 0.75, 1.0} x 6 directions x N development images (default 3),
   fixed seed, ``--steps`` denoising steps (default 30: the calibration needs the
   motion, not final polish, and 72 runs at 50 steps would cost ~40 % more GPU time);
2. measures g for every video, a_obs = g * u;
3. per direction fits an isotonic (PAV) ``a_obs(u)`` through the origin and inverts it to
   the map ``u(s)`` with ``a_obs(u(s)) = s``; caps and records ``saturates_at`` when the
   adapter never reaches s = 1;
4. writes the calibration JSON (schema: ``fpcamera.strength``), one plot per direction
   (requested vs observed, before/after), a markdown table and, with ``--verify``, re-runs
   s in {0.33, 0.66, 1.0} through the fitted map and reports |a_obs - s|.

Resumable: a run directory with ``run.json`` is not regenerated, one with
``measure.json`` is not re-tracked. ``--no-generate`` only measures/fits what exists (CPU).

    python scripts/calibrate_strength.py --project-root . \\
        --checkpoint artifacts/v2_full/checkpoints/adapter_001000.safetensors --verify
"""

from __future__ import annotations

import argparse
import contextlib
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

from fpcamera import strength as st  # noqa: E402
from fpcamera.session_deadline import add_deadline_args, deadline_from_args, should_stop  # noqa: E402

DEFAULT_US = (0.25, 0.5, 0.75, 1.0)
DEFAULT_VERIFY_S = (0.33, 0.66, 1.0)
GAIN_KEY = "signed_motion_gain_pathwide"
SOURCES_MANIFEST = "data/v2/sources_v2.json"


# --------------------------------------------------------------------------- images
def parse_images(items, root):
    """Parse ``name=path`` / ``path`` items into ``{name: Path}`` (path relative to the project root)."""
    images = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep:
            path, name = item, Path(item).stem
        p = Path(path)
        images[name] = p if p.is_absolute() else root / p
    return images


def dev_images(root, n, manifest=SOURCES_MANIFEST):
    """``n`` evenly spread development-split sources of the v2 manifest -> {name: path}."""
    path = Path(manifest) if Path(manifest).is_absolute() else root / manifest
    data = json.loads(path.read_text(encoding="utf8"))
    records = data["records"] if isinstance(data, dict) else data
    dev = sorted(
        (r for r in records if r.get("split") in ("development", "dev")),
        key=lambda r: str(r["source_id"]),
    )
    if not dev:
        raise SystemExit(f"No development-split sources in {path}")
    n = min(int(n), len(dev))
    picks = [dev[int(i)] for i in np.linspace(0, len(dev) - 1, n).round()] if n > 1 else dev[:1]
    out = {}
    for r in picks:
        name = f"dev{r.get('source_image_id', len(out))}"
        out[name] = root / r["source_image"]
    return out


# --------------------------------------------------------------------------- measuring
def measure_run_gain(run_dir, gain_key=GAIN_KEY, force=False):
    """Track ``video.mp4`` and compare with ``request.json``; cached in ``measure.json``.

    Uses the repository's tracker (``measure_flow_video.track_video``) and
    ``summarize_flow_adherence.summarize`` unchanged. Returns a small dict with the signed
    gain ``g`` (pathwide), its cosine, endpoint gain and the run's ``u``/``s`` if recorded.
    """
    run_dir = Path(run_dir)
    cache = run_dir / "measure.json"
    if cache.is_file() and not force:
        return json.loads(cache.read_text(encoding="utf8"))
    import measure_flow_video as mfv
    import summarize_flow_adherence as sfa

    report, tracks, visible, initial = mfv.track_video(run_dir / "video.mp4")
    tracks_path = run_dir / "tracks.npz"
    np.savez_compressed(
        tracks_path,
        tracks_xy=tracks,
        visibility=visible,
        initial_query_xy=initial,
        frame_indices=np.arange(len(tracks), dtype=np.int32),
    )
    summary = sfa.summarize(tracks_path, run_dir / "request.json")
    (run_dir / "path_adherence.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    request = json.loads((run_dir / "request.json").read_text(encoding="utf8"))
    control = request.get("strength_control") or {}
    motion = summary["motion"]
    result = {
        "run_dir": str(run_dir),
        "gain": motion.get(gain_key),
        "gain_key": gain_key,
        "gain_pathwide": motion.get("signed_motion_gain_pathwide"),
        "gain_endpoint": motion.get("signed_motion_gain_endpoint"),
        "cosine_pathwide": motion.get("motion_cosine_pathwide"),
        "tracks_used_pathwide": motion.get("tracks_used_pathwide"),
        "ade_missing_penalty": summary["adherence"].get("ADE_over_image_diagonal_missing_penalty"),
        "track_coverage_final": report["summary"].get("final_track_coverage"),
        "motion": control.get("motion") or request.get("mode"),
        "strength_s": control.get("strength_s"),
        "strength_u": control.get("strength_u", request.get("strength_u")),
    }
    cache.write_text(json.dumps(clean(result), indent=2), encoding="utf8")
    with contextlib.suppress(OSError):
        tracks_path.unlink()  # 100+ MB of tracks per sweep is not worth keeping
    return result


def clean(value):
    """Recursively convert numpy scalars / non-finite floats into JSON-safe values."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


# --------------------------------------------------------------------------- generation
class Generator:
    """Lazy owner of the GPU lease and the single loaded pipeline."""

    def __init__(self, root, checkpoint, steps, seed, guide_fill_mode=None, log_name="calibrate_library.log",
                 use_lease=True, attn_backend="auto", compile_blocks=False):
        self.root, self.checkpoint = Path(root), Path(checkpoint)
        self.steps, self.seed, self.fill = steps, seed, guide_fill_mode
        self.attn_backend, self.compile_blocks = attn_backend, compile_blocks  # opt-in speed options (flow_infer.generate)
        self.log_name = log_name
        self.use_lease = use_lease
        self.stack = contextlib.ExitStack()
        self.pipe = None
        self.calls = 0
        self.deadline = None  # absolute epoch; set by main from --session-hours etc.
        self.deadline_hit = False

    def _ensure(self):
        if self.pipe is not None:
            return
        from fpcamera.flow_infer import load_inference_pipeline
        from fpcamera.runtime import gpu_lease

        if self.use_lease:
            self.stack.enter_context(gpu_lease(self.root))
        print("loading Wan pipeline (once)...", flush=True)
        t0 = time.perf_counter()
        self.pipe = load_inference_pipeline(
            flow_shift=5.0, log_path=self.root / "artifacts" / "runtime" / self.log_name
        )
        print(f"pipeline loaded in {time.perf_counter() - t0:.0f}s", flush=True)

    def run(self, jobs, label=""):
        """jobs: dicts with image, run_dir and either motion/strength/calibration (v2) or
        generate_kwargs (any ``generate`` argument, e.g. v1 mode/magnitude); an optional
        ``checkpoint`` overrides the generator's. Returns {run_dir: traceback}."""
        from fpcamera.flow_infer import generate

        errors, done_t = {}, []
        for i, job in enumerate(jobs):
            if self.deadline is not None and should_stop(
                    time.time(), self.deadline, float(np.mean(done_t)) if done_t else 300.0):
                self.deadline_hit = True
                print(f"DEADLINE_STOP {label}done={i}/{len(jobs)} (re-run to continue; finished runs are kept)", flush=True)
                break
            self._ensure()
            run_dir = Path(job["run_dir"])
            eta = ""
            if done_t:
                eta = f" (avg {np.mean(done_t):.0f}s/run, ~{np.mean(done_t) * (len(jobs) - i) / 60:.0f} min left)"
            print(f"[{label}{i + 1}/{len(jobs)}] {run_dir.relative_to(self.root) if run_dir.is_relative_to(self.root) else run_dir}{eta}", flush=True)
            t0 = time.perf_counter()
            try:
                kwargs = dict(
                    seed=self.seed,
                    steps=self.steps,
                    guide_fill_mode=self.fill,
                    pipe=self.pipe,
                    verify_backbone=(self.calls == 0),
                    prompt=job.get("prompt"),
                    attn_backend=self.attn_backend,
                    compile_blocks=self.compile_blocks,
                )
                if "motion" in job:
                    kwargs.update(
                        motion=job["motion"],
                        strength=job["strength"],
                        calibration=job.get("calibration"),
                    )
                kwargs.update(job.get("generate_kwargs", {}))  # e.g. v1: mode=, magnitude=
                generate(self.root, job.get("checkpoint", self.checkpoint), job["image"], run_dir, **kwargs)
                done_t.append(time.perf_counter() - t0)
            except Exception:
                errors[str(run_dir)] = traceback.format_exc()
                print("    FAILED\n" + errors[str(run_dir)], flush=True)
                with contextlib.suppress(Exception):
                    import torch

                    torch.cuda.empty_cache()
            self.calls += 1
        return errors

    def close(self):
        """Close the exit stack (releases the loaded pipeline and the GPU lease)."""
        self.stack.close()


# --------------------------------------------------------------------------- planning
def sweep_dir(out, image, direction, u):
    """Run folder of one sweep job: ``<out>/sweep/<image>/<direction>/u<u>``."""
    return Path(out) / "sweep" / image / direction / f"u{u:.2f}"


def verify_dir(out, image, direction, s):
    """Run folder of one verification job: ``<out>/verify/<image>/<direction>/s<s>``."""
    return Path(out) / "verify" / image / direction / f"s{s:.2f}"


def plan_jobs(kind, out, images, directions, levels, calibration=None):
    """Enumerate the (image, direction, level) jobs of a sweep or a verification pass."""
    jobs = []
    for image, path in images.items():
        for direction in directions:
            for level in levels:
                run_dir = (sweep_dir if kind == "sweep" else verify_dir)(out, image, direction, level)
                jobs.append(
                    dict(image=path, run_dir=run_dir, motion=direction, strength=float(level),
                         calibration=calibration, kind=kind, name=image, level=float(level))
                )
    return jobs


def pending(jobs):
    """Jobs whose run folder has no ``run.json`` yet (resume skips finished ones)."""
    return [j for j in jobs if not (Path(j["run_dir"]) / "run.json").is_file()]


# --------------------------------------------------------------------------- fitting
def collect_rows(jobs, gain_key=GAIN_KEY):
    """Measure every finished job; returns rows with u, s, gain and a_obs."""
    rows = []
    for job in jobs:
        run_dir = Path(job["run_dir"])
        if not (run_dir / "run.json").is_file() or not (run_dir / "video.mp4").is_file():
            continue
        m = measure_run_gain(run_dir, gain_key)
        g = m.get("gain")
        u = m.get("strength_u")
        if g is None or u is None:
            continue
        rows.append(
            {
                "image": job["name"],
                "direction": job["motion"],
                "level": job["level"],
                "u": float(u),
                "s": float(m.get("strength_s", job["level"])),
                "gain": float(g),
                "cosine": m.get("cosine_pathwide"),
                "a_obs": float(g) * float(u),
            }
        )
    return rows


def fit_calibration(rows, *, checkpoint_sha256=None, max_amplitudes=None, extra_fit=None):
    """Fit a calibration dict from sweep rows ``{direction, u, a_obs}``.

    Per direction: isotonic ``a_obs(u)`` through the origin (PAV on per-u means), then
    inversion ``u(s)`` on s = 0, 0.05, ..., 1 (smallest u reaching s; capped at the largest
    tested u with ``saturates_at`` = max observed amplitude if s = 1 is unreachable).
    Returns ``(calibration_dict, diagnostics)``.
    """
    directions, diag = {}, {}
    for d in st.DIRECTIONS:
        pts = [r for r in rows if r["direction"] == d]
        if not pts:
            continue
        u = np.array([r["u"] for r in pts])
        a = np.array([r["a_obs"] for r in pts])
        entry, info = st.build_direction_entry(u, a)
        info["n_points"] = len(pts)
        info["mean_gain"] = float(np.mean([r["gain"] for r in pts]))
        info["per_u_mean_a_obs"] = {
            f"{x:.2f}": float(np.mean(a[np.isclose(u, x)])) for x in sorted(set(np.round(u, 6)))
        }
        directions[d] = entry
        diag[d] = info
    fit = {
        "method": "isotonic (pool-adjacent-violators) a_obs(u) through the origin, inverted to u(s)",
        "observed_amplitude": "a_obs = g * u, g = signed_motion_gain_pathwide (tracker vs requested homographies)",
        "per_direction": diag,
        **(extra_fit or {}),
    }
    cal = st.new_calibration(
        directions, checkpoint_sha256=checkpoint_sha256, max_amplitudes=max_amplitudes, fit=fit
    )
    return cal, diag


def verify_report(rows):
    """|a_obs - s| per direction/level from verify rows."""
    out = {"per_direction": {}, "all_abs_errors": []}
    for d in st.DIRECTIONS:
        pts = [r for r in rows if r["direction"] == d]
        if not pts:
            continue
        levels = {}
        for lvl in sorted({r["level"] for r in pts}):
            sel = [r for r in pts if r["level"] == lvl]
            errs = [abs(r["a_obs"] - r["s"]) for r in sel]
            levels[f"{lvl:.2f}"] = {
                "mean_a_obs": float(np.mean([r["a_obs"] for r in sel])),
                "mean_abs_error": float(np.mean(errs)),
                "n": len(sel),
            }
            out["all_abs_errors"].extend(errs)
        out["per_direction"][d] = {
            "levels": levels,
            "mean_abs_error": float(np.mean([v["mean_abs_error"] for v in levels.values()])),
        }
    errs = out.pop("all_abs_errors")
    out["overall_mean_abs_error"] = float(np.mean(errs)) if errs else None
    out["overall_max_abs_error"] = float(np.max(errs)) if errs else None
    return out


# --------------------------------------------------------------------------- reporting
def plot_direction(path, direction, rows, entry_diag, calibration, verify_rows=()):
    """Plot requested vs observed amplitude (before/after calibration) for one direction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c_before, c_after, c_pts = "#c0504d", "#1f6f8b", "#6b6b6b"
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color=c_pts, lw=1, label="ideal (observed = requested)")
    pts = [r for r in rows if r["direction"] == direction]
    ax.scatter([r["u"] for r in pts], [r["a_obs"] for r in pts], s=26, color=c_before,
               alpha=0.75, label="measured (each image)")
    if entry_diag:
        ax.plot([0] + entry_diag["u_knots"][1:], [0] + entry_diag["a_obs_isotonic"][1:], "-o",
                color=c_before, lw=1.6, ms=4, label="isotonic fit")
    ax.set_title("Before: requested u vs observed amplitude")
    ax.set_xlabel("requested strength u (1 = trained max)")
    ax.set_ylabel("observed amplitude a_obs = g * u")
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "--", color=c_pts, lw=1, label="ideal")
    d = calibration["directions"].get(direction)
    if d and entry_diag:
        sm = st.StrengthMap(calibration)
        s_grid = np.linspace(0, 1, 101)
        u_of_s = np.array([sm.s_to_u(direction, s) for s in s_grid])
        uk = np.array(entry_diag["u_knots"])
        ak = np.array(entry_diag["a_obs_isotonic"])
        ax.plot(s_grid, np.interp(u_of_s, uk, ak), color=c_after, lw=1.8,
                label="predicted a_obs(u(s)) (fit)")
    vp = [r for r in verify_rows if r["direction"] == direction]
    if vp:
        ax.scatter([r["s"] for r in vp], [r["a_obs"] for r in vp], s=30, color=c_after,
                   edgecolor="black", linewidth=0.5, zorder=3, label="verify runs")
    ax.set_title("After: UI strength s vs observed amplitude")
    ax.set_xlabel("UI strength s")
    ax.set_ylabel("observed amplitude")
    for a in axes:
        a.set_xlim(0, 1.02)
        a.set_ylim(0, max(1.1, a.get_ylim()[1]))
        a.grid(alpha=0.25)
        a.legend(fontsize=8, loc="upper left")
    sat = (entry_diag or {}).get("saturates_at")
    fig.suptitle(f"{st.MOTION_LABELS[direction]}" + (f"  (saturates at {sat:.2f})" if sat else ""))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def markdown_report(calibration, diag, rows, verify=None, meta=None):
    """Render the calibration report (fit diagnostics, per-run gains, verification errors) as markdown."""
    lines = ["# Strength calibration report", ""]
    if meta:
        lines += [f"- {k}: {v}" for k, v in meta.items()] + [""]
    lines += [
        "a_obs = g * u with g the signed path-wide motion gain; u = 1 is the trained maximum.",
        "",
        "## Sweep (before calibration)",
        "",
        "| direction | n | mean g | " + " | ".join(f"a_obs(u={u:g})" for u in DEFAULT_US) + " | max a_obs | saturates_at |",
        "|---|---|---|" + "---|" * (len(DEFAULT_US) + 2),
    ]
    for d, info in diag.items():
        per = info["per_u_mean_a_obs"]
        cells = [f"{per[f'{u:.2f}']:.3f}" if f"{u:.2f}" in per else "n/a" for u in DEFAULT_US]
        sat = info["saturates_at"]
        lines.append(
            f"| {d} | {info['n_points']} | {info['mean_gain']:.3f} | " + " | ".join(cells)
            + f" | {info['a_obs_max']:.3f} | {'%.3f' % sat if sat else '-'} |"
        )
    lines += ["", "## Fitted map u(s)", "", "| direction | " + " | ".join(f"u(s={s:g})" for s in (0.25, 0.5, 0.75, 1.0)) + " |",
              "|---|---|---|---|---|"]
    sm = st.StrengthMap(calibration)
    for d in diag:
        lines.append(f"| {d} | " + " | ".join(f"{sm.s_to_u(d, s):.3f}" for s in (0.25, 0.5, 0.75, 1.0)) + " |")
    if verify:
        lines += ["", "## Verification (calibrated runs)", "",
                  "| direction | " + " | ".join(f"a_obs(s={s:g}) / abs err" for s in DEFAULT_VERIFY_S) + " | mean abs err |",
                  "|---|" + "---|" * (len(DEFAULT_VERIFY_S) + 1)]
        for d, v in verify["per_direction"].items():
            cells = []
            for s in DEFAULT_VERIFY_S:
                lv = v["levels"].get(f"{s:.2f}")
                cells.append(f"{lv['mean_a_obs']:.3f} / {lv['mean_abs_error']:.3f}" if lv else "n/a")
            lines.append(f"| {d} | " + " | ".join(cells) + f" | {v['mean_abs_error']:.3f} |")
        lines += ["", f"Overall mean |a_obs - s| = {verify['overall_mean_abs_error']:.3f}, "
                  f"max = {verify['overall_max_abs_error']:.3f}."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- main
def main(argv=None):
    """CLI entry: sweep (GPU) -> measure gains -> fit + invert -> write calibration json, plots, report -> optional verify."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--images", nargs="+", default=None, help="name=path or path (overrides dev split)")
    ap.add_argument("--n-images", type=int, default=3, help="dev-split sources from data/v2/sources_v2.json")
    ap.add_argument("--us", type=float, nargs="+", default=list(DEFAULT_US))
    ap.add_argument("--directions", nargs="+", default=list(st.DIRECTIONS), choices=list(st.DIRECTIONS))
    ap.add_argument("--steps", type=int, default=30, help="denoising steps for the sweep (default 30)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--guide-fill-mode", choices=("zeros", "replicate"), default=None,
                    help="default: recorded by the checkpoint")
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/calibration_v2"))
    ap.add_argument("--calibration-out", type=Path, default=Path("configs/strength_calibration_v2.json"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing calibration file")
    ap.add_argument("--gain-key", default=GAIN_KEY, choices=(GAIN_KEY, "signed_motion_gain_endpoint"))
    ap.add_argument("--verify", action="store_true", help="re-run s in --verify-s through the fitted map")
    ap.add_argument("--verify-s", type=float, nargs="+", default=list(DEFAULT_VERIFY_S))
    ap.add_argument("--no-generate", action="store_true", help="CPU only: measure/fit/report existing runs")
    ap.add_argument("--attn-backend", choices=("auto", "cudnn", "flash", "efficient"), default="auto",
                    help="opt-in SDPA kernel priority for the transformer (speed only, not bit-identical); auto = unchanged")
    ap.add_argument("--compile-blocks", action="store_true",
                    help="opt-in regional torch.compile of the transformer blocks, once per generated video "
                         "(minutes of compile each; speed only, not bit-identical)")
    add_deadline_args(ap)
    args = ap.parse_args(argv)
    deadline = deadline_from_args(args)

    root = args.project_root.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    cal_out = args.calibration_out if args.calibration_out.is_absolute() else root / args.calibration_out
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    images = parse_images(args.images, root) if args.images else dev_images(root, args.n_images)
    for name, p in images.items():
        if not p.is_file():
            sys.exit(f"Image not found: {name} -> {p}")
    out.mkdir(parents=True, exist_ok=True)
    gain_key = "signed_motion_gain_pathwide" if args.gain_key == GAIN_KEY else "signed_motion_gain_endpoint"

    fit_needed = args.force or not cal_out.exists()
    if not fit_needed:
        print(f"{cal_out} exists and is kept (use --force to refit); --verify runs against it.")

    gen = None
    if not args.no_generate:
        if not checkpoint.is_file():
            sys.exit(f"Checkpoint not found: {checkpoint}")
        gen = Generator(root, checkpoint, args.steps, args.seed, args.guide_fill_mode,
                        attn_backend=args.attn_backend, compile_blocks=args.compile_blocks)
        gen.deadline = deadline
    errors = {}
    try:
        sweep = plan_jobs("sweep", out, images, args.directions, args.us)
        todo = pending(sweep)
        print(f"sweep: {len(sweep)} runs, {len(todo)} pending (steps={args.steps}, seed={args.seed})")
        if todo and gen is not None and fit_needed:
            errors.update(gen.run(todo, "sweep "))
            if gen.deadline_hit:  # never fit a calibration from a partial sweep
                return 0
        rows = collect_rows(sweep, gain_key)
        print(f"measured {len(rows)} sweep videos")
        cal = None
        diag = {}
        ck_sha = None
        if checkpoint.is_file():
            ck_sha = st.file_sha256(checkpoint)
        if fit_needed:
            if not rows:
                sys.exit("No measured sweep runs; nothing to fit.")
            cal, diag = fit_calibration(
                rows, checkpoint_sha256=ck_sha, max_amplitudes=st.load_max_amplitudes(root),
                extra_fit={"steps": args.steps, "seed": args.seed, "images": {k: str(v) for k, v in images.items()},
                           "us": list(args.us), "n_runs": len(rows), "gain_key": gain_key},
            )
            cal_out.parent.mkdir(parents=True, exist_ok=True)
            cal_out.write_text(json.dumps(cal, indent=2) + "\n", encoding="utf8")
            print(f"wrote {cal_out}")
        else:
            cal = json.loads(cal_out.read_text(encoding="utf8"))
            if rows:  # diagnostics for plots/report only; the file on disk is not rewritten
                _, diag = fit_calibration(rows, checkpoint_sha256=ck_sha)

        verify_rows, verify = [], None
        if args.verify:
            vjobs = plan_jobs("verify", out, images, args.directions, args.verify_s, calibration=str(cal_out))
            vtodo = pending(vjobs)
            print(f"verify: {len(vjobs)} runs, {len(vtodo)} pending")
            if vtodo and gen is not None:
                errors.update(gen.run(vtodo, "verify "))
                if gen.deadline_hit:
                    return 0
            verify_rows = collect_rows(vjobs, gain_key)
            if verify_rows:
                verify = verify_report(verify_rows)
                (out / "verify_report.json").write_text(json.dumps(clean(verify), indent=2), encoding="utf8")

        plots = out / "plots"
        plots.mkdir(exist_ok=True)
        for d in diag:
            plot_direction(plots / f"{d}.png", d, rows, diag[d], cal, verify_rows)
        meta = {"checkpoint": str(checkpoint), "checkpoint_sha256": ck_sha, "steps": args.steps, "seed": args.seed,
                "images": ", ".join(images), "calibration_file": str(cal_out)}
        md = markdown_report(cal, diag, rows, verify, meta)
        (out / "calibration_report.md").write_text(md, encoding="utf8")
        (out / "sweep_rows.json").write_text(json.dumps(clean(rows), indent=2), encoding="utf8")
        print(md)
    finally:
        if gen is not None:
            gen.close()
    if errors:
        (out / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf8")
        print(f"{len(errors)} runs failed; see {out / 'errors.json'}; re-run to retry them")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
