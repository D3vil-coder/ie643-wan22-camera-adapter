"""Fit the strength calibration from existing v2 videos of any origin (CPU; no GPU sweep).

``scripts/calibrate_strength.py`` generates its own 72-video sweep. This script instead pools **every finished v2
video that is already available** (e.g. the G1 and E1 evaluation videos: different images, amplitudes requested by the
slider or as physical amplitudes), measures each with the same tracker (``calibrate_strength.measure_run_gain``, cached
in ``measure.json``) and fits the same per-direction isotonic map (``calibrate_strength.fit_calibration``), so the file
format and the run-time use are identical.

Selection of videos: a run folder (``run.json`` + ``request.json`` + ``video.mp4``) is used when its adapter hash starts
with ``--checkpoint-sha``, its guide fill is ``replicate`` and it has a positive requested strength ``u``. ``--exclude``
drops folders whose path contains a substring (e.g. a duplicate run of the same video with another attention kernel). Two
runs of the same input, direction and ``u`` count once (first wins).

Directions with too little data fall back to the identity map (``u = s``): a direction is fitted only when it has at least
``--min-videos`` videos on at least ``--min-levels`` distinct ``u`` levels and a level >= ``--top-level`` (so the top of the
slider is measured); otherwise it stays the identity and the report says so.

    python scripts/fit_calibration_from_runs.py --root G1 --root E1/showcase --exclude attn_cudnn \\
        --checkpoint-sha 83df1e3f --out configs/strength_calibration_v2.json --report-md G1/calibration_fit.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibrate_strength as cs  # noqa: E402
from fpcamera import strength as st  # noqa: E402

S_GRID = np.round(np.linspace(0.0, 1.0, 21), 6)
FAMILY = {"pan_left": "pan", "pan_right": "pan", "tilt_up": "tilt", "tilt_down": "tilt", "zoom_in": "zoom", "zoom_out": "zoom"}


def find_runs(roots):
    """Run folders below the roots that contain ``video.mp4``, ``request.json`` and ``run.json``."""
    out = []
    for root in roots:
        out += sorted(p.parent for p in Path(root).rglob("run.json")
                      if (p.parent / "video.mp4").is_file() and (p.parent / "request.json").is_file())
    return out


def collect(roots, checkpoint_sha, exclude=(), measure=True):
    """Rows ``{run_dir, direction, u, gain, a_obs, ...}`` of the usable runs, plus the list of skipped ones with reasons."""
    rows, skipped, seen = [], [], set()
    for rd in find_runs(roots):
        sp = str(rd).replace("\\", "/")
        run = json.loads((rd / "run.json").read_text(encoding="utf8"))
        why = None
        if any(x in sp for x in exclude):
            why = "excluded"
        elif not str(run.get("checkpoint_sha256", "")).startswith(checkpoint_sha):
            why = "other checkpoint"
        elif run.get("guide_fill_mode") != "replicate":
            why = f"guide fill {run.get('guide_fill_mode')}"
        if why:
            skipped.append((sp, why))
            continue
        m = cs.measure_run_gain(rd) if measure else json.loads((rd / "measure.json").read_text(encoding="utf8"))
        direction = m.get("motion") or run.get("mode")
        u, g = m.get("strength_u"), m.get("gain")
        if direction not in st.DIRECTIONS or u is None or g is None or not u > 0:
            skipped.append((sp, f"no usable direction/u/gain ({direction}, {u}, {g})"))
            continue
        key = (run.get("input_sha256"), direction, round(float(u), 4))
        if key in seen:
            skipped.append((sp, "duplicate of an earlier run (same input, direction, u)"))
            continue
        seen.add(key)
        sc = run.get("strength_control") or {}
        rows.append({
            "run_dir": sp, "direction": direction, "family": FAMILY[direction], "u": float(u),
            "s": m.get("strength_s"), "gain": float(g), "cosine": m.get("cosine_pathwide"),
            "a_obs": float(g) * float(u), "source": sc.get("amplitude_source"), "label": sc.get("label"),
            "input": str(run.get("input_sha256", ""))[:8],
        })
    return rows, skipped


def identity_entry():
    """The identity map entry (s = u on the standard grid) used for directions without usable data."""
    return {"s": [float(x) for x in S_GRID], "u": [float(x) for x in S_GRID]}


LEVEL_TOL = 0.02  # u levels closer than this (e.g. 0.500 and 0.508) count as one level for the decision rule


def n_levels(us, tol=LEVEL_TOL):
    """Number of distinct u levels after merging values closer than ``tol``."""
    n, last = 0, None
    for u in sorted(us):
        if last is None or u - last > tol:
            n += 1
        last = u
    return n


def decide(rows, min_videos, min_levels, top_level):
    """Per direction: ``(fit_rows or None, reason)``; None = identity fallback."""
    out = {}
    for d in st.DIRECTIONS:
        pts = [r for r in rows if r["direction"] == d]
        levels = sorted({round(r["u"], 4) for r in pts})
        k = n_levels(levels)
        if len(pts) < min_videos:
            out[d] = (None, f"only {len(pts)} video(s) (< {min_videos})")
        elif k < min_levels:
            out[d] = (None, f"only {k} u level(s) (< {min_levels})")
        elif max(levels) < top_level:
            out[d] = (None, f"no level >= {top_level} (max u {max(levels):.2f})")
        else:
            out[d] = (pts, f"{len(pts)} videos on {k} u levels")
    return out


def build(rows, decisions, checkpoint_sha256, extra_fit):
    """Fit + invert every direction from the measured rows (identity where data are insufficient); returns ``(directions, diagnostics)``."""
    directions, diag = {}, {}
    for d in st.DIRECTIONS:
        pts, _ = decisions[d]
        if pts is None:
            directions[d] = identity_entry()
            continue
        u = np.array([r["u"] for r in pts])
        a = np.array([r["a_obs"] for r in pts])
        entry, info = st.build_direction_entry(u, a, s_grid=S_GRID)
        info["n_points"] = len(pts)
        info["mean_gain"] = float(np.mean([r["gain"] for r in pts]))
        info["per_u_mean_a_obs"] = {f"{x:.4f}": float(np.mean(a[np.isclose(u, x)])) for x in sorted(set(np.round(u, 6)))}
        directions[d] = entry
        diag[d] = info
    fit = {
        "method": "isotonic (pool-adjacent-violators) a_obs(u) through the origin, inverted to u(s); "
                  "fitted from pre-existing v2 videos (G1/E1), not from a 72-video sweep; identity for under-sampled directions",
        "observed_amplitude": "a_obs = g * u, g = signed_motion_gain_pathwide (tracker vs requested homographies)",
        "per_direction": diag,
        "identity_fallback": {d: decisions[d][1] for d in st.DIRECTIONS if decisions[d][0] is None},
        **extra_fit,
    }
    return st.new_calibration(directions, checkpoint_sha256=checkpoint_sha256,
                              max_amplitudes=st.load_max_amplitudes(ROOT), fit=fit), diag


def report(rows, skipped, decisions, cal, diag, meta):
    """Markdown report of the videos used, the fit per direction and the resulting map."""
    sm = st.StrengthMap(cal)
    L = ["# Strength calibration fit from existing v2 videos", ""]
    L += [f"- {k}: {v}" for k, v in meta.items()] + [""]
    L += ["## Videos used", "", "| direction | u | requested | gain g | cosine | a_obs = g u | input | run |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["direction"], r["u"], r["run_dir"])):
        L.append(f"| {r['direction']} | {r['u']:.4f} | {r['label'] or ''} | {r['gain']:.3f} | "
                 f"{(r['cosine'] if r['cosine'] is not None else float('nan')):.3f} | {r['a_obs']:.3f} | {r['input']} | `{r['run_dir'].split('/')[-3:][0] + '/.../' + '/'.join(r['run_dir'].split('/')[-2:])}` |")
    L += ["", f"{len(rows)} videos used, {len(skipped)} skipped.", ""]
    L += ["## Gain per family versus requested strength u", "",
          "Mean signed path-wide gain (1 = the requested motion is realised); n = videos. A gain above 1 over-delivers.", "",
          "| family | direction | u level | n | mean g | min g | max g |", "|---|---|---|---|---|---|---|"]
    for fam in ("pan", "tilt", "zoom"):
        for d in [x for x in st.DIRECTIONS if FAMILY[x] == fam]:
            for lv in sorted({round(r["u"], 4) for r in rows if r["direction"] == d}):
                g = [r["gain"] for r in rows if r["direction"] == d and round(r["u"], 4) == lv]
                L.append(f"| {fam} | {d} | {lv:.4f} | {len(g)} | {np.mean(g):.3f} | {np.min(g):.3f} | {np.max(g):.3f} |")
    L += ["", "Family pooled (both directions of a family):", "", "| family | u level | n | mean g |", "|---|---|---|---|"]
    for fam in ("pan", "tilt", "zoom"):
        for lv in sorted({round(r["u"], 4) for r in rows if r["family"] == fam}):
            g = [r["gain"] for r in rows if r["family"] == fam and round(r["u"], 4) == lv]
            L.append(f"| {fam} | {lv:.4f} | {len(g)} | {np.mean(g):.3f} |")
    L += ["", "## Fit decision per direction", "", "| direction | decision | n | mean g | a_obs at the u knots (isotonic) | max a_obs | saturates_at |",
          "|---|---|---|---|---|---|---|"]
    for d in st.DIRECTIONS:
        pts, why = decisions[d]
        if pts is None:
            L.append(f"| {d} | **identity** ({why}) | {len([r for r in rows if r['direction'] == d])} | | | | |")
        else:
            i = diag[d]
            knots = ", ".join(f"u={u:.3f}: {a:.3f}" for u, a in zip(i["u_knots"][1:], i["a_obs_isotonic"][1:]))
            sat = f"{i['saturates_at']:.3f}" if i["saturates_at"] else "-"
            L.append(f"| {d} | fitted ({why}) | {i['n_points']} | {i['mean_gain']:.3f} | {knots} | {i['a_obs_max']:.3f} | {sat} |")
    L += ["", "## Fitted map u(s)", "", "| direction | " + " | ".join(f"u(s={s:g})" for s in (0.25, 0.5, 0.75, 1.0)) + " | max abs(u - s) |", "|---|---|---|---|---|---|"]
    for d in st.DIRECTIONS:
        us = [sm.s_to_u(d, s) for s in (0.25, 0.5, 0.75, 1.0)]
        dev = max(abs(sm.s_to_u(d, s) - s) for s in S_GRID)
        L.append(f"| {d}{' (identity)' if decisions[d][0] is None else ''} | " + " | ".join(f"{x:.3f}" for x in us) + f" | {dev:.3f} |")
    before, after = [], []
    for d in st.DIRECTIONS:
        if d in diag:
            uk, ak = np.array(diag[d]["u_knots"]), np.array(diag[d]["a_obs_isotonic"])
            for r in rows:
                if r["direction"] == d:
                    before.append(abs(r["a_obs"] - r["u"]))
                    after.append(abs(r["a_obs"] - float(np.interp(r["u"], uk, ak))))
    if before:
        L += ["", "## In-sample residuals (no new GPU verification run was possible)", "",
              f"Mean |a_obs - u| of the {len(before)} videos with no calibration (u = s): **{np.mean(before):.3f}** (max {np.max(before):.3f}). "
              f"Mean |a_obs - A(u)| around the fitted isotonic curve: {np.mean(after):.3f} (max {np.max(after):.3f}). "
              "The second number is an in-sample fit residual, not a held-out verification; it is bounded below by the image-to-image "
              "scatter of the gain at equal u (each knot is the mean of 1 to 3 videos)."]
    if skipped:
        L += ["", "## Skipped runs", ""] + [f"- `{p}`: {w}" for p, w in skipped]
    return "\n".join(L) + "\n"


def plots(rows, cal, diag, out_dir):
    """One requested-vs-observed plot per direction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sm = st.StrengthMap(cal)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), constrained_layout=True)
    cols = {"pan_left": "#1f6f8b", "pan_right": "#c0504d", "tilt_up": "#1f6f8b", "tilt_down": "#c0504d",
            "zoom_in": "#1f6f8b", "zoom_out": "#c0504d"}
    for j, fam in enumerate(("pan", "tilt", "zoom")):
        ax = axes[0, j]
        ax.axhline(1.0, color="#888", ls="--", lw=1)
        for d in [x for x in st.DIRECTIONS if FAMILY[x] == fam]:
            pts = [r for r in rows if r["direction"] == d]
            ax.scatter([r["u"] for r in pts], [r["gain"] for r in pts], s=36, color=cols[d], alpha=0.8, label=d)
        ax.set_title(f"{fam}: tracker gain g vs requested u")
        ax.set_xlabel("requested strength u")
        ax.set_ylabel("gain g (1 = requested motion realised)")
        ax.set_xlim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax = axes[1, j]
        ax.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="ideal")
        sg = np.linspace(0, 1, 101)
        for d in [x for x in st.DIRECTIONS if FAMILY[x] == fam]:
            ax.plot(sg, [sm.s_to_u(d, s) for s in sg], color=cols[d], lw=1.8, label=f"u(s) {d}")
        ax.set_title(f"{fam}: calibration map s -> u")
        ax.set_xlabel("slider s")
        ax.set_ylabel("requested u")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(out_dir / "calibration_fit.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    """CLI entry: collect runs -> measure gains -> fit -> write the calibration json, report and plots (CPU)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, action="append", required=True, help="folder scanned recursively for run folders")
    ap.add_argument("--exclude", action="append", default=[], help="skip run folders whose path contains this text")
    ap.add_argument("--checkpoint-sha", required=True, help="prefix of the adapter sha256 the videos must come from")
    ap.add_argument("--out", type=Path, default=ROOT / "configs" / "strength_calibration_v2.json")
    ap.add_argument("--report-md", type=Path, default=None)
    ap.add_argument("--rows-json", type=Path, default=None)
    ap.add_argument("--plots-dir", type=Path, default=None)
    ap.add_argument("--min-videos", type=int, default=3)
    ap.add_argument("--min-levels", type=int, default=2)
    ap.add_argument("--top-level", type=float, default=0.9)
    ap.add_argument("--note", default="", help="free text stored in the fit block (what the data are)")
    args = ap.parse_args(argv)
    rows, skipped = collect(args.root, args.checkpoint_sha, args.exclude)
    if not rows:
        sys.exit("no usable runs")
    sha = None
    for rd in (Path(r["run_dir"]) for r in rows):
        sha = json.loads((rd / "run.json").read_text(encoding="utf8")).get("checkpoint_sha256")
        break
    decisions = decide(rows, args.min_videos, args.min_levels, args.top_level)
    extra = {"steps": 50, "seed": 42, "n_runs": len(rows), "gain_key": cs.GAIN_KEY, "note": args.note,
             "rules": {"min_videos": args.min_videos, "min_levels": args.min_levels, "top_level": args.top_level},
             "runs": [{"direction": r["direction"], "u": round(r["u"], 4), "gain": round(r["gain"], 4),
                       "run": "/".join(r["run_dir"].split("/")[-4:])} for r in rows]}
    cal, diag = build(rows, decisions, sha, extra)
    args.out.write_text(json.dumps(cal, indent=2) + "\n", encoding="utf8")
    meta = {"checkpoint_sha256": sha, "roots": ", ".join(str(r) for r in args.root), "calibration_file": str(args.out),
            "rules": f"fit a direction only with >= {args.min_videos} videos, >= {args.min_levels} u levels and a level >= {args.top_level}"}
    md = report(rows, skipped, decisions, cal, diag, meta)
    print(md)
    if args.report_md:
        args.report_md.write_text(md, encoding="utf8")
    if args.rows_json:
        args.rows_json.write_text(json.dumps(cs.clean(rows), indent=2) + "\n", encoding="utf8")
    if args.plots_dir:
        plots(rows, cal, diag, args.plots_dir)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
