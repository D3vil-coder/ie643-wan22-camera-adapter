"""Ground-truth reveal evaluation on the v2 development split (Molab GPU for generation; metrics CPU/GPU).

For N balanced dev (source, motion) pairs the real clip is rendered exactly like the cache builder; each
model gets the GT first frame and the SAME homographies and generates a video (50 steps, fixed seed).
The newly revealed region R is then compared with the real content (LPIPS, PSNR, SSIM, Lab-ab histogram
distance, HF deviation, artifact_score_local), with the valid region V as a control. See docs/v2/EVALUATION.md.

    python scripts/eval_dev_reveal.py --project-root . --model v2=artifacts/v2_full/checkpoints/adapter_001000.safetensors
    python scripts/eval_dev_reveal.py --project-root .                      # default models: v1 and v1+rep only
    python scripts/eval_dev_reveal.py --project-root . --models v2 --model v2=...   # subset of the given models
    python scripts/eval_dev_reveal.py --project-root . --no-generate        # metrics/tables/sheets from existing runs
    python scripts/eval_dev_reveal.py --project-root . --dry-run             # selection + time estimate

``--model name=ckpt[:fill[:recipe]]`` (repeatable): fill in zeros|replicate|auto (auto = recorded by the
checkpoint), recipe in v1|default (v1 = camera_residual_anchor teacher + camera_min_sigma 0.6, the
Submission-1 recipe). Resumable: finished runs (run.json) and metrics are never recomputed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fpcamera import reveal_eval as re_  # noqa: E402
from fpcamera.flow_cache_v2 import SourceRejected, contract_of, load_config, load_sources  # noqa: E402
from fpcamera.session_deadline import add_deadline_args, deadline_from_args  # noqa: E402

V1_CHECKPOINT = "artifacts/flow_poc_all/adapter_000300.safetensors"
SEC_PER_VIDEO = 480.0
MODEL_RE = re.compile(r"^(?P<name>[^=:]+)=(?P<ckpt>.+?)(?::(?P<fill>zeros|replicate|auto)(?::(?P<recipe>v1|default))?)?$")


def clean(v):
    """Recursively convert numpy scalars / non-finite floats into JSON-safe values."""
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def parse_model(text):
    """Parse ``--model name=ckpt[:fill[:recipe]]`` into a model spec dict."""
    m = MODEL_RE.match(text)
    if not m:
        raise argparse.ArgumentTypeError(f"--model expects name=ckpt[:fill[:recipe]], got {text!r}")
    fill = m["fill"] or "auto"
    return {"name": m["name"], "ckpt": m["ckpt"], "fill": None if fill == "auto" else fill,
            "recipe": m["recipe"] or "default"}


def default_models():
    """The two default models: the v1 checkpoint with zeros fill and with replicate fill."""
    return [
        {"name": "v1", "ckpt": V1_CHECKPOINT, "fill": "zeros", "recipe": "v1"},
        {"name": "v1+rep", "ckpt": V1_CHECKPOINT, "fill": "replicate", "recipe": "v1"},
    ]


def generate_kwargs(model, request):
    """``generate()`` keyword arguments of a model spec (v1 checkpoints use the teacher anchor and sigma 0.6)."""
    kw = {"camera_request": request, "guide_fill_mode": model["fill"]}
    if model["recipe"] == "v1":
        kw.update(camera_residual_anchor="teacher", camera_min_sigma=0.6)
    return kw


def safe(name):
    """Filesystem-safe version of a model name."""
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", name)


def read_frames(video, indices):
    """Decode the listed frame indices of a video as RGB arrays."""
    import cv2

    want, out, i = set(int(x) for x in indices), {}, 0
    cap = cv2.VideoCapture(str(video))
    while want and True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            want.discard(i)
        i += 1
    cap.release()
    return out


def main(argv=None):
    """CLI entry: for every dev source/clip render the ground-truth reveal, generate each model's video from the GT first
    frame with the same homographies, score R vs GT (``fpcamera.reveal_eval``) and write json/markdown; resumable.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--config", type=Path, default=None, help="data_v2.json (default configs/data_v2.json)")
    ap.add_argument("--model", type=parse_model, action="append", default=None, dest="model_specs")
    ap.add_argument("--models", nargs="+", default=None, help="subset of model names to run/evaluate")
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--min-u", type=float, default=0.5)
    ap.add_argument("--frames", type=int, nargs="+", default=list(re_.DEFAULT_FRAMES))
    ap.add_argument("--frame-stride", type=int, default=None,
                    help="evaluate every k-th frame (e.g. 4) instead of --frames; sheets still use --frames")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/reveal_eval"))
    ap.add_argument("--lpips-device", default="cpu")
    ap.add_argument("--no-lpips", action="store_true", help="skip LPIPS (CPU debugging only)")
    ap.add_argument("--no-lease", action="store_true", help="do not take the GPU lease")
    ap.add_argument("--no-generate", action="store_true", help="metrics/tables/sheets from existing runs only")
    ap.add_argument("--dry-run", action="store_true", help="print the selection and the time estimate; exit")
    ap.add_argument("--sec-per-video", type=float, default=SEC_PER_VIDEO, help="for the time estimate")
    ap.add_argument("--attn-backend", choices=("auto", "cudnn", "flash", "efficient"), default="auto",
                    help="opt-in SDPA kernel priority for the transformer (speed only, not bit-identical); auto = unchanged")
    ap.add_argument("--compile-blocks", action="store_true",
                    help="opt-in regional torch.compile of the transformer blocks, once per generated video "
                         "(minutes of compile each; speed only, not bit-identical)")
    add_deadline_args(ap)
    args = ap.parse_args(argv)
    deadline = deadline_from_args(args)

    root = args.project_root.resolve()
    cfg = load_config(args.config or root / "configs" / "data_v2.json")
    contract = contract_of(cfg)
    manifest = root / cfg["sources"]["manifest"]
    records = load_sources(manifest)
    rec_by_id = {r["source_id"]: r for r in records}
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir

    models = args.model_specs or default_models()
    if args.models:
        unknown = set(args.models) - {m["name"] for m in models}
        if unknown:
            sys.exit(f"--models {sorted(unknown)} not among the defined models {[m['name'] for m in models]}")
        models = [m for m in models if m["name"] in args.models]
    for m in models:
        p = Path(m["ckpt"])
        m["ckpt_path"] = p if p.is_absolute() else root / p

    plans = re_.select_pairs(records, cfg, n=args.pairs, min_u=args.min_u)
    n_vid = len(plans) * len(models)
    print(f"{len(plans)} pairs x {len(models)} models = {n_vid} videos; "
          f"~{n_vid * args.sec_per_video / 3600:.1f} h GPU at {args.sec_per_video:.0f} s/video "
          f"(~{len(plans) * args.sec_per_video / 3600:.1f} h per model)")
    for p in plans:
        print(f"  {p.id}  family={p.family} dir={p.direction} u={p.u:.3f}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps(
        [dict(id=p.id, source_id=p.source_id, family=p.family, direction=p.direction, u=p.u) for p in plans],
        indent=2), encoding="utf8")
    if args.dry_run:
        return 0

    from PIL import Image

    frames_sheet = tuple(args.frames)
    eval_frames = tuple(range(0, contract.frames, args.frame_stride)) if args.frame_stride else frames_sheet
    eval_frames = tuple(sorted(set(eval_frames) | (set(frames_sheet) if args.frame_stride else set())))

    # ---- GT first frames + requests (CPU, ~1.5 s/pair), cached on disk
    pair_dir = lambda p: out / "pairs" / safe(p.id)  # noqa: E731
    gt_cache, usable = {}, []

    def gt_of(plan):
        """Ground-truth clip of a plan (re-rendered from the source photograph; only one clip is kept in memory)."""
        if plan.id not in gt_cache:
            image = Image.open(root / rec_by_id[plan.source_id]["source_image"]).convert("RGB")
            gt_cache.clear()  # keep one clip (~300 MB) in memory
            gt_cache[plan.id] = re_.render_gt(image, plan, cfg, contract)
        return gt_cache[plan.id]

    for p in plans:
        d = pair_dir(p)
        try:
            if not (d / "gt_first.png").is_file():
                rgb, h, overscan = gt_of(p)
                d.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb[0]).save(d / "gt_first.png")
                (d / "request.json").write_text(json.dumps(re_.camera_request_for(h, contract)), encoding="utf8")
                (d / "pair.json").write_text(json.dumps(
                    dict(id=p.id, source_id=p.source_id, family=p.family, direction=p.direction, u=p.u,
                         overscan=overscan)), encoding="utf8")
            usable.append(p)
        except SourceRejected as exc:
            print(f"skip {p.id}: source rejected by the cache builder ({exc})")

    # ---- generation
    jobs = []
    for m in models:
        for p in usable:
            run_dir = out / "models" / safe(m["name"]) / safe(p.id)
            if (run_dir / "run.json").is_file():
                continue
            request = json.loads((pair_dir(p) / "request.json").read_text(encoding="utf8"))
            jobs.append(dict(image=pair_dir(p) / "gt_first.png", run_dir=run_dir, checkpoint=m["ckpt_path"],
                             generate_kwargs=generate_kwargs(m, request), model=m["name"]))
    print(f"{len(jobs)} videos pending")
    if jobs and not args.no_generate:
        missing = sorted({str(j["checkpoint"]) for j in jobs if not Path(j["checkpoint"]).is_file()})
        if missing:
            sys.exit(f"checkpoint(s) not found: {missing}")
        import calibrate_strength as cs

        gen = cs.Generator(root, jobs[0]["checkpoint"], args.steps, args.seed, None,
                           log_name="reveal_eval_library.log", use_lease=not args.no_lease,
                           attn_backend=args.attn_backend, compile_blocks=args.compile_blocks)
        gen.deadline = deadline
        try:
            errors = gen.run(jobs)
        finally:
            gen.close()
        if errors:
            print(f"{len(errors)} run(s) failed; see output above")
        if gen.deadline_hit:
            print("DEADLINE_STOP: re-run the same command to continue (finished runs are kept)")
            return 0

    # ---- metrics
    lpips_fn = None
    if not args.no_lpips:
        try:
            lpips_fn = re_.LpipsScorer(args.lpips_device)
        except ImportError:
            sys.exit("pip install lpips (needed for LPIPS; use --no-lpips to skip)")
    records_out = []
    for m in models:
        for p in usable:
            run_dir = out / "models" / safe(m["name"]) / safe(p.id)
            if not (run_dir / "video.mp4").is_file():
                continue
            metrics_path = run_dir / "reveal_metrics.json"
            key = dict(frames=list(eval_frames), lpips=lpips_fn is not None)
            cached = json.loads(metrics_path.read_text(encoding="utf8")) if metrics_path.is_file() else None
            if cached is None or cached.get("key") != key:
                rgb, h, _ = gt_of(p)
                gen_frames = read_frames(run_dir / "video.mp4", eval_frames)
                got = [t for t in eval_frames if t in gen_frames]
                res = re_.evaluate_video(gen_frames, rgb, h, got, lpips_fn)
                cached = clean(dict(key=key, model=m["name"], pair=p.id, family=p.family, direction=p.direction,
                                    u=p.u, **res))
                metrics_path.write_text(json.dumps(cached, indent=2), encoding="utf8")
                print(f"[metrics] {m['name']} {p.id}")
            cached["mean"] = {k: v for k, v in cached["mean"].items()}
            records_out.append(dict(model=m["name"], pair=p.id, family=p.family, direction=p.direction, u=p.u,
                                    mean=_nan(cached["mean"])))
    agg = re_.aggregate(records_out)
    result = dict(
        protocol=dict(pairs=[p.id for p in usable], steps=args.steps, seed=args.seed, frames=list(eval_frames),
                      models=[dict(name=m["name"], ckpt=str(m["ckpt"]), fill=m["fill"], recipe=m["recipe"])
                              for m in models], min_u=args.min_u),
        per_pair=records_out, aggregate=agg)
    (out / "reveal_eval.json").write_text(json.dumps(clean(result), indent=2), encoding="utf8")
    table = re_.markdown(agg)
    (out / "reveal_eval.md").write_text(table, encoding="utf8")
    print(table)

    # ---- contact sheets (GT / models; columns frames 40/80/120)
    for p in usable:
        rows_dirs = [(m["name"], out / "models" / safe(m["name"]) / safe(p.id) / "video.mp4") for m in models]
        rows_dirs = [(n, v) for n, v in rows_dirs if v.is_file()]
        sheet = out / "sheets" / f"{safe(p.id)}.png"
        if not rows_dirs or sheet.is_file():
            continue
        rgb, h, _ = gt_of(p)
        rows = [("GT", {t: rgb[t] for t in frames_sheet})]
        for n, v in rows_dirs:
            fr = read_frames(v, frames_sheet)
            if all(t in fr for t in frames_sheet):
                rows.append((n, fr))
        re_.contact_sheet(rows, h, frames_sheet, sheet)
    return 0


def _nan(mean):
    """JSON None -> NaN so aggregation can ignore it."""
    def conv(v):
        """Replace None by NaN recursively (for averaging)."""
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        return float("nan") if v is None else v
    return conv(mean)


if __name__ == "__main__":
    raise SystemExit(main())
