"""Overfitting monitor: held-out (dev) vs train-subset losses of v2 checkpoints at fixed evaluation noise.

  python scripts/validate_v2.py --run-dir artifacts/v2_full --cache data/v2/cache_1280x704_f121_v2/index.jsonl \
      --steps 77,250,477            # or --all-local, or --checkpoint PATH[=LABEL] (repeatable)

Writes <run-dir>/validation/{validation.json, validation.md, validation.png, rows_<step>.json}.
Rerunning skips checkpoints already in validation.json. Exit 0 also on a clean DEADLINE_STOP.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    """CLI entry: evaluate saved checkpoints on dev vs train-subset at fixed noise via ``flow_validate_v2`` (resumable)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, help="training run dir (config.json, checkpoints/)")
    ap.add_argument("--cache", type=Path, required=True, help="v2 cache index.jsonl (needs development + train rows)")
    ap.add_argument("--steps", help="comma-separated update numbers in <run-dir>/checkpoints, e.g. 77,250,477")
    ap.add_argument("--all-local", action="store_true", help="every adapter_*.safetensors in <run-dir>/checkpoints")
    ap.add_argument("--checkpoint", action="append", default=[], metavar="PATH[=LABEL]",
                    help="explicit adapter .safetensors (repeatable)")
    ap.add_argument("--out-dir", type=Path, help="default <run-dir>/validation")
    ap.add_argument("--draws", type=int, default=1, help="fixed (sigma, eps) draws per row (default 1; 4 = 4x slower)")
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--train-sources", type=int, default=28, help="size of the fixed random train-source subset")
    ap.add_argument("--dev-moving-per-source", type=int, default=0, help="0 = all moving rows of each dev source")
    ap.add_argument("--train-moving-per-source", type=int, default=1, help="moving rows per train-subset source")
    ap.add_argument("--teacher-cache-gib", type=float, default=16.0, help="CPU RAM for cached teacher outputs")
    ap.add_argument("--est-checkpoint-minutes", type=float, default=10.0,
                    help="initial per-checkpoint time estimate for the deadline rule")
    ap.add_argument("--attn-backend", choices=("auto", "cudnn", "flash", "efficient"), default="auto",
                    help="opt-in SDPA kernel priority for the forward passes (speed only, kernel-level, not "
                         "bit-identical to earlier rows in validation.json); auto = unchanged")
    from fpcamera.session_deadline import add_deadline_args, deadline_from_args

    add_deadline_args(ap)
    args = ap.parse_args(argv)
    from fpcamera.flow_validate_v2 import discover_checkpoints, run_validation

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    steps = [int(s) for s in args.steps.split(",") if s.strip()] if args.steps else []
    cps = discover_checkpoints(run_dir, steps, args.all_local, args.checkpoint)
    if not cps:
        ap.error("no checkpoints selected (use --steps, --all-local or --checkpoint)")
    status = run_validation(
        run_dir=run_dir, index=args.cache if args.cache.is_absolute() else ROOT / args.cache,
        out_dir=args.out_dir, checkpoints=cps, draws=args.draws, seed=args.seed,
        n_train_sources=args.train_sources, dev_moving_per_source=args.dev_moving_per_source,
        train_moving_per_source=args.train_moving_per_source, teacher_cache_gib=args.teacher_cache_gib,
        deadline=deadline_from_args(args), est_checkpoint_seconds=args.est_checkpoint_minutes * 60,
        attn_backend=args.attn_backend,
        progress=lambda line: print(line, flush=True))
    print(status, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
