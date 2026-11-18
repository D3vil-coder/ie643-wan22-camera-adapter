"""v2 hybrid-objective trainer launcher (fresh start, warm start from v1, auto-resume).

Direct:      python scripts/train_v2.py --config configs/train_v2.json --run-dir artifacts/v2_full --stop-after-updates 150
             (--cache INDEX[:WEIGHT] repeatable; --init-from v1 adapter .safetensors for the warm start)
Notebook:    subprocess.run([sys.executable, "scripts/train_v2.py", ...]) in a loop until the return code is 0.

Exit codes: 0 = target reached, --check-only ok, or clean DEADLINE_STOP (run_status.json stopped_reason); 75 = interrupted, checkpoint saved, rerun the
SAME command to continue; anything else = failure (rerunning resumes from the last checkpoint).
Rerunning an identical command is always safe: if <run-dir>/latest.json exists the run resumes
(optimizer, schedule, RNG and data order included) unless --no-resume is given.

Speed-only options (never part of the run identity, defaults = unchanged behaviour): --ckpt-blocks K,
--attn-backend {auto,cudnn,flash,efficient}, --compile-blocks; pick them with --check-only --memory-probe or
--check-only --speed-probe (see docs/v2/TRAINING_V2.md section 7).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_cache(text: str):
    """Parse ``--cache INDEX[:WEIGHT]`` into ``(path, weight)`` (weight defaults to 1.0)."""
    path, weight = text, 1.0
    if ":" in text:
        head, tail = text.rsplit(":", 1)
        try:
            path, weight = head, float(tail)
        except ValueError:
            pass  # a Windows drive letter, not a weight
    return {"index": path, "weight": weight}


def _list_arg(parser_fn):
    """argparse ``type=`` adapter: speed_options parsers raise ValueError."""

    def conv(text):
        """argparse type converting a comma list with the given parser."""
        try:
            return parser_fn(text)
        except ValueError as e:
            raise argparse.ArgumentTypeError(str(e)) from e

    return conv


def build_parser() -> argparse.ArgumentParser:
    """Argument parser: run dir, config, caches, warm start, resume/identity flags, deadline flags, speed options
    (``--ckpt-blocks``, ``--attn-backend``, ``--compile-blocks``, probes).
    """
    from fpcamera import speed_options as so

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--cache", action="append", default=[], metavar="INDEX[:WEIGHT]",
                    help="cache index.jsonl (v1 or v2 schema) with optional mixing weight; repeatable. "
                         "Overrides config 'caches'.")
    ap.add_argument("--init-from", type=Path, help="v1 adapter .safetensors for a warm start (fresh run only)")
    ap.add_argument("--stop-after-updates", type=int, help="stop at this global update (schedule length unchanged)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore (archive, never delete) existing checkpoints in --run-dir and start over")
    ap.add_argument("--check-only", action="store_true",
                    help="load everything, run one forward/backward without an optimizer step, print memory/time")
    ap.add_argument("--ckpt-blocks", type=int, default=None, metavar="K",
                    help="activation-checkpoint only the first K of 30 blocks (30 = all = default behaviour, 0 = none). "
                         "Speed/memory only; not part of the run identity, so resuming with another K is allowed")
    ap.add_argument("--memory-probe", action="store_true",
                    help="with --check-only: one forward+backward (both passes) at ckpt-blocks 30/20/15/10/0, print "
                         "peak GiB and seconds per setting and the fastest one under --probe-limit-gib")
    ap.add_argument("--probe-limit-gib", type=float, default=85.0)
    ap.add_argument("--attn-backend", choices=so.ATTN_BACKENDS, default="auto",
                    help="SDPA kernel priority for the transformer forward/backward (incl. checkpoint recompute): auto = "
                         "torch default (unchanged); cudnn = cuDNN, then flash, then memory-efficient; flash / efficient "
                         "= that kernel only (fails loudly if unsupported). Kernel-level change, not bit-identical; "
                         "speed only, not part of the run identity")
    ap.add_argument("--compile-blocks", action="store_true",
                    help="regional torch.compile of every transformer block (mode=default, dynamic=False), inside the "
                         "activation-checkpoint wrapper. One-off compile time is logged separately (compile_first_update). "
                         "Kernel-level change, not bit-identical; speed only, not part of the run identity")
    ap.add_argument("--speed-probe", action="store_true",
                    help="with --check-only: time one full update on one fixed batch for a grid of --probe-ckpt x "
                         "--probe-attn x --probe-compile (discarded warm-up per row), compare loss_A/loss_B/grad-norm "
                         "with the baseline row (K=20, auto, no compile) and print the recommended flags. Use "
                         "--init-from <trained adapter> so the gradient check is meaningful")
    ap.add_argument("--probe-ckpt", type=_list_arg(so.parse_int_list), default=so.SpeedProbeConfig().ckpt_blocks,
                    metavar="K[,K]", help="speed-probe grid: checkpointed-block counts (default 20,15)")
    ap.add_argument("--probe-attn", type=_list_arg(so.parse_attn_list), default=so.SpeedProbeConfig().attn_backends,
                    metavar="B[,B]", help="speed-probe grid: attention backends (default auto,cudnn)")
    ap.add_argument("--probe-compile", type=_list_arg(so.parse_onoff_list), default=so.SpeedProbeConfig().compile,
                    metavar="off|on[,..]", help="speed-probe grid: block compilation (default off,on)")
    ap.add_argument("--probe-budget-minutes", type=float, default=so.SpeedProbeConfig().budget_minutes,
                    help="no new speed-probe row starts after this many minutes (default 20; the baseline always runs)")
    ap.add_argument("--probe-timed-updates", type=int, default=1,
                    help="measured updates per speed-probe row after the discarded warm-up (default 1)")
    ap.add_argument("--probe-equiv-tol", type=float, default=so.EQUIV_TOL,
                    help="relative loss/grad-norm difference above which a row is flagged NOT lossless (default 1e-2)")
    ap.add_argument("--break-lock", action="store_true", help="remove a stale artifacts/gpu.lock")
    ap.add_argument("--allow-config-change", action="store_true",
                    help="resume even though the training identity differs from the checkpoint's")
    ap.add_argument("--allow-identity-change", default="", metavar="KEY[,KEY]",
                    help="on resume, permit only these run-identity keys to differ (e.g. caches); old/new values are "
                         "logged and recorded in launches.jsonl. Narrower than --allow-config-change")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                    help="override a top-level config key, e.g. --set updates=300 --set fill_mode='\"zeros\"'")
    from fpcamera.session_deadline import add_deadline_args

    add_deadline_args(ap)
    return ap


def main(argv=None) -> int:
    """CLI entry: build the config, resolve the deadline and call ``flow_train_v2.run_training``; exits 0 on complete / clean
    deadline stop, 75 when interrupted after saving a checkpoint (relaunch the same command).
    """
    ap = build_parser()
    args = ap.parse_args(argv)
    from fpcamera.session_deadline import deadline_from_args

    deadline = deadline_from_args(args)

    if args.memory_probe and not args.check_only:
        ap.error("--memory-probe requires --check-only")
    if args.speed_probe and not args.check_only:
        ap.error("--speed-probe requires --check-only")
    if args.speed_probe and args.memory_probe:
        ap.error("--speed-probe and --memory-probe are separate runs; pick one")
    speed_cfg = None
    if args.speed_probe:
        from fpcamera import speed_options as so

        try:
            speed_cfg = so.SpeedProbeConfig(
                ckpt_blocks=args.probe_ckpt, attn_backends=args.probe_attn, compile=args.probe_compile,
                limit_gib=args.probe_limit_gib, budget_minutes=args.probe_budget_minutes,
                equiv_tol=args.probe_equiv_tol, timed_updates=args.probe_timed_updates)
        except ValueError as e:
            ap.error(str(e))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for item in args.set:
        key, _, value = item.partition("=")
        config[key] = json.loads(value)
    config.setdefault("project_root", str(ROOT))

    from fpcamera.flow_train_v2 import EXIT_COMPLETE, EXIT_INTERRUPTED, run_training

    status = run_training(
        config,
        run_dir=args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir,
        caches=[parse_cache(c) for c in args.cache] or None,
        init_from=args.init_from,
        stop_after=args.stop_after_updates,
        resume=not args.no_resume,
        check_only=args.check_only,
        break_lock=args.break_lock,
        allow_config_change=args.allow_config_change,
        allow_identity_change=args.allow_identity_change,
        deadline=deadline,
        ckpt_blocks=args.ckpt_blocks,
        probe=args.memory_probe,
        probe_limit_gib=args.probe_limit_gib,
        attn_backend=args.attn_backend,
        compile_blocks=args.compile_blocks,
        speed_probe_config=speed_cfg,
        progress=lambda line: print(line, flush=True),
    )
    print(json.dumps({"final": {k: v for k, v in status.items() if k != "micro"}}, default=str), flush=True)
    if status["status"] in ("complete", "check_only", "deadline"):
        return EXIT_COMPLETE
    if status["status"] == "interrupted":
        return EXIT_INTERRUPTED
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
