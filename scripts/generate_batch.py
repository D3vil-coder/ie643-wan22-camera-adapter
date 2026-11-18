"""Queue runner for GPU generation: one loaded Wan pipeline, jobs dropped into a folder (Molab GPU).

The pipeline (about a minute to load) is loaded once; then the runner repeatedly scans ``--queue-dir`` for
``*.json`` job files (each a JSON list of jobs), runs the first pending job (files in name order, jobs in list
order, so a file named ``00_*.json`` pre-empts ``50_*.json``), and polls for new files when nothing is pending.
It exits when everything is done and ``<queue-dir>/STOP`` exists. A job whose ``run_dir/run.json`` exists is
finished and never regenerated, so the runner is resumable and idempotent. A job that raised is not retried
until the next launch.

Job fields (all paths relative to ``--project-root`` unless absolute)::

    {"image": "examples/inputs/ball_pan_left.png",
     "run_dir": "artifacts/g1/matched/v2/ball_pan_left",
     "checkpoint": "artifacts/flow_poc_all/adapter_000300.safetensors",     # optional, default --checkpoint
     "generate_kwargs": {"motion": "pan_left", "amplitude_deg": 2.5}}       # any flow_infer.generate argument

``generate_kwargs`` examples: v2 slider ``{"motion": "pan_right", "strength": 1.0, "calibration": null}``;
v2 physical amplitude ``{"motion": "zoom_out", "zoom_ratio": 1.12}`` (``amplitude_deg`` for pan/tilt);
v1 recipe ``{"mode": "pan_right", "magnitude": 0.5, "guide_fill_mode": "zeros", "camera_residual_anchor": "teacher",
"camera_min_sigma": 0.6}``; kernel choice ``{"attn_backend": "cudnn"}``.

    python scripts/generate_batch.py --project-root . --queue-dir artifacts/g1/queue \\
        --checkpoint artifacts/v2_full/checkpoints/adapter_000415.safetensors --steps 50 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibrate_strength as cs  # noqa: E402


def _abs(root, p):
    p = Path(p)
    return p if p.is_absolute() else root / p


def pending_jobs(root, queue_dir, failed):
    """Pending jobs in priority order: ``[(job_file_name, index, job)]`` (finished and failed ones skipped)."""
    out = []
    for f in sorted(Path(queue_dir).glob("*.json")):
        try:
            jobs = json.loads(f.read_text(encoding="utf8"))
        except (OSError, ValueError) as exc:  # half-written file: look again next scan
            print(f"[queue] cannot read {f.name}: {exc}", flush=True)
            continue
        for i, job in enumerate(jobs):
            run_dir = _abs(root, job["run_dir"])
            if (run_dir / "run.json").is_file() or str(run_dir) in failed:
                continue
            out.append((f.name, i, job))
    return out


def main(argv=None):
    """CLI entry: load the pipeline once, then repeatedly run the first pending job of the ``*.json`` job files in ``--queue-dir`
    `` (finished runs have ``run.json`` and are never repeated); polls for new files and exits when idle and ``STOP`` exists.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--queue-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True, help="default adapter for jobs without 'checkpoint'")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--poll-seconds", type=float, default=10.0)
    ap.add_argument("--no-lease", action="store_true", help="do not take the GPU lease")
    args = ap.parse_args(argv)

    root = args.project_root.resolve()
    queue = _abs(root, args.queue_dir)
    queue.mkdir(parents=True, exist_ok=True)
    ckpt = _abs(root, args.checkpoint)
    if not ckpt.is_file():
        sys.exit(f"checkpoint not found: {ckpt}")
    gen = cs.Generator(root, ckpt, args.steps, args.seed, None, log_name="generate_batch_library.log",
                       use_lease=not args.no_lease)
    failed, n_done, t_start = {}, 0, time.time()
    try:
        while True:
            todo = pending_jobs(root, queue, failed)
            if not todo:
                if (queue / "STOP").exists():
                    print(f"[queue] STOP and nothing pending: exiting after {n_done} jobs "
                          f"({(time.time() - t_start) / 60:.1f} min)", flush=True)
                    break
                time.sleep(args.poll_seconds)
                continue
            fname, idx, job = todo[0]
            run_dir = _abs(root, job["run_dir"])
            gj = {"image": _abs(root, job["image"]), "run_dir": run_dir,
                  "generate_kwargs": dict(job.get("generate_kwargs", {}))}
            if job.get("checkpoint"):
                gj["checkpoint"] = _abs(root, job["checkpoint"])
            if job.get("prompt"):
                gj["prompt"] = job["prompt"]
            print(f"[queue] {time.strftime('%H:%M:%S')} {fname}#{idx} -> {run_dir} ({len(todo)} pending)", flush=True)
            errors = gen.run([gj], label="job ")
            if errors:
                failed.update({k: True for k in errors})
            else:
                n_done += 1
    finally:
        gen.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
