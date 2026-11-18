# Session chunking runbook (Molab, ~3 h hard kill)

Molab sandboxes are hard-killed about 3 h after they start (observed 2026-11-16: ~17:40 to 20:42 IST). The v2 run
(1000 updates at ~35-41 s/update) therefore runs in chunks of one session each. Every chunk must stop cleanly so the
last checkpoint is complete and on HF.

## Budget per session

| item | minutes |
|---|---|
| sandbox lifetime | 180 |
| restore env / cache / run (`tools/molab_recover.py`), model load, probe | 15-20 |
| deadline margin (`--deadline-margin-minutes`) | 15 |
| per-stop reserve (last update + checkpoint write + 90 s upload grace, enforced by the stop rule) | ~3-4 |
| usable training time | ~130-135 (about 2.2 h) |
| updates per session at 35-41 s | ~200-220 |
| sessions for 1000 updates | ~5 |

The sandbox age at session start is not necessarily 0 (a sandbox may have been up a while before you attach). The
trainer measures it itself (`/proc/1` start time; fallback env `MOLAB_SESSION_START` = unix seconds; fallback process
start, which is the least safe). Check it at the start of the session:

```bash
python - <<'EOF'
import sys, time; sys.path.insert(0, "src")
from fpcamera.session_deadline import sandbox_start_epoch
print("sandbox age (min):", round((time.time() - sandbox_start_epoch()) / 60, 1))
EOF
```

If the age looks wrong (for example 0 when you know the sandbox is older), export `MOLAB_SESSION_START=<epoch>` or pass
`--deadline-epoch <epoch>` explicitly.

## Runbook

1. Start the sandbox, get the pairing code.
2. Run `tools/molab_recover.py` (owned by the recovery tooling; see `docs/experiments/BACKUP_AND_RESTORE.md`). It restores
   the environment, the cache and the run directory (latest checkpoint) from HF.
3. Note the sandbox age (above). Launch the backup daemon (it uploads within ~60 s of `latest.json` changing), then:

   ```bash
   python scripts/train_v2.py --config configs/train_v2.json --run-dir artifacts/v2_full \
     --cache data/v2/cache_1280x704_f121_v2/index.jsonl:1.0 \
     --cache data/flow_poc/cache_1280x704_f121_v1/index.jsonl:0.25 \
     --init-from artifacts/flow_poc_all/adapter_000300.safetensors \
     --session-hours 3 --deadline-margin-minutes 15
   ```
   Resume is automatic if `latest.json` exists; `--init-from` only applies to a fresh run.
3b. (First session after the speed options landed, or whenever the GPU type changes.) Before the long run, once:
   `python scripts/train_v2.py --config configs/train_v2.json --run-dir artifacts/v2_full --cache data/v2/cache_1280x704_f121_v2/index.jsonl:1.0 --init-from artifacts/flow_poc_all/adapter_000300.safetensors --check-only --memory-probe`
   (about 1-2 minutes; see `docs/v2/TRAINING_V2.md` section 7). Pick the fastest row under ~85 GiB (the printed
   `recommended_flag`), then add that `--ckpt-blocks K` to the launch command in step 3 (resume is allowed; it is not part of the run identity).
   Make sure `max_peak_gib` (config, default 80) is above that row's peak, e.g. `--set max_peak_gib=88`.
   Optional (once per GPU type, up to ~20 min of GPU time, training stopped): the speed probe times K x attention backend x
   block compile on one fixed batch and checks loss_A / loss_B / grad norm against the K=20 baseline (flags anything above
   1e-2 as NOT lossless), then prints the recommended flags:
   `python scripts/train_v2.py --config configs/train_v2.json --run-dir artifacts/v2_full --cache data/v2/cache_1280x704_f121_v2/index.jsonl:1.0 --init-from artifacts/v2_full/checkpoints/adapter_<latest>.safetensors --check-only --speed-probe --probe-limit-gib 85`
   (see `docs/v2/TRAINING_V2.md` section 7). Add the printed `--attn-backend ... --compile-blocks` to the launch command as well;
   none of them is part of the run identity.
4. Expect ~200-220 updates. Periodic checkpoints (every 25 updates / 10 min) continue as usual.
5. At the end you see `DEADLINE_STOP step=N` (twice: the decision, then `checkpoint_saved=true`), exit code 0,
   `run_status.json` has `stopped_reason: "deadline"`. Do NOT kill the process before this line.
6. Confirm HF has the checkpoint: wait ~60-90 s after the line, then check that HF `latest.json` reports step N and that
   `adapter_00000N.safetensors` / `resume_00000N.pt` / `ckpt_00000N.json` are present (daemon log or the HF repo
   page). Only then let the sandbox expire. If the upload is missing, run the backup tool once manually.
7. Next session: repeat from step 1. Final session ends by itself at update 1000 (`status: complete`).

If the process exits 0 but the step is below 1000 and `stopped_reason` is `deadline`, that is the normal chunk end. A
non-zero exit or a hard kill loses at most 25 updates / 10 minutes (last periodic checkpoint).

## Stop rule

Before each update: stop if `now + est_update + est_ckpt + 90 s > deadline`, where
`deadline = min(--deadline-epoch, now + --max-minutes, sandbox_start + --session-hours) - margin`, `est_update` is an EMA
of measured s/update (initial guess 45 s) and `est_ckpt` the last measured checkpoint write time (initial 60 s).

## Other long jobs

`scripts/build_cache_v2.py`, `scripts/generate_examples_v2.py` and `scripts/calibrate_strength.py` take the same four
flags and stop between items (clips / runs) with a `DEADLINE_STOP ...` line and exit 0; they are resumable, so rerun the
same command next session. Calibration deliberately does not fit or write the calibration file from a partial sweep.
