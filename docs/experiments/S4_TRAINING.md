# Session 4: training only (v2_full, steps 77 -> 201)

Sandbox started 21:11 UTC (3 h hard kill ~00:11). Training-only session, no GPU eval. Paused early by request at step 201 to start a faster Session 5.

## Recovery
- `molab_recover.py --resume-training v2_full`: probe/code/deps/token/model OK within about 2 min (model download ~112 s).
- HF rate limits (429, 1000 requests per 5 min) hit the cache restore; the bulk restore (3182 files, 1908/1908 rows verified) took about 15 min (21:15 to 21:30 UTC) and kept sleeping in backoff.
- Bugs fixed (commits 9d3e99f, and the `--allow-config-change` commit): (1) the token step treated a whoami 429 as "token rejected"; now a 429 is tolerated. (2) Resume was refused with "Run identity changed (keys: ['caches'])": the identity holds the cache index sha, and the cache grew from the partial index used in S3 to the full 1908 rows. `molab_recover` now launches train_v2 with `--allow-config-change`.
- Note: a restore-step re-run while the first detached restore was still running caused only a spurious whoami 429 failure (the first restore finished fine).
- Training resumed at about 21:45 UTC (`{"event": "resumed", "step": 77}`); a second backup daemon was started by the re-run (two watchers; they raced on pruning, harmless 404 log lines).

## Training
| window | loss_a | loss_b | s/update |
|---|---|---|---|
| resume, steps 78-82 | 0.191 | 0.090 | 41.5 |
| steps 78-127 (+50) | 0.159 | 0.091 | 40.7 |
| steps 128-177 (+100) | 0.137 | 0.085 | 39.8 |
| steps 178-201 | 0.131 | 0.078 | 37.6 |
| all 78-201 | 0.145 | 0.086 | 39.8 |

Single points: step 77 0.184/0.107, step 127 0.152/0.076, step 177 0.064/0.051, step 201 0.163/0.077. No NaN/inf/OOM. 124 updates in about 83 min.

## Pause
Single SIGTERM to train_v2.py (PID 8955) at about 23:08 UTC: finished the update, wrote an emergency checkpoint at **step 201**, exit code 75. HF `runs/v2_full` latest.json = step 201 (sha256-verified on pull). Backup daemons stopped with one SIGTERM each. Step 250 was not reached (retention keeps every 250th step; no milestone beyond 201 yet).
Laptop: `<checkpoints_dir>\v2_full\` holds steps 77 and 201.
