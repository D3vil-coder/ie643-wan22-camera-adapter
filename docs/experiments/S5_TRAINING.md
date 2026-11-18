# Session 5: training with speed options (v2_full, steps 201 -> 415)

Sandbox (new, GPU attached): NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB. Sandbox start 23:17:54 UTC; deadline = start + 3 h - 15 min = 02:02:54 UTC.
Trainer stopped itself at `DEADLINE_STOP step=415` (reason `deadline`, exit 0).

## Recovery timings (molab_recover --only probe code deps token model restore extras)
| step | time |
|---|---|
| probe | 4 s |
| code | 4 s |
| deps | 66 s |
| token | 1 s |
| model | 80 s |
| restore (cache + sources, 1908 clips, 1908 index rows verified; HF 429 backoff, bulk path since no shards on HF yet) | 875 s |
| extras | 4 s |
| run restore `v2_full` (step 201, sha256 verified) | 15 s |

First restore attempt failed in 1 s: the local (uncommitted) molab_recover passed `--prefer-shards` while the shipped bundle was an older HEAD. Re-shipped with `--force code` after the shard commit landed. Training was gated until `s5_go` (content 201) existed.

## Memory probe (`--check-only --memory-probe`, limit 85 GiB)
| --ckpt-blocks | status | peak GiB | s (probe update) |
|---|---|---|---|
| 30 (all) | ok | 23.1 | 41.5 |
| 20 | ok | 71.7 | 38.5 |
| 10 | OOM | - | - |

Chosen K = 20 (peak 71.8 GiB in training, below the 80 GiB limit; no `max_peak_gib` change). Real training peak stayed at 71.81 GiB for all 214 updates.

## Training (`--ckpt-blocks 20 --allow-identity-change caches --session-hours 3 --deadline-margin-minutes 15`)
Resumed at step 201 exactly. One `hf_backup watch-run` daemon. `pack-cache` (21 shards, 1 GB each, `cache_v2_shards` on HF) was run once at nice 19 during training; it finished in about 3 min with no visible slowdown.

| window | loss_a | loss_b | s/update |
|---|---|---|---|
| steps 202-206 | 0.155 | 0.099 | 36.8 |
| 202-251 | 0.132 | 0.087 | 37.1 |
| 252-301 | 0.106 | 0.079 | 36.8 |
| 302-351 | 0.102 | 0.079 | 36.3 |
| 352-415 | 0.089 | 0.079 | 36.9 |
| all 202-415 (214 updates, 7837 s) | 0.106 | 0.081 | 36.8 |

Single points: 202 0.095/0.068, 250 0.173/0.070, 300 0.099/0.077, 350 0.096/0.067, 415 0.077/0.065. No NaN/inf/OOM.

Speedup: S4 averaged 39.8 s/update (all blocks checkpointed) vs 36.8 s now (K=20): about 7.5% faster. 214 updates in 130 min vs 124 in 83 min before. Note: K=10 does not fit in 85 GiB.

## End state
HF `runs/v2_full` latest.json = step 415 (sha256 verified on pull: adapter 83df1e3f...). Backup daemon stopped (SIGTERM). Laptop `<checkpoints_dir>\v2_full\` holds steps 77, 201, 250, 415.
