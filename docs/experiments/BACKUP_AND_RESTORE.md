# Backup and restore (Hugging Face) for the ephemeral Molab sandbox

Molab keeps nothing across a restart (see `MOLAB_SETUP.md`). Checkpoints and the latent cache are
therefore mirrored continuously to two **private** Hugging Face repos. No token appears in the repo:
it lives in `$HF_HOME/token` (Molab, mode 600) and in a local `KEY=value` keys file outside git (laptop; `$FPCAMERA_KEYS_ENV`, default `<repo root>/keys.env`).

## Local configuration (laptop tools)

No machine-specific path is hard-coded. `tools/local_config.py` reads two environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `FPCAMERA_KEYS_ENV` | path of a `KEY=value` file holding the HF tokens (`HF_WRITE_TOKEN`, `HF_DOWNLOAD_TOKEN`, ...); read at run time, never printed | `<repo root>/keys.env` (git-ignored) |
| `FPCAMERA_CHECKPOINTS_DIR` | folder of the pulled run folders and evaluation outputs (`<dir>/v2_full/adapter_000415.safetensors`, `<dir>/eval_E1`, `<dir>/G1`) | `<repo root>/artifacts/checkpoints/v2` (git-ignored) |

A missing keys file stops the tools with a message that says how to set the variable. PowerShell: `$env:FPCAMERA_KEYS_ENV = 'C:\path\to\keys.env'`
for the session, or `setx FPCAMERA_KEYS_ENV 'C:\path\to\keys.env'` to keep it; bash: `export FPCAMERA_KEYS_ENV=/path/to/keys.env`.

## Repos

| Repo | Type | Content |
|---|---|---|
| `D3vil-Reap3r/ie643-camera-adapter-v2` | model | `runs/<name>/{latest.json,config.json,progress.json,progress.jsonl,launches.jsonl,run_status.json}` and `runs/<name>/checkpoints/{adapter_NNNNNN.safetensors,resume_NNNNNN.pt,ckpt_NNNNNN.json}` |
| `D3vil-Reap3r/ie643-camera-v2-cache` | dataset | `cache_v2/{clips/*.safetensors,references/*,index.jsonl,config_snapshot.json,...}` and `cache_v2/sources_root/{sources_v2.json,sources/**}` |

Remote retention mirrors `checkpointing.py`: newest 3 checkpoints plus every 250th update. The remote
`index.jsonl` lists only clips whose files are already uploaded. The sources images are COCO photos
(not redistributable), which is fine in a private repo.

## Tools

* `tools/hf_backup.py` (Molab daemon; stdlib + `huggingface_hub`)
  * `watch-run RUN_DIR --path-in-repo runs/<name>` polls every 60 s. When a new complete checkpoint
    exists (adapter and resume sha256 verified against `ckpt_NNNNNN.json`) it uploads that checkpoint,
    `latest.json`, config and progress files in ONE commit and deletes out-of-policy remote checkpoints in
    the same commit. At most about one commit per minute, retry with exponential backoff, `*.tmp` never uploaded.
    Small metadata files are refreshed at most every 10 min.
  * `watch-cache CACHE_DIR --path-in-repo cache_v2 --sources-root data/v2` uploads finished clips (sha256 checked
    against the index row) in batches of 50 or more files, or whatever is pending after 10 min, then the merged
    index. A local ledger (`<cache parent>/hf_backup_ledger_<path>.json`) plus the remote tree make restarts idempotent.
  * `once-run` / `once-cache` (or `--once`): a single pass that flushes everything; use at the end of a run.
* `scripts/restore_from_hf.py`: `--run NAME --run-dir DIR`, `--cache --cache-dir DIR`, `--sources --sources-root DIR`.
  Verifies every sha256, skips what is already present, writes `latest.json` last.
* `tools/pull_milestones.py` (laptop): see the last section.

The tools are in the repo (`tools/hf_backup.py`, `scripts/restore_from_hf.py`); run them from `/home/marimo/work/IE643-Novelty` after `sync-code`.

## One-time setup of the token on Molab

Write the token to `/home/marimo/work/hf/token` (chmod 600) through the exec channel (do not echo it), then check
`huggingface_hub.whoami()['name']` (expected `D3vil-Reap3r`). `HF_HOME=/home/marimo/work/hf` comes from `env.sh`; a
detached process needs it exported (or `HF_TOKEN`).

## Launching the daemons (on Molab, detached, CPU/network only)

```bash
source /home/marimo/work/env.sh; mkdir -p /home/marimo/work/logs
setsid nohup python tools/hf_backup.py watch-run \
  /home/marimo/work/IE643-Novelty/artifacts/v2_full --path-in-repo runs/v2_full \
  > /home/marimo/work/logs/backup_v2_full.log 2>&1 &

setsid nohup python tools/hf_backup.py watch-cache \
  /home/marimo/work/IE643-Novelty/data/v2/cache_1280x704_f121_v2 --path-in-repo cache_v2 \
  --sources-root /home/marimo/work/IE643-Novelty/data/v2 \
  > /home/marimo/work/logs/backup_cache.log 2>&1 &
```

Check: `tail -n 20 /home/marimo/work/logs/backup_*.log`, `ps aux | grep hf_backup`. Daemons are safe to kill and
restart at any time. When the run is complete, finish with `once-run` and `once-cache` (same arguments) so the last
checkpoint, `run_status.json` and the final index are on the remote. Use one `--path-in-repo` per run name (`v2_pilot`, `v2_full`).

Caveats: if a run is restarted with `--no-resume` (old state archived, step counter restarts) the remote still
holds the higher steps; the daemon never deletes steps newer than the local latest and `latest.json` then follows the
new run, so use a new run name instead. The cache daemon uploads the index rows as they are; if the builder is run
in shards, `index.shard*.jsonl` files are merged into the remote `index.jsonl`.

## One-shot recovery (preferred)

From the laptop, with a fresh sandbox (commit local work first, the repo is shipped as a git bundle):

```bash
export MOLAB_URL=https://sb-<id>.sb.molab.run MOLAB_TOKEN=<token>      # never commit
python tools/molab_recover.py --resume-cache-build [--resume-training v2_full]   # --plan only prints the plan
```

`tools/molab_recover.py` runs, each skipped if already done: probe (GPU, torch, sandbox age) + `env.sh`; repo push (bundle + clone,
or `sync-code`); pinned deps with torch/torchvision constrained to the preinstalled build (`require_pinned_runtime()` must pass);
HF token from the local keys file to `$HF_HOME/token`; Wan2.2 base model download; `restore_from_hf.py --sources --cache` with verification;
optionally `build_cache_v2.py` + `hf_backup.py watch-cache` (batches of 25 files / 5 min); optionally run restore + `train_v2.py` + `watch-run`.
Long steps are detached and polled; a final status summary lists PIDs and logs (`/home/marimo/work/logs/`).

Incident note: 2026-11-16 sandbox terminated (HTTP 410) mid cache build at ~640/1902 clips HF-backed; recovered via tools/molab_recover.py.

Sources layout: `cache_v2/sources_root/` holds only `sources_v2.json`, `sources/**` and `_work/annotations*.zip` (allow-list in
`hf_backup.py` and `restore_from_hf.py`). The cache dir lives inside `data/v2`, so an earlier whole-folder sources upload duplicated it
(7 GB, 1072 files); that copy was verified identical (every clip/reference size + sha256) and deleted in one commit, the dataset repo went
from 14.40 GB to 7.39 GB. Only its `encoder.log` was unique and was moved to `cache_v2/encoder.log`.

## Disaster recovery (sandbox wiped, manual steps)

1. Rebuild the environment as in `MOLAB_SETUP.md` (bundle push or `sync-code`, `uv pip install` with torch pinned, `env.sh`,
   Wan2.2 model download).
2. Install the HF token at `/home/marimo/work/hf/token` (chmod 600), verify with `whoami()`.
3. `restore_from_hf.py` needs only `huggingface_hub`.
4. Restore, inside `/home/marimo/work/IE643-Novelty` after `source /home/marimo/work/env.sh`:
   ```bash
   python scripts/restore_from_hf.py --sources --sources-root data/v2
   python scripts/restore_from_hf.py --cache --cache-dir data/v2/cache_1280x704_f121_v2
   python scripts/restore_from_hf.py --run v2_full --run-dir artifacts/v2_full
   ```
   The cache restore verifies every clip and reference against the index and rewrites a local `index.jsonl` containing
   only verified rows (it exits non-zero if any row fails). If the cache was incomplete at backup time, rerun
   `scripts/build_cache_v2.py`: it adopts the restored files and builds only what is missing. The v1 warm-start adapter is in the model repo under `v1/flow_poc_all/`:
   `python -c "from huggingface_hub import snapshot_download as d; d('D3vil-Reap3r/ie643-camera-adapter-v2', allow_patterns='v1/flow_poc_all/*', local_dir='artifacts_dl')"` then copy the three files to `artifacts/flow_poc_all/` (sha256 of the safetensors: `9e2963f264ae5dcd9c406f12e32194782ba523908fb4d2220d8f7774906efdbc`). The v1 cache is not backed up.
5. Relaunch the identical `train_v2.py` command (see `docs/v2/TRAINING_V2.md` section 6). It finds `artifacts/v2_full/latest.json`
   and resumes. `--step N` on the restore picks a specific older checkpoint (must still be on the remote).
6. Restart the two `watch-*` daemons.

## Laptop: milestone checkpoints

```powershell
python tools\pull_milestones.py v2_full            # to <checkpoints_dir>\v2_full\  ($FPCAMERA_CHECKPOINTS_DIR, default artifacts/checkpoints/v2)
python tools\pull_milestones.py v2_full --with-resume --keep 4 --every 250
```

Downloads `adapter_NNNNNN.safetensors` (+ `ckpt_NNNNNN.json`, `config.json`, `progress.json`, `latest.json`) for every
250th update and the newest step, verifies sha256, and keeps only the newest 4 steps locally. The token comes from `HF_TOKEN`
or from the keys file (`$FPCAMERA_KEYS_ENV`, default `<repo root>/keys.env`) at run time (`HF_DOWNLOAD_TOKEN` first); it is never printed or stored.

## Evaluation outputs (`watch-dir` / `--eval`)

Evaluation scripts (`eval_dev_reveal.py`, `generate_examples_v2.py`, `calibrate_strength.py`, `validate_v2.py`) are resumable
from their output folders, so those folders are backed up to the dataset repo under `eval/<name>/`:

```bash
python tools/hf_backup.py watch-dir artifacts/reveal_eval --repo-type dataset --path-in-repo eval/reveal_v2 [--repo USER/ie643-camera-v2-cache]
python tools/hf_backup.py once-dir  artifacts/reveal_eval --repo-type dataset --path-in-repo eval/reveal_v2   # single flush
```

- A run folder (any dir with `run.json`/`video.mp4`) is uploaded only when COMPLETE: `run.json` (written last by `flow_infer`) and
  `video.mp4` exist. Half-written folders, `*.tmp`/`*.partial` and `tracks.npz` are skipped. The sidecars that appear later
  (`reveal_metrics.json`, `border_quality.json`, `path_adherence.json`) are picked up by the next pass.
- Aggregates (json/jsonl/md/png/html/js/csv/txt outside run folders: `reveal_eval.json`, `validation.json`, `selection.json`,
  `pairs/*/pair.json`, `summary.md`, ...) are uploaded once older than `--settle-seconds` (10 s) and re-uploaded when they change.
- Ledger `<dir>/../hf_backup_ledger_eval_<name>.json` maps path -> `size:mtime:sha256`; unchanged files are never re-hashed or
  re-uploaded, and a file already on the remote with the same sha256 is only recorded. Batched (`--batch-files 20`,
  `--batch-seconds 300`, `--min-commit-interval 60`).

Restore on a fresh sandbox (existing same-size files are kept, so the resumable script continues where it stopped):

```bash
python scripts/restore_from_hf.py --eval reveal_v2:artifacts/reveal_eval --eval validate_v2full:artifacts/v2_full/validation
python scripts/restore_from_hf.py --run v2_full --run-dir artifacts/v2_full --checkpoints 77,250   # adapter-only, latest.json untouched
```

`--checkpoints` fails (non-zero) for steps pruned by the retention policy (newest 3 + every 250th); 77 survives only if it was
pulled to the laptop or kept before pruning.

`tools/molab_recover.py` wraps both: `--resume-eval NAME:DIR` (repeatable; restores and launches a `watch-dir` daemon per dir,
log `logs/backup_eval_<NAME>.log`) and `--checkpoints 77,250 [--checkpoints-run RUN]` (defaults to `--resume-training`). The deps step
installs `requirements-evaluation.txt` (incl. `lpips==0.1.4`, `opencv-python==5.0.0.93`, same as `requirements.txt`) under the torch
constraints and pre-fetches the AlexNet backbone for LPIPS.

## Daemon dedup and single-instance locks

- `molab_recover` launches every daemon/trainer through `Remote.launch_once`: it reads `/home/marimo/work/pids/<name>.pid`
  (the detached wrapper shell, whose cmdline contains the command), checks the pid is alive and its cmdline matches the target,
  and otherwise falls back to a `/proc/*/cmdline` scan. A live match means skip (logged `[dedup]`); otherwise it launches and
  rewrites the pidfile. Safe to rerun `molab_recover` any number of times.
- `hf_backup.py` (all watch-/once- commands) takes an exclusive OS lock per (repo, path-in-repo) in
  `$HF_BACKUP_LOCK_DIR` (default `<tmp>/hf_backup_locks/`); a second instance for the same target exits with code 3. The OS
  releases the lock when the process dies.

## Resuming with a changed run identity

`train_v2.py --allow-identity-change KEY[,KEY]` permits only the listed identity keys to differ from the checkpoint's
(e.g. `caches`, when the cache index grew); old/new values are logged and appended to `launches.jsonl` as an
`identity_change_accepted` record. Any other differing key still aborts. `--allow-config-change` remains the explicit
all-keys escape hatch; `molab_recover` now passes `--allow-identity-change caches`.

## Sharded cache transfer (few large files instead of ~3800 small ones)

Pack once (CPU/network only, can run concurrently with training; resumable, takes the same per-target lock):

```bash
python tools/hf_backup.py pack-cache data/v2/cache_1280x704_f121_v2 --shard-gb 1.0 --path-in-repo cache_v2_shards --sources-root data/v2
```

- Members: `clips/*`, `references/*` (+ png), a filtered `index.jsonl` (rows whose clip and reference exist), `config_snapshot.json`,
  `rejected.jsonl`, `progress.json`, and `sources_root/sources_v2.json`. Sorted by name, split greedily into ~1 GB uncompressed tar
  shards `shard_0000.tar ...` (fixed mtime/owner, GNU format, so identical input gives byte-identical shards).
- `shards.json` (uploaded last) lists every shard with size + sha256 and every member with size + sha256; member sha256 is computed
  while writing and must equal the index row, otherwise packing aborts. Its presence on HF means the set is complete.
- Resume: shards are rebuilt locally (cheap) and uploaded only when the remote LFS sha256 differs; rerun after the cache grows.
  Each shard is one commit; temp shards live in `<cache>/../.hf_pack_cache_v2_shards` and are deleted after upload.

Restore: `python scripts/restore_from_hf.py --cache-shards --cache-dir DIR --sources-root data/v2 [--shards-path cache_v2_shards]`
downloads shards (up to 4 parallel, 429 backoff via `fetch`), checks shard sha256, extracts (regular files only, no path
escapes, atomic placement) verifying each member's sha256, and finally checks that every index row's files exist. Shards whose
members are already valid locally are skipped. `--cache --prefer-shards` uses shards when `cache_v2_shards/shards.json` exists on
HF and falls back to the per-file restore otherwise; `molab_recover`'s restore step uses it.
