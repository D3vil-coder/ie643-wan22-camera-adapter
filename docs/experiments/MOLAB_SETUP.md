# Molab GPU notebook setup (recorded 2026-11-16)

No URL or token appears in this file. Credentials are passed through the
`MOLAB_URL` / `MOLAB_TOKEN` environment variables (or `--url` / `--token`).

## Machine probe

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB (95 GiB), driver 595.71.05 (CUDA 13.2) |
| CPU / RAM | 20 vCPU, 160 GB RAM, no swap |
| OS / Python | Debian 13, gVisor sandbox, running as root. Python 3.13.11 in `/tmp/uv-venv` (the marimo kernel env), system site-packages are also on the path |
| torch | 2.11.0+cu130, `cuda.is_available()` True. Matches the 2.11.x pin. |
| Other tools | git 2.47, ffmpeg 7.1, nvcc 13.3, uv, jq absent |
| Network | HF hub, PyPI and download.pytorch.org reachable. `images.cocodataset.org` works over **http** (https has a cert-name mismatch, so use `http://` or `curl -k`) |

Disk: every path (`/`, `/tmp`, `~`, `/marimo`) is the same overlay root that reports a
virtual 8 EiB free. A 3 GB `dd` test succeeded at about 4.5 GB/s. The real quota is unknown.
The 32 GB model download plus all checkpoints fit without issue.

## Quick recovery after a sandbox loss

`python tools/molab_recover.py --resume-cache-build` (see `BACKUP_AND_RESTORE.md`) redoes the whole setup below and restores
sources + cache from HF. Incident 2026-11-16: sandbox terminated (HTTP 410) mid cache build at ~640/1902 clips HF-backed; recovered via
tools/molab_recover.py. Sandboxes are hard-killed about 3 h after start; the recovery script logs the measured age at start.

## Persistence

Nothing is documented as persistent. The box is a `sb-...` sandbox. Only
`/marimo/notebook.py` is backed by the Molab notebook. Treat **everything else, including
`/home/marimo/work` and the HF cache, as lost when the sandbox restarts.** Plan on re-running
this setup (about 5 min: bundle push, `uv pip install`, 70 s model download at
about 450 MB/s) after every restart. Copy outputs you care about back to the local machine.

## Layout

- Project root: `/home/marimo/work/IE643-Novelty` (branch `D3vil-coder/Main`, cloned from a git bundle)
- HF cache: `HF_HOME=/home/marimo/work/hf`. Wan2.2-TI2V-5B-Diffusers rev `b8fff731...` is
  downloaded, 31.9 GiB, 29 files, at `$HF_HOME/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers`.
  The code (`src/fpcamera/wan.py`) uses the default HF cache, so `HF_HOME` must be exported in
  any process that loads the model.
- `/home/marimo/work/env.sh` sets `HF_HOME`, `PYTHONPATH=<root>/src`, puts `/tmp/uv-venv/bin` on `PATH`
  and `cd`s into the root. There is no pyproject/setup.py, so there is no editable install. Use `source env.sh`.
- Uploaded (gitignored): `artifacts/flow_poc_all/{adapter_000300.safetensors, adapter_000300.json, latest.json, config.json, progress.json}`
  (safetensors sha256 `9e2963f2...efdbc`, verified), `examples/inputs/*` (5 jpg + 11 png), `data/flow_poc/reference_blue_ball.png`.

## Dependencies

Installed with `uv pip install --python /tmp/uv-venv/bin/python -r <requirements.txt minus torch/torchvision> + requirements-ui.txt + requirements-evaluation.txt`
(minus numpy/Pillow/imageio lines from the evaluation file; since 2026-11-17 `requirements-evaluation.txt` pins
`opencv-python==5.0.0.93` like `requirements.txt` (the old 4.10 pin conflicted, resolved) and adds `lpips==0.1.4`;
`tools/molab_recover.py` does this automatically.)

Result: diffusers 0.36.0, transformers 4.57.6 (the image shipped 5.14.1), accelerate 1.15.0,
safetensors 0.8.0, huggingface-hub 0.36.2, numpy 2.4.6, Pillow 12.3.0, imageio 2.38.0,
opencv-python 5.0.0.93, marimo 0.25.2, torch 2.11.0, torchvision 0.26.0.
`require_pinned_runtime()` passes.

Pitfall: the venv did not contain torch (it lives in the system site-packages), so uv pulled
torch 2.15.0 plus new nvidia-* wheels into the venv. These were uninstalled again so the original
2.11.0+cu130 is visible. If you reinstall, **pass `torch==2.11.0 torchvision==0.26.0` explicitly or
exclude them, and check `python -c "import torch; print(torch.__version__)"` afterwards.**

## Tests on Molab

`python -m pytest tests -q`: 63 passed, 3 failed. `compileall src scripts`: OK.
- `test_disk_check_refuses_when_insufficient`: sandbox reports 8 EiB free, so the "insufficient disk" branch can never fire. Environment artifact.
- `test_dry_run_needs_no_torch_and_writes_previews`, `test_build_resume_crash_safety_and_load`:
  `scripts/build_cache_v2.py` synthetic-image helper calls `cv2.putText` on a non-uint8 image,
  which OpenCV 5.0.0 rejects (`img.depth() == CV_8U`). Fix: convert the image to uint8 before drawing.
  This is relevant only to the synthetic/dry-run path.

## Driving the notebook and syncing files

`tools/molab_sync.py` (stdlib only, uses the same HTTP API as the marimo-pair skill's
`execute-code.sh`, needed because `jq` is not available on the local Windows box):

```bash
export MOLAB_URL=https://sb-<id>.sb.molab.run MOLAB_TOKEN=<token>   # never commit
python tools/molab_sync.py exec -c "print(1)"            # run in the kernel scratchpad; also: exec file.py | exec -
python tools/molab_sync.py push LOCAL REMOTE             # chunked (768 KB), resumable, sha256-verified
python tools/molab_sync.py pushdir LOCAL_DIR REMOTE_DIR --glob '*.png'
```

On Git Bash for Windows set `MSYS_NO_PATHCONV=1` and give the local path in `C:/...` form, or
`/home/...` remote paths get rewritten. Code updates: `git bundle create x.bundle <base>..HEAD`
(or `--all`), `push` it, then `git fetch x.bundle D3vil-coder/Main:...` / `git pull` from the
bundle on Molab. Code runs in marimo's scratchpad (no notebook cells are created; bindings
are discarded after each call).

## Long-running jobs

Anything longer than about 60 s goes in a detached subprocess with a log, never in the kernel call:

```python
import subprocess, os
subprocess.Popen(["nohup", "bash", "-c", "source /home/marimo/work/env.sh; python -u scripts/X.py > /home/marimo/work/X.log 2>&1"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
```

Check with `exec -c "print(open('/home/marimo/work/X.log').read()[-2000:])"`, `ps aux | grep python`, `nvidia-smi`.
The model download ran this way (`/home/marimo/work/dl_model.py`, log `/home/marimo/work/dl_model.log`).

## Smoke test results 2026-11-16

Code synced with `python tools/molab_sync.py sync-code` (incremental git bundle + `merge --ff-only`; `pull` / `pulldir`
download with sha256 verification were added alongside). `python -m pytest tests -q` on Molab: **66 passed** (the three
earlier environment failures are fixed, see below). All GPU jobs ran as detached processes; peak GPU memory is the
`nvidia-smi` maximum sampled every 2 s for the whole job (includes the CUDA context and caching allocator), so it is an
upper bound on tensor memory. Models load in 15-21 s (page cache is warm).

| Smoke test | Wall | Peak GPU (nvidia-smi) | Result |
|---|---|---|---|
| P1 diagnostics, 10 steps, `heldout` zoom_out, `v1_baseline` | 194 s total (21 s load, generate 149 s, decode, metrics) | 51.6 GiB (during x0-hat / final fp32 VAE decode); 32.7 GiB during sampling | OK, no OOM. `debug_meta.json` captured x0-hat at step 9 of 10, sigma 0.555 (<= 0.6); the three `x0hat_sigma_0.60_frame_{040,080,120}.png` decoded |
| `build_cache_v2.py --limit 6` | 67 s total | 15.4 GiB | 6 clips built, **6.82 s/clip** average (VAE encode 6.4 s + render 0.4 s), **10.5 MB per clip file**, 11.0 MB/clip incl. references, 66 MB total |
| `train_v2.py --check-only` (warm start from `adapter_000300`, 2 micro-batches) | 74 s total | 26.5 GiB | OK. 41 s for the one update, torch peak **23.1 GiB**, grad norm 2.13, warm start architecture matches (162 tensors, none missing) |
| 2 real updates (`--stop-after-updates 2`) | 111 s | 26.6 GiB | updates took 41.5 s and 41.3 s (2 clips each) = **about 41 s/update**, torch peak 23.2 GiB |
| SIGTERM during update 3, then resume | | | Process finished update 3 (32 s), wrote an `emergency` checkpoint at step 3 (`ckpt_000003.json`) and exited with code 75. Relaunch with `--stop-after-updates 3` logged `resumed step 3` and completed. Backbone hash unchanged. Orphaned (un-checkpointed) progress lines were dropped on resume (`orphaned_log_lines: 1`) |

Observation: the update times are about 41 s for the pilot config. 150 pilot updates is about 1.7 h; the plan is now 1,000 updates (pilot = first 150 of the same run, no separate pilot config), about 9-11 h at this speed (the third update was faster, 32 s, so this may improve after warm-up).

### Cache projections (measured 6.82 s/clip, 10.98 MB/clip)

| Plan | Clips | Time | Disk |
|---|---|---|---|
| 636 sources x (1 static + 2 moving), K=2 | 1908 | 3.6 h | 21 GB |
| 636 sources x (1 + 1), K=1 | 1272 | 2.4 h | 14 GB |
| 400 sources x (1 + 2) | 1200 | 2.3 h | 13 GB |

`--project` assumed 45 s/clip (3.6x too pessimistic); use `--s-per-clip 6.8`. The 6 clips built by the smoke test live in
the default cache dir `data/v2/cache_1280x704_f121_v2` and are reused (resumable) by the full build.

### COCO source selection

`scripts/select_coco_sources_v2.py --project-root .` ran in 58 s on Molab (the config URLs were already `http://`, no
change needed). Output `data/v2/sources_v2.json`: **636 sources (train 608, development 28)** = 36 v1 sources + 600 new
(24 new development, 576 new train). Rejected during download checks: grayscale 25, duplicate (aHash) 9, decode errors 0.

### Bugs fixed

- `scripts/build_cache_v2.py` synthetic helper: image cast to uint8 before `cv2.putText` (OpenCV 5 requires CV_8U).
- `tests/test_geometry_v2.py::test_disk_check_refuses_when_insufficient` now monkeypatches `shutil.disk_usage`.
- `tools/molab_sync.py`: added `pull`, `pulldir`, `sync-code`.
- No GPU-side bugs were found; no code change was needed for the VAE decode (fp32, 121 frames fits easily in 95 GiB).

### Operational notes

- Never `pkill -f "scripts/train_v2.py"` from a shell whose own command line contains that string (it kills itself);
  use `kill -TERM <pid>`. A second SIGTERM while the first is still finishing the update/checkpoint raises KeyboardInterrupt
  and **skips the emergency checkpoint** (one signal only; the first attempt in this session lost the step-3 checkpoint
  that way).
- A 50-step P1 variant took 503 s for `v1_baseline` (3-4 transformer forwards per step) including the fp32 decode.

## Session 2 (2026-11-16)

Recovery after the HTTP 410 sandbox loss, run with `python tools/molab_recover.py --resume-cache-build`.

- New sandbox started 15:19:18 UTC (measured by the probe step: age 11 min at 20:30 IST start of recovery, estimated 169 min left of the 3 h session). Hard deadline about 18:19 UTC; the build uses `--session-hours 3 --deadline-margin-minutes 15` (stops cleanly about 18:04 UTC).
- Per-step wall time: probe 4 s, code 3 s, deps 73 s (torch stayed 2.11.0+cu130, `require_pinned_runtime()` OK), token 1 s (HF user verified via `whoami`), model download started in the first run (which crashed on the logging bug) and finished detached, so it was skipped on the re-run (about 1-2 min), restore 276 s (711 + 1070 files fetched, one burst of HF 429 rate limits that the downloader retried; 640 clip files, 640 sha256-verified index rows, 0 bad), build launch 2 s. Total about 6 min from start to a running build.
- Cache build: 1268 clips remaining of 1908 at 6.9 s/clip (about 2.4 h); the ETA (~135 min at 88/1268) is right at the session deadline, so it may stop a few dozen clips short and resume next session. The cache watcher (`--batch-files 25 --batch-seconds 300`) uploaded its first batches successfully (batches of 41-46 files because the threshold is a minimum and the poll is 120 s).
- Bug fixed: `tools/molab_recover.py` crashed with `UnicodeEncodeError` (cp1252 stdout, redirected to a file) when echoing a remote download progress-bar glyph; `log()` now reconfigures stdout to UTF-8 with `errors="replace"`. Re-running the script skipped the completed steps.

## Session 3 (2026-11-16)

Sandbox started 18:10:12 UTC (hard kill ~21:10). Details, timings and the evaluation table: `S3_TRAINING_AND_EVAL100.md`.

- `molab_recover.py` now has an `extras` step (v1 warm-start adapter with sha256 check, `examples/inputs` from the HF dataset repo `examples_inputs/`) and starts `--resume-training` fresh when the run is not on HF yet.
- HF API limit: the restore (16 workers) can hit 1000 requests / 5 min and fail 46 files with 429; re-run the step (idempotent).
- The cache build finished at 19:42 (580 clips, 4413 s); v2 training ran 77 updates at 41.5 s (23.2 GiB), 76-92 s/update while an inference job shared the GPU.
- `generate_examples_v2.py --no-lease` shares the GPU with the trainer. The HF run backup prunes older checkpoints (latest 3 + current): pull a checkpoint you want to evaluate later before it rotates out.
- Launch detached jobs with `Remote.detach` (nohup, own session); a plain `nohup ... &` inside a one-off exec call is killed when the call's stream ends.
