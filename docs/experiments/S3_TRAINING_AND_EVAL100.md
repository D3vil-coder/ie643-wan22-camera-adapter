# Session 3: cache completion, v2 training start, early evaluation (2026-11-16)

Sandbox started 18:10:12 UTC (hard kill about 21:10 UTC). Everything below is UTC. Despite the file name, the early
evaluation could not be run at update 100 (see "Deviations"): it used the update-50 checkpoint (3 cases) and update-77 (1 case).

## Timeline

| time | event |
|---|---|
| 18:11 | `molab_recover.py --resume-cache-build` started (sandbox age 1 min) |
| 18:14-18:27 | code, deps (73 s), token, model download (97 s); restore from HF hit the 1000 requests / 5 min HF API limit: 46 of 2216 files failed with 429 (the script exits non-zero but is idempotent) |
| 18:27 | restore re-run: 1328/1328 index rows verified. 580 clips were still missing |
| 18:28 | cache build resumed (6.88 s/clip) and cache watcher started |
| 19:42 | cache build complete: 580 built in 4413 s, 1908/1908 rows |
| 19:43 | training started: `train_v2.py --config configs/train_v2.json --run-dir artifacts/v2_full --init-from artifacts/flow_poc_all/adapter_000300.safetensors --session-hours 3 --deadline-margin-minutes 15`, plus the `watch-run` backup daemon |
| 20:00 | HF holds 1908 clips and 1908 index rows (checked via the HF API); cache watcher stopped (single SIGTERM, exact PID) |
| 20:20-20:57 | evaluation jobs (concurrent with training until 20:53) |
| 20:53 | `DEADLINE_STOP step=77 checkpoint_saved=true`, exit code 0; HF `latest.json` = step 77 (adapter sha256 `dc71ac8a666d...`), checked via the HF API |
| 21:04 | last evaluation video pulled; adapter 77 pulled to `<checkpoints_dir>\v2_full\` |

Training used only the v2 cache (`--cache data/v2/...:1.0`, 608 train sources); the 0.25-weight v1 cache mix of the
SESSION_CHUNKING runbook was not used, as instructed for this session.

## Recovery fixes (folded into `tools/molab_recover.py`)

- New step `extras` (between `restore` and `build`): restores the v1 warm-start adapter from HF (`v1/flow_poc_all/`) and verifies
  sha256 `9e2963f2...efdbc`; restores `examples/inputs/` (11 case PNGs + 5 COCO jpgs) from the HF dataset repo, folder `examples_inputs/`
  (uploaded once this session).
- `--resume-training RUN` now starts fresh when `runs/RUN` does not exist on HF (the HF tree call raises a 404, which used to abort).
- Not fixed: HF restore rate limit. The 16-worker restore exceeds 1000 API requests / 5 min; re-running the step finishes it. Lowering `--workers` or adding
  backoff in `restore_from_hf.py` would remove the manual re-run.
- `scripts/generate_examples_v2.py --no-lease` (via `Generator(use_lease=...)` in `calibrate_strength.py`): needed to share the GPU with the trainer.

## Training health (progress.jsonl)

| step | loss_A | loss_B | R fraction | grad norm | s/update | peak GiB |
|---|---|---|---|---|---|---|
| 1 | 0.711 | 0.096 | 0.082 | 3.74 | 41.6 | 23.1 |
| 5 | 0.575 | 0.145 | 0.089 | 2.35 | 41.5 | 23.2 |
| 25 | 0.197 | 0.114 | 0.083 | 0.82 | 41.5 | 23.2 |
| 50 | 0.228 | 0.067 | 0.063 | 1.12 | 41.5 | 23.2 |
| 77 (last) | 0.184 | 0.107 | 0.132 | 1.38 | 92.0 | 23.2 |

Single-update values are noisy (2 micro-batches). Window means: steps 1-10 A 0.396 / B 0.128; 11-25 A 0.274 / B 0.120; 26-50 A 0.178 / B 0.106;
51-77 A 0.227 / B 0.097. No NaN/inf, no OOM. Loss A (the original objective) fell about 2x and loss B (the new Pass B) about 25% in 50 updates; this is the warm-start
transient, not a convergence statement. 41.5 s/update alone; **76-92 s/update (about 2.2x slower) while an inference job shared the GPU**
(steps 52-77), so concurrent evaluation cost roughly 20 updates. Peak training memory 23.2 GiB.

Checkpoints: 15, 25, 39 (10-minute periodic), 50, 67, 75, 77 (deadline). The HF backup prunes old ones (keeps the latest 3 plus the current);
step 50 was therefore no longer available on HF or locally after step 77 was written.

## Early evaluation

Setup: 50 steps, seed 42, default prompt, replicate guide fill (checkpoint's recorded mode), v2 at the **physical amplitude of the Submission 1 case**
(`--match-v1`: zoom x1.12 out, pan 2.5 deg, tilt 1.75 deg). v1 and v1+replicate numbers are from `P1b_REPLICATE_BASELINE.md` (same
metric code). Artifact score: lower is better. Colorfulness / HF are R/V ratios (1.0 = revealed region like valid region).
Tracker: path-wide signed gain (1.0 = requested motion fully realised) and cosine.

| case | model | artifact | colorfulness R/V | HF R/V | gain | cosine |
|---|---|---|---|---|---|---|
| heldout_zoom_out | v1 | 0.603 | 0.725 | 0.738 | 1.245 | 0.996 |
| heldout_zoom_out | v1+rep | 0.359 | 0.974 | 0.715 | 1.336 | 0.986 |
| heldout_zoom_out | v2@50 | 0.397 | 0.900 | 0.769 | 1.128 | 0.999 |
| motorcycle_pan_right | v1 | 0.760 | 1.040 | 0.262 | 1.106 | 0.999 |
| motorcycle_pan_right | v1+rep | 0.680 | 0.902 | 0.458 | 1.147 | 0.997 |
| motorcycle_pan_right | v2@50 | 0.720 | 1.172 | 0.285 | 1.056 | 1.000 |
| computers_pan_left | v1 | 1.255 | 0.695 | 0.068 | 0.915 | 0.997 |
| computers_pan_left | v1+rep | 0.567 | 1.504 | 0.458 | 0.962 | 0.999 |
| computers_pan_left | v2@50 | 0.953 | 0.988 | 0.121 | 0.998 | 0.996 |
| flowers_tilt_up | v1 | 0.181 | 1.119 | 1.567 | 0.925 | 0.997 |
| flowers_tilt_up | v1+rep | 0.126 | 1.268 | 2.646 | 1.019 | 0.997 |
| flowers_tilt_up | v2@77 | 0.167 | 1.261 | 3.142 | 1.201 | 0.999 |

Revealed fraction R is small (zoom_out 11 %, pans/tilt 3 %), n frames with R > 1 %: 26-30, so these are small-sample single-seed numbers.
Generation time with the GPU shared with training: 13 min per video (16 s/step); alone about 6.5 s/step (6 min per video).

Observations (visual review of the contact sheets, rows v1 / v1+rep / v2, frames 40/80/120):

- **No gray bands in any v2 video**; the zero-fill artefact is gone as for v1+rep. This was expected from the fill change alone.
- heldout_zoom_out: v2@50 has a clean continuation of the pink pattern around the box; v1+rep shows odd extruded carrot-like fragments at the left edge.
  The artifact score (0.397) is close to v1+rep (0.359) and well below v1 (0.603).
- computers_pan_left: v1+rep shows a stretched teal smear at the revealed edge; v2@50 shows a narrower cyan wall region similar to v1 but without gray. The artifact score (0.953) is between
  v1+rep and v1, so the score alone does not favour v2 here; the score is dominated by the cyan/white contrast and is not a complete judge.
- motorcycle_pan_right: v2 and v1+rep both continue the door; v2 has a slightly blurrier brown streak on the right edge at frame 120. Score 0.720 (v1 0.760, v1+rep 0.680).
- flowers_tilt_up (update 77): generated flower mass fills the top band, with high-frequency texture (HF R/V 3.1, above v1+rep 2.6, i.e. more detailed than the valid region); artifact score 0.167 (v1 0.181, v1+rep 0.126).
- Motion: v2 gain is -9 % to +30 % relative to v1 (zoom_out 1.13 vs 1.25, pan_right 1.06 vs 1.11, pan_left 1.00 vs 0.92, tilt_up 1.20 vs 0.93) and cosine is at least 0.996, so the camera motion is preserved at matched physical amplitude.
- Honest reading: at update 50-77 (about 7 % of the planned 1000 updates, warm-started from v1) v2 is on par with the training-free v1+replicate baseline on these artifact numbers
  and is not yet clearly better. This is an early, noisy snapshot; it shows that the training path, checkpoint loading and the v2 inference pipeline work end to end and do not harm motion.

Pulled artefacts (local scratch, not in git): `scratchpad\eval_s3\v2_50_<case>\` (and `v2_77_flowers_tilt_up`) with `video.mp4`, `run.json`, `request.json`,
`border_quality.json`, `measure.json` (tracker gain; `path_adherence.json` is not written by the examples script), frames 40/80/120 at 640 px;
`scratchpad\eval_s3\sheets\<case>_contact_sheet.jpg` (rows v1 / v1+rep / v2) and `scratchpad\eval_s3\table.md`.

## Deviations from the plan

- Training reached **77 updates, not 100**. The cache build finished only at 19:42 (580 of 1908 clips were missing at session start, about 67 min of VAE encoding),
  leaving 70 min before the training stop (~20:53). Update 100 was out of reach even without the GPU sharing.
- Evaluation therefore ran on the **update-50 checkpoint** (first 3 matched cases). After training stopped, the HF backup rotation had pruned step 50, so the 4th case (flowers_tilt_up)
  used **update 77**.
- **Not run (no time):** the extended runs (heldout_zoom_out at u=1 x1.25, motorcycle_pan_right at u=1 8 deg). The `--session-hours 3 --deadline-margin-minutes 15` stop was relaxed to a 1-minute margin
  for the last video, because the ~21:10 hard kill, not the 15-minute margin, was the real limit once training had stopped.
- A first smoke generation (4 steps, update 25) confirmed that the v2 checkpoint loads and that metrics run, before the real jobs; it is not part of the table.

## Next session

`python tools/molab_recover.py --resume-training v2_full` (add `--resume-cache-build` only if the cache is incomplete; it is complete now). Expect ~200 updates per session at 41.5 s.
Avoid concurrent evaluation unless needed: it roughly halves training throughput. The HF backup keeps the latest checkpoints only; pull milestones to the laptop
(`tools/pull_milestones.py v2_full`) after each session. Run the extended-range and the remaining matched cases against a later checkpoint.

## Symmetric artifact score

`artifact_score_v2 = |ln clip(colorfulness_ratio,0.2,5)| + |ln clip(hf_ratio,0.2,5)| + glow` penalizes over-saturated / over-sharp borders as well as desaturated / blurred ones (v1 = `max(0,1-c)+max(0,1-h)+glow` only penalized deficits). Mean over frames with R > 1%, stride 4. v1 = Submission-1 video, v1+rep = v1 checkpoint with replicate guide fill, v2 = 50-step eval (flowers_tilt_up: 77 updates).

| case | run | artifact_v1 | artifact_v2 | colorfulness ratio | HF ratio | glow |
|---|---|---|---|---|---|---|
| heldout_zoom_out | v1 | 0.603 | 0.753 | 0.725 | 0.738 | 0.038 |
| heldout_zoom_out | v1+rep | 0.359 | 0.419 | 0.974 | 0.715 | 0.047 |
| heldout_zoom_out | v2 | 0.397 | 0.440 | 0.900 | 0.769 | 0.064 |
| motorcycle_pan_right | v1 | 0.760 | 1.403 | 1.040 | 0.262 | 0.020 |
| motorcycle_pan_right | v1+rep | 0.680 | 0.977 | 0.902 | 0.458 | 0.023 |
| motorcycle_pan_right | v2 | 0.720 | 1.422 | 1.172 | 0.285 | 0.005 |
| computers_pan_left | v1 | 1.255 | 1.992 | 0.695 | 0.068 | 0.018 |
| computers_pan_left | v1+rep | 0.567 | 1.213 | 1.504 | 0.458 | 0.026 |
| computers_pan_left | v2 | 0.953 | 1.693 | 0.988 | 0.121 | 0.054 |
| flowers_tilt_up | v1 | 0.181 | 0.742 | 1.119 | 1.567 | 0.181 |
| flowers_tilt_up | v1+rep | 0.126 | 1.332 | 1.268 | 2.646 | 0.126 |
| flowers_tilt_up | v2 (77 upd) | 0.167 | 1.539 | 1.261 | 3.142 | 0.167 |

Note: motorcycle/computers v1+rep and flowers show where v1 hid over-sharp or over-saturated borders (e.g. flowers HF ratio 1.5-3.1 scored v1 ~0.15 but v2 0.74-1.54).
