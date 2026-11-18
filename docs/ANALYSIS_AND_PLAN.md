# v2 Analysis & Plan — fixing gray borders and adding calibrated motion strength

Living research log: the diagnosis, the plan and a dated status log of what was run and measured, kept as it was written.
The **Status log** at the bottom was updated whenever a phase finished. Machine-specific locations are written as
placeholders: `<checkpoints_dir>` is the local checkpoint/evaluation folder (`$FPCAMERA_CHECKPOINTS_DIR`) and `<course_folder>`
the course material folder. Neither is part of the repository. Commit hashes quoted in this log refer to the author's original
development history, which is not part of this repository's history; the code they describe is the tree at tag `v2-final-415`.

## 0. Context (v1 = Submission 1, tag `v1-submission1`)

- Base: frozen `Wan-AI/Wan2.2-TI2V-5B-Diffusers` @ `b8fff73…`. Output 1280×704, 121 frames, 24 fps.
  Latent `[48, 31, 44, 80]`. Tokens `[3072, 31, 22, 40]`.
- Adapter `FlowControlLoRA` (`src/fpcamera/flow_control.py`), 14.26 M parameters:
  - 51-channel guide = 48-ch first-image latent warped by inverse homography + 1 validity + 2 displacement.
  - The guide goes through a zero-init Conv3d added to the `patch_embedding` output. The first latent time is zeroed.
  - LoRA r=16 on Q/K/V and FFN in/out of blocks 0–15.
- Data: 36 COCO val2017 photos × {static, zoom_in, zoom_out} = 108 cached clips.
  - Synthetic homography warps with 1.35× overscan (real pixels in revealed margins).
  - Magnitude is always 1.0 (zoom ratio 1.12). Pan/tilt were never trained.
- Loss: teacher-relative velocity MSE (`flow_train_delta.py`):
  `target = teacher(x_t^static) + (z_static − z_moving)/σ`, with `x_t = (1−σ) z_static + σ ε`.
  - σ ~ TruncNormal(0.8, 0.075, [0.6, 1]).
  - The loss is unmasked over future planes.
- Inference: the adapter is active only for σ ≥ 0.6. Teacher-anchored residual
  `v = v_cfg + g (v_adapted − v_frozen)`.
- 300 of 1000 updates, about 20.7 s/update, 23.3 GiB peak, Molab RTX PRO 6000.

## 1. The two gaps

1. **Gray inpainting in newly revealed regions.** Zoom-out and pan expose border regions that have no source pixels.
   These come out as gray, desaturated, noisy bands with a sharp rectangular inner edge on the validity boundary.
   - Clearest example: `gallery/contact_sheets/heldout_zoom_out.png` (Submission 1 folder).
2. **Motion strength is not controllable or calibrated.**
   - Training saw a single magnitude, so the 0–2 slider extrapolates.
   - Observed gain varies from 0.915 to 1.13 across scenes and directions. Tilt is weaker than pan.
   - For drawn paths the `magnitude` argument is silently ignored.

## 2. Diagnosis (hypotheses — confirm in Phase 1)

| ID | Hypothesis | Mechanism | How to test |
|---|---|---|---|
| H1 | **Zero-guide shortcut** | `grid_sample(padding_mode="zeros")` plus `guide*valid` puts latent **0** in invalid regions. Normalized latent 0 ≈ dataset-mean colour, i.e. gray. The adapter largely learns "output ≈ guide", so it paints gray exactly where valid = 0. | Compare final-latent mean/std and decoded saturation in revealed vs valid regions. If revealed latents are ≈ 0 mean and low-variance, H1 holds. Re-run inference with replicate fill: if the bands turn into smeared colour, H1 is confirmed. |
| H2 | **Deterministic regression target** | In revealed regions the true content cannot be predicted from the inputs, and the teacher-relative target does not depend on ε. MSE therefore learns the conditional mean: blurry mean colour, no texture. Nothing ever trains the adapter to *generate* new content. | Decode the x̂₀ prediction at σ≈0.6. Revealed regions should be smooth mean colour. |
| H3 | **Adapter hand-off at σ=0.6** | Below 0.6 the frozen base refines whatever low-frequency layout it inherited. It treats the gray band as content, keeps it, and adds noise texture. | Ablate `camera_min_sigma` ∈ {0.6, 0.3, 0.0}. |
| H4 | **Data scarcity** | Only 36 zoom-out clips, one magnitude, 300 updates. | Addressed by the v2 data plan. |

The literature agrees on what fixes disocclusion and outpainting: TrajectoryCrafter, GEN3C, ViewCrafter, CamCo and ReCamMaster all
condition on a warped render **plus an explicit validity mask** and train with an **ordinary generative (diffusion/flow) loss on real targets**.
The hole content is then *sampled*, not regressed to a mean. v1 lacks exactly that generative term in revealed regions.

## 3. v2 design (adapter-first; decided on 2026-11-16)

### 3.1 Guide fill (fixes H1)
- `build_image_condition(..., fill_mode)` with `fill_mode ∈ {"zeros" (v1), "replicate"}`.
- `replicate` samples with `padding_mode="border"`, so invalid positions carry the nearest edge features instead of latent 0.
- The validity channel is unchanged, so the model still knows which features are extrapolated.
- **Displacement channels stay as they are.** They encode motion magnitude, which is what the strength control needs.

### 3.2 Hybrid objective (fixes H2)
Per update, for one training source with a sampled motion request (`R` = revealed mask = `1 − valid` at latent resolution, computed from the homographies):

- **Pass A — teacher-relative (motion).** Same as v1. Per-location weight `w_A = (1 − R) + α_R · R`, with α_R = 0.1, so the mean-regression target barely acts on holes.
- **Pass B — ordinary conditional flow matching (generation).**
  - Input: `x_B = (1−σ_B) z_moving + σ_B ε_B`. Target: `ε_B − z_moving`.
  - Weight: `w_B = λ_B (1 + β R)`, with λ_B = 1 and β = 4.
  - σ_B ~ TruncNormal(0.7, 0.15, [0.2, 1.0]). This is wider than pass A so the adapter can stay active lower in the schedule at inference.
- First latent plane: timestep 0 and clean image latent in both passes (anchor), and excluded from the loss, as in v1.
- Cost: teacher fwd (no grad) + 2 adapted fwd/bwd. This is about the same as v1's paired step, so expect roughly 20–21 s/update.
- Static requests (10 %): pass A only, which acts as identity regularization.
- Weighted MSE is normalized by the sum of weights.

### 3.3 Data v2 (720p COCO, fixes H4 and the strength gap)
- Sources: COCO 2017, roughly 600 train images plus the 36 v1 images.
  - Filter: min side ≥ 427 px, not grayscale, not near-duplicates.
  - Category round-robin. Keep the v1 4-image dev split and add a 24-image dev split.
- Motions: pan_left/right, tilt_up/down, zoom_in/out (one motion per clip; the UI asks for one motion at a time), plus static.
- **Continuous magnitude** u ~ U[0.15, 1.0] of the trained max:
  - yaw/pitch max = 8°;
  - zoom max ratio = 1.25 (log-scale interpolation).
  - Easing: linear (the UI uses linear).
- **Overscan is set per source, not a fixed 1.45** (see `docs/v2/DATA_V2.md`).
  - Derivation: dense-grid bisection over the max-strength paths.
  - 8° tilt on 16:9 images needs 1.51, so 1.45 is insufficient.
  - Each source gets `max(1.35, required + 0.02)`: about 1.38 for 4:3 COCO and 1.53 for 16:9. Sources needing more than 1.6 are rejected.
  - Per-pixel coverage is asserted, and `BORDER_REFLECT_101` is a safety net.
  - Revealed fraction at the last frame for u = 1: pan 0.16, tilt 0.24, zoom-out 0.36.
- Per source: 1 static + K=2 moving clips (random family + random u). About 636 × 3 ≈ 1.9 k clips × ~10.5 MB ≈ 20 GB.
  **Check Molab disk first.** If it is short, use K=1 or ~400 images.
- Cache row schema (`index.jsonl`):
  `{id, source_id, split, family, direction, magnitude_u, yaw_deg, pitch_deg, zoom_ratio, overscan, latent_file, sha256}`.
  - Tensors in the file: `latents` [48,31,44,80] bf16, `homographies` [121,3,3] f32.
- The cache build must be **resumable** (skip rows whose file exists and whose hash matches) and sharded (`--shard i/n`).

### 3.4 Training run
- Warm start from v1 step-300 weights. The guide conv and LoRA shapes are unchanged.
- Run 1,000 updates (cosine horizon), lr 1e-4, 25 warm-up, cosine to 0.1×.
- **Fault tolerance (Molab sessions drop):**
  - Checkpoint every 25 updates **and** every 10 min.
  - Atomic writes: temp file then rename.
  - `latest.json` pointer.
  - On start, automatically resume from `latest.json` if present (optimizer, scheduler, RNG, data-order cursor).
  - Keep the last 3 checkpoints plus every 250th.
- Cost: measured ~41 s/update (32 s warm) on RTX PRO 6000, so 1,000 updates is about 9-11 h.
- The pilot is the first 150 updates of this same run (`--stop-after-updates 150`, config `configs/train_v2.json`; the separate pilot config was removed), then a visual check on heldout zoom-out, kitchen zoom-out and the pans, then continue the run.

### 3.5 Strength control (one motion at a time, one slider)
- The UI offers a motion dropdown (pan left/right, tilt up/down, zoom in/out) and **one strength slider s ∈ [0, 1]**.
  - s = 0 is static. s = 1 is the trained max (8° or zoom ratio 1.25).
- **Calibration**:
  - Sweep s ∈ {0.25, 0.5, 0.75, 1.0} × 6 directions × 3 dev images.
  - Measure observed amplitude with the existing tracker (`summarize_flow_adherence.py` signed gain).
  - Fit a monotone (isotonic / piecewise-linear) map s → requested u per family, so that observed ≈ s × max.
  - Saved as `configs/strength_calibration_v2.json` and applied by the UI and CLI.
- Inference camera gain g stays at 1. Strength enters only through geometry, which is what the adapter was trained on.

### 3.6 Sampling-time mask-aware step (ablation only, by decision)
- Optionally lower `camera_min_sigma`.
- Optional RePaint-style blending in revealed regions.
- Reported as ablations, not part of the main method.

## 4. Evaluation (lightweight — path adherence is already good)

- **Border quality on revealed regions R** (new script `scripts/measure_border_quality.py`):
  - colourfulness ratio (Hasler–Süsstrunk) R/valid;
  - high-frequency energy ratio (Laplacian variance) R/valid;
  - "gray-band score" = fraction of R pixels with chroma < 12 and 80 < luma < 176, minus the same fraction in the valid region;
  - seam score = gradient magnitude along the validity boundary vs nearby.
  - Report per frame 40/80/120, v1 vs v2.
- **Strength**: observed-vs-requested amplitude curve per direction, before and after calibration.
- **Regression check**: the existing ADE/FDE/gain on the 5 unseen and 6 regression cases. It should not get worse.

## 5. Phases

| Phase | Where | What | Output |
|---|---|---|---|
| P0 | local | Repo import (done, `v1-submission1`). This plan. v2 code: data/cache, guide fill, hybrid trainer, eval, diagnostics script | commits |
| P1 | Molab | Diagnostics on v1: H1–H3 tests; measure disk, VAE encode s/clip, s/update | `docs/experiments/P1_*.md` |
| P2 | Molab | Build the v2 cache (resumable) | cache + index |
| P3 | Molab | Pilot 150 updates, visual check, then the full 1,500-update run | checkpoints |
| P4 | Molab | Calibration sweep, before/after examples on all v1 example images | calibration json, videos |
| P5 | local+Molab | marimo interface v2 (motion + one strength slider), docs, gallery | notebook, gallery |

## 6. Conventions
- Code lives in `src/fpcamera/`. v2 modules are suffixed `_v2` or gated by config flags so v1 stays reproducible.
- Every run writes `config.json`, git commit hash and source hashes next to its checkpoints.
- Large assets are never committed. See `ASSETS.md`.
- Experiment notes go in `docs/experiments/` (one file per phase), and the status log below is updated.

## 7. Status log
- 2026-11-16: v1 imported (commit `624af49`, tag `v1-submission1`). Diagnosis and plan written. Decisions taken:
  - long but resumable Molab runs;
  - 720p COCO data;
  - adapter-first fix, with sampling tricks as ablation only;
  - one motion + one strength slider in the UI.
- Known v1 bug: `notebooks/flow_camera_interface.py:279-280` IndentationError (fix in v2 notebook).
- 2026-11-16: **P0 code complete (CPU-tested only; nothing run on GPU yet).** 66 tests pass.
  - `d3c202e`, `65de7bd`: `build_image_condition(fill_mode)`.
    - `flow_infer.generate`: `guide_fill_mode`, `debug_dir` (x̂0 = x_t − σ·v captured by wrapping `scheduler.step`), and a reusable `pipe=` so the model isn't reloaded.
    - `border_metrics.py`, `scripts/measure_border_quality.py`, `scripts/diagnose_v1_borders.py`.
    - Protocol in `docs/experiments/P1_DIAGNOSTICS_PROTOCOL.md`.
  - `9a7bf7c`, `720e6c9`: `camera_path_v2`, `motion_sampling.py`, `flow_cache_v2.py`, `scripts/select_coco_sources_v2.py`, `scripts/build_cache_v2.py`, `configs/data_v2.json`, `docs/v2/DATA_V2.md`.
  - `7deaf79`: `flow_train_v2.py` (hybrid A+B loss), `checkpointing.py` (atomic, auto-resume, corrupt-checkpoint fallback, SIGTERM checkpoint), `scripts/train_v2.py`, pilot and full configs, `docs/v2/TRAINING_V2.md`. Kill/resume is bit-identical on CPU.
- **Open risks for Molab** (check in this order):
  1. Diagnostics 10-step smoke test. Confirm x̂0 capture fires and that the fp32 VAE decode of 121 frames fits in memory.
  2. `build_cache_v2.py --limit 6`. Measure VAE s/clip, then size the cache (K=1 or 2; ~400 or ~636 sources).
  3. `train_v2.py --check-only`. Measure s/update and peak memory.
  4. The warm start was trained on zero-filled guides while v2 uses replicate fill. Watch the early pilot loss.
  5. COCO selection has only run against a mock. The real annotations need several GB of RAM.
  6. Verdict thresholds in the diagnosis script are heuristic. Sanity-check them on the first real outputs.
  7. Delete a stale `artifacts/gpu.lock` after any crash.
- 2026-11-16: **Molab bootstrapped and smoke-tested.** See `docs/experiments/MOLAB_SETUP.md`.
  - Measured: VAE 6.8 s/clip; training ~41 s/update (32 s warm), 23 GiB; GPU resume works.
  - Risks 1–3 and 5 above are cleared.
  - Molab has no persistence. Checkpoints and cache are backed up to private HF repos (`docs/experiments/BACKUP_AND_RESTORE.md`), and milestones are pulled to the laptop.
  - Run length reduced to 1,000 updates. The pilot is the first 150 updates of the same run.
- 2026-11-16: **P1 diagnostics on v1 done** (`docs/experiments/P1_RESULTS.md`). Conclusions from visual review of the contact sheets plus the ratio metrics; the gray-score metric was miscalibrated and is being replaced:
  - **H1 (zero-filled guide) confirmed.** Replicate fill at *inference only* removes the bands.
    - Heldout zoom-out: colourfulness R/V 0.67 → 0.97, HF R/V 0.39 → 0.72.
    - Motorcycle pan-right: the gray-white blob becomes a clean door/brick continuation.
  - **H3 (σ=0.6 hand-off) rejected.** The bands already exist in x̂0 at σ=0.6, and min_sigma 0.3 or 0.0 doesn't help (slightly worse).
  - **H2 (deterministic mean target) consistent but not isolated.** x̂0 HF R/V is 0.12 with zero fill. Replicate fill gives *stretched edge streaks* rather than generated content, which is what Pass B targets.
  - New ablation row for the paper: **"v1 + replicate fill at inference"**, a training-free baseline between v1 and v2.
  - Pending check: whether replicate fill reduces motion gain.
- In flight: full v2 cache build (1,902 clips) → pilot (updates 1–150) → pilot evaluation → automatic continuation to 1,000 updates.
- 2026-11-16: **Session 1 sandbox was killed at the ~3 h Molab limit.** 640/1902 clips survived on HF. HF duplicate removed (14.4 → 7.4 GB).
  - Added `tools/molab_recover.py` (one-shot, idempotent; ~6 min of steps) and deadline-aware clean stops (`--session-hours 3 --deadline-margin-minutes 15`) on train, cache build, example generation and calibration.
- 2026-11-16: **Session 2** (sandbox start 15:19 UTC, hard kill ≈ 18:19 UTC).
  - Recovery OK. Cache build resumed: 1,268 clips left, about 6.9 s/clip, clean stop ≈ 18:04 UTC.
  - Running concurrently: "v1 + replicate fill" on all 11 example cases (P1b), to measure motion gain and provide the training-free baseline row.
- **Session schedule (plan):**
  - S3: recover → finish cache → start training (deadline-aware). At update 150, run the pilot evaluation *concurrently* on the same GPU (train ~27 GiB + infer ≤ 52 GiB < 95 GiB). Continue training to the deadline.
  - S4–S6: recover + `--resume-training v2_full` (~200 updates/session). After each session, review the eval at the latest checkpoint and decide whether to stop early (plateau) or continue to 1,000.
  - Final session: strength calibration sweep (~3 h, 3 strengths × 6 directions × 2 images at 30 steps; may span 2 sessions), then final examples on all 11 cases plus strength sweeps, then the marimo UI demo on Molab.
- 2026-11-16: **P1b ("v1 + replicate fill", 8/11 cases; `docs/experiments/P1b_REPLICATE_BASELINE.md`).**
  - **Motion is not weakened.** Path-gain ratio vs v1 is 0.98–1.19; family means zoom ×1.08, pan ×1.04, tilt ×1.06. Cosine ≥ 0.986.
  - The artifact score improves in 7/7 applicable cases.
  - Visual review: the gray is gone, but **stretch streaks** appear (teal smear at the computers pan-left edge, smeared top band in flowers tilt-up). This is H2 made visible: replicated edge features, not generated content. It supports the ablation ladder v1 → v1+fill → v2 (fill + Pass B + data).
  - **Important:** the Submission-1 showcase used small amplitudes: pan magnitude 0.5 (2.5°), tilt 0.35 (1.75°), zoom 1.0 (×1.12).
    - Final before/after must run v2 at **matched physical amplitude** (`generate_examples_v2.py --match-v1`).
    - The larger strengths (up to 8° and ×1.25) are shown separately as "v2 extends the range".
  - The cache build stopped cleanly at about 18:03 UTC (session-2 deadline). Verify the HF clip count at the start of S3.
- 2026-11-17: **S3 done** (`docs/experiments/S3_TRAINING_AND_EVAL100.md`). The cache is complete: 1,908/1,908 rows, all on HF.
  - Training ran from 19:43 UTC and stopped cleanly with `DEADLINE_STOP` at step 77. The checkpoint is on HF and the laptop.
  - Healthy: loss_A 0.71 → 0.20 by step 25 (recovered from the zero→replicate fill switch); loss_B ~0.07–0.14. 41.5 s/update alone, 23 GiB.
  - Eval of v2@50/77 at matched amplitude:
    - The automatic score says v2 is level with v1+replicate.
    - **A visual review judged v2 already best** (corrected in the next entry).
      - Heldout zoom-out: clean pattern extension, while v1+rep extrudes fragments.
      - Computers pan-left: a much softer edge band, while v1+rep has a strong teal streak.
  - Metric flaw: the artifact score did not penalise over-saturated streaks. A symmetric `|ln ratio|` version is being added.
  - **Decisions:**
    1. Concurrent eval halved training speed (76–92 s/update), so training sessions now run *training only*.
    2. Next checkpoint review is at ~step 477 (after S4+S5, ~200 updates each). Then decide between stopping around 500 and continuing toward 1,000.
    3. HF pruning keeps the last 3 plus every 250th checkpoint. Pull to the laptop after every session.
- 2026-11-17: **Correction to the S3 reading.** With the symmetric `|ln ratio|` artifact score (`ea314e3`), v2@50/77 is **not** better than v1+replicate:
  - heldout: 0.44 vs 0.42;
  - motorcycle: 1.42 vs 0.98;
  - computers: 1.69 vs 1.21;
  - flowers: 1.54 vs 1.33 (v1 0.74).
  
  The earlier "v2 best" was a visual impression from 2 cases at 50 updates. **Status: inconclusive; do not claim improvement yet.**
  - The reference-free metric itself is flawed: it compares R with the *whole* valid region, so plausible new content that differs from the frame average is penalised (flowers).
  - **Evaluation upgrade (in progress):**
    - (A) A local-ring reference-free score, comparing R with valid pixels 8–48 px from the boundary.
    - (B) A **ground-truth-referenced reveal evaluation** on 12 held-out dev clips. Real overscan pixels exist, so we can report LPIPS, colour-histogram distance and HF deviation for R against the truth for v1 / v1+rep / v2.
    - (C) A dev-set validation loss (L_B on R) per checkpoint as the overfitting monitor.
    - (B) and (C) are the primary evidence for the paper. Visual review is supporting evidence only.
- 2026-11-17: **Evaluation infrastructure ready** (`051fe0d` validate_v2, `f53ced0` GT reveal eval + local-ring score, `56381ac` eval backup/restore + lpips). Session plan:
  - S4 → ~277, then S5 → ~477.
  - **E1:** `molab_recover --checkpoints 250 --resume-eval ...`. Push step 77 from the laptop copy (`<checkpoints_dir>\v2_full\adapter_000077.safetensors`), since HF retention prunes it.
    - Run validate_v2 on step 0 (v1 adapter), 77, 250 and latest.
    - Then eval_dev_reveal for v2@latest.
  - **E2(/E3):** eval_dev_reveal for v1 and v1+rep (run once).
  - **Decision gate:**
    - If dev L_B|R at latest ≥ step 0 → revisit the loss weights or data before training further.
    - If it improves and still trends down → continue toward 1,000.
    - If it plateaus → stop at the best checkpoint.
- 2026-11-17: **S4 (77→201, paused for speedups) and S5 (201→415)** done (`docs/experiments/S4_TRAINING.md`, `S5_TRAINING.md`).
  - Probe: K=20 checkpointed blocks, 72 GiB, 36.8 s/update (~7.5% faster); K=10 OOMs.
  - Losses (S5 windows): loss_A 0.132 → 0.089 (still falling); **loss_B flat at ~0.079 since step ~250**.
  - Cache is packed into 21 shards on HF.
  - Checkpoints 77/201/250/415 are on the laptop.
- **Decision: next session = E1 evaluation at step 415** (loss_B plateau → need held-out evidence before more training). Order:
  1. validate_v2 on step 0 (v1 adapter), 77, 201, 250, 415;
  2. showcase matched-amplitude videos (heldout zoom-out, motorcycle, computers, flowers) + extended u=1 (heldout zoom-out ×1.25, motorcycle 8°);
  3. GT reveal eval v2@415 on 8 pairs, resumable.
- 2026-11-18: **E1 done** (`docs/experiments/E1_EVAL_415.md`; outputs in `<checkpoints_dir>\eval_E1\`).
  - **Held-out (dev) L_B|R:** 0.230 (step 0 / v1) → 0.174 (77) → 0.150 (201) → 0.146 (250) → **0.140 (415)**.
    - Paired 415 vs 250: −0.0058, SE 0.0006, so still significantly improving, but decelerating: −0.019 → −0.008 → −0.0035 per 100 updates.
    - The dev − train gap stays at about −0.009 throughout, so there is no overfitting.
    - The "loss_B plateau" in the S5 training logs was σ/sample noise, not a real plateau.
  - **Visual review** (native-resolution border crops, frame 120):
    - v2@415 removes v1's gray/noisy bands (heldout zoom-out, flowers top strip, motorcycle screen-door texture).
    - It also removes v1+rep's stretch streaks and duplicated edges (computers cyan smear, motorcycle doubled frame edge), and continues the scene sensibly (bento rim and cloth, door frame, wall).
    - Weakest: extended u=1 zoom-out ×1.25 (soft, low-detail orange fill in the corner) and a mild horizontal smear on one plank at motorcycle 8°. No gray anywhere.
  - **Metric caveat:** artifact_score_v2/local disagree with the visual read on flowers (v1 "best" despite speckle noise) and motorcycle (v1+rep "best" despite a doubled edge).
    - For the paper, the quantitative claim rests on validation L_B|R plus the GT reveal eval (LPIPS_R etc.) with baselines. The border scores are secondary.
  - **Strength:** tracker gain at matched amplitude for v2@415 is 0.95–1.17 (v1 0.92–1.25), and 0.99 at u=1. Motion is not weakened.
  - **GT reveal eval v2@415:** 7/8 pairs. LPIPS_R 0.276, PSNR_R 19.4, valid-region control LPIPS_V 0.068. No baselines yet (E2).
  - **Bug found and fixed:** generate() silently defaulted to zero-fill when the run config.json was missing (`7387f6a`). The interface must always pass or resolve `replicate`.
  - **Speedups** (opt-in `--attn-backend`, `--compile-blocks`, `--speed-probe`) were developed on a side branch and merged after E1. They are untested on GPU; the Molab GPU is an RTX PRO 6000 Blackwell, not Hopper, so the cuDNN-attention gain is uncertain.
- **Decision (gate → "improving, decelerating"): one more training session (S6), then stop training.**
  - **S6:** speed probe (≤ 20 min, `--init-from` step 415), then resume v2_full with the recommended flags until the deadline (≈ step 650–700).
  - Rationale: the largest residual weakness is large reveals (u near 1), which is where more Pass-B training helps most. The returns are diminishing, so we cap at one session instead of going to 1000.
  - **E2 + calibration session:**
    - validate_v2 on the S6 final vs 415 and pick the best checkpoint;
    - GT reveal eval for v1, v1+rep and the final model on the same 8 pairs (fill the missing pair);
    - strength calibration sweep.
  - **Final session:** all 11 cases (matched + extended-range gallery) + marimo interface demo on Molab.
- 2026-11-18 (later): **Training stopped at step 415** (after reviewing the E1 images the inpainting was judged solved; S6 dropped). Final model = v2_full step 415.
  - **Final deliverables:**
    1. old-v1 vs new-v2 comparison gallery on the same images;
    2. test on a user-supplied test photo (not distributed: never public, never in git) for pan/tilt/zoom;
    3. calibrated strength slider;
    4. NEW self-sufficient marimo notebook on Molab with code visible as web cells. It installs deps and downloads Wan + adapter + code. Features: image upload, draw trajectory → detect pan/tilt direction, zoom in/out (disables drawing), strength slider, progress, preview + download;
    5. documented codebase (clear comments, what does what, how to use);
    6. ~2-page report: new vs old approach, fixes, experiments, outputs.
  - **Public HF model repo: `D3vil-Reap3r/ie643-wan22-camera-adapter-v2`** (`c4ac081`, `85da288`; `tools/publish_public_hf.py`, re-runnable/idempotent, allow-list + secret scan).
    - Contents: adapter 415 + ckpt json + run config (→ replicate), code zip, inference config, calibration (identity for now).
    - Anonymous download verified. At that point no licence was set; Apache-2.0 was chosen for the public release.
  - **In flight when this entry was written:**
    - **G1 GPU session** (Molab; configured to keep running past the nominal deadline until the sandbox dies, saving continuously). Outputs → `<checkpoints_dir>\G1\` + private HF backup. Tooling commit `9522d30`. Tasks:
      - attn-kernel check (auto vs cudnn);
      - test photo s=1 × 6 directions + s=0.5 × 3;
      - v2 matched for the remaining 7 gallery cases;
      - v1 vs v2 matched on the test photo;
      - then v1 GT reveal eval, extended s=1;
      - calibration fit → `configs/strength_calibration_v2.json`;
      - writes `docs/experiments/G1_GENERATION_CALIBRATION.md`.

      Whatever was missing afterwards went into the next GPU session.
    - **Notebook build** (separate branch, later merged).
      - Builds `notebooks/camera_studio_v2.py` with `FPCAMERA_DRY_RUN=1`, plus `src/fpcamera/trajectory_direction.py` and an optional `progress_callback` in `generate()`.
      - Merged into `D3vil-coder/Main` with a small conflict in `flow_infer.py`.
  - **Next after those:**
    1. If cudnn passes G1's check, enable it as the notebook default.
    2. Re-run `tools/publish_public_hf.py` after the calibration fit and the notebook merge, so the code zip includes `trajectory_direction` and `progress_callback`, and the real calibration is published.
    3. Deploy session: paste the notebook into a BRAND-NEW Molab notebook as visible cells (marimo-pair), cold-start test end to end, and generate with the sample + a drawn trajectory.
    4. Build the gallery: `gallery/index.html` + `gallery_data.json` scaffold exists; v1 vs v2 for 11 cases + photo section + slider section.
    5. Documentation pass (module docstrings, README code map + usage).
    6. 2-page report (PDF).
- 2026-11-18: **G1 done** (`docs/experiments/G1_GENERATION_CALIBRATION.md`, outputs in `<checkpoints_dir>\G1\`).
  - Test photo: all 6 directions are clean, with no gray. Weakest is the tilt_up top-left (hazy invented structure).
  - Gallery: v2 matched for all 11 cases (v1 originals exist).
  - Calibration: fitted, near identity, within ±3% (`configs/strength_calibration_v2.json`).
  - cudnn attention: 3% faster, not adopted.
  - **Open defect:** all 4 `ball_*` v2 videos dim globally to about 0.65× luma with a brown shift. v1 with identical settings does not. Every other video stays within ±5%.
    - Hypothesis: the Pass-B absolute-velocity training at high σ pulls global colour statistics toward the data mean on uniform bright scenes. Could also be seed-specific: all 4 use seed 42.
- **Merged** the Camera Studio v2 notebook (`3ec022c`); 269 tests pass. Public HF re-published (`36fd0487`) with the fitted calibration and code `4af97e5` (trajectory_direction + progress_callback). Verified anonymously.
- **Next (two parallel Molab sessions):**
  - **G2 (GPU):** ball dimming diagnosis on ball_pan_left:
    - seeds 43/44;
    - checkpoints 77/250;
    - base Wan without adapter;
    - v2 s=0 (static);
    - camera_min_sigma 0.8.

    Then the v1 GT reveal eval (8 pairs plus the missing v2 pair), then extended s=1 for the gallery cases. Record per-frame luma for every video.
  - **D1 (fresh notebook):** deploy `notebooks/camera_studio_v2.py` as visible cells into a brand-new Molab notebook; cold-start test; real generations (drawn pan/tilt + zoom).
- 2026-11-18: **D1 done.** The Camera Studio v2 notebook was deployed as 27 visible cells on a fresh Molab sandbox. It installs everything itself, and 3 real generations worked (`361246c`).
  - Tested on additional images: people and detailed photos are great. **Flat or low-texture images show global darkening/colour drift and occasional per-frame pixel distortion**, which 3-frame sheets missed.
  - **The interface is accepted.**
- **Freeze:** tag `v2-final-415` (= `361246c`). Current code + adapter are final on `D3vil-coder/Main` → `master`.
- **Flat-image fix:** separate branch `exp/flat-image-fix`, never merged into the frozen code. Any new adapter goes to private HF paths only, and the public adapter stays frozen.
  - Session F1 (`docs/experiments/F1_FLAT_IMAGE_FIX.md` on that branch): phase 1 diagnosis → inference-time fixes (residual low-frequency/mean removal, adapter window) → training fix only if needed.
- **Submission:** poster + max-2-page report (course instructions are in `<course_folder>`, not part of this repository), written under `docs/submission/`.
- **Then:**
  1. documentation pass: inline comments/docstrings, plus `docs/CODEBASE_GUIDE.md` (files, key functions, how they link, entry points);
  2. move `master` to the final `D3vil-coder/Main`;
  3. publish a **public** GitHub repository (Apache-2.0) with `master`, the `v1-submission1` and `v2-final-415` tags and `exp/flat-image-fix`;
  4. check that no secrets, personal photos or large data are tracked.
