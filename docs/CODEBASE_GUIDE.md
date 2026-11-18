# Codebase guide (v2 camera adapter)

A map of the repository for someone who wants to understand, run or modify it. It is written for the author's own orientation, not as a
submission document. Everything here is derived from the code and the existing docs; where a topic has a deeper document, it is linked.
Paths are relative to the repository root. All code runs with `PYTHONPATH=src` (or the scripts add `src/` themselves).

Contents: [1. Big picture](#1-big-picture) | [2. Directory map](#2-directory-map) | [3. Per-file reference](#3-per-file-reference) |
[4. Main entry points](#4-main-entry-points-recipes) | [5. Glossary](#5-key-concepts-glossary) | [6. Where results and docs are](#6-where-results-and-docs-are) |
[7. Tests](#7-tests) | [8. Conventions and gotchas](#8-conventions-and-gotchas)

---

## 1. Big picture

**What it is.** A 14.26 M parameter adapter (a zero-initialised 51 -> 3072 guide projection plus rank-16 LoRA on 16 of the 30 transformer blocks)
that sits on the *frozen* `Wan-AI/Wan2.2-TI2V-5B-Diffusers` image-to-video model and gives it trajectory control of a virtual camera (pan, tilt, zoom)
from a single photograph. Output is 1280 x 704, 121 frames, 24 fps; latents are `[48, 31, 44, 80]`.

**v1 -> v2 in one paragraph.** v1 (Submission 1, tag `v1-submission1`, 300 of 1000 planned updates) trained on 36 photographs x {static, zoom-in, zoom-out}
at one fixed magnitude with a single teacher-relative velocity loss. It had two visible gaps: newly revealed regions (zoom-out, pan) came out as gray,
desaturated bands with a sharp edge on the validity boundary, and the motion amount was not controllable or calibrated (one trained magnitude, a 0-2 slider
that merely extrapolated). v2 (tag `v2-final-415`, trained to update 415 of a planned 1000, warm-started from the v1 adapter) fixes both: (1) the guide
is filled by *replicating the nearest edge feature* instead of latent 0 (`fill_mode="replicate"`), removing the "output the guide" shortcut for holes;
(2) a **hybrid objective** adds, next to the v1 teacher-relative pass A (down-weighted on holes), an ordinary conditional **flow-matching pass B** on the real
moving clip, weighted 5x on the revealed region, so holes are *generated* rather than regressed to a mean; (3) a new data pipeline renders ~600 COCO
photographs with six directions and many strengths `u` (per-source overscan so no filled pixel ever reaches a target); (4) strength is exposed as one slider
`s in [0,1]` mapped to the trained strength `u` through a per-direction, isotonic-regression **calibration** fitted from measured motion. Inference keeps the
frozen base for text guidance and adds only the adapter's *residual* above the frozen prediction ("teacher-anchored"), and only for noise levels >= 0.6.
A self-installing marimo notebook (Camera Studio v2) wraps all of it for users; the adapter is published on the Hugging Face Hub as
`D3vil-Reap3r/ie643-wan22-camera-adapter-v2`.

### Data flow

```
 COCO 2017 photos                                              user photo + camera request
        |                                                                  |
 [1] select_coco_sources_v2.py  -> data/v2/sources_v2.json                 |
        |                          (+ configs/data_v2.json drives everything below)
 [2] clip synthesis   fpcamera.flow_geometry (camera_path_v2, render_source, overscan)  <- shared with inference
        |             fpcamera.motion_sampling (balanced direction + u per clip)
 [3] VAE latent cache fpcamera.flow_cache_v2.build  <- scripts/build_cache_v2.py
        |             data/v2/cache_*/clips/*.safetensors, references/*, index.jsonl   (resumable, sharded)
        v
 [4] training         scripts/train_v2.py -> fpcamera.flow_train_v2.run_training
        |               Pass A (teacher-relative) + Pass B (flow matching), weights from R = 1 - validity
        |               fpcamera.flow_control.FlowControlLoRA (+ control_lora.ControlLoRALinear), guide from build_image_condition
        v
 [5] checkpoints      fpcamera.checkpointing: latest.json, adapter_N.safetensors, resume_N.pt, ckpt_N.json (atomic, sha256)
        |               + session_deadline (Molab ~3 h kill) + tools/hf_backup.py (daemon) / scripts/restore_from_hf.py
        v
 [6] inference        python -m fpcamera.flow_infer  (fpcamera.flow_infer.generate)
        |               request: motion + strength s --calibration--> u --camera_path_v2--> homographies
        |               guide: build_image_condition(first-image latent, homographies, fill_mode="replicate")
        |               adapter attach -> frozen_camera_difference (flow_anchor): CFG(F0) + gain * (F_adapter - F_frozen), sigma >= 0.6
        |               -> video.mp4 + run.json + request.json
        v
 [7] measurement      border_metrics / measure_border_quality (gray band, seam, ...)   measure_flow_video (tracker) -> gain g
        |            reveal_eval / eval_dev_reveal (generated R vs real photo)      flow_validate_v2 / validate_v2 (dev vs train loss)
        v
 [8] calibration      calibrate_strength.py (72-run sweep) or fit_calibration_from_runs.py -> configs/strength_calibration_v2.json
        |               fpcamera.strength: isotonic fit a_obs(u), inversion u(s), StrengthMap
        v
 [9] notebook         notebooks/camera_studio_v2.py  (draw stroke -> direction, slider -> physical label, generate in-process)
                      published bundle on HF: adapter + ckpt manifest + config + calibration + code zip (tools/publish_public_hf.py)
```

Cross-cutting: `fpcamera.runtime` (GPU lease, atomic json), `fpcamera.wan` (pinned frozen model, hashing), `fpcamera.speed_options` (opt-in kernel options),
`tools/molab_*` (drive the ephemeral GPU sandbox), `tools/hf_backup.py` (survive its death), `tools/local_config.py` (machine-specific paths from the environment).

---

## 2. Directory map

| Path | What lives there |
|---|---|
| `src/fpcamera/` | The Python package: geometry, data cache, adapter, trainer, inference, evaluation metrics, run infrastructure. See section 3.1. |
| `src/fpcamera/provenance/executed_source/` | Byte-exact copies of the v1 trainer sources that produced the frozen step-300 checkpoint (used by `scripts/train_fresh.py` / `train_resume.py` to verify hashes). Do not edit. |
| `scripts/` | Command-line entry points: data selection, cache building, training launch, generation batches, measurement, calibration, evaluation, restore, plus Submission-1 (v1) helpers. See 3.2. |
| `tools/` | Operations around the ephemeral GPU sandbox and the Hugging Face Hub: `molab_sync`, `molab_recover`, `hf_backup`, `pull_milestones`, `publish_public_hf` (+ `model_card_v2.md`), `local_config`. See 3.3. |
| `notebooks/` | marimo notebooks: `camera_studio_v2.py` (final user-facing), `camera_interface_v2.py` (developer UI), `flow_camera_interface.py` (v1 UI). See 3.4. |
| `configs/` | JSON configs: data, training, inference, strength calibration, v1 profiles; `provenance/` holds v1 runtime/profile records. See 3.5. |
| `docs/` | `ANALYSIS_AND_PLAN.md` (living plan + status log), `v1/` (Submission 1 docs), `v2/` (design docs), `experiments/` (session runbooks and results), `submission/` (report, poster, figures, build scripts), this guide. See 6. |
| `tests/` | 288 CPU-only pytest tests (no GPU, no model). See 7. |
| `gallery/` | Offline HTML gallery template (`index.html`, `gallery_data.json`); generated media/sheets are git-ignored. |
| `data/` | Two small COCO source manifests are tracked: `data/flow_poc/pilot_sources.json` (v1) and `data/v2/sources_v2.json` (v2: 636 COCO 2017 sources with image ids, Flickr/COCO URLs, licences, captions and sha256 of the original files; the images themselves are not redistributed, `select_coco_sources_v2.py` re-downloads them). The v2 images and caches (`sources/`, `_work/`, `cache_*/`) are generated on the GPU host by `select_coco_sources_v2.py` / `build_cache_v2.py` and are git-ignored. |
| top level | `README.md`, `ASSETS.md` (large assets kept out of git and where they live), `requirements*.txt` (base / UI / evaluation, pinned), `LICENSE` (Apache-2.0) and `NOTICE` (third-party licences). |

Git-ignored local state (never committed): `artifacts/` (runs, checkpoints, GPU lock), `outputs/`, `runs/`, `data/**/cache_*/`, `data/**/sources/`, `data/**/_work/`,
`*.safetensors`, `*.pt`, `*.mp4`, `*.npz`, `__marimo__/`, `.playwright-mcp/` and local tool folders (see `.gitignore`).

---

## 3. Per-file reference

Notation: "Used by" lists the main importers/callers; "CLI" is the main command if the file has one. Every module now starts with a docstring stating its place in
the pipeline, and every public function/class has a docstring, so `help(module)` or reading the file head is the fastest way to a detailed explanation.

### 3.1 `src/fpcamera/` by pipeline stage

**Geometry, motion and data**

| File | Purpose | Important names | Imports / used by |
|---|---|---|---|
| `flow_geometry.py` | Explicit virtual pan/tilt/zoom geometry, independent of model weights. Homographies H[t] map first view -> frame t; H[0] = I. Also source crop (overscan) and renders. | `VideoContract` (1280x704, 121 f, 24 fps, strides), `camera_path` (v1 modes), `camera_path_v2` (strength u), `motion_amplitudes`, `render_source`, `source_to_first_view`, `coverage_slack` / `assert_source_coverage` / `required_overscan`, `revealed_mask` / `revealed_fraction`, `project_points`, `V2_MOTIONS` (6 directions), `V2_MAX_*` (8 deg, 8 deg, x1.25) | Used by nearly everything: `flow_cache*`, `flow_control` (convention), `flow_infer`, `strength`, `motion_sampling`, `flow_preview`, `reveal_eval`, scripts |
| `motion_sampling.py` | Deterministic, stratified (family, direction, u) per clip: balanced directions and a near-uniform u marginal by construction; independent of call order/shards. | `MotionSampler.sample`, `MotionRequest`, `U_MIN=0.15`, `U_MAX=1.0` | `flow_cache_v2.plan_clips` |
| `flow_cache.py` | **v1** synthetic cache (static / zoom-in / zoom-out) and the shared VAE encoding. | `encode_rgb` (native FP32 VAE, posterior mode, normalise, bf16; reused by v2), `prepare`, `load_cached`, `NORMALIZATION` | `flow_cache_v2`, `flow_train_delta`, v1 scripts. CLI: `python -m fpcamera.flow_cache` |
| `flow_cache_v2.py` | **v2** resumable, sharded cache: one static + K moving clips per source, per-source overscan, atomic files, index rebuilt from files. | `build`, `plan_clips`, `source_overscan`, `render_clip`, `make_row`, `reconcile` / `rebuild_index`, `load_cached_v2`, `critical_config`, `check_disk`, `SourceRejected` | `scripts/build_cache_v2.py`; read by `flow_train_v2` (index format) |
| `flow_view_path.py` | Convert a user-drawn normalised view path to a camera request (v1 trajectory UI). | `camera_view_path` | v1 notebook |
| `trajectory_direction.py` | Stroke -> pan/tilt direction (+ confidence, warnings) for Camera Studio; pure Python, unit-tested. | `detect_direction`, `StrokeDirection` | `notebooks/camera_studio_v2.py` |
| `flow_preview.py` | CPU-only preview of the first-frame crop and requested geometry (SVG arrows). | `conditioning_crop`, `geometry_preview_svg` | notebooks |

**Model (adapter)**

| File | Purpose | Important names | Imports / used by |
|---|---|---|---|
| `control_lora.py` | Frozen-linear + zero-start low-rank delta building block; also the older ray-conditioned `ControlLoRA` (v0, kept for reference). | `ControlLoRALinear`, `ControlLoRAConfig`, `ControlLoRA`, `attach` | `flow_control` |
| `flow_control.py` | **The v2/v1 adapter.** Guide construction and attachment to the transformer. | `build_image_condition` (48 warped + 1 validity + 2 displacement channels; `fill_mode` zeros/replicate), `FlowControlConfig`, `FlowControlLoRA` (`set_condition`, `disabled()` = teacher, `adapter_state_dict`, `close`) | `flow_train_v2`, `flow_infer`, `flow_anchor`, validation |
| `wan.py` | Pinned, frozen Wan loading (`MODEL`, `REVISION`), runtime version check, parameter hashing (proof the backbone never changes), VAE/T5 encoders. | `load_pipeline`, `load_encoders`, `require_pinned_runtime`, `hash_module`, `encode_first_image`, `encode_text`, `pad_text_embeddings` | everything that touches the model |

**Training**

| File | Purpose | Important names | Imports / used by |
|---|---|---|---|
| `flow_train_v2.py` | The v2 hybrid trainer: weights/targets (pure, CPU-testable), config + identity, cache table + resumable sampler + loader, `HybridTrainer`, `run_training` (GPU), speed/memory probes. | `build_pass_a` / `build_pass_b`, `run_microbatch`, `pass_a_weight` / `pass_b_weight`, `weighted_mse`, `sample_truncated_normal`, `derive_seed`, `SourceSampler`, `HybridTrainer`, `run_identity` / `check_identity_change`, `train_until` (deadline rule), `run_training`, `gpu_lease` (stale-lock aware) | `scripts/train_v2.py`; `flow_validate_v2` reuses its loss/loader |
| `checkpointing.py` | Crash-safe checkpoints (tmp + fsync + rename; manifest with sha256; `latest.json` written last; verified fallback; retention). | `CheckpointManager` (`due`, `save`, `load_latest`, `prune`, `archive_existing`), `atomic_write_json` | trainer, tests |
| `session_deadline.py` | Molab ~3 h hard-kill handling: effective deadline, "does one more item fit" rule, EMA, shared CLI flags. | `resolve_deadline`, `should_stop`, `Ema`, `add_deadline_args`, `deadline_from_args`, `UPLOAD_GRACE_SECONDS` | trainer, `build_cache_v2`, `calibrate_strength`, `generate_examples_v2`, `validate_v2`, `eval_dev_reveal` |
| `speed_options.py` | Opt-in, speed-only options: SDPA kernel priority, regional `torch.compile`, probe planning/reporting. | `resolve_attn_backend`, `attention_backend`, `set_block_compile`, `SpeedProbeConfig`, `plan_probe_rows`, `recommend`, `flags_for` | trainer, `flow_infer` |
| `runtime.py` | GPU lease (`artifacts/gpu.lock`), atomic JSON, file sha256. | `gpu_lease`, `write_json`, `file_sha256` | inference, cache building |
| `flow_train_delta.py` | **v1** trainer (paired teacher-relative objective); kept for provenance and as the warm start. | `train`, `camera_velocity_target`, `token_timestep` | `scripts/train_fresh.py` / `train_resume.py` (via the provenance snapshot) |

**Inference**

| File | Purpose | Important names | Imports / used by |
|---|---|---|---|
| `flow_infer.py` | Generate one video from one image; checkpoint + fill-mode resolution; camera request; debug captures; CLI. | `generate` (the big one), `load_inference_pipeline`, `build_camera_request`, `resolve_checkpoint_meta`, `resolve_guide_fill_mode`, `main` | notebooks, `generate_batch`, `calibrate_strength`, `eval_dev_reveal`. CLI: `python -m fpcamera.flow_infer` |
| `flow_anchor.py` | **Teacher-anchored residual** (v2 default): native text CFG on the frozen model + `camera_gain * (F_requested - F_reference)`; separate-pass CFG order asserted; optional batch-2 frozen forwards. | `frozen_camera_difference` (context manager yielding call counters) | `flow_infer` |
| `flow_guidance.py` | Legacy camera guidance `v_static + gain (v_req - v_static)` with sigma gating. | `camera_prediction_guidance` | `flow_infer` (non-anchored path) |
| `strength.py` | Strength calibration: `s -> u` maps, isotonic fit, inversion, physical amplitude labels; pure numpy. | `StrengthMap.s_to_u`, `load_calibration`, `fit_amplitude_curve`, `invert_amplitude_curve`, `build_direction_entry`, `new_calibration`, `physical_amplitude` / `physical_to_u` / `amplitude_label`, `DIRECTIONS` | `flow_infer`, notebooks, calibration scripts |

**Evaluation and measurement**

| File | Purpose | Important names | Imports / used by |
|---|---|---|---|
| `flow_validate_v2.py` | Overfitting monitor: held-out dev vs train-subset losses of saved checkpoints at fixed noise; resumable; reports. | `run_validation`, `evaluate_item`, `fixed_draw`, `TeacherCache`, `render_markdown`, `write_reports` | `scripts/validate_v2.py` |
| `reveal_eval.py` | Ground-truth reveal evaluation: compare generated R with the real photograph. | `select_pairs`, `render_gt`, `evaluate_video`, `region_metrics`, `LpipsScorer`, `aggregate` | `scripts/eval_dev_reveal.py`, `render_reveal_sheets.py` |
| `border_metrics.py` | Reference-free border quality (pure numpy/cv2): gray-band fraction, seam score, colorfulness, high-frequency energy, artifact scores. | `revealed_mask_pixels`, `artifact_score_v2`, `artifact_score_local`, `frame_metrics`, `mean_metrics`, `latent_stats` | measurement scripts, `reveal_eval` |
| `visual_metrics.py` | Semantic/appearance preservation (DINOv2 CLS cosine, first-frame SSIM, flow-compensated temporal L1) and a joint acceptance helper. | `evaluate_visuals`, `joint_acceptance`, `DINOv2SemanticScorer`, `PreservationThresholds` | v1-era; not imported by any current script or test |
| `flow_validate.py`, `flow_validate_delta.py` | **v1** paired denoising diagnostics (`flow_validate.py` is legacy and imports the absent `flow_train`). | `validate` | v1 only |

`__init__.py` only carries the package docstring and `__version__`.

### 3.2 `scripts/`

Run from the repository root; most accept `--project-root .`. "GPU" = needs the pinned Wan runtime and a CUDA GPU (Molab); "CPU" = runs anywhere.

**Data (v2)**

| Script | Purpose | Notes |
|---|---|---|
| `select_coco_sources_v2.py` | Select ~600 COCO 2017 photographs (category round-robin, size/colour/near-duplicate filters, keeps the 36 v1 sources), split train/dev, download, write `data/v2/sources_v2.json`. | internet; resumable; `--plan-only` |
| `build_cache_v2.py` | Build / dry-run the v2 latent cache via `flow_cache_v2.build`. | GPU (dry run CPU); `--dry-run --preview-dir`, `--shard i/n`, `--merge-index`, `--derive-overscan`, deadline flags |

**Training and checkpoints**

| Script | Purpose | Notes |
|---|---|---|
| `train_v2.py` | Launcher of the v2 hybrid trainer. Rerunning the same command resumes. Exit 0 = done / clean deadline stop, 75 = interrupted after checkpoint. | GPU; `--check-only`, `--memory-probe`, `--speed-probe` |
| `validate_v2.py` | Overfitting monitor (dev vs train subset) for saved checkpoints. | GPU |
| `restore_from_hf.py` | Restore a run, the cache (per file or shards), sources, eval outputs or exact checkpoints from the private HF repos. Idempotent, sha256-verified. | no torch needed |

**Generation**

| Script | Purpose | Notes |
|---|---|---|
| `generate_batch.py` | Queue runner: one loaded pipeline, JSON job files dropped into a folder; idempotent. | GPU |
| `generate_examples_v2.py` | Matched before/after cases (v1 vs v2 at equal physical amplitude) + strength sweeps + offline gallery. | GPU + CPU metrics |
| `generate_camera.py` | v1-style single-video CLI (`configs/inference_default.json`); for v2 use `python -m fpcamera.flow_infer`. | GPU |
| `diagnose_v1_borders.py` | Phase P1: variants (zeros/replicate fill, camera_min_sigma, frozen) to diagnose the v1 gray borders. | GPU |

**Measurement and evaluation**

| Script | Purpose | Notes |
|---|---|---|
| `measure_border_quality.py` | Border metrics of run folders (`border_quality.json`, CSV/markdown). | CPU |
| `measure_flow_video.py` | **The tracker**: Shi-Tomasi + pyramidal Lucas-Kanade sparse flow, optionally compared with the request. | CPU |
| `summarize_flow_adherence.py` | Tracks vs requested homographies: `signed_motion_gain_pathwide` etc. | CPU |
| `measure_flow_preservation.py`, `measure_temporal_consistency.py` | Appearance preservation / temporal consistency after compensating observed motion. | CPU |
| `measure_runs.py` | Border metrics + tracker gain for many run folders in parallel, one table. | CPU |
| `eval_dev_reveal.py` | Ground-truth reveal evaluation on the dev split for several models; resumable. | GPU gen + metrics |
| `render_reveal_sheets.py`, `compare_videos.py` | Contact sheets with R outlined / border crops; per-frame PSNR of two videos (kernel-equivalence check). | CPU |
| `plot_flow_learning.py`, `verify_flow_cache_roundtrip.py` | v1 learning-curve plot; VAE round-trip check of cached clips. | v1 |

**Calibration**

| Script | Purpose | Notes |
|---|---|---|
| `calibrate_strength.py` | 72-run sweep (u x 6 directions x N dev images), measure gain, fit isotonic `a_obs(u)`, invert to `u(s)`, write JSON + plots + report, optional `--verify`. | GPU sweep, CPU fit |
| `fit_calibration_from_runs.py` | Same fit from existing v2 videos of any origin (no sweep). Produced `configs/strength_calibration_v2.json`. | CPU |

**Submission 1 (v1) and packaging helpers**

`bootstrap_artifacts.py`, `freeze_submission_manifest.py`, `verify_frozen_package.py`, `train_fresh.py`, `train_resume.py`, `render_camera_suite.py`,
`render_flow_comparison.py`, `build_submission_gallery.py`: restore/verify/freeze the v1 package and re-render v1 figures.
They read `submission_manifest.json` and `artifacts/flow_poc_all/` which are not in git (see `ASSETS.md`).

### 3.3 `tools/`

| Tool | Purpose | Run where |
|---|---|---|
| `molab_sync.py` | Drive a remote marimo (Molab) kernel from the laptop: `exec`, `push`/`pushdir`, `pull`/`pulldir` (chunked, resumable, sha256), `sync-code` (incremental git bundle). Class `Molab`. Needs `MOLAB_URL` and `MOLAB_TOKEN` in the environment (never committed). | laptop |
| `molab_recover.py` | One-shot recovery of a fresh sandbox: probe, code, deps, token, model, restore, extras, (build), (train); every step idempotent; long work detached and polled. | laptop |
| `hf_backup.py` | Daemon next to the trainer: uploads verified checkpoints (retention newest 3 + every 250th), the latent cache (batched, ledger) or any eval directory to private HF repos; `pack-cache` builds ~1 GB shards. Stdlib + `huggingface_hub` only, so it can live outside the repo. | sandbox |
| `pull_milestones.py` | Pull milestone checkpoints (every 250th + latest) from the backup to the laptop, verify, prune. Output folder `--out-dir` (default `$FPCAMERA_CHECKPOINTS_DIR`). | laptop |
| `publish_public_hf.py` | Publish the final adapter to the PUBLIC HF model repo from an explicit allow-list (adapter, manifest, config, calibration, inference defaults, code zip, model card), with secret scan, one commit, anonymous verification. `model_card_v2.md` is its card template. | laptop |
| `local_config.py` | The only place that knows machine-specific locations, both from the environment: `FPCAMERA_KEYS_ENV` (the `KEY=value` file with the Hugging Face tokens; default `<repo root>/keys.env`, git-ignored) and `FPCAMERA_CHECKPOINTS_DIR` (pulled run folders and evaluation outputs; default `<repo root>/artifacts/checkpoints/v2`). Used by `molab_recover`, `pull_milestones`, `publish_public_hf` and `docs/submission/build/make_figures.py`. | laptop |

### 3.4 `notebooks/`

| File | Purpose |
|---|---|
| `camera_studio_v2.py` | **Final user-facing notebook.** Self-installing (packages, base model, adapter bundle), loads the pipeline once, draw-a-stroke direction detection, strength slider with physical label, live progress, mp4 + download. Cells: S0 config, S1 helpers, S2 GPU check, S3 packages, S4 base model, S5 adapter + code bundle, S6 modules + calibration, pipeline, `run_generation`, then the UI. `FPCAMERA_DRY_RUN=1` gives a GPU-less placeholder. |
| `camera_interface_v2.py` | Developer/evaluation UI that expects a repository checkout and a local checkpoint; runs `flow_infer` in a child process. |
| `flow_camera_interface.py` | v1 UI (drawn path, 0-2 magnitude). |

### 3.5 `configs/`

JSON cannot hold comments, so the configs are described here (changing a file changes its sha256, which is recorded in `run.json` for calibrations; do not annotate in place).

| File | What it is |
|---|---|
| `data_v2.json` | Schema `fpcamera_data_v2`: video contract, camera maxima (8 deg, 8 deg, x1.25, hfov 60), sampling (`u_min` 0.15, `u_max` 1, 1 static + 2 moving clips per source), overscan policy (`per_source`, floor 1.35, margin 0.02, cap 1.6), render, sources manifest, cache dir, COCO selection rules. `strength.load_max_amplitudes` reads its `camera` section at inference. |
| `train_v2.json` | The v2 run `v2_full`: seed, 1000-update cosine schedule, effective batch 2, lr 1e-4, objective weights and sigma windows, `fill_mode: replicate`, checkpoint cadence, backbone hash, caches, `init_from` (v1 adapter). |
| `inference_v2.json` | Defaults for `flow_infer --config` and the developer UI (has a `_comment`). |
| `inference_default.json` | v1 defaults (mode/magnitude) for `scripts/generate_camera.py`. |
| `strength_calibration_v2.json` | The fitted calibration (G1 data, checkpoint 415): per-direction `s -> u` knots plus fit diagnostics. `strength_calibration_v2.identity.json`: identity map (`u = s`). |
| `train_fresh_300.json`, `train_resume_300.json` | v1 training profiles (Submission 1). |
| `provenance/*.json` | v1 runtime/profile records (verified runtime, native reference, transformer config). |

### 3.6 `docs/`, `tests/`, `gallery/`, `data/`

Described in sections 6, 7 and 2.

---

## 4. Main entry points (recipes)

All commands assume the pinned runtime (`requirements.txt`: torch 2.11.x, diffusers 0.36.0, transformers 4.57.6) and, for GPU steps, a CUDA GPU with ~48 GB+ (Molab RTX PRO 6000 used).
`export PYTHONPATH=src` once per shell.

**Build the cache** (details: `docs/v2/DATA_V2.md`)
```bash
python scripts/select_coco_sources_v2.py --project-root .                      # sources_v2.json (internet, resumable)
python scripts/build_cache_v2.py --project-root . --dry-run --limit 12 --preview-dir artifacts/preview_v2   # geometry check, no model
python scripts/build_cache_v2.py --project-root .                              # full build, interrupt and rerun at will
```

**Train, with resume** (details: `docs/v2/TRAINING_V2.md`)
```bash
python scripts/train_v2.py --config configs/train_v2.json --run-dir artifacts/v2_full \
  --cache data/v2/cache_1280x704_f121_v2/index.jsonl:1.0 \
  --init-from artifacts/flow_poc_all/adapter_000300.safetensors [--stop-after-updates 150] [--session-hours 3]
# Resume = run the SAME command again (auto-resumes from <run-dir>/latest.json). --no-resume archives the old state first.
# exit code 0 = finished or clean deadline stop, 75 = interrupted after a checkpoint (rerun), other = failure (rerun resumes).
# On a new sandbox: python scripts/restore_from_hf.py --run v2_full --run-dir artifacts/v2_full   then the same train command.
```
Speed-only options (never part of the run identity): `--ckpt-blocks K`, `--attn-backend {auto,cudnn,flash,efficient}`, `--compile-blocks`; choose with `--check-only --memory-probe` / `--speed-probe`.

**Generate**
```bash
# CLI (one video)
python -m fpcamera.flow_infer --project-root . --checkpoint artifacts/v2_full/checkpoints/adapter_000415.safetensors \
  --image photo.jpg --output outputs/run1 --motion pan_left --strength 0.6 --calibration configs/strength_calibration_v2.json
# or --config configs/inference_v2.json for the defaults; --amplitude-deg 3 / --zoom-ratio 1.12 give physical amplitudes (bypass the calibration)
```
```python
# Python API (load Wan once)
from fpcamera.flow_infer import generate, load_inference_pipeline
from fpcamera.runtime import gpu_lease
with gpu_lease(root):
    pipe = load_inference_pipeline()
    generate(root, ckpt, "photo.jpg", "outputs/run1", motion="pan_right", strength=0.75,
             calibration="configs/strength_calibration_v2.json", pipe=pipe, verify_backbone=False)
```
Outputs per run folder: `video.mp4`, `run.json` (everything + hashes, written last), `request.json` (homographies), `input.png`, `frame_*.png`, `progress.json`.
For many videos use `scripts/generate_batch.py` (queue) or `scripts/generate_examples_v2.py`.

**Evaluate** (details: `docs/v2/EVALUATION.md`)
```bash
python scripts/validate_v2.py --run-dir artifacts/v2_full --cache data/v2/cache_1280x704_f121_v2/index.jsonl --steps 77,250,415   # dev vs train loss, fixed noise
python scripts/eval_dev_reveal.py --project-root . --model v2=artifacts/v2_full/checkpoints/adapter_000415.safetensors              # generated R vs real photo
python scripts/measure_border_quality.py run1 run2 --out-md border.md                                                              # gray band / seam / ... (CPU)
python scripts/measure_flow_video.py --video run1/video.mp4 --output run1/flow.json --request run1/request.json                    # the tracker (CPU)
python scripts/measure_runs.py --root outputs --workers 4 --md table.md                                                            # border + tracker gain for many runs
```

**Calibrate** (details: `docs/v2/CALIBRATION.md`)
```bash
python scripts/calibrate_strength.py --project-root . --checkpoint artifacts/v2_full/checkpoints/adapter_000415.safetensors --verify   # sweep + fit + verify
python scripts/fit_calibration_from_runs.py --root G1 --checkpoint-sha 83df1e3f --out configs/strength_calibration_v2.json              # fit from existing videos (CPU)
```
(`--out`, `--report-md`, `--rows-json`, `--plots-dir` control the outputs; `--exclude` skips folders; needs `--checkpoint-sha`, the adapter hash prefix the videos must come from).

**Publish to Hugging Face**
```bash
python tools/publish_public_hf.py --dry-run     # stage + secret scan + list files; no upload
python tools/publish_public_hf.py               # one commit to <user>/ie643-wan22-camera-adapter-v2, then anonymous download + import check
```
Needs `HF_WRITE_TOKEN`, read from the `KEY=value` file named by `FPCAMERA_KEYS_ENV` (default `<repo root>/keys.env`, git-ignored; see `tools/local_config.py`); the checkpoint folder comes from `FPCAMERA_CHECKPOINTS_DIR`.

**Run the notebook** (details: `docs/v2/INTERFACE_V2.md`)
```bash
FPCAMERA_DRY_RUN=1 marimo edit notebooks/camera_studio_v2.py   # local, no GPU, no downloads: placeholder videos, full UI
marimo run  notebooks/camera_studio_v2.py                      # real: needs GPU; installs, downloads ~32 GB base + adapter bundle, then generates
python -m marimo check notebooks/camera_studio_v2.py           # lint (must stay clean)
```
On Molab: open the file in a GPU sandbox and run all cells (first run ~10 min of setup, then ~6 min per 50-step video).
Environment switches: `FPCAMERA_DRY_RUN`, `FPCAMERA_WORK_DIR`, `FPCAMERA_USE_REPO_SRC`, `FPCAMERA_ADAPTER_REPO`.

**Molab and backup tooling** (details: `docs/experiments/MOLAB_SETUP.md`, `BACKUP_AND_RESTORE.md`, `SESSION_CHUNKING.md`)
```bash
export MOLAB_URL=https://sb-<id>.sb.molab.run MOLAB_TOKEN=<token>             # never commit
python tools/molab_sync.py exec -c "print(1+1)"                               # run code in the sandbox kernel
python tools/molab_sync.py push LOCAL REMOTE ; python tools/molab_sync.py pull REMOTE LOCAL ; python tools/molab_sync.py sync-code
python tools/molab_recover.py --resume-training v2_full [--resume-cache-build] [--plan]   # rebuild a fresh sandbox, restore, relaunch training
python tools/hf_backup.py watch-run artifacts/v2_full --path-in-repo runs/v2_full         # daemon: back up checkpoints (also watch-cache, watch-dir, pack-cache)
python scripts/restore_from_hf.py --run v2_full --run-dir artifacts/v2_full               # restore (also --cache, --sources, --eval, --checkpoints)
python tools/pull_milestones.py v2_full                                                   # pull milestone checkpoints to the laptop
```
Because Molab hard-kills a sandbox ~3 h after it starts, long jobs take `--session-hours 3 --deadline-margin-minutes 15` (stop cleanly before the kill),
the trainer checkpoints atomically, and `hf_backup` mirrors everything off the sandbox.

**Quickstart for a user (adapter from the Hub)**: the bundle `D3vil-Reap3r/ie643-wan22-camera-adapter-v2` holds `adapter/adapter_000415.safetensors`,
`adapter/ckpt_000415.json`, `adapter/config.json` (keep the three together), `configs/*`, and `code/fpcamera_src.zip`. The simplest route is the notebook; a manual route is
`huggingface_hub.snapshot_download(...)` then `python -m fpcamera.flow_infer --checkpoint <bundle>/adapter/adapter_000415.safetensors --calibration <bundle>/configs/strength_calibration_v2.json ...`.

---

## 5. Key concepts glossary

| Term | Meaning |
|---|---|
| **R / validity** | `R = 1 - valid`. *Valid* (channel 48 of the 51-channel guide, also `border_metrics` for pixels) marks cells whose source point lies inside the first image; `R` is the **revealed** region: newly exposed content that does not exist in the conditioning photo and must be generated. Used as the weight map of both training passes and as the evaluation region. |
| **Guide** | The 51-channel condition `[B,51,T,H,W]` built by `build_image_condition` from the first-image latent and the homographies: 48 warped latent channels, 1 validity, 2 forward-displacement channels. Added (through a zero-init Conv3d) to the patch-embedding output; the first latent plane is never touched. |
| **Replicate fill** | `fill_mode="replicate"`: outside the first image the warp copies the nearest edge feature (`padding_mode="border"`) instead of latent 0 (v1 `zeros`, which decodes to mid-gray and let v1 "output the guide" for holes). Validity still marks those cells. Recorded in the checkpoint's `config.json` and read back by `flow_infer`. |
| **Pass A / Pass B** | Training objective per microbatch. **A**: teacher-relative velocity loss (v1's): `target = teacher(x_A) + (z_static - z_moving)/s_A`, weight `(1-R) + 0.1 R`, `s_A ~ TruncNormal(0.8, 0.075, [0.6,1])`. **B**: plain conditional flow matching on the moving clip, `target = eps - z_moving`, weight `1 + 4R`, `s_B ~ TruncNormal(0.7, 0.15, [0.2,1])`. Static requests (10 %) run A only. Noise level is called `s` in the training docs and `sigma` in the code; do not confuse it with the strength `s` below. |
| **Teacher** | The frozen Wan model with the adapter disabled (`FlowControlLoRA.disabled()`); pass A's target and the inference baseline. |
| **u vs s** | `u` = requested strength in *training units* in [0,1] (1 = the trained maximum: 8 deg pan/tilt or x1.25 zoom; clips use u in [0.15, 1]). `s` = the UI strength in [0,1] meant to equal the *observed* fraction of that maximum. A per-direction monotone map `s -> u` (the **calibration**) corrects the adapter's imperfect realisation. `s = 0` is a static camera. |
| **gain `g`** | The tracker's signed motion gain: observed point displacement projected on the requested displacement (`signed_motion_gain_pathwide`). `a_obs(u) = g(u) * u` is the realised amplitude; calibration inverts a monotone fit of `a_obs(u)`. |
| **L_B\|R** | Held-out pass-B loss restricted to the revealed region (weight `R`): the number that says whether holes are being filled with real content. Reported by `validate_v2.py` next to `L_A`, `L_A\|valid`, `L_B`. |
| **Teacher-anchored residual** | Inference rule (`flow_anchor.frozen_camera_difference`): `v = CFG_text(F0) + camera_gain * (F_adapter(requested guide) - F0_positive)`. The adapter only contributes the camera-induced difference to the *positive* prediction; negative-text and low-noise steps use the frozen model. Matches pass A's target construction. |
| **`camera_min_sigma`** | Noise level below which the adapter is switched off (default 0.6 = lower bound of pass A's window): fine-detail steps stay pure Wan. Read from the checkpoint's recorded policy. |
| **Overscan** | Extra real source margin around the first view (1.35-1.6x per source) so the strongest path in every direction renders from real pixels; clips are additionally coverage-asserted. This is what makes the training target for revealed regions *real* content. |
| **Static / moving / family** | Per source: one static clip (no camera motion) and K moving clips; `family` in {pan, tilt, zoom} with a direction; six directions in `V2_MOTIONS`. |
| **Run identity** | The part of the config that defines what is trained (schedule, objective, architecture, fill mode, contract, cache index hashes + weights). A resume with a different identity is refused unless `--allow-identity-change KEY` / `--allow-config-change`. Speed options and checkpoint cadence are excluded. |
| **`--ckpt-blocks K`** | Activation-checkpoint only the first K of the 30 transformer blocks (speed/memory trade-off; session 5 used K=20: 71.8 GiB peak, about 7.5 % faster than checkpointing all blocks; K=10 does not fit in 85 GiB). Not part of the run identity. |
| **DEADLINE_STOP** | Greppable line logged when a job stops itself before the Molab kill; `run_status.json` has `stopped_reason: "deadline"`, exit code 0, next launch resumes. |
| **Run folder** | One generation: `run.json` (completion marker, written last), `request.json`, `video.mp4`, `input.png`, measurement JSONs (`border_quality.json`, `measure.json`). Never regenerated if `run.json` exists. |
| **Molab** | marimo's cloud notebook with GPUs; sandboxes are ephemeral (hard kill ~3 h after start), hence deadline handling, atomic checkpoints and HF backup/restore. |

---

## 6. Where results and docs are

| Location | Content |
|---|---|
| `docs/ANALYSIS_AND_PLAN.md` | The living plan and **status log**: context, the two gaps, hypotheses H1-H4, v2 design decisions, phase results, current state. Start here for "why". |
| `docs/v2/` | Design docs of the final system: `DATA_V2.md` (sources, motion model, overscan, cache), `TRAINING_V2.md` (objective, fault tolerance, commands, monitor), `INFERENCE_V2.md` (checkpoint layouts, CLI, Python), `EVALUATION.md` (metrics and caveats, protocol), `CALIBRATION.md` (s/u, fit, inversion, file format), `INTERFACE_V2.md` (both notebooks). |
| `docs/experiments/` | Dated runbooks and results: `P1_*` (v1 gray-border diagnosis, replicate baseline), `S3/S4/S5_*` (training sessions), `E1_EVAL_415.md` (held-out loss, reveal evaluation, showcase at step 415), `G1_GENERATION_CALIBRATION.md` (generation + calibration data), `MOLAB_SETUP.md`, `BACKUP_AND_RESTORE.md`, `SESSION_CHUNKING.md`. |
| `docs/v1/` | Submission 1: architecture, data, training, inference, interface, reproducibility (`PROJECT_README_v1.md` = the original README). |
| `docs/submission/` | Submission 2 deliverables: 2-page report (docx + PDF), A0 poster (HTML + PDF), `figures/`, `assets/`, `build/` scripts, own `README.md` (rebuild instructions). |
| `gallery/` | Offline HTML gallery template + data; media are git-ignored. |
| Hugging Face | Public model repo `D3vil-Reap3r/ie643-wan22-camera-adapter-v2` (adapter step 415 + code zip). Private backup repos hold runs, the latent cache and eval outputs (see `BACKUP_AND_RESTORE.md`). |
| Tags | `v1-submission1` (baseline import), `v2-final-415` (frozen model and code). Branch `exp/flat-image-fix` = separate flat-image drift experiment. |

---

## 7. Tests

`python -m pytest -q` runs 288 CPU-only tests (about 2 minutes). Nothing needs a GPU or the Wan weights; the model is replaced by a tiny mock transformer (plus the real `FlowControlLoRA`) where needed.

| Test file | Covers |
|---|---|
| `test_train_v2_logic.py` | Pass A/B targets and weights (incl. agreement with v1), truncated normal, only the adapter changes in an update, kill-and-resume is bit-identical, checkpoint layout / retention / corruption fallback, exact sampler resume, v1 and v2 cache loading, strict warm start, log truncation on resume, session-deadline stop + identical resume, targeted identity changes. |
| `test_validate_v2.py` | Overfitting monitor: fixed draws, teacher cache, results resume/refusal, reports. |
| `test_condition_fill.py` | `build_image_condition` fill modes (zeros vs replicate), validity/displacement invariants. |
| `test_geometry_v2.py` | v2 geometry: amplitudes, coverage/overscan, motion sampling balance, cache I/O and reconcile, source selection helpers. |
| `test_strength.py` | Isotonic fit, inversion, `StrengthMap`, calibration file validation, gain pipeline. |
| `test_infer_v2_args.py`, `test_progress_callback.py` | `flow_infer` argument plumbing (`build_camera_request`, fill-mode and checkpoint resolution) and the progress hook, without a model. |
| `test_trajectory_direction.py` | Stroke -> pan/tilt detection, confidence, warnings. |
| `test_border_metrics.py`, `test_reveal_eval.py` | Border/local-ring metrics and the ground-truth reveal evaluation on synthetic frames. |
| `test_examples_v2.py` | Examples pipeline: metrics, sheets, summary, offline gallery. |
| `test_speed_options.py`, `test_attn_compile_options.py` | Partial activation checkpointing, batched frozen forwards (lossless equivalence), attention backend and compile options, speed probe. |
| `test_hf_backup_dir.py`, `test_hf_backup_sources.py`, `test_cache_shards.py`, `test_restore_backoff.py` | Backup daemon logic (dirs, sources allow-list, shards) and restore retry/backoff. |
| `test_molab_recover.py`, `test_publish_public_hf.py` | Recovery planning/dedup and the public-publish allow-list and secret scanner. |
| `test_local_config.py` | Keys-file and checkpoint-folder resolution from the environment, token loaders, and guards that no laptop path or personal photo name is committed in code, docs or configs. |
| `test_diagnose_p1.py` | P1 diagnosis case building and verdict logic. |

---

## 8. Conventions and gotchas

* **The frozen base is never modified.** `hash_module` before/after proves it; trainers and `generate` raise if it changes.
* **Everything is resumable and write-once.** Atomic writes (tmp + fsync + rename), completion markers last (`run.json`, `latest.json`), sha256 on every artefact. Rerun a command to continue.
* **GPU lease.** `artifacts/gpu.lock` serialises GPU jobs (training, cache build, generation). The trainer replaces a stale lock; `runtime.gpu_lease` does not (delete the file or use `--break-lock`).
* **Train/inference consistency.** Fill mode and the pinned runtime must match training; `flow_infer` reads the fill mode from the checkpoint's `config.json` and refuses to fall back to zeros silently.
* **v1 code is kept** (`flow_cache.py`, `flow_train_delta.py`, `flow_validate*.py`, many scripts) for provenance and as the warm start; v2 modules are the ones suffixed `_v2` plus `flow_control`, `flow_infer`, `flow_anchor`, `strength`, `motion_sampling`.
* **Pinned runtime** (`wan.require_pinned_runtime`): diffusers 0.36.0, transformers 4.57.6, torch 2.11.x.
* **Line endings.** Many files are stored with CRLF in the working tree (autocrlf on Windows); comments/docstrings edits preserve whatever a file already uses.
* **Docs are numbers-of-record.** `docs/experiments/*` quote single runs (seed 42) with their caveats; read the caveats before quoting a figure.
