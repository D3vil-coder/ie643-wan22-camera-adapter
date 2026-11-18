# P1 — Diagnostics of the v1 gray borders (Molab GPU)

Goal: decide which of H1 (zero-guide shortcut), H2 (regression target) and H3
(hand-off at sigma 0.6) explain the gray bands in newly revealed regions R of the v1
adapter (see `docs/ANALYSIS_AND_PLAN.md` section 2). Tooling added for this phase:

| File | Purpose |
|---|---|
| `src/fpcamera/flow_control.py::build_image_condition(fill_mode=)` | `zeros` (v1, bit-identical) or `replicate` (border padding, guide not multiplied by validity) |
| `src/fpcamera/flow_infer.py::generate(guide_fill_mode, debug_dir, debug_sigmas, pipe, verify_backbone)` | inference switches + latent capture; `load_inference_pipeline()` loads Wan once |
| `src/fpcamera/border_metrics.py` | revealed masks (pixel + latent), colourfulness / HF / gray-band / seam, latent R-vs-valid stats |
| `scripts/measure_border_quality.py` | metrics for any run folder(s) (also for v2 later) |
| `scripts/diagnose_v1_borders.py` | the P1 driver (this protocol) |

## 0. Assumptions about the Molab session

- Repo cloned at `$ROOT` (below `/workspace/IE643-Novelty` is only an example).
- v1 artifacts copied so that `$ROOT/artifacts/flow_poc_all/adapter_000300.safetensors`
  **and** `adapter_000300.json` exist (the sidecar is required; sha256 is checked,
  expected `9e2963f2...efdbc`, see `ASSETS.md`).
- `Wan-AI/Wan2.2-TI2V-5B-Diffusers` revision `b8fff73...` is in the HF cache (the code
  uses `local_files_only=True`, so there is no download at run time) and the pinned
  runtime is installed (`torch 2.11.x`, `diffusers 0.36.0`, `transformers 4.57.6`;
  `requirements.txt`). Plus `opencv-python`, `imageio`, `imageio-ffmpeg`, `safetensors`.
- Input images (not tracked in git, they live in the Submission 1 folder under
  `examples/inputs/`): the **heldout** image (zoom-out case; its sha256 is
  `91e259d5...91370` in `configs/provenance/camera_control_evaluation_profile.json`) and the
  **motorcycle** image (pan-right case). Copy them anywhere and pass the paths
  with `--images name=path`. If they are under `$ROOT/examples/inputs/` with
  `heldout` / `motorcycle` in the file name, `--images` can be omitted.
  Any other photo works too (e.g. a COCO val image from `data/flow_poc/`), but the
  verdict thresholds were chosen for these two.

## 1. Environment probe (run first, in a notebook cell)

```python
import os, shutil, sys, pathlib, importlib.metadata as md
import torch
ROOT = pathlib.Path(os.environ.get("FPCAMERA_ROOT", ".")).resolve()
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda)
print("diffusers", md.version("diffusers"), "| transformers", md.version("transformers"))
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU", p.name, f"{p.total_memory/2**30:.1f} GiB;",
          f"free {torch.cuda.mem_get_info()[0]/2**30:.1f} GiB")
else:
    print("NO CUDA")
hf = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".cache/huggingface"))
for label, path in (("repo", ROOT), ("hf cache", hf)):
    path.mkdir(parents=True, exist_ok=True)
    u = shutil.disk_usage(path)
    print(f"disk[{label}] {path}: free {u.free/2**30:.0f} GiB of {u.total/2**30:.0f} GiB")
wan = list(hf.glob("hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/*"))
print("Wan snapshot:", [w.name for w in wan] or "MISSING")
for f in ("artifacts/flow_poc_all/adapter_000300.safetensors",
          "artifacts/flow_poc_all/adapter_000300.json"):
    print(f, "OK" if (ROOT / f).is_file() else "MISSING")
print("stale GPU lock:", (ROOT / "artifacts/gpu.lock").exists(), "(delete it if no job runs)")
```

Need: ~96 GB GPU (peak per 50-step run is ~23 GiB for v1 inference plus the VAE decode),
~2 GB free disk for the diagnostics, the Wan snapshot listed, both adapter files `OK`.

## 2. Run the diagnostics

```bash
cd $ROOT
python -m pytest tests -q            # CPU sanity, ~20 s (optional)
python scripts/diagnose_v1_borders.py \
    --project-root . \
    --checkpoint artifacts/flow_poc_all/adapter_000300.safetensors \
    --images heldout=/path/to/heldout.jpg motorcycle=/path/to/motorcycle.jpg \
    --output-root artifacts/p1_diagnostics \
    --steps 50 --seed 42
```

Without `--modes` and with the default images the pairs are heldout -> `zoom_out` and
motorcycle -> `pan_right`. With explicit `--images` and no `--modes` every image is run
with both `zoom_out` and `pan_right`; pass `--modes zoom_out` to restrict.
`--variants v1_baseline fill_replicate` runs a subset. Smoke test first with
`--steps 10 --variants v1_baseline --images heldout=... --modes zoom_out --output-root artifacts/p1_smoke`
(about 1.5 min plus model load) and check that `artifacts/p1_smoke/heldout__zoom_out/v1_baseline/debug/` contains
`debug_latents.safetensors`, `debug_meta.json` and `x0hat_sigma_0.60_frame_{040,080,120}.png`.

Resumable: a variant whose `run.json` exists is skipped; if the Molab session drops just
re-run the same command. (Stale lock after a crash: `rm artifacts/gpu.lock`.) A failing
variant is logged, the loop continues, and the traceback is appended to the summary;
the exit code is 1 if anything failed.

`python scripts/diagnose_v1_borders.py ... --summarize-only` recomputes metrics + the
summary from existing runs without touching the GPU (add `rm .../border_quality.json` to force).

### Runtime

| Step | Time |
|---|---|
| Wan load (once) | 2-5 min |
| each 50-step video (frozen base is cheaper: 2 forwards/step instead of 3-4) | about 6 min |
| x0-hat decode (full VAE decode, 3 PNGs) | about 0.5 min |
| border metrics per run (CPU, every 4th frame) | about 15-30 s |
| 2 cases x 5 variants = 10 runs | about 70-80 min total |

The 5B backbone digest is verified on the first and last generation only
(`verify_backbone`); intermediate runs record `backbone_unchanged: null`.

## 3. Outputs

```
artifacts/p1_diagnostics/
  diagnosis_summary.md / .json
  <image>__<mode>/<variant>/
      video.mp4, input.png, request.json, run.json, frame_*.png, border_quality.json
      debug/debug_latents.safetensors   (final_latent, x0_hat_sigma_0.60, image_latent, validity)
      debug/debug_meta.json, x0hat_sigma_0.60_frame_{040,080,120}.png
```

`run.json` records `guide_fill_mode` and a `debug` block (including the x0-hat convention).

### x0-hat convention

Wan2.2 flow matching: `x_t = (1 - sigma) x0 + sigma eps`, model output `v = eps - x0`, so
`x0_hat = x_t - sigma * v`. This is what diffusers' `UniPCMultistepScheduler`
(`prediction_type="flow_prediction"`) computes: `x0_pred = sample - sigmas[step_index] * model_output`
with `sigmas` the flow-shifted schedule (shift 5) and `timestep = 1000 * sigma`. The capture
wraps `scheduler.step`, so `v` is the final text-CFG + camera-anchored prediction and
`x_t` the exact noisy latent of that step. It is saved at the first step with
`sigma <= 0.6` (actual step and sigma in `debug_meta.json`). The first latent plane is
replaced by the image latent (the pipeline does the same before decode).

## 4. Variants and what each one tests

| Variant | Setting | Question |
|---|---|---|
| `v1_baseline` | zeros fill, `min_sigma` 0.6 | Reproduces the artefact; reference for everything |
| `fill_replicate` | replicate fill, 0.6 | H1: does feeding edge features instead of latent 0 remove the gray? Caveat: v1 saw only zero fill in training, so the adapter's reaction to replicate input is a distribution shift. A positive result is strong evidence for H1; a null result is weak evidence against it |
| `min_sigma_0.3` | zeros, adapter active down to sigma 0.3 | H3: is the gray created or kept after the hand-off at 0.6? |
| `min_sigma_0.0` | zeros, adapter always active | H3 (extreme) |
| `frozen_base` | adapter off | Control: what the frozen model does with the same image (it ignores the camera; R is only where a border *would* be) |

## 5. Reading the verdicts

`diagnosis_summary.md` has one table per case and an automatic verdict per hypothesis.
Metrics are means over frames with R > 1 %; `R@120` is the revealed fraction at frame 120.

- `gray score` = fraction of R pixels that are gray (chroma < 12, 80 < luma < 176) minus the same in the valid region. Higher is worse.
- `color ratio` / `hf ratio` = Hasler-Susstrunk colourfulness / Laplacian variance of R divided by valid. Healthy content is near 1; the gray artefact is far below 1.
- `seam` = gradient at the R/valid boundary over gradient just inside the valid region (about 1 = no visible seam, much larger = hard edge).
- `lat std R/V`, `lat mean-abs R/V` = ratio of std / mean |z| of the **final latent** in R versus valid (first latent plane excluded). `x0hat ...` the same for x0-hat at sigma 0.6, plus the decoded-PNG hf ratio.

Thresholds (constants at the top of `scripts/diagnose_v1_borders.py`):

- A case is analysed only if the baseline `gray score` > 0.15 and R exists; otherwise `NOT APPLICABLE` (e.g. the pan case may show little gray).
- "Gray reduced" = `gray score` drops by at least 25 % relative to `v1_baseline`.
- **H1 SUPPORTED** = baseline R latents collapsed (R/valid std **and** mean |z| both < 0.5) **and** `fill_replicate` reduces gray. `PARTIAL` = exactly one of the two. Expect decoded colour in R after replicate to be smeared edge colour, not new texture: inspect `video.mp4` / frames, the metric alone cannot tell colourful smear from plausible content.
- **H2 SUPPORTED (indicative)** = x0-hat at sigma 0.6 is smooth in R relative to valid (hf ratio < 0.25 or latent std ratio < 0.5). Only indicative because x0-hat is blurry everywhere.
- **H3 SUPPORTED** = `min_sigma_0.3` or `min_sigma_0.0` reduces gray by >= 25 %.
- H4 (data scarcity) cannot be tested in P1.

Decision guide: H1 supported -> v2 replicate fill + hybrid loss as planned (the main fix
stands). H1 not supported but H3 supported -> prioritise training the adapter active at lower
sigma (pass B range). Only H2 -> the hybrid generative loss is the key fix. In all cases also
look at the videos (frames 40/80/120) before deciding. Report back: `diagnosis_summary.md`,
GPU/disk numbers from the probe, and the s per 50-step run from `run.json["seconds"]`.

## 6. Measuring any other run later

```bash
python scripts/measure_border_quality.py runA runB --out-csv border.csv --out-md border.md
# optional latent stats if the run was made with --debug-dir runX/debug:
python scripts/measure_border_quality.py runX --debug-subdir debug --force
```

Homographies are read from `request.json["homographies_first_to_current"]` (always written by
`generate`); older runs without them are regenerated from `run.json` mode / magnitude.
