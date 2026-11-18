# Inference v2 (CLI and Python)

Entry points: `python -m fpcamera.flow_infer` (one video), `scripts/calibrate_strength.py` (strength sweep and fit),
`scripts/generate_examples_v2.py` (before/after examples and gallery), `notebooks/camera_interface_v2.py` and `notebooks/camera_studio_v2.py` (UIs, see `INTERFACE_V2.md`).
All need the pinned Wan runtime with a CUDA GPU (Molab). Run from the repository root with `PYTHONPATH=src`.
v1 behaviour is the default and is unchanged: without `--motion` the `--mode`/`--magnitude` path of v1 is used.

## 1. Checkpoints

`flow_infer` accepts both layouts:

| layout | files | metadata used |
|---|---|---|
| v1 | `adapter_000300.safetensors` + `adapter_000300.json` (sidecar) | sidecar (digest, architecture, backbone audit, contract, `inference_policy`) |
| v2 trainer | `<run>/checkpoints/adapter_NNNNNN.safetensors` + `ckpt_NNNNNN.json` | manifest (`adapter_sha256`, `architecture`, `step`) + `<run>/config.json` (`fill_mode`, `video_contract`, `expected_backbone_sha256`, sigma window) + `<run>/run_status.json` if present |

For v2 checkpoints the missing sidecar is synthesised from these files (`resolve_checkpoint_meta`). The backbone audit of a mid-run
checkpoint is "start-of-launch hash only" (the trainer writes `run_status.json` when a launch ends); it is recorded as
`backbone_audit`. Defaults derived for v2: `camera_min_sigma = objective.sigma_A.lower` (0.6, as v1), residual anchor `teacher` (as v1's configs/UI).

**Guide fill mode.** `--guide-fill-mode` defaults to the mode the checkpoint was trained with (`replicate` for the v2 trainer,
`zeros` when nothing is recorded, i.e. v1). An explicit value overrides; a value that differs from the recorded one raises a
warning (also stored in `run.json["notes"]`). `run.json` records `guide_fill_mode` and `guide_fill_mode_source`
(`explicit` / `checkpoint` / `default`).

## 2. CLI

Motion + strength (v2, calibrated):

```bash
export PYTHONPATH=src
python -m fpcamera.flow_infer --project-root . \
  --checkpoint artifacts/v2_full/checkpoints/adapter_001000.safetensors \
  --image examples/inputs/kitchen_zoom_out.png --output artifacts/runs/kitchen_zoom_out_s075 \
  --motion zoom_out --strength 0.75 --calibration configs/strength_calibration_v2.json --steps 50 --seed 42
```

* `--motion` is one of `pan_left pan_right tilt_up tilt_down zoom_in zoom_out static`, `--strength` is `s` in [0, 1]; both are required together.
* `s` is mapped to `u` through `--calibration` (default: identity, `u = s`), then `camera_path_v2(family, direction, u)` with the trained maximum
  amplitudes of `configs/data_v2.json` (8 degrees, 1.25x). `s = 0` and `static` give the identity path.
* `request.json` gets a `strength_control` block (`motion`, `strength_s`, `strength_u`, `amplitude`, `label`, `max_amplitudes`, `calibration{path,sha256,identity,checkpoint_sha256}`);
  `run.json` repeats it and sets `mode = motion`, `magnitude = u`. Existing tools (`measure_border_quality.py`, `measure_flow_video.py --request`) read `request.json` unchanged.
* Physical amplitude instead of strength: `--amplitude-deg D` (pan/tilt) or `--zoom-ratio R` (zoom) with `--motion`, mutually exclusive with `--strength`.
  They map to `u` exactly (`u = D / 8`, `u = ln R / ln 1.25`; a zoom-out takes the same `R > 1` or its reciprocal) and bypass the calibration; the request/run
  `strength_control` records `amplitude_source: "physical"` (`strength_s = null`, `calibration = null`; the calibrated path records `"calibrated"`). Amplitudes above the trained
  maximum are rejected. `camera_path_v2` at that `u` equals v1 `camera_path(mode, magnitude=m)` to 1e-6 (pan m = 0.5 = 2.5 deg, tilt m = 0.35 = 1.75 deg, zoom m = 1.0 = x1.12; `tests/test_examples_v2.py`).
* A warning is issued when the calibration was fitted for another checkpoint or other maximum amplitudes.

Defaults from a JSON (keys are the option names with underscores, explicit flags win):

```bash
python -m fpcamera.flow_infer --config configs/inference_v2.json --project-root . \
  --checkpoint <adapter> --image <png> --output <dir> --motion pan_right --strength 1.0
```

`configs/inference_v2.json` points to the identity calibration until `configs/strength_calibration_v2.json` exists; change that entry afterwards.

Other useful flags: `--camera-min-sigma X` (ablation; default from the checkpoint), `--camera-guidance g` (camera gain, 1 = trained),
`--camera-residual-anchor {auto,none,frozen,teacher}`, `--component`, `--guidance`, `--prompt` (default: the neutral training caption),
`--debug-dir` (final / x0-hat latents), `--frozen-baseline` (adapter off).
v1 call, unchanged: `--mode zoom_in --magnitude 1.0 --checkpoint artifacts/flow_poc_all/adapter_000300.safetensors`.

### Speed options

- `--batch-forwards` (default on; `--no-batch-forwards` restores the old sequential calls). With
  `--camera-residual-anchor frozen|teacher` each step previously made separate batch-1 forwards (text-cond, text-uncond, adapted).
  Now the frozen cond and uncond predictions run as one batch-2 forward (adapter disabled, so no per-sample adapter logic);
  the adapted pass stays batch-1 (the guide injection is per-call state, not per-sample). The negative text tensor is learned
  from step 1, so batching starts at step 2, and the cached uncond half is used only if the negative call carries exactly the
  same latents/timestep/text, otherwise it is recomputed. Equivalence to sequential calls is tested on CPU (fp32, 1e-5).
  Expect a modest gain (the GPU is already saturated at 1280x704x121) and higher transient memory at the batched call; if
  that call OOMs use `--no-batch-forwards`. The execution record gains `batched_forward_calls`.
- `--attn-backend {auto,cudnn,flash,efficient}` (default `auto` = torch's own SDPA dispatch, unchanged; `generate(attn_backend=...)`).
  The Wan attention is `F.scaled_dot_product_attention` via diffusers' `dispatch_attention_fn`, so
  `torch.nn.attention.sdpa_kernel` selects the kernel: `cudnn` = cuDNN, then flash, then memory-efficient; `flash` / `efficient` =
  that kernel only. Applied to every transformer forward, including the camera-guidance passes, but not to the VAE decode.
  Kernel-level, so output is not bit-identical to `auto`; a torch without `SDPBackend.CUDNN_ATTENTION` warns and uses `auto`.
  Recorded in `run.json` under `speed_options`. Pick it with the training speed probe (`docs/v2/TRAINING_V2.md` section 7).
- `--compile-blocks` (default off): regional `torch.compile` of the 30 transformer blocks (`mode="default"`, `dynamic=False`),
  applied after the adapter is attached and undone when `generate()` returns, so the compile cost (minutes; batch-1 and
  batch-2 forwards and adapter on/off are separate graphs) is paid per call. Worth it only for a long single run; not
  bit-identical. Scripts that loop over many videos on one pipeline (`calibrate_strength.py`, `eval_dev_reveal.py`,
  `generate_examples_v2.py`) accept the same flags but recompile per video.
- TF32 is not enabled (not bit-lossless). Not adopted: FlashAttention-3 (fragile source build on a non-persistent sandbox) and
  SageAttention (int8, lossy); no flash-attn install.

## 3. Python

```python
from fpcamera.flow_infer import generate, load_inference_pipeline
from fpcamera.runtime import gpu_lease

with gpu_lease(root):
    pipe = load_inference_pipeline()                       # load Wan once
    for s in (0.25, 0.5, 0.75, 1.0):
        generate(root, ckpt, image, f"out/s{s:.2f}", motion="pan_right", strength=s,
                 calibration="configs/strength_calibration_v2.json", pipe=pipe,
                 verify_backbone=(s == 0.25))               # hash the 5B backbone once
```

New `generate` arguments: `motion`, `strength`, `calibration` (path, dict, `StrengthMap` or `None`), `max_amplitudes` (`{yaw_deg, pitch_deg, zoom_ratio}`
override), `guide_fill_mode=None` (auto), `progress_callback=None` (optional `f(step, total)` called in the calling thread after every denoising step, for live UI progress; raising from it
aborts the run). `build_camera_request(...)` builds the homographies without a GPU and is what the tests exercise.

## 4. Calibration and examples (Molab)

```bash
# 72-run sweep (3 dev images x 6 directions x 4 strengths, 30 steps), fit, plots, then the verification runs
python scripts/calibrate_strength.py --project-root . \
  --checkpoint artifacts/v2_full/checkpoints/adapter_001000.safetensors --verify

# matched before/after for the 11 cases (default --match-v1: v2 at each v1 case's physical amplitude; v1 and v1+replicate rows optional)
# + strength sweep (extended range) + gallery
python scripts/generate_examples_v2.py --project-root .   --checkpoint artifacts/v2_full/checkpoints/adapter_001000.safetensors   --calibration configs/strength_calibration_v2.json --include-v1 --include-v1-replicate
```

Both load the pipeline once, take the GPU lease, skip finished runs (`run.json`) and can be re-run after a session drop. Method: `CALIBRATION.md`.

`generate_examples_v2.py` writes runs to `artifacts/v2_examples/matched/{v2,v1,v1_replicate}/<case>` and `artifacts/v2_examples/sweep/<case>/sNN`, `artifacts/v2_examples/summary.md`
(per case: gray-band score, colorfulness ratio, HF ratio, seam score, gain, for v1 and v2; strength-sweep table with observed amplitude),
contact sheets `gallery/contact_sheets/<case>_before_after.png` and `<case>_strength.png` (rows = v1/v2 or strengths; columns = input, frames 40, 80, 120),
and the offline gallery `gallery/index.html` + `gallery_data.json` (+ `gallery_data.js`, so it opens from `file://`) with copied media in `gallery/media`.
`.gitignore` keeps only `gallery/index.html` and `gallery/gallery_data.json` (empty template) under version control. The v1 baseline uses the original recipe:
mode, magnitude 1.0, zeros fill, teacher anchor, `camera_min_sigma` 0.6. Inputs are `examples/inputs/<case>.png`; the motion is parsed from the case name.
`--no-generate` rebuilds metrics, sheets and gallery on CPU; `--write-gallery-template` only rewrites `index.html`.

### Matched vs extended sections

* **Matched** (`--match-v1`, default; `--no-match-v1` skips it): the Submission 1 showcase used pan 0.5 (2.5 deg of v1's 5 deg), tilt 0.35 (1.75 deg), zoom 1.0 (x1.12), not 1.0 everywhere,
  while v2's `u` is relative to 8 deg / x1.25. For a fair comparison each case takes the v1 mode and magnitude (table in `generate_examples_v2.py`; `--v1-runs-root DIR` reads `DIR/<case>/run.json`
  `mode`/`magnitude` instead) and runs v2 at the identical physical amplitude via `amplitude_deg`/`zoom_ratio` (no calibration).
  Rows: v1 (`--include-v1` regenerates it with the original recipe; `--v1-videos-root DIR` reuses existing run dirs `DIR/<case>/{video.mp4,run.json,request.json,input.png}` instead),
  v1 + replicate fill (`--include-v1-replicate`: v1 checkpoint, `guide_fill_mode=replicate`, same magnitudes) and v2 (matched amplitude), so one run produces all three rows.
* **Extended range**: v2 only, calibrated strength s in {0.25, 0.5, 0.75, 1.0} up to 8 deg / x1.25, beyond what v1 was shown at.
  The summary, contact sheets and gallery label each block as "matched" or "extended".
