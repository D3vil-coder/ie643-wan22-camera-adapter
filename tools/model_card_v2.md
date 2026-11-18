---
license: apache-2.0
base_model: Wan-AI/Wan2.2-TI2V-5B-Diffusers
base_model_relation: adapter
library_name: diffusers
pipeline_tag: image-to-video
tags:
  - wan2.2
  - image-to-video
  - camera-control
  - lora
  - safetensors
---

# Camera adapter v2 for Wan2.2-TI2V-5B (pan / tilt / zoom with a strength control)

A lightweight camera-control adapter ({{PARAMS}} trainable parameters, {{ADAPTER_MB}} MB, `float32`) for the
frozen [`Wan-AI/Wan2.2-TI2V-5B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) image-to-video model
(pinned revision `{{WAN_REVISION}}`). Given one still image it produces a 121-frame, 24 fps, 1280x704 video in which the
virtual camera pans left/right, tilts up/down or zooms in/out. A single strength value `s` in [0, 1] sets how far the
camera moves (up to 8 degrees of pan/tilt or a 1.25x zoom at `s = 1`).

The Wan backbone, VAE and text encoder are never modified. This repository holds only the adapter weights, the small
amount of Python code needed to run it, and the configuration files. **It does not contain the Wan weights**; they are
downloaded separately from the base-model repository.

Source repository (full code, documentation, tests, report): <https://github.com/D3vil-coder/ie643-wan22-camera-adapter> (its tag `v2-final-415` is the code bundled here).

License: Apache-2.0 (the adapter weights, the code and the configuration in this repository). The base model `Wan-AI/Wan2.2-TI2V-5B-Diffusers` is Apache-2.0 under its own terms and is not part of this repository. The training photographs (COCO 2017, Flickr images under Creative Commons licences) are not redistributed here.

## Contents

| path | what |
|---|---|
| `adapter/adapter_{{STEP6}}.safetensors` | adapter weights, training step {{STEP}}, sha256 `{{ADAPTER_SHA256}}` |
| `adapter/ckpt_{{STEP6}}.json` | checkpoint manifest (digest, architecture, step) |
| `adapter/config.json` | resolved training config of the run (records `fill_mode: replicate`, video contract, backbone digest) |
| `configs/strength_calibration_v2.json` | strength-to-amplitude map ({{CALIBRATION_NOTE}}) |
| `configs/inference_v2.json` | default inference options (motion, steps, guidance, seed) |
| `code/fpcamera_src.zip` | the `fpcamera` Python package (`src/fpcamera/**`), `requirements.txt`, `requirements-ui.txt`, `VERSION` (source commit `{{CODE_COMMIT}}`) |

The three files in `adapter/` must stay together in one folder: the loader reads the manifest and `config.json` that sit
next to the weights.

## How to use

You need a CUDA GPU (about 43 GiB peak memory and about 6 minutes per 50-step video in our runs) and the pinned
runtime from `requirements.txt` (`torch 2.11.x`, `diffusers 0.36.0`, `transformers 4.57.6`; the code refuses other
versions of the last two).

```python
import sys, zipfile
from pathlib import Path
from huggingface_hub import snapshot_download

# 1. Download this repository (public, no token needed) and unpack the code.
local = Path(snapshot_download("{{REPO_ID}}", local_dir="ie643_adapter_v2"))
zipfile.ZipFile(local / "code" / "fpcamera_src.zip").extractall("fpcamera_code")
sys.path.insert(0, str(Path("fpcamera_code") / "src"))      # makes `import fpcamera` work
# (pip install -r fpcamera_code/requirements.txt if the runtime is not installed yet)

# 2. Fetch the frozen base model once, at the pinned revision (the code loads it with local_files_only=True).
snapshot_download("Wan-AI/Wan2.2-TI2V-5B-Diffusers", revision="{{WAN_REVISION}}")

# 3. Generate.
from fpcamera import flow_infer

record = flow_infer.generate(
    ".",                                                    # root: a gpu lock is kept in ./artifacts/
    local / "adapter" / "adapter_{{STEP6}}.safetensors",
    "my_photo.jpg",                                         # any RGB image; it is center-cropped to 1280x704
    "out/run1",                                             # new folder: video.mp4, run.json, request.json
    motion="zoom_in",                                       # pan_left pan_right tilt_up tilt_down zoom_in zoom_out static
    strength=0.6,                                           # s in [0, 1], used together with motion
    calibration=local / "configs" / "strength_calibration_v2.json",
    max_amplitudes=dict(yaw_deg=8.0, pitch_deg=8.0, zoom_ratio=1.25),   # the trained maxima
    steps=50, seed=42,
)
print(Path("out/run1/video.mp4").exists(), record["guide_fill_mode"])   # -> True replicate
```

`generate(root, checkpoint, image_path, output, *, motion=None, strength=None, calibration=None, max_amplitudes=None,
steps=50, guidance=5.0, seed=42, prompt=None, pipe=None, verify_backbone=True, ...)` returns the run record
(`run.json`). Pass an already loaded pipeline with `pipe=` (see `flow_infer.load_inference_pipeline`) to avoid reloading
the 5B model between videos, and `verify_backbone=False` on repeated calls to skip the (slow) backbone hash check.
Instead of `strength`, `amplitude_deg=` (pan/tilt) or `zoom_ratio=` (zoom) request a physical amplitude directly.

### The guide fill mode must be `replicate`

This adapter was trained with `fill_mode: replicate` (regions revealed by the camera move are filled with the nearest edge
feature in the guide, not with zeros). Running it with a zero-filled guide produces a flat gray frame around the image.
`generate()` reads the mode from `adapter/config.json` and uses `replicate` automatically, and it refuses to fall back
to zeros if that file is missing. If you build your own folder, keep `config.json` and the `ckpt_*.json` manifest next to
the weights, or pass `guide_fill_mode="replicate"` explicitly.

## Training summary

- **Base model:** Wan2.2-TI2V-5B (Diffusers), frozen; pinned revision `{{WAN_REVISION}}`.
- **Adapter:** a camera-guide projection (51 channels: 48 warped first-image latent channels, 1 validity channel, 2
  displacement channels) plus rank-16 LoRA updates on the attention Q/K/V and feed-forward projections of blocks 0 to 15
  (80 wrapped linear layers). {{PARAMS}} trainable parameters.
- **Data:** synthetic camera clips derived from COCO 2017 train photographs (608 training sources, 28 held-out
  development sources). Each clip is a still image warped along a pan, tilt or zoom path (homographies, linear easing
  over 121 frames, strength drawn from [0.15, 1] per clip), encoded with the Wan VAE. The photographs themselves are not
  redistributed here. All clips use the same neutral caption, so the camera guide is the only varying condition.
- **Objective:** a hybrid of two flow-matching terms per update. Pass A is the teacher-relative objective (the frozen
  model's velocity plus the latent difference between the static and the moving clip). Pass B is ordinary conditional flow
  matching on the moving clip. Regions newly revealed by the camera get 5x weight in pass B and 0.1x weight in pass A,
  so the model is pushed to generate new content there instead of predicting a gray average.
- **Schedule:** AdamW, learning rate 1e-4 with warm-up and cosine decay, effective batch 2, warm start from an earlier
  (v1) adapter. The run was planned for 1000 updates and stopped by a compute-session limit; this checkpoint is
  **update {{STEP}}**.

## Limitations

- **Early checkpoint.** {{STEP}} of a planned 1000 updates; the held-out loss was still decreasing when training stopped.
- **Strength calibration is light and in sample.** `configs/strength_calibration_v2.json` is a per-direction isotonic map fitted on 28 videos
  (it differs from the identity by at most 0.05 in the strength `u`) and it was not verified on held-out images. The pooled measured motion was 0.96x to 1.02x of the requested amplitude
  at every tested strength level; single videos at small amplitudes range from 0.76x to 1.17x.
- **Single motion per video**, one of six directions (plus static), up to 8 degrees of pan/tilt or 1.25x zoom. No
  combined or custom paths in this release.
- **Fixed output format:** 1280x704, 121 frames, 24 fps (the trained contract). Inputs are center-cropped to that aspect.
- **Flat or low-texture images** can drift in brightness and colour (one bright test image dims to about 0.63x of the input luma at the
  end of the clip) and can show a short burst of frame distortion in the first frames. Base Wan does not do this on that image, so it comes from
  the adapter. An opt-in inference option that removes the low-frequency part of the adapter residual (`--residual-lowfreq blur:8`) fixed the two
  flat images tested, but its effect on detailed images is not verified, so it is not in this release; see branch `exp/flat-image-fix` of the source repository.
- **Revealed regions are the hard part.** Content outside the original frame is generated by the model; it can look
  smeared, repeated or inconsistent, and results vary with the seed and the image. We saw fewer gray borders than with
  the previous version, not their absence.
- **Evaluation is small:** a handful of showcase clips and seven ground-truth reveal pairs, single seed. Numbers do not
  support claims of superiority over other camera-control methods.
- **Compute:** needs a large CUDA GPU (about 43 GiB peak) and the pinned library versions.
- **No safety evaluation** beyond what the base model provides. Use of the base model is governed by its own license.
