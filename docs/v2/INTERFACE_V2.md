# Interface v2 (marimo)

Two notebooks: `notebooks/camera_studio_v2.py` is the self-sufficient, user-facing one (section "Camera Studio v2" at the end of this file);
`notebooks/camera_interface_v2.py` (below) is the developer/evaluation UI that expects a repository checkout and a local checkpoint.

File: `notebooks/camera_interface_v2.py`. One motion at a time, one strength slider, a preview of what has to be invented, and a
subprocess-based generation that can be cancelled. v1's drawn-path canvas and the 0-2 magnitude slider are gone; v1's
IndentationError (`notebooks/flow_camera_interface.py`, monitor loop) is not carried over (`marimo check` is clean on the new file).

## Launch

From the repository root in the pinned Wan runtime (CUDA GPU, base model in the local Hugging Face cache):

```bash
pip install -r requirements-ui.txt            # marimo 0.25.2, anywidget
marimo edit notebooks/camera_interface_v2.py  # editable; or: marimo run notebooks/camera_interface_v2.py
```

On Molab open the file and run it. Needs at least one `adapter_*.safetensors` under `artifacts/` and no `artifacts/gpu.lock`.
Generation runs `python -m fpcamera.flow_infer ...` in a child process (the UI process never imports the model).

## Walkthrough

1. **Scene.** Upload a photograph (png, jpg, webp; EXIF rotation applied; saved under `artifacts/ui_v2_uploads/<sha>.png`) or pick one of
   `examples/inputs/*`. An upload wins over the example. The image is centre-cropped to 16:9 (1280 x 704) by the inference code, as in training.
2. **Motion and strength.** *Motion*: pan left/right, tilt up/down, zoom in/out. *Strength*: 0 to 1 in steps of 0.05. The callout below
   shows the amount after calibration, for example **Pan right · 5.6°** or **Zoom out · ×0.86**, plus `s` and the calibrated `u`;
   with the identity file it says "not calibrated". 0 is a static camera, 1 the trained maximum (8°, ×1.25).
3. **Model.** *Adapter checkpoint*: all `artifacts/**/adapter_*.safetensors`, v2 runs first and newest step first, labelled with updates and the
   recorded guide fill. *Strength calibration*: `configs/strength_calibration_v2*.json`, fitted files first.
   "Rescan" refreshes both lists.
4. **Advanced** (accordion): scene description (default: the training caption, `A photograph of a scene. One continuous camera shot.`), seed (42),
   denoising steps (50), camera gain (1 = trained), `camera_min_sigma` (default: the checkpoint's value; untick to set one), guide fill mode
   (`auto` = what the checkpoint was trained with, or `zeros` / `replicate`).
5. **Preview.** Left the input; right the geometry at frame 120: the photograph warped by the final homography, the **newly revealed region
   in red** (what the model must generate; its share of the frame is printed) and cyan outlines of the first view at frames 40, 80 and 120.
   It updates with motion, strength and calibration, on the CPU, without the model.
6. **Generate / Cancel / Status.** *Generate* starts the worker (refused if a job runs or `artifacts/gpu.lock` exists). The status block refreshes every
   2 s (selector: 2 s or 5 s) and shows a progress bar (denoising step, elapsed seconds). *Cancel* sends SIGINT to the worker's process group (the GPU lease is
   released by its `finally`), then SIGTERM after 45 s; a lease left by a terminated worker is removed. A 40 minute limit cancels automatically.
   On failure or cancellation the last log lines are shown.
7. **Result.** The video player, **Download MP4**, and three accordions:
   * *Run summary*: requested label, `s -> u`, calibration file and hash, checkpoint updates, guide fill and its source, `camera_min_sigma`, anchor, gain,
     seed/steps, time, peak memory and warnings (for example a fill-mode mismatch or a calibration fitted for another checkpoint).
   * *Border quality and motion*: gray-band score, colorfulness R/valid, high-frequency R/valid, seam score (from `border_metrics`, mean over frames with a
     revealed fraction above 1 %), the tracker gain `g` and the observed amplitude `a_obs = g * u`. Zoom in reveals nothing, so the border metrics are n/a.
   * `run.json` and `request.json` (homographies removed) for reproduction.

Every run folder `artifacts/ui_v2_runs/<timestamp>-<id>/` holds `input.png`, `request.json`, `run.json`, `video.mp4`, `progress.json`, `worker.log`,
`border_quality.json`, `measure.json`. Re-running a notebook cell never changes a finished run.

## Notes

* The measurements are the same code as the paper's tables (`scripts/measure_border_quality.py`, `scripts/measure_flow_video.py`,
  `scripts/summarize_flow_adherence.py`), run on the output after generation. They describe apparent 2D motion and border statistics; they are not an acceptance test.
* Calibration is per checkpoint. If the label says "not calibrated", or the run summary warns about another checkpoint, the physical amount is only approximate.
* The notebook has no state outside `artifacts/`; deleting that folder resets it.

---

# Camera Studio v2 (`notebooks/camera_studio_v2.py`)

The final user-facing notebook: pan / tilt by **drawing a trajectory**, zoom in / out, one strength slider with a live physical label, in-process
generation with **live progress**, an mp4 preview and a download button. It is **self-sufficient**: on a cold GPU sandbox (Molab) running it top to bottom
installs what is missing, downloads the base model and the adapter bundle, loads the pipeline once and generates. All code is in visible cells.
`camera_interface_v2.py` is untouched.

```bash
marimo run  notebooks/camera_studio_v2.py     # app view
marimo edit notebooks/camera_studio_v2.py     # editable, code visible (Molab: open the file and run all cells)
```

## What it needs

* An NVIDIA GPU with about 48 GB or more (about 43 GiB peak at 1280 x 704 x 121 frames, about 6 min per 50-step video on an RTX PRO 6000).
  Without a CUDA GPU the GPU cell stops the notebook with a message before anything is downloaded.
* Network access to PyPI and Hugging Face. About 32 GB for `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (29 files, revision `b8fff7315c76...`) plus about 55 MB for the adapter bundle.
* The **adapter bundle**, a public Hugging Face model repo (`ADAPTER_REPO` in the first setup cell, currently `D3vil-Reap3r/ie643-wan22-camera-adapter-v2`, no token):

| file in the repo | used for |
|---|---|
| `adapter/adapter_000415.safetensors`, `adapter/ckpt_000415.json`, `adapter/config.json` | the adapter. Keep the three together: `generate()` reads the checkpoint digest, the architecture and the guide fill mode (`replicate`) from them |
| `code/fpcamera_src.zip` | `src/fpcamera/...` in repository layout; unpacked once, `<extract>/src` goes on `sys.path` |
| `configs/inference_v2.json` | defaults of the Advanced panel (steps, seed, guidance, camera gain, ...) |
| `configs/strength_calibration_v2*.json` | strength calibration; a fitted file wins over the identity file (see below) |
| `examples/inputs/*.png` or `*.jpg` (optional) | extra photos in the sample menu |

**The code zip must be built from a commit that contains `src/fpcamera/trajectory_direction.py` and the `progress_callback` argument of `flow_infer.generate`.**
The notebook stops with a "code bundle is older than this notebook" message if the module is missing and runs without step progress if the argument is missing.
Rebuild it with, for example, `git archive --format=zip -o fpcamera_src.zip HEAD src requirements.txt requirements-ui.txt` and upload it to `code/`.

## Cells, top to bottom

| # | cell | purpose |
|---|---|---|
| - | intro | what it does and how to use it |
| S0 | configuration | constants (`ADAPTER_REPO`, base model and revision, pinned runtime), work folder (`~/fpcamera_studio`), env switches |
| S1 | helpers | size formatting, `fail()` (red stop message), `download_with_progress()` (live bar from the bytes on disk) |
| S2 | GPU check | `nvidia-smi`; clear message and stop if there is no GPU; warns below 48 GiB |
| S3 | Python packages | `uv pip` / `pip install` of the pinned runtime **only where missing** (see below); status table |
| - | Hugging Face client | imports `huggingface_hub` after the install cell |
| S4 | base model | `snapshot_download` of Wan2.2-TI2V-5B-Diffusers at the pinned revision with a progress bar; idempotent |
| S5 | adapter and code | `snapshot_download` of `ADAPTER_REPO` into `~/fpcamera_studio/bundle`, unzip the code, `sys.path` |
| S6 | modules, defaults, calibration | imports `fpcamera`, reads `inference_v2.json`, picks the calibration, defines the strength label |
| - | pipeline | `load_inference_pipeline()` once, cached in the kernel (survives a re-run of the cell) |
| - | imports | numpy, Pillow, OpenCV (after the install cell) |
| - | engine | `run_generation()`: the real `generate()` call, or the dry-run stand-in |
| - | sample image | the built-in sample (generated landscape; the repo ships no photographs) plus optional bundle examples |
| 1 | photo | `mo.ui.file` (images only) and a sample menu |
| - | scene | EXIF-rotate, centre-crop to 16:9 (1280 x 704), save PNG, preview data |
| 2 | motion | radio: Pan / Tilt (draw a trajectory), Zoom in, Zoom out |
| - | canvas widget | anywidget class: stroke canvas for mouse and touch (pointer events), Clear button, badge with the detected direction, veil when disabled |
| - | canvas | shows the canvas over the photo |
| - | direction | runs `detect_direction()`, sets the badge, draws the arrow card and the confidence, warns on short or diagonal strokes |
| 3 | strength | slider 0..1 step 0.05, live label ("0.50 → 4.0° pan right"), preview of the last frame with the revealed region in red |
| 4 | settings | Advanced accordion: seed, steps (default 50), text guidance (default 5.0), prompt override, attention kernel (`auto`, `cudnn`, `flash`, `efficient`; default `auto`) |
| 5 | generate | button; generation cell with a live progress bar; the finished run is kept in `mo.state` |
| - | result | video, **Download MP4**, run summary table |
| - | how it works | collapsed explanation |

## Direction convention and detection

The drawn stroke is a **camera-view path**, exactly as in the v1 trajectory UI (`flow_camera_interface.py`, `classify_camera_path`; `flow_view_path.py`):
*drag in the direction the camera should move*. Drag right = **pan right** (the scene slides left in the frame), left = pan left, down = **tilt down**, up = tilt up
(canvas y grows downward). Zoom in / out has no stroke; selecting it turns the canvas off (grey veil) and nothing is read from it.

`src/fpcamera/trajectory_direction.py` (`detect_direction`, pure Python, unit-tested in `tests/test_trajectory_direction.py`):

* The direction is the **net displacement** (last minus first point), which tolerates a wobbly path and a hook at release. The widget sends the stroke on release as points normalised
  to the image; x is multiplied by the aspect ratio (1280 / 704) before the axes are compared, because a normalised dx is longer on screen than the same dy.
* The dominant axis chooses pan (horizontal) or tilt (vertical), its sign the side.
* `confidence = dominant / (dominant + secondary)`, from 100 % (axis aligned) to 50 % (exact diagonal). Below 62 % (within about 12 degrees of the diagonal) the stroke is
  flagged **ambiguous**: a best guess is returned with a warning that names the alternative axis.
* A net displacement shorter than 0.08 image heights (about 56 px of 704) is **too short**: no motion, Generate refuses.
* A curvy stroke (net / path length below 0.5) only adds a warning; the direction still comes from start to end.
* If `anywidget` is unavailable the notebook shows a direction radio group instead of the canvas.

## Strength label

The slider value `s` goes through the calibration (`StrengthMap.s_to_u`, `fpcamera/strength.py`) and the label shows the physical amount of the requested path, as in the
developer UI: pan / tilt `u * 8` degrees, zoom `exp(+-u ln 1.25)`, for example **0.50 → 4.0° pan right** or **0.50 → ×1.12 zoom in**; a second line gives `u` and whether the map is the identity.
The calibration file is chosen from the bundle: a fitted file beats the identity one; among fitted files the one whose `checkpoint_sha256` matches the adapter, then the one named in
`inference_v2.json`. With only the identity map the label says "not calibrated".

## Generation

`run_generation()` calls `flow_infer.generate(bundle_root, adapter_path, image, fresh_run_dir, motion=..., strength=s, calibration=<file>, max_amplitudes={8, 8, 1.25},
guide_fill_mode="replicate", pipe=<cached pipeline>, verify_backbone=False, attn_backend=..., progress_callback=...)`.

* **Fill mode is always passed explicitly as `replicate`** and re-checked on the returned record (a fall back to `zeros` once produced gray borders).
* The pipeline is loaded once by `load_inference_pipeline()` and kept in the kernel (also parked in `sys.modules["fpcamera_studio_cache"]`, so re-running the cell does not load a second copy);
  no subprocess per video, no GPU lease file (the notebook owns the sandbox's GPU).
* `progress_callback(step, total)` is the new optional argument of `generate()` (default `None`, called after every denoising step in the calling thread; an exception from it aborts the run).
  The notebook drives an `mo.status.progress_bar` with step, elapsed time and ETA; at the last step it says that the VAE decode and mp4 write follow (30 to 60 s).
* Every run goes to a fresh `~/fpcamera_studio/runs/<timestamp>-<id>/` (`generate()` refuses a folder that already has `run.json`).
* Result: `mo.video`, `mo.download` of the mp4 and a summary (motion, strength `s -> u`, direction source, seed, steps, guidance, guide fill, generation time, peak GPU memory, notes).

## Setup details (what makes it idempotent)

* **Python packages.** `torch` (any 2.11.x), `diffusers 0.36.0` and `transformers 4.57.6` must match exactly (`fpcamera.wan.require_pinned_runtime`); every other pin of
  `requirements.txt` / `requirements-ui.txt` is installed only when the package is absent. `uv pip install --python <kernel python>` is used when `uv` exists (Molab), else `pip`;
  an installed torch / torchvision is passed as a constraint so the resolver cannot replace it (the Molab pitfall in `docs/experiments/MOLAB_SETUP.md`). If a changed package was
  already imported in the session the cell asks for a kernel restart. Heavy imports happen only after this cell.
* **Base model.** `state/base_model.json` lists every file with its size; if all are present in the Hugging Face cache the cell finishes without network access. Otherwise it fetches the
  file list, checks the free disk space, downloads with `snapshot_download` in a thread and shows a bar fed from the bytes on disk (rate and ETA), and verifies the sizes at the end.
  A broken download is resumed by running the cell again.
* **Bundle.** `snapshot_download(local_dir=...)` only fetches what changed; offline, the copy of an earlier run is used. The code zip is unpacked once per content digest.

## Dry run (no GPU, no downloads)

`FPCAMERA_DRY_RUN=1 marimo run notebooks/camera_studio_v2.py` skips the installs, the downloads and the model. Configs and code come from the repository checkout. *Generate* writes a
placeholder mp4 (the photo warped by the **same homographies** the real path would use, half resolution, a "DRY RUN" watermark, fake step progress), so the whole UI flow can be tried:
stroke, direction, label, preview, progress, video, download, summary. Other switches: `FPCAMERA_WORK_DIR`, `FPCAMERA_USE_REPO_SRC=1` (use `src/` of the checkout instead of the code zip),
`FPCAMERA_ADAPTER_REPO`.

## Molab deployment test (D1, 2026-11-18)

The notebook was deployed into a **fresh, empty Molab notebook** as 27 visible cells and run on a cold GPU sandbox. Nothing in `notebooks/camera_studio_v2.py` or `src/fpcamera` had to change
(no code fix, no re-publish of the HF bundle; the public bundle `D3vil-Reap3r/ie643-wan22-camera-adapter-v2` with code 4af97e5 and the fitted calibration was used as is).

**How it was deployed.** The cells were created through the marimo-pair `marimo._code_mode` API (`ctx.create_cell(code, after=..., hide_code=...)`), not run hidden in the kernel; the repo file was parsed
with `marimo._ast.parse.parse_notebook` and each cell's code was sent unchanged. Check afterwards: all 27 cells in the live notebook **and** in the file Molab saves (`/marimo/notebook.py`) have the same
sha256 per cell and the same `hide_code` flag as the repo file (only the intro and "How it works" cells are collapsed, exactly as in the source). The only difference is the file header
(`marimo.App(width="medium", auto_download=["html"])` on Molab versus `app_title="Camera Studio v2"` in the repo).

**Sandbox.** Molab GPU sandbox (gVisor, 20 CPUs, 160 GB RAM): RTX PRO 6000 Blackwell Server Edition, 96 GiB; Python 3.13.11 in `/tmp/uv-venv`, running as root with `HOME=/home/marimo`; `uv` 0.12.1, `pip`, `ffmpeg`
and Playwright + Chromium are present; the disk reports 8 EB free (no space problem). Preinstalled: torch 2.11.0+cu130, torchvision 0.26.0, numpy 2.4.6, Pillow 13.0.0, OpenCV 5.0.0.93, anywidget 0.11.0,
marimo 0.25.2, **transformers 5.20.1 and huggingface-hub 2.2.0** (both clash with the pins); missing: diffusers, accelerate, imageio-ffmpeg, sentencepiece, ftfy. The sandbox was created at 10:29 UTC and is
hard-killed about 3 h later; the notebook file itself persists on Molab, the kernel state, the downloaded model and the loaded pipeline do not.

| step | measured | note |
|---|---|---|
| Python packages (cell S3, `uv pip`) | **52 s** | installed diffusers 0.36.0, transformers 4.57.6 (replacing 5.20.1), accelerate, imageio-ffmpeg, sentencepiece, ftfy; no kernel restart needed because none of them was imported yet |
| Base model (cell S4), 34.2 GB / 31.9 GiB, 29 files | clean speed **164 MB/s** (a 5 GB shard in 30 s), so about 4 to 6 min | the real cold run was interrupted twice by the driver (gotcha 3): 23.7 GB in the first ~4 min, then the resumed cell took 13.1 min for the last ~8.5 GB because the three half-finished `.incomplete` files were fetched again slowly. An uninterrupted download should not show this. |
| Adapter bundle (cell S5) | **0.9 s**, 57 MB | |
| Pipeline load | **30 s**, 22.6 GiB in use | cached in the kernel afterwards (a re-run says "already loaded, reused") |
| Everything else, "run all" on a warm kernel | 23 s | install, base model and bundle cells all finish as no-ops |
| One 50-step video | **361 to 364 s** (3 runs), peak **43.0 GiB** | about 7 s per step, 30 to 40 s for VAE decode and mp4 write; same as the documented ~6 min |

So an uninterrupted cold start is about 1 min (install) + 4 to 6 min (download) + 0.5 min (pipeline) = **roughly 6 to 8 min** (not measured end to end in one piece), in line with the "about 10 minutes" in the intro cell.

After the install cell: `torch 2.11.0+cu130`, `torch.cuda.is_available() == True`, transformers 4.57.6, huggingface-hub 0.36.2 (downgraded by the resolver as a dependency of transformers 4.57.6), diffusers 0.36.0.
Side effects of the resolver that were harmless here but are worth knowing: numpy 2.4.6 -> 2.5.5 (the pin is only enforced for torch / diffusers / transformers), tokenizers 0.23.3 -> 0.22.2, and uv added the
`nvidia-*` CUDA 13 wheels that the preinstalled torch declares. The torch constraint file (`state/constraints.txt`) kept torch / torchvision where they were.

**Generations** (built-in sample image, seed 42, 50 steps, guidance 5.0, adapter step 415; all started with the real **Generate** button path, see below; files in `<checkpoints_dir>\D1\`):

| run | motion and strength | summary label | wall | run.json |
|---|---|---|---|---|
| 1 | drawn stroke to the right, s = 0.50 | Pan right 4.0 deg, u = 0.495, direction source stroke 98 % | 364 s | `guide_fill_mode = replicate (explicit)`, 415 adapter updates |
| 2 | Zoom out, s = 1.00, canvas disabled | Zoom out x0.80, u = 0.990, zoom preset | 363 s | replicate (explicit) |
| 3 | drawn stroke upward, s = 0.75 | Tilt up 5.9 deg, u = 0.737, stroke 100 % | 361 s | replicate (explicit) |

Frames 0 / 60 / 120 were extracted from each mp4: pan right moves the scene left (phase correlation 0 -> 120: dx = -88 px), zoom out shrinks the house to roughly 75 to 80 % and fills the border with new meadow and sky,
tilt up moves the scene down (dy = +122 px) and reveals sky. **No gray borders** in any run (edge pixels are blue sky or green meadow). Observation unrelated to the interface: on this synthetic sample the sky
becomes more saturated blue in every run (the first frame is the exact input), and tilt up shows two faint blotches in the newly revealed sky at frame 120.
The progress bar was checked by recording every `ProgressBar.update` call (a temporary wrapper in the scratchpad, removed afterwards): 50 step updates (subtitles such as "step 48/50 · 319 s elapsed · about 13 s left"),
then the "decoding the video (VAE)" message, then the close. The result cell contains the `<video>`, the **Download MP4** button and the run summary (in-sandbox screenshots in `D1/screens/`). The bar itself
was not watched in a live browser.

**How the UI was driven without a browser.** Widgets were set through the pair API: `ctx.set_ui_value(canvas, {**canvas.value, "points": [...]})`, `ctx.set_ui_value(motion_type, "Zoom out")`,
`ctx.set_ui_value(strength_slider, 0.75)`; the direction, label and preview cells then reacted as in the browser. Generate was pressed with `POST /api/kernel/set_ui_element_value`
(`objectIds=[generate_button._id]`, `values=[true]`), the message the browser sends for a `run_button`, so the real Generate cell ran. Not driven through the live page: the mouse drag itself. It was
tested separately: the notebook's own `TrajectoryCanvas._esm` was loaded in headless Chromium inside the sandbox with a fake model; a real mouse drag right / up produced strokes that `detect_direction`
turned into `pan_right` 97.8 % / `tilt_up` 100 %, with no JavaScript errors, and the stroke and arrow render (`D1/screens/harness_pan_right.png`).

### Gotchas (Molab, and driving it from outside)

1. **Pins versus the image.** The image ships transformers 5.x / huggingface-hub 2.x; the install cell fixes this in about a minute. Importing `transformers` / `huggingface_hub` before the install cell would make it
   ask for a kernel restart; the notebook imports them only afterwards, and the setup cells are ordered so that nothing heavy is imported earlier.
2. **`execute-code.sh` needs `jq`**, which Git Bash on Windows does not have; `tools/molab_sync.py exec` speaks the same protocol (`MOLAB_URL` / `MOLAB_TOKEN` environment variables, never in files).
3. **Do not run long cells through the scratchpad request.** `async with cm.get_context(): ctx.run_cell(...)` (or `set_ui_value(generate_button, True)`) keeps the HTTP request open until the cell and all its
   dependents finish. When that stream ends (the first attempt died after a couple of minutes, probably a proxy limit) marimo **interrupts the kernel**: the execute endpoint interrupts on every client disconnect
   (`marimo/_server/api/endpoints/execution.py`, console message "Cancellation requested"). The same happened once when a second `/api/kernel/execute` call was made while a run was in progress
   (it ended with "stream ended without done event" and the running cell was interrupted; the exact cause was not isolated). The running cell ends as `interrupted`, its descendants as `cancelled`, and cells that are not descendants of the re-run cell stay `stale`. Use the fire-and-forget endpoints instead:
   `POST /api/kernel/run {"cellIds": [...], "codes": [...]}` for cells, `POST /api/kernel/set_ui_element_value` for buttons, poll `GET /api/kernel/status`, and only call `/api/kernel/execute` while it says `idle`.
   After an interruption, run all cells (all ids and codes in one `/api/kernel/run`) rather than single cells.
4. **An interrupted download resumes slowly** (13 min for the last 8.5 GB versus about 3.5 min for the whole model uninterrupted). The cell is idempotent and finishes correctly when run again, so just let it run.
5. **Programmatic widget state versus a freshly opened page.** A browser (or kiosk screenshot) that opens the notebook later builds the anywidget canvas from its initial state (no stroke), even if Python was given
   a stroke with `set_ui_value`. After testing, the stroke and strength were therefore reset to their defaults (empty stroke, 0.5, Pan / Tilt) so that the page and the kernel agree.
6. The saved notebook on Molab is `/marimo/notebook.py` (the kernel keeps it up to date as cells change); `pyproject.toml` next to it is read-only and is not touched.

### What the user does on a fresh Molab sandbox

Open the notebook, choose a **GPU machine**, press **Run all** (or run cells top to bottom) and wait about 6 to 10 min (install, 34 GB download, pipeline load); the *Setup* cells show live progress. Then upload a photo
(or keep the sample), choose Pan / Tilt and drag on the photo, or Zoom, move the strength slider and press **Generate video**; about 6 min per 50-step video.
