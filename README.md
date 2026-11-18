# Trajectory-Controlled Camera Motion Adapter for Wan2.2 I2V (IE 643, v2)

Turn **one photograph** into a 5-second, 1280 x 704, 24 fps video with a **virtual camera move** (pan, tilt, zoom in / out) and one
strength slider. A lightweight 14.26M-parameter adapter (guide projection + rank-16 LoRA) steers the **frozen**
`Wan-AI/Wan2.2-TI2V-5B-Diffusers` image-to-video model; the base weights are never modified.

**v2 (tag `v2-final-415`, adapter at update 415)** closes the two gaps of the v1 submission:

- **No more gray borders.** Newly revealed regions are filled from the nearest edge features instead of latent 0 (replicate-fill guide),
  and training adds an ordinary flow-matching pass on the real moving clip, weighted 5x on the revealed region, next to the v1 teacher-relative pass.
  Training data are ~600 COCO photographs with per-source overscan, so revealed regions have real targets.
- **Calibrated strength.** One slider `s` in [0, 1] (1 = the trained maximum: 8 degrees of pan/tilt or a 1.25x zoom) mapped to the trained
  strength through a per-direction isotonic-regression calibration; the notebook shows the real physical amount ("Pan right: 5.6 degrees").
- **Teacher-anchored inference.** Text guidance stays native; the adapter only adds its camera residual above the frozen prediction, for noise levels >= 0.6.

Held-out check (pass-B loss on the revealed region, 28 dev sources, fixed noise): 0.230 at the v1 adapter -> 0.140 at update 415, with no sign of
overfitting in the dev-vs-train monitor. Numbers and caveats: [`docs/experiments/E1_EVAL_415.md`](docs/experiments/E1_EVAL_415.md).

Repository: <https://github.com/D3vil-coder/ie643-wan22-camera-adapter> (`master`; tags `v1-submission1`, `v2-final-415`; branch `exp/flat-image-fix` for the flat-image drift experiment).

## Quickstart: download and run

### What you need

| | |
|---|---|
| GPU | NVIDIA, at least 48 GB of memory (peak about 43 GiB per video; developed on a 95 GiB RTX PRO 6000 Blackwell). Without a GPU you can only run the tests and the notebook dry run. |
| Disk | about 35 GB: the base model is about 34 GB, the adapter bundle under 0.1 GB |
| Software | Python 3.12 (used for development), PyTorch 2.11.0 and torchvision 0.26.0 built for your CUDA version (the GPU host used the cu130 build), `git`. The code pins `diffusers 0.36.0` and `transformers 4.57.6` and refuses other versions. |
| Accounts | none: the base model and the adapter are public, no Hugging Face token is needed |

### 1. Get the code and the environment

```bash
git clone https://github.com/D3vil-coder/ie643-wan22-camera-adapter.git
cd ie643-wan22-camera-adapter
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
# install the PyTorch 2.11.0 + torchvision 0.26.0 wheels for your CUDA version first (pytorch.org), then:
pip install -r requirements.txt                          # pinned runtime
pip install -r requirements-ui.txt                       # marimo + anywidget, only for the notebook
export PYTHONPATH=src                                    # Windows PowerShell: $env:PYTHONPATH = "src"
```

### 2. Download the weights

```bash
# base model at the pinned revision (about 34 GB, goes to the default Hugging Face cache; the code loads it with local_files_only=True)
python -c "from huggingface_hub import snapshot_download as d; d('Wan-AI/Wan2.2-TI2V-5B-Diffusers', revision='b8fff7315c768468a5333511427288870b2e9635')"
#   same with the CLI:  huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --revision b8fff7315c768468a5333511427288870b2e9635

# the adapter bundle (adapter/, configs/, code/), public: https://huggingface.co/D3vil-Reap3r/ie643-wan22-camera-adapter-v2
python -c "from huggingface_hub import snapshot_download as d; d('D3vil-Reap3r/ie643-wan22-camera-adapter-v2', local_dir='ie643_adapter_v2')"
#   same with the CLI:  huggingface-cli download D3vil-Reap3r/ie643-wan22-camera-adapter-v2 --local-dir ie643_adapter_v2
```

Keep `adapter/adapter_000415.safetensors`, `adapter/ckpt_000415.json` and `adapter/config.json` together in one folder: the loader reads the
manifest and `config.json` (which records the replicate guide fill) next to the weights.

### 3a. Notebook (easiest): Camera Studio v2

The notebook `notebooks/camera_studio_v2.py` installs what is missing, downloads the base model and the adapter bundle itself, loads the pipeline
once and generates with live progress. Upload a photo, draw a stroke (pan or tilt) or pick zoom in / out, set the strength slider (0 to 1), press Generate.

```bash
marimo edit notebooks/camera_studio_v2.py      # code visible;  marimo run  notebooks/camera_studio_v2.py = app view only
FPCAMERA_USE_REPO_SRC=1 marimo run notebooks/camera_studio_v2.py   # use this checkout's src/ instead of the code zip from the Hub
FPCAMERA_DRY_RUN=1 marimo edit notebooks/camera_studio_v2.py       # no GPU, no installs, no downloads: UI test with placeholder videos
```

On **Molab** (marimo's hosted notebooks): create a notebook on a GPU sandbox with at least 48 GB, put the contents of
`notebooks/camera_studio_v2.py` into it (upload the file or paste it) and press **Run all**. It installs the pinned runtime and downloads the
model by itself (first run: the 34 GB download plus loading, several minutes; every later video: about 6 minutes). Nothing persists across a sandbox restart.
Optional switches: `FPCAMERA_WORK_DIR` (downloads, inputs, videos; default `~/fpcamera_studio`), `FPCAMERA_ADAPTER_REPO`.

### 3b. Command line

```bash
python -m fpcamera.flow_infer --project-root . \
  --checkpoint ie643_adapter_v2/adapter/adapter_000415.safetensors \
  --calibration ie643_adapter_v2/configs/strength_calibration_v2.json \
  --image photo.jpg --output outputs/run1 \
  --motion pan_left --strength 0.6 --steps 50 --seed 42
```

`--motion` is one of `pan_left pan_right tilt_up tilt_down zoom_in zoom_out static`; `--strength` is the slider in [0, 1] (1 = 8 degrees of pan or tilt,
or a 1.25x zoom, mapped through the calibration file). Instead of `--strength` you can ask for a physical amount with `--amplitude-deg` (pan, tilt) or
`--zoom-ratio` (zoom). `--prompt`, `--guidance` (default 5.0) and `--config configs/inference_v2.json` (defaults from a JSON) are also available.
The input is centre-cropped to 1280 x 704.

### 3c. Python API

```python
from pathlib import Path
from fpcamera import flow_infer          # needs PYTHONPATH=src

record = flow_infer.generate(
    ".",                                                       # project root (a GPU lock is kept in ./artifacts/)
    Path("ie643_adapter_v2/adapter/adapter_000415.safetensors"),
    "photo.jpg", "outputs/run1",                               # image, new output folder
    motion="zoom_in", strength=0.6,
    calibration=Path("ie643_adapter_v2/configs/strength_calibration_v2.json"),
    max_amplitudes=dict(yaw_deg=8.0, pitch_deg=8.0, zoom_ratio=1.25),
    steps=50, seed=42,
)
```

Pass an already loaded pipeline with `pipe=` (see `flow_infer.load_inference_pipeline`) to avoid reloading the 5B model between videos.
Python API, batch generation and the other options: [`docs/v2/INFERENCE_V2.md`](docs/v2/INFERENCE_V2.md).

### Time, memory and outputs

- One video (121 frames, 24 fps, 1280 x 704, 50 steps): about 6 minutes (355 to 364 s measured) and about 43 GiB peak GPU memory on an RTX PRO 6000 Blackwell; model loading and the first download come on top.
- Command line and API: the `--output` folder gets `video.mp4`, `run.json` (all settings and hashes) and `request.json`.
  Notebook: one folder per video under `~/fpcamera_studio/runs/` (or `$FPCAMERA_WORK_DIR/runs/`).
- Tests (CPU only, no model needed): `python -m pytest -q` (288 tests, about 2 minutes).

## Where to read next

| If you want | Read |
|---|---|
| a map of the code: pipeline diagram, every file, entry points, glossary | [`docs/CODEBASE_GUIDE.md`](docs/CODEBASE_GUIDE.md) |
| the story and the decisions behind v2 | [`docs/ANALYSIS_AND_PLAN.md`](docs/ANALYSIS_AND_PLAN.md) |
| design details | [`docs/v2/`](docs/v2): data, training, inference, evaluation, calibration, interface |
| results and session runbooks | [`docs/experiments/`](docs/experiments): training sessions, E1 evaluation, G1 generation + calibration, Molab and backup setup |
| the report and poster | [`docs/submission/`](docs/submission) |
| the v1 baseline (Submission 1, tag `v1-submission1`) | [`docs/v1/`](docs/v1) |

## Layout

- `src/fpcamera/` - the Python package (geometry, data cache, adapter, hybrid trainer, inference, evaluation, run infrastructure)
- `scripts/` - data selection, cache building, training launch, generation, measurement, calibration, evaluation, restore
- `tools/` - Molab sandbox and Hugging Face tooling (`molab_sync`, `molab_recover`, `hf_backup`, `publish_public_hf`, ...)
- `notebooks/` - marimo notebooks (`camera_studio_v2.py` is the user-facing one)
- `configs/` - data, training, inference and strength-calibration JSON
- `tests/` - CPU-only pytest suite
- `gallery/` - offline HTML gallery template (media are generated, not stored)
- `data/flow_poc/pilot_sources.json`, `data/v2/sources_v2.json` - the v1 and v2 COCO source manifests (ids, URLs, licences); the images and caches are generated (see `docs/v2/DATA_V2.md`)
- `ASSETS.md` - large assets kept out of git (checkpoints, caches, media) and where they live; `requirements*.txt` - pinned base, UI and evaluation dependencies

Tags: `v1-submission1` (baseline import), `v2-final-415` (frozen model and code).

## Local settings and licence

The operational tools (`tools/`) read two optional environment variables instead of hard-coded paths: `FPCAMERA_KEYS_ENV` (a `KEY=value` file
with Hugging Face tokens; default `<repo root>/keys.env`, git-ignored) and `FPCAMERA_CHECKPOINTS_DIR` (pulled checkpoints and evaluation outputs;
default `artifacts/checkpoints/v2`). Details: [`docs/experiments/BACKUP_AND_RESTORE.md`](docs/experiments/BACKUP_AND_RESTORE.md).

Licensed under the [Apache License 2.0](LICENSE) (see [`NOTICE`](NOTICE) for third-party licences: the Wan2.2 base model, COCO / Flickr
photographs and the course logos).
