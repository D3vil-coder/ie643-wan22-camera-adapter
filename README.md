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

## Quickstart

**Easiest: the notebook** (a GPU sandbox such as marimo's Molab, about 48 GB of GPU memory). It installs what is missing, downloads the
base model (~32 GB) and the public adapter bundle, loads the pipeline once and generates with live progress:

```bash
pip install -r requirements-ui.txt
marimo run  notebooks/camera_studio_v2.py        # app view   (marimo edit = code visible)
FPCAMERA_DRY_RUN=1 marimo edit notebooks/camera_studio_v2.py   # laptop UI test: no GPU, no downloads
```

**Command line** (pinned runtime from `requirements.txt`, CUDA GPU, base model in the local Hugging Face cache). The adapter bundle is on the Hub:
[`D3vil-Reap3r/ie643-wan22-camera-adapter-v2`](https://huggingface.co/D3vil-Reap3r/ie643-wan22-camera-adapter-v2) (`adapter/`, `configs/`, `code/`).

```bash
pip install -r requirements.txt                  # after the platform's torch 2.11.0 wheel
python -c "from huggingface_hub import snapshot_download as d; print(d('D3vil-Reap3r/ie643-wan22-camera-adapter-v2'))"   # prints the bundle folder
export PYTHONPATH=src
python -m fpcamera.flow_infer --project-root . \
  --checkpoint <bundle>/adapter/adapter_000415.safetensors --calibration <bundle>/configs/strength_calibration_v2.json \
  --image photo.jpg --output outputs/run1 --motion pan_left --strength 0.6 --steps 50 --seed 42
```

`--motion` is one of `pan_left pan_right tilt_up tilt_down zoom_in zoom_out`; `--strength` is in [0, 1]. The output folder holds `video.mp4`
and `run.json` (all settings and hashes). Python API, batch generation and the other options: [`docs/v2/INFERENCE_V2.md`](docs/v2/INFERENCE_V2.md).

Tests (CPU only, no model needed): `python -m pytest -q` (288 tests).

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
