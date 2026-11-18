# Trajectory-Controlled Camera Motion Adapter for Wan2.2 I2V (IE 643)

A lightweight 14.26M-parameter LoRA + guide adapter on the frozen
`Wan-AI/Wan2.2-TI2V-5B-Diffusers` image-to-video model, giving trajectory
control of camera motion (pan, tilt, zoom).

**Status:** v1 baseline (Submission 1, step 300) imported; v2 work in progress -
see docs/ANALYSIS_AND_PLAN.md

## Layout
- `src/fpcamera/` - the Python package (adapter, training, inference, provenance)
- `scripts/` - training, evaluation and utility scripts
- `configs/` - configuration files
- `notebooks/` - marimo interface notebooks
- `docs/v1/` - v1 documentation (architecture, data, training, inference, interface, reproducibility; `PROJECT_README_v1.md` is the original project README)
- `data/flow_poc/pilot_sources.json` - source manifest (images and cache are not in git)
- `requirements*.txt` - base, UI and evaluation dependencies
- `ASSETS.md` - large assets kept out of git (checkpoint, cache, media) and where they live

Tag `v1-submission1` marks the baseline import.
