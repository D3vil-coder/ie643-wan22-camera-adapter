# Large assets (not in git)

All paths are relative to the `submission 1` folder of the original project
(`project/` there is the source of this repo's imported code).

## v1 checkpoint (step 300)
Directory: `project/artifacts/flow_poc_all/`
- `adapter_000300.safetensors` (about 57 MB) - the v1 adapter weights
  - sha256 (from `adapter_000300.json`, `checkpoint_sha256`): `9e2963f264ae5dcd9c406f12e32194782ba523908fb4d2220d8f7774906efdbc`
- `adapter_000300.json` - sidecar metadata (also duplicated as `latest.json`)
- `resume.pt` (about 114 MB) - optimizer/training resume state
- Also present: `config.json`, `progress.json`, `source_snapshot.zip`

## Latent cache
`project/data/flow_poc/cache_1280x704_f121_v1/` - 108 rows (`index.jsonl`),
with `videos/` (108 files), `references/` (72 files) and `progress.json`.
Resolution 1280x704, 121 frames.

## Source images
`project/data/flow_poc/sources/` - 36 JPGs in `train/` and `development/`.
The manifest `data/flow_poc/pilot_sources.json` is tracked in git.
Also not imported: `project/data/flow_poc/reference_blue_ball.png`.

## Example videos and gallery
- `examples/` - inputs, regression, unseen, ui_test, ui_revision outputs and JSON summaries
- `gallery/` - static gallery (`index.html`, `contact_sheets/`, `gallery_data.json`)

## Not imported
`project/submission_manifest.json`, and all media/binary files (mp4, png, pt, safetensors, zip).
