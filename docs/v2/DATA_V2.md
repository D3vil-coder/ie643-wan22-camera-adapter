# Data v2: sources, motions and latent cache

v2 trains on a larger, magnitude-diverse, six-direction synthetic camera cache so that the adapter
(a) learns to *generate* content in newly revealed borders and (b) responds proportionally to a
strength slider. All parameters live in `configs/data_v2.json`; code is in
`src/fpcamera/{flow_geometry,motion_sampling,flow_cache_v2}.py` and `scripts/{select_coco_sources_v2,build_cache_v2}.py`.

## 1. Sources

* Dataset: COCO 2017 **train** (annotations `annotations_trainval2017.zip`, about 252 MB). Only the
  selected images are fetched from `http://images.cocodataset.org/train2017/<file>`; the 18 GB image zip is never downloaded.
* Selection (`scripts/select_coco_sources_v2.py`, deterministic given `seed` = 20261116):
  1. a candidate for category *c* is an image with a non-crowd *c* annotation covering at least 5 % of the image and `min(width, height) >= 427 px`;
  2. category round-robin in waves: each wave offers every still-short category its next candidate (seeded-hash order); it is accepted only after download and passing: decodable, real colour (mean per-pixel channel std >= 3, rejects grayscale), and 8x8 average-hash Hamming distance > 5 to every accepted image and to the v1 images;
  3. target 600 new images; the 36 v1 sources (COCO val2017; excluded by COCO id) are added with their original splits.
* Splits: of the 600 new images the 24 with the smallest `sha256(seed|id)` are `development`, the other 576 `train`. v1 keeps 32 train / 4 development. Total 636 sources: **608 train, 28 development**.
* Manifest `data/v2/sources_v2.json` extends the v1 `pilot_sources.json` schema (same record keys: `source_id`, `source_image_id`, `target_category`, `original_image_sha256`, `source_image`, `official_captions`, `training_caption`, `original_license`, `provenance`, `split`; plus `origin` (`v1`/`v2`) and `ahash`). `source_image` is relative to the project root. Files are `data/v2/sources/train2017/*.jpg` (v1 images are used in place from `data/flow_poc/sources/` when present and checksum-correct, else re-fetched from their COCO URL into `data/v2/sources/val2017/`).
* Licences: COCO images are Flickr photographs under CC licences (ids 1-8 in the annotations; each record keeps its licence name and URL and the Flickr URL). The images are used unchanged as inputs for synthetic warps; they are not redistributed by this repository (`data/**/sources/` is git-ignored). Annotations are CC BY 4.0 (COCO Consortium).
* Captions: all clips use the neutral caption of v1 (`"A photograph of a scene. One continuous camera shot."`), so the camera hint is the only conditioning difference; the original COCO captions are kept as metadata.
* Resumability: `data/v2/sources/_ledger.json` maps each downloaded file to its sha256; matching files are skipped. Downloads retry with exponential backoff, 4 workers, generic user agent.

## 2. Motion model

One motion per clip (the UI asks for one motion at a time): family `pan` (left/right), `tilt` (up/down), `zoom` (in/out), plus `static`.
Strength `u` in [0, 1] maps to the physical *end-of-clip* amplitude, with **linear easing over 121 frames** (same `tau` as v1):

| quantity | value at strength u |
|---|---|
| yaw (pan), pitch (tilt) | `u * 8 deg` (right/up positive) |
| focal zoom ratio | `exp(+-u * ln 1.25)`, i.e. 1.25x at `u = 1`; zoom-out is the exact reciprocal |

`camera_path_v2(family, direction, u, contract, max_yaw_deg=8, max_pitch_deg=8, max_zoom_ratio=1.25, hfov_degrees=60)` returns the same `[121,3,3]` first-view -> frame-t homographies (`H[0] = I`) as v1 `camera_path` (it delegates to it with `magnitude=u`). Tests assert equality (< 1e-6) with v1 for zoom at ratio 1.12 (`u = ln 1.12 / ln 1.25`) and for pan/tilt at 5 deg (`u = 5/8`).

Sampling (`MotionSampler`, `src/fpcamera/motion_sampling.py`): per source one static clip and K = 2 moving clips (config `moving_clips_per_source`). Draws are a deterministic function of (seed, source population, source id, clip index); they never depend on call order or shard.

* Direction balance by construction: within each split, sources are ordered by a seeded hash and consecutive blocks of six receive a seeded permutation of the six directions, so every direction count differs by at most one. The second clip of a source is shifted to a different *family*; the shift is a bijection so balance also holds per clip slot.
* `u ~ U[0.15, 1.0]`, stratified within each (split, direction) into equal bins assigned by a seeded permutation plus hash jitter, so the marginal is near-uniform per direction (5-bin histogram within +-15 % of its mean in the tests).

For 636 sources: 636 static + 1272 moving = **1908 clips**, about 212 moving clips per direction.

## 3. Overscan and coverage (no filled pixel may reach a target)

The first view (frame 0) is a centred crop of the photograph, `1/overscan` times the largest 16:9-ish window that fits (`source_to_first_view`); the remainder is real source margin reserved for revealed regions. `overscan` must be large enough that, for the **maximum-strength** path in every direction, every output pixel of every frame maps inside the source (with a 1 px bilinear margin).

Derivation (`python scripts/build_cache_v2.py --derive-overscan`, bisection on a dense output grid including all borders; a projective map takes the convex output rectangle to a convex quadrilateral, so border-inclusive grids bound the extremum): required overscan at `u = 1`, by image aspect ratio (width/height):

| aspect | pan | tilt | zoom in | zoom out | max |
|---|---|---|---|---|---|
| 0.5 - 1.5 (portrait to 3:2) | 1.355 - 1.358 | 1.06 - 1.25 | 1.00 | 1.25 | 1.358 |
| 1.667 | 1.355 | 1.387 | 1.00 | 1.25 | 1.387 |
| >= 1.818 (16:9 and wider) | 1.355 / 1.10 (very wide) | **1.5125** | 1.00 | 1.25 | **1.5125** |

Pan is x-limited (about 1.355 for any image not wider than 16:9); tilt is y-limited and bites for wide images. **The plan's 1.45 is insufficient** for 16:9 images at 8 deg tilt (a test asserts this).

Policy in `configs/data_v2.json` (`overscan`):

* `per_source` (default): `overscan = max(1.35, required + 0.02)`, rounded up to 1e-3 and constant across the clips of a source (all clips share one first-view crop, which the reference latent requires). Typical COCO 4:3 photos get about 1.377, 16:9 photos about 1.533; sources needing more than `cap = 1.6` are skipped and logged to `rejected.jsonl`. The value is stored per row.
* `fixed = 1.53`: a single value that is safe for any aspect ratio (1.5125 + margin), selectable by `policy: "fixed"`.

Safety nets in the renderer: `cv2.warpPerspective(..., borderMode=BORDER_REFLECT_101)` (so even a bug could not produce black fill) **and** `render_source(..., assert_coverage=True)` which raises if any grid pixel of any frame maps outside the source (`coverage_slack`, `assert_source_coverage`); the clip is then rejected, not written. v1 defaults of `render_source` (constant border, no assertion) are unchanged.

Revealed-region sanity (fraction of output pixels at the last frame whose first-view preimage is outside the frame, `u = 1`): pan 0.164, tilt 0.239, zoom out 0.359 (analytic `1 - 1/1.25^2 = 0.36`), zoom in 0.

## 4. Cache format

Cache root `data/v2/cache_1280x704_f121_v2/` (git-ignored):

```
clips/<id>.safetensors          latents [48,31,44,80] bf16, homographies [121,3,3] f32
references/<source>.safetensors image_latent [48,1,44,80] bf16, text_embeddings, negative_text_embeddings  (v1 keys)
references/<source>.png         rendered first view
index.jsonl                     one row per clip
config_snapshot.json  progress.json  rejected.jsonl  encoder.log
```

VAE path is exactly v1 (`flow_cache.encode_rgb`): FP32 native VAE, posterior mode, per-channel normalisation in FP32, store bf16 (`normalization = native_fp32_vae_mode_normalize_then_bf16`). The video's first latent plane is checked against the image-only reference latent.

Index row: `{id, source_id, split, family, direction, magnitude_u, yaw_deg, pitch_deg, zoom_ratio, overscan, latent_file, reference_file, sha256}` plus compatibility/bookkeeping fields: `mode` (v1 string such as `pan_left`, `static`), `record_id`, `cache_path`/`reference_path` (aliases of `latent_file`/`reference_file`, relative to the cache root), `video_contract`, `normalization`, `caption`, `reference_sha256`, `source_sha256`, `revealed_fraction_last`, `first_latent_max_abs_diff`. `yaw_deg`/`pitch_deg` are signed end-of-clip values; `zoom_ratio` is the end focal scale (< 1 for zoom out). Static rows have `family = "static"`, `direction = null`, `magnitude_u = 0`. Load with `flow_cache_v2.load_cached_v2(row, cache_root)`; it returns the same keys as v1 `load_cached`.

Crash safety: files are written to `*.partial` then atomically renamed; the index row is appended (fsync) only after the clip file is complete. On restart the builder deletes partials, drops index rows whose file is missing or whose sha256 mismatches, **adopts** complete files that lack a row (the row JSON is embedded in the safetensors metadata), and rebuilds only what is missing. `--merge-index` rewrites `index.jsonl` from the files. `config_snapshot.json` pins the config and source population; a changed config is refused (`--force-config` overrides).

## 5. Commands (Molab, from the repo root)

```bash
pip install opencv-python pillow numpy safetensors      # if missing; torch/diffusers come with the Wan runtime
export PYTHONPATH=src

# 1. sources (internet; about 252 MB annotations + about 100 MB of images). Resumable: just rerun.
python scripts/select_coco_sources_v2.py --project-root .

# 2. geometry check without any model (writes preview strips; no torch import)
python scripts/build_cache_v2.py --project-root . --dry-run --limit 12 --preview-dir artifacts/preview_v2

# 3. smoke test of the real encode path; read s_per_clip and the disk line
python scripts/build_cache_v2.py --project-root . --limit 6

# 4. full build; interrupt and rerun at will
python scripts/build_cache_v2.py --project-root .

# optional: 2 workers on 2 GPUs/processes (each writes index.shard<i>of<n>.jsonl), then
python scripts/build_cache_v2.py --project-root . --shard 0/2 --no-lease
python scripts/build_cache_v2.py --project-root . --shard 1/2 --no-lease
python scripts/build_cache_v2.py --project-root . --merge-index      # writes index.jsonl
```

Before encoding, the builder compares free disk with `remaining clips x ~11 MB x 1.1 + 2 GiB` and refuses with a message if short (use K = 1 via `moving_clips_per_source`, fewer sources via `--limit`, or `--force-disk`). It prints per-clip render/encode seconds, a running s/clip, an ETA and free disk.

## 6. Expected size and time

* 636 sources x (1 + 2) = 1908 clips x 10.5 MB (latent `48*31*44*80*2` bytes = 10.48 MB) = **about 20.0 GB**, plus about 0.7 GB references/PNGs. K = 1 gives about 13.3 GB.
* Render about 1.5 s/clip on CPU (measured locally, includes the coverage check). VAE encode time of 121 frames at 1280x704 in FP32 has **not been measured** (the VAE cannot run locally): plan 20-60 s/clip, i.e. about 11-32 h for 1908 clips. Measure with step 3 and decide K/sources accordingly (`--project --s-per-clip <x>` prints the projection).
