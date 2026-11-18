# Dataset and cache

## Source selection

`data/flow_poc/pilot_sources.json` records the selection from local COCO train and development manifests: 32 training photographs and four development photographs, with one image per selected category. The copied source images are byte copies with their source IDs, original image SHA-256 values, license fields, original captions and neutral training captions in the manifest. This is a pilot with 36 images, not the full COCO dataset.

The neutral caption policy isolates the camera hint. Original captions and source provenance are retained as metadata. The public source images are in `data/flow_poc/sources/`.

## Synthetic camera cache

Each source has three paired variants: `static`, `zoom_in` and `zoom_out`. This gives 108 indexed rows:

| Split | Sources | Rows | Modes per source |
|---|---:|---:|---|
| Train | 32 | 96 | `static`, `zoom_in`, `zoom_out` |
| Development | 4 | 12 | `static`, `zoom_in`, `zoom_out` |
| Total | 36 | 108 | 3 |

The RGB video contract is 1280×704 at 24 fps for 121 frames. Latents have shape `[48, 31, 44, 80]`. The cache stores BF16 latents after native FP32 VAE normalization, plus camera homographies. Each source has a safetensors first image and text reference and a rendered first view PNG. No generated video or camera pose is a captured physical trajectory: the zoom paths are synthetic geometric targets rendered from the source still.

The index is `data/flow_poc/cache_1280x704_f121_v1/index.jsonl`. It includes SHA-256 values for each latent clip, reference tensor and original source image, as well as split, source ID, mode, normalization and video contract. The checkpoint identity binds to the exact index SHA-256.

## Rebuilding the cache

The already encoded cache with 108 rows is included so normal training and evaluation do not require VAE and text encoding again. Encoding the cache again requires the pinned Wan VAE and T5 files in the local model cache and the source images/manifests above. `src/fpcamera/flow_cache.py` uses model loading from local files only. Cache regeneration is deliberately a separate action from using the packaged cache.

The cache builder has a CLI for a CUDA host. For a new scratch project root with a copy of `data/flow_poc/pilot_sources.json` and the corresponding `data/flow_poc/sources/` images, run:

```bash
PYTHONPATH=src python -m fpcamera.flow_cache --project-root /path/to/scratch --frames 121 --max-new-clips 24 --modes zoom_in zoom_out static
```

The command loads the pinned base files from the local Hugging Face cache and writes new references, videos, and index rows below that scratch root. `--max-new-clips` limits new encoded clips per invocation. Rerunning against that same scratch index continues incrementally and validates already indexed cache files. Use a scratch root so cache generation cannot alter this frozen package with 108 rows.




