# Training walkthrough

## Frozen inputs and objective

Training uses the cached first image latent, text features, static clip latent and two camera conditioned motion targets from the same source. A single random noise tensor and one truncated normal σ are shared across a pair of opposite requested commands. The first temporal latent plane is fixed to the input image latent and excluded from the MSE on future planes.

For each command, the frozen transformer produces a teacher prediction on the same noisy static input. The adapter target is

```text
teacher_prediction + (static_clean - requested_clean) / sigma
```

The optimizer minimizes squared error on future latents for each requested condition. A 10% probability of a static pair provides additional supervision with no requested change. The recorded configuration uses AdamW, learning rate `1e-4`, weight decay `0.01`, warmup 25, effective batch 2 and full activation checkpointing. Noise is truncated normal with mean 0.8, standard deviation 0.075 and bounds [0.6, 1.0].

## Frozen step 300 state

The recorded schedule has 1,000 total updates. The run stopped at update 300. The checkpoint sidecar reports 14,261,248 adapter parameters, unchanged backbone hash, the training source hashes, training history, and a digest for the optimizer and RNG resume state. `resume.pt` retains the exact optimizer and RNG state. The original project config is preserved in `configs/provenance/`. `configs/train_resume_300.json` changes only the absolute Molab paths to relative paths while preserving the training identity.

## Resume command

The project launcher resolves paths relative to itself, so no source edits or Molab paths are required. Resume with the archived runtime versions in `requirements.txt` and an already available local Wan model revision. The launcher validates every training source SHA from the checkpoint sidecar and imports the exact trainer source bytes from `src/fpcamera/provenance/executed_source/`. It does not run the readable formatted package copy. This preserves the source identity stored in the optimizer state.

```powershell
python scripts/train_resume.py --stop-after-updates 300
```

The stop update must be explicit. To verify the source identity without loading the model or writing state, run `python scripts/train_resume.py --stop-after-updates 300 --check-only`. Choose a larger value to intentionally continue the schedule with 1,000 updates. A real resume loads the included optimizer state and writes new checkpoint metadata to `artifacts/flow_poc_all/`.

## Fresh run without a warm start

Use `python scripts/train_fresh.py --run-name fresh_zoom_v1 --stop-after-updates 300` to create a separate run from zero. The launcher rejects an existing output name, verifies the source hashes and cache index digest, removes resume and warm start fields, and imports the exact executed trainer snapshot. It keeps the recorded 1,000-update cosine schedule and explicitly stops the first run at update 300. Its output is isolated below `artifacts/fresh_runs/<run-name>/`, and the data cache is mounted with a symlink instead of copied. `--check-only` runs the source/config audit without CUDA, model loading, or file writes. No GPU training was run to prepare this package.

## What training statistics mean

Loss, target edit RMS, prediction edit RMS, target projection and command separation ratio describe agreement in the cached latent objective. They are useful for monitoring optimization. They are not optical flow, pixel displacement, video quality, physical camera motion or evidence of generalization to held out sources. Those require generated videos and separate evaluation.





