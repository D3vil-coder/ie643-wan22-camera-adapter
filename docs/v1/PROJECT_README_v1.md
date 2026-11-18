# Trajectory conditioned camera control for Wan I2V

This folder is a self contained project copy for Submission 1. It contains the frozen step 300 Flow Control LoRA, its exact resume optimizer/RNG state, the 108 encoded training examples, the 36 source photographs and manifests, readable source copies plus byte exact executed source snapshots, and CPU evaluation tools.

The pretrained Wan2.2 TI2V 5B backbone is intentionally not copied: it is a multi gigabyte model. Inference and training require the pinned model revision to be available in the local Hugging Face cache. The project code requests local files only and does not download model files.

## Current artifact

- Adapter: 14,261,248 trainable parameters, 16 Control LoRA blocks, rank 16, alpha 16.
- Training checkpoint: update 300, `artifacts/flow_poc_all/adapter_000300.safetensors`.
- Exact resume state: `artifacts/flow_poc_all/resume.pt`, with sidecar, `latest.json`, runtime config and source snapshot.
- Frozen base: `Wan-AI/Wan2.2-TI2V-5B-Diffusers`, revision `b8fff7315c768468a5333511427288870b2e9635`.
- Native video contract: 1280×704, 121 frames, 24 fps, inference default 50 UniPC steps, text guidance 5, flow shift 5.

Training progress and generated videos are not a camera control acceptance test. Evaluate each output with the supplied motion and preservation tools and inspect the video, especially for newly exposed regions.

## Package layout

- `src/fpcamera/`: frozen training, cache, geometry, adapter, inference and UI helper modules.
- `notebooks/`: interactive camera interface.
- `data/flow_poc/`: 36 public source images and metadata, 108 native format cache rows, latent references and the pilot selection manifest.
- `artifacts/flow_poc_all/`: only the final step 300 adapter and its training/resume provenance, earlier checkpoints are excluded.
- `configs/provenance/`: original Molab path config files as used for the experiment.
- `configs/train_resume_300.json`: same training identity with package relative paths.
- `requirements.txt`: recorded runtime pins.
- `scripts/`: project root launchers, metric tools, plots and renderers.
- `docs/`: architecture, dataset, training, inference and reproducibility walkthrough.
- `submission_manifest.json`: SHA 256 inventory and frozen identity summary.

## Code walkthrough

`src/fpcamera/flow_geometry.py` turns a mode and magnitude into a frame aligned homography path. `flow_cache.py` renders each source photograph along that path, encodes the resulting 121 frame clip through Wan's native VAE, and records the latent, image reference, path, source hashes and contract in the cache index. `flow_train_delta.py` pairs opposite zoom commands for the same source, static noisy input and timestep, it obtains a no gradient frozen teacher prediction and trains only the 14.26M parameter adapter toward the teacher relative camera velocity target. `flow_control.py` builds the image condition and applies the adapter inside the frozen transformer. `flow_infer.py` loads the verified checkpoint, runs the requested camera path during the configured denoising interval, decodes frames, and writes request and run metadata. The CPU scripts measure motion and image preservation separately, neither a training loss nor a successful decode proves camera adherence.

The `src/fpcamera/provenance/executed_source/` directory preserves the exact source bytes recorded by the completed training run. The corresponding files in `src/fpcamera/` are the readable package copy, formatting is checked for AST equivalence to the preserved files. Two files are left in their original formatting because the available formatter rewrites multi name `del` statements into a different Python AST shape. The verifier checks both source hashes and AST equivalence.

## Verify the package

From this directory:

```powershell
python scripts/verify_frozen_package.py
```

The verifier checks every listed file digest, the checkpoint/sidecar/latest/resume linkage, the adapter parameter count, the dataset index digest and row/split counts, and the source hashes embedded in the checkpoint sidecar. It hashes files but does not unpickle the resume state.

## Generate a video

After the pinned model is already present in the local Hugging Face cache, run from this directory:

```powershell
python scripts/generate_camera.py --image data/flow_poc/reference_blue_ball.png --output artifacts/demo_zoom_in --mode zoom_in
```

The output directory must not already exist. The launcher resolves the project root from its own file location, selects the archived checkpoint by default, and writes the request, input copy, generated MP4, run metadata and frames. A completed call means a video was generated and decoded, it does not establish camera control success.

## Interactive camera interface

The interface is [notebooks/flow_camera_interface.py](notebooks/flow_camera_interface.py). It runs in the existing GPU inference environment because generation uses the local Wan cache and frozen adapter. One drawn straight path UI run was exercised end to end on the verified host, this confirms that single UI to generation workflow, not general path adherence. Install only the UI pins into that environment, then launch it from the project root:

```powershell
python -m pip install -r requirements-ui.txt
marimo edit notebooks/flow_camera_interface.py
```

Keep the UI dependencies separate from `requirements-evaluation.txt`, which is for CPU only metrics and figures. The Motion type selector offers **Pan or tilt path**, **Zoom in**, and **Zoom out**. Draw a line across the image to set horizontal and vertical camera view movement. When zoom is selected, the path canvas is visibly disabled and its points are ignored. A drawn path must contain at least two points before generation can start. Zoom presets use the checkpoint matched empirical profile, every output still needs its own review.

The CLI supports all six path families through the same geometric API. Training used zoom in and zoom out only. Pan and tilt are exploratory zero shot commands unless separately evaluated on the supplied output.

## Reproduce the exact training state

The original training used a 1,000 update schedule and stopped at update 300. The package includes the exact optimizer and RNG state. Resume with the archived runtime pins in `requirements.txt` and the already available local Wan model revision. The launcher verifies sidecar source hashes and imports the byte exact executed trainer under `src/fpcamera/provenance/executed_source/`, it does not resume through the readable formatted source copy. Check source identity without model loading using `--check-only`. To resume, specify a stop update explicitly:

```powershell
python scripts/train_resume.py --stop-after-updates 300
```

Use a value above 300 only when you intend to continue training. The training script writes into `artifacts/flow_poc_all`, preserve a copy of this package before continuing if the frozen Submission 1 state must remain unchanged. The resume file is only for this package's trusted provenance and is loaded by PyTorch's optimizer state deserializer.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/DATA.md](docs/DATA.md), [docs/TRAINING.md](docs/TRAINING.md), and [docs/INFERENCE.md](docs/INFERENCE.md) for details.

## Start a separate fresh training run

The resume launcher continues the archived update 300 optimizer state. To start from zero while keeping the original checkpoint untouched, use the separate fresh launcher on the GPU host:

```powershell
python scripts/train_fresh.py --run-name fresh_zoom_v1 --stop-after-updates 300
```

The launcher's source hash and cache index checks run before model loading. It requires a previously unused run name, verifies the exact executed trainer snapshot, and mounts the existing `data/flow_poc` folder into a new run root with a local directory symlink. The output goes under `artifacts/fresh_runs/<run-name>/`, it does not write to the frozen `artifacts/flow_poc_all/`. The schedule remains 1,000 updates with an explicit initial stop at 300. This is an instruction only, no fresh training was run to prepare this package.

Check the source and config on any host without CUDA or output writes:

```powershell
python scripts/train_fresh.py --run-name fresh_zoom_v1 --stop-after-updates 300 --check-only
```

For a source only repository checkout, `scripts/bootstrap_artifacts.py` restores the full local artifacts from a separate full package and checks the pinned base model on the GPU host. It supports `--target-project` so the bootstrap can be launched from either the source repository or this full package. See the repository level README for its host side usage.

## Adjacent frame consistency diagnostic

`scripts/measure_temporal_consistency.py` compares each frame with the one before it. It fits a robust homography from point tracks visible in both frames, warps the earlier frame, and measures normalized RGB mean absolute error over pixels shared by both frames. This compensates for global two dimensional motion before measuring the remaining pixel change.

The measure is descriptive. It cannot judge newly exposed regions, semantic drift, or whether the generated video followed the requested path. It has no pass or fail threshold. The five unseen examples had 600 successful frame pairs out of 600. Their pooled mean error was 0.00561 and median error was 0.00521 on a zero to one scale. The mean shared region fraction was 0.99932. These values describe only the five videos and do not establish broad temporal quality.
