# Inference walkthrough

## Preconditions

1. Use the recorded Python, PyTorch, and Diffusers versions in `requirements.txt` in an environment with CUDA support.
2. Make the exact Wan TI2V model revision from `src/fpcamera/wan.py` available in the local Hugging Face cache.
3. Run `python scripts/verify_frozen_package.py` from the project root.

The base model is not packaged, and the inference path requests local files only. No HF token or credential file is needed when the model is already cached.

## Default command

```powershell
python scripts/generate_camera.py --image data/flow_poc/reference_blue_ball.png --output artifacts/demo_zoom_in --mode zoom_in --magnitude 1
```

The launcher defaults to the audited step 300 adapter, seed 42, 50 denoising steps, text CFG 5, flow shift 5, camera gain 1, minimum active sigma 0.6, and the frozen positive teacher reference. It writes the request geometry, input copy, video, decoded keyframes, run record and logs under the new output directory.

For another image, replace `--image`. Available geometric presets are zoom in, zoom out, pan left, pan right, tilt up, and tilt down. Training included `zoom_in` and `zoom_out` only. Pan and tilt remain zero shot controls and require their own evaluation. The magnitude scales the geometry. A custom homography request can be supplied with `--camera-request`.

## Output checks

Generation verifies the decoded dimensions, frame count and fps and records the checkpoint SHA, frozen backbone SHA, guidance schedule, adapter component and source code hashes. This proves that a video with the expected format was produced. It does not establish that the requested path was followed or that newly exposed areas are realistic.

## Evaluation commands

```powershell
python scripts/measure_flow_video.py --video artifacts/demo_zoom_in/video.mp4 --request artifacts/demo_zoom_in/request.json --output artifacts/demo_zoom_in/motion_diagnostics.json
python scripts/measure_flow_preservation.py --input artifacts/demo_zoom_in/input.png --video artifacts/demo_zoom_in/video.mp4 --tracks artifacts/demo_zoom_in/motion_diagnostics.tracks.npz --request artifacts/demo_zoom_in/request.json --samples 13 --output artifacts/demo_zoom_in/preservation_diagnostics.json
python scripts/summarize_flow_adherence.py --tracks artifacts/demo_zoom_in/motion_diagnostics.tracks.npz --request artifacts/demo_zoom_in/request.json --output artifacts/demo_zoom_in/path_adherence.json
```

The first command tracks actual video pixels before comparing to the request. The summary includes direction and strength measures, plus missing track penalties. The preservation metric uses aligned overlapping regions and cannot certify unseen margins. Review the complete MP4 and contact sheet alongside the numbers.



