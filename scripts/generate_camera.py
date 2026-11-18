"""Run the pinned native Wan flow-camera adapter from the clean project root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/inference_default.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--mode",
        choices=(
            "zoom_in",
            "zoom_out",
            "pan_left",
            "pan_right",
            "tilt_up",
            "tilt_down",
            "static",
        ),
    )
    parser.add_argument("--magnitude", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--flow-shift", type=float)
    parser.add_argument("--camera-guidance", type=float)
    parser.add_argument("--camera-min-sigma", type=float)
    parser.add_argument(
        "--camera-residual-anchor", choices=("teacher", "frozen", "none")
    )
    parser.add_argument("--prompt")
    parser.add_argument("--camera-request", type=Path)
    parser.add_argument(
        "--component",
        choices=("full", "guide_only", "attention_only", "lora_only"),
    )
    args = parser.parse_args()
    config_path = (
        args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    )
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    mode = args.mode if args.mode is not None else cfg["mode"]
    magnitude = args.magnitude if args.magnitude is not None else cfg["magnitude"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    steps = args.steps if args.steps is not None else cfg["steps"]
    guidance = args.guidance if args.guidance is not None else cfg["guidance"]
    flow_shift = args.flow_shift if args.flow_shift is not None else cfg["flow_shift"]
    camera_guidance = (
        args.camera_guidance
        if args.camera_guidance is not None
        else cfg["camera_guidance"]
    )
    camera_min_sigma = (
        args.camera_min_sigma
        if args.camera_min_sigma is not None
        else cfg["camera_min_sigma"]
    )
    camera_residual_anchor = (
        args.camera_residual_anchor
        if args.camera_residual_anchor is not None
        else cfg["camera_residual_anchor"]
    )
    prompt = (
        args.prompt
        if args.prompt is not None
        else cfg.get("prompt", "A photograph of a scene. One continuous camera shot.")
    )
    component = (
        args.component if args.component is not None else cfg.get("component", "full")
    )
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    checkpoint_arg = (
        args.checkpoint if args.checkpoint is not None else Path(cfg["checkpoint"])
    )
    checkpoint = (
        checkpoint_arg
        if checkpoint_arg.is_absolute()
        else PROJECT_ROOT / checkpoint_arg
    )
    image = args.image if args.image.is_absolute() else PROJECT_ROOT / args.image
    camera_request = args.camera_request
    if camera_request is not None and not camera_request.is_absolute():
        camera_request = PROJECT_ROOT / camera_request
    if output.exists():
        parser.error(f"Output directory already exists: {output}")
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from fpcamera.flow_infer import generate

    result = generate(
        PROJECT_ROOT,
        checkpoint,
        image,
        output,
        mode=mode,
        prompt=prompt,
        seed=seed,
        steps=steps,
        guidance=guidance,
        flow_shift=flow_shift,
        magnitude=magnitude,
        camera_request=camera_request,
        camera_guidance=camera_guidance,
        camera_min_sigma=camera_min_sigma,
        camera_residual_anchor=camera_residual_anchor,
        component=component,
    )
    print(
        f"Generated {result['video']['decoded_frames']} frames at {result['video']['fps']} fps; "
        f"motion and preservation still require evaluation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
