"""Submission 1 (v1) helper: render verified camera runs as input plus frames 40, 80 and 120 (one PNG).

Reads fixed-name case folders (``CASES``) below ``--suite-root``, checks they share one checkpoint/profile and draws
one row per case.  Never overwrites an existing output.

    python scripts/render_camera_suite.py --suite-root artifacts/suite/step300 --output suite.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

CASES = {
    "ball_pan_left",
    "ball_pan_right",
    "ball_tilt_up",
    "ball_tilt_down",
    "heldout_zoom_in",
    "heldout_zoom_out",
    "blue_ball_zoom_in_frozen",
    "blue_ball_zoom_out_frozen",
    "blue_ball_direct_zoom_in",
    "blue_ball_direct_zoom_out",
    "blue_ball_zoom_in",
    "blue_ball_zoom_out",
    "blue_ball_frozen",
}
NATIVE_CONTRACT = {"width": 1280, "height": 704, "frames": 121, "fps": 24}
COMMON_PROFILE_KEYS = (
    "completed_updates",
    "checkpoint_sha256",
    "seed",
    "steps",
    "guidance",
    "flow_shift",
    "camera_guidance",
    "camera_residual_anchor",
    "camera_min_sigma",
    "camera_sigma_schedule",
    "camera_guidance_reference",
    "prompt",
    "negative_prompt",
)
FRAME_INDICES = (40, 80, 120)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _load_case(folder: Path, name: str) -> dict:
    """Load and verify one case folder (run.json, video, input) for the suite figure."""
    for required in ("run.json", "input.png", "video.mp4"):
        if not (folder / required).is_file():
            raise FileNotFoundError(f"Case {name} is incomplete: {folder / required}")
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    if run.get("backbone_unchanged") is not True:
        raise ValueError(f"Case {name} lacks backbone unchanged attestation")
    contract = run.get("contract", {})
    if any(contract.get(key) != value for key, value in NATIVE_CONTRACT.items()):
        raise ValueError(f"Case {name} does not match native 1280x704/121/24 format")
    video = run.get("video", {})
    if _sha(folder / "video.mp4") != video.get("sha256") or run.get(
        "input_sha256"
    ) != _sha(folder / "input.png"):
        raise ValueError(f"Case {name} input or video hash mismatch")
    with Image.open(folder / "input.png") as source:
        input_image = source.convert("RGB")
    if input_image.size != (NATIVE_CONTRACT["width"], NATIVE_CONTRACT["height"]):
        raise ValueError(f"Case {name} conditioning input has unexpected dimensions")
    capture = cv2.VideoCapture(str(folder / "video.mp4"))
    if not capture.isOpened():
        raise ValueError(f"Cannot open case video: {folder / 'video.mp4'}")
    decoded = 0
    selected = {}
    selected_targets = set(FRAME_INDICES)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape != (704, 1280, 3):
            capture.release()
            raise ValueError(f"Case {name} frame has wrong dimensions at {decoded}")
        if decoded in selected_targets:
            selected[decoded] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        decoded += 1
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if (
        decoded != 121
        or int(video.get("decoded_frames", -1)) != decoded
        or abs(fps - 24.0) > 0.01
    ):
        raise ValueError(f"Case {name} video failed the 121-frame/24-fps decode check")
    if set(selected) != selected_targets:
        raise ValueError(f"Case {name} is missing a requested contact-sheet frame")
    return {
        "name": name,
        "folder": folder,
        "run": run,
        "input": input_image,
        "frames": selected,
        "input_sha256": _sha(folder / "input.png"),
    }


def _common_profile(cases: list[dict]) -> dict:
    first = cases[0]["run"]
    profile = {key: first.get(key) for key in COMMON_PROFILE_KEYS}
    profile["contract"] = {
        key: first.get("contract", {}).get(key) for key in NATIVE_CONTRACT
    }
    for case in cases[1:]:
        run = case["run"]
        if any(run.get(key) != profile[key] for key in COMMON_PROFILE_KEYS):
            raise ValueError(f"Case {case['name']} has a different inference profile")
        contract = {key: run.get("contract", {}).get(key) for key in NATIVE_CONTRACT}
        if contract != profile["contract"]:
            raise ValueError(f"Case {case['name']} has a different video contract")
    return profile


def _render(cases: list[dict], output: Path, profile: dict) -> None:
    """Draw the suite figure (input + frames 40/80/120 per case)."""
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite contact sheet: {output}")
    margin, image_w, image_h, title_h = 18, 384, 211, 56
    row_h = title_h + image_h + 19
    canvas = Image.new(
        "RGB", (margin * 2 + 4 * image_w, 58 + len(cases) * row_h), "#f2f4f7"
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(20)
    label_font = _font(16)
    small_font = _font(14)
    digest = str(profile["checkpoint_sha256"] or "")[:12]
    draw.text(
        (margin, 12),
        f"Camera suite | step {profile['completed_updates']} | checkpoint {digest}",
        font=title_font,
        fill="#17212b",
    )
    columns = (
        "Conditioning input",
        "Generated frame 40",
        "Generated frame 80",
        "Generated frame 120",
    )
    y_header = 36
    for col, label in enumerate(columns):
        draw.text(
            (margin + col * image_w, y_header), label, font=small_font, fill="#46525f"
        )

    input_hashes = sorted({case["input_sha256"] for case in cases})
    group = {sha: f"source {chr(65 + i)}" for i, sha in enumerate(input_hashes)}
    for index, case in enumerate(cases):
        run = case["run"]
        y = 58 + index * row_h
        control = run.get("camera_guidance_reference") or "unspecified"
        label = (
            f"{case['name']}  |  mode={run.get('mode')}  |  anchor={run.get('camera_residual_anchor')}  |  "
            f"magnitude={run.get('magnitude')}"
        )
        draw.text((margin, y + 1), label, font=title_font, fill="#17212b")
        profile_line = (
            f"{group[case['input_sha256']]} · control reference={control} · "
            f"guidance={run.get('camera_guidance')} · sigma floor={run.get('camera_min_sigma')}"
        )
        draw.text((margin, y + 29), profile_line, font=small_font, fill="#46525f")
        images = [case["input"], *(case["frames"][frame] for frame in FRAME_INDICES)]
        for col, image in enumerate(images):
            thumb = image.resize((image_w, image_h), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (margin + col * image_w, y + title_h))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG")


def main() -> int:
    """CLI entry; renders the PNG (never overwrites)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-root",
        type=Path,
        required=True,
        help="step directory containing fixed-name case folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new PNG path; existing output is never overwritten",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(CASES),
        help="optional fixed case subset; default includes all configured folders found",
    )
    args = parser.parse_args()
    root = args.suite_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"--suite-root is not a directory: {root}")
    present = []
    selected_names = args.cases if args.cases else sorted(CASES)
    for name in selected_names:
        folder = root / name
        if folder.exists():
            present.append(_load_case(folder, name))
    if not present:
        raise FileNotFoundError(
            f"No configured complete camera cases found under {root}"
        )
    profile = _common_profile(present)
    _render(present, args.output, profile)
    print(
        json.dumps(
            {
                "status": "rendered",
                "output": str(args.output.expanduser().resolve()),
                "cases": [case["name"] for case in present],
                "rows": len(present),
                "checkpoint_sha256": profile["checkpoint_sha256"],
                "completed_updates": profile["completed_updates"],
                "profile_checked": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
