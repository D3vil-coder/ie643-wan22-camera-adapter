"""Render a matched input/zoom contact sheet from local, completed I2V folders."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

SAMPLE_FRAMES = (40, 80, 120)
CASE_NAMES = ("baseline", "zoom_in", "zoom_out")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_case(name: str, folder: Path, *, expect_adapter: bool) -> dict:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    run_path, input_path, video_path = (
        folder / "run.json",
        folder / "input.png",
        folder / "video.mp4",
    )
    for path in (run_path, input_path, video_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    contract = run.get("contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{name} has no I2V video contract")
    width, height = int(contract["width"]), int(contract["height"])
    frame_count, fps = int(contract["frames"]), float(contract["fps"])
    if (width, height, frame_count, fps) != (1280, 704, 121, 24.0):
        raise ValueError(
            f"{name} is not the expected 1280x704, 121-frame, 24-fps format"
        )
    updates = run.get("completed_updates")
    mode = run.get("mode")
    if not isinstance(updates, int) or updates < 0:
        raise ValueError(f"{name} lacks a completed checkpoint step")
    if mode not in ("zoom_in", "zoom_out"):
        raise ValueError(f"{name} has no zoom camera anchor: {mode!r}")
    residual_anchor = run.get("camera_residual_anchor", "none")
    if residual_anchor not in ("none", "frozen", "teacher"):
        raise ValueError(
            f"{name} has an unknown camera residual anchor: {residual_anchor!r}"
        )
    if run.get("adapter_enabled") is not expect_adapter:
        raise ValueError(
            f"{name} adapter_enabled does not match its baseline/adapter role"
        )
    if (
        not isinstance(run.get("seed"), int)
        or not isinstance(run.get("steps"), int)
        or not run.get("prompt")
    ):
        raise ValueError(
            f"{name} lacks matched seed, prompt, or inference-step metadata"
        )
    if run.get("video", {}).get("sha256") != _sha(video_path):
        raise ValueError(f"{name} video hash does not match run.json")
    image = Image.open(input_path).convert("RGB")
    if image.size != (width, height):
        raise ValueError(
            f"{name} input.png dimensions {image.size} do not match contract"
        )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open {video_path}")
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    sampled = {}
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (height, width, 3):
                raise ValueError(
                    f"{name} decoded frame {decoded} has shape {frame.shape}"
                )
            if decoded in SAMPLE_FRAMES:
                sampled[decoded] = Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
            decoded += 1
    finally:
        capture.release()
    expected_video = run["video"]
    if (
        decoded != frame_count
        or int(expected_video.get("decoded_frames", -1)) != decoded
    ):
        raise ValueError(
            f"{name} decoded {decoded} frames; run/contract expect {frame_count}"
        )
    if (
        abs(actual_fps - fps) > 0.01
        or abs(float(expected_video.get("fps", -1)) - fps) > 0.01
        or int(expected_video.get("width", -1)) != width
        or int(expected_video.get("height", -1)) != height
    ):
        raise ValueError(f"{name} decoded FPS/dimensions disagree with run.json")
    if set(sampled) != set(SAMPLE_FRAMES):
        raise RuntimeError(f"{name} did not decode all requested sample frames")
    return {
        "name": name,
        "folder": str(folder),
        "run": run,
        "input": image,
        "samples": sampled,
        "input_sha256": _sha(input_path),
        "video_sha256": expected_video["sha256"],
        "decoded_frames": decoded,
        "decoded_fps": actual_fps,
        "dimensions": [width, height],
        "camera_residual_anchor": residual_anchor,
    }


def _contact_sheet(cases: list[dict], output: Path) -> None:
    cell_w, cell_h = 480, 264
    row_label_w, margin, gap = 112, 22, 14
    title_h, header_h, row_caption_h = 46, 58, 24
    rows = ("input", *SAMPLE_FRAMES)
    width = margin * 2 + row_label_w + len(cases) * cell_w + (len(cases) - 1) * gap
    height = (
        margin * 2 + title_h + header_h + len(rows) * (cell_h + row_caption_h + gap)
    )
    sheet = Image.new("RGB", (width, height), "#121822")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    draw.text(
        (margin, margin),
        "Matched input and camera-motion comparison",
        fill="#f2f5f8",
        font=font,
    )
    y0 = margin + title_h
    for column, case in enumerate(cases):
        run = case["run"]
        role = "Frozen baseline" if case["name"] == "baseline" else "Trained adapter"
        mode_label = (
            "camera command disabled"
            if case["name"] == "baseline"
            else f"mode {run['mode']}"
        )
        label = (
            f"{role} · step {run['completed_updates']}\n{mode_label} · "
            f"residual anchor {case['camera_residual_anchor']}"
        )
        x = margin + row_label_w + column * (cell_w + gap)
        draw.text((x, y0), label, fill="#d4e4f4", font=font)
    for row_index, row in enumerate(rows):
        y = y0 + header_h + row_index * (cell_h + row_caption_h + gap)
        row_label = "I2V input" if row == "input" else f"Frame {row}"
        draw.text((margin, y + cell_h // 2), row_label, fill="#f2f5f8", font=font)
        for column, case in enumerate(cases):
            x = margin + row_label_w + column * (cell_w + gap)
            image = case["input"] if row == "input" else case["samples"][row]
            thumb = image.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            sheet.paste(thumb, (x, y))
            draw.text((x, y + cell_h + 4), row_label, fill="#aebdca", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite contact sheet: {output}")
    output = output.resolve()
    temp = output.with_name(output.stem + ".partial" + output.suffix)
    sheet.save(temp, format="PNG", optimize=True)
    temp.replace(output)


def render(
    baseline: Path,
    zoom_in: Path,
    zoom_out: Path,
    output: Path,
    report: Path | None = None,
) -> dict:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite contact sheet: {output}")
    if report is not None:
        report = report.expanduser().resolve()
        if report.exists():
            raise FileExistsError(f"Refusing to overwrite report: {report}")
        if report == output:
            raise ValueError("PNG output and JSON report paths must differ")
    cases = [
        _read_case("baseline", baseline, expect_adapter=False),
        _read_case("zoom_in", zoom_in, expect_adapter=True),
        _read_case("zoom_out", zoom_out, expect_adapter=True),
    ]
    reference = cases[0]["run"]
    matched_fields = ("seed", "steps", "prompt", "contract")
    for case in cases[1:]:
        for field in matched_fields:
            if case["run"].get(field) != reference.get(field):
                raise ValueError(
                    f"Matched comparison {field} differs: baseline vs {case['name']}"
                )
        if case["input_sha256"] != cases[0]["input_sha256"]:
            raise ValueError(
                f"Input image bytes differ between baseline and {case['name']}"
            )
    if zoom_in.resolve() == zoom_out.resolve() or baseline.resolve() in (
        zoom_in.resolve(),
        zoom_out.resolve(),
    ):
        raise ValueError("Baseline, zoom-in, and zoom-out folders must be distinct")
    _contact_sheet(cases, output)
    record = {
        "status": "contact_sheet_rendered_matched_inference_settings",
        "output": str(output.resolve()),
        "sample_frames": list(SAMPLE_FRAMES),
        "matched_fields": list(matched_fields),
        "cases": [
            {
                key: case[key]
                for key in (
                    "name",
                    "folder",
                    "input_sha256",
                    "video_sha256",
                    "decoded_frames",
                    "decoded_fps",
                    "dimensions",
                )
            }
            | {
                "completed_updates": case["run"]["completed_updates"],
                "mode": case["run"]["mode"],
                "camera_residual_anchor": case["camera_residual_anchor"],
                "adapter_enabled": case["run"]["adapter_enabled"],
            }
            for case in cases
        ],
        "interpretation": "Static visual comparison of the same I2V input and selected decoded frames; visual inspection is still required.",
    }
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
        record["report"] = str(report)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="Local frozen-baseline output folder",
    )
    parser.add_argument(
        "--zoom-in",
        required=True,
        type=Path,
        help="Local trained zoom-in output folder",
    )
    parser.add_argument(
        "--zoom-out",
        required=True,
        type=Path,
        help="Local trained zoom-out output folder",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New contact-sheet PNG path"
    )
    parser.add_argument(
        "--report-json", type=Path, help="Optional small machine-readable audit"
    )
    args = parser.parse_args()
    record = render(
        args.baseline, args.zoom_in, args.zoom_out, args.output, args.report_json
    )
    print(
        json.dumps(
            {
                "status": record["status"],
                "output": record["output"],
                "sample_frames": record["sample_frames"],
                "cases": record["cases"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
