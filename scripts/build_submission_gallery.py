"""Build a fully offline HTML gallery and contact sheets for Submission 1."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil

import cv2
from PIL import Image, ImageDraw, ImageFont


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
UNSEEN_REQUIRED = ("input.png", "video.mp4", "run.json", "request.json")
REGRESSION_CASES = (
    "ball_pan_left",
    "ball_pan_right",
    "ball_tilt_up",
    "ball_tilt_down",
    "heldout_zoom_in",
    "heldout_zoom_out",
)
SAMPLE_FRAMES = (40, 80, 120)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _frame(video: Path, frame_index: int) -> Image.Image:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not decode frame {frame_index} from {video}")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def make_contact_sheet(input_image: Path, video: Path, output: Path) -> None:
    """Create an unannotated four-panel sheet: input and native RGB frames 40/80/120."""
    width, thumb_height, label_height = 384, 224, 38
    font = _font(19)
    panels = [("Input", Image.open(input_image).convert("RGB"))]
    panels.extend(
        (f"Output frame {index}", _frame(video, index)) for index in SAMPLE_FRAMES
    )
    canvas = Image.new(
        "RGB", (width * len(panels), thumb_height + label_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for column, (label, image) in enumerate(panels):
        image.thumbnail((width, thumb_height), Image.Resampling.LANCZOS)
        x = column * width + (width - image.width) // 2
        y = label_height + (thumb_height - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text((column * width + 12, 9), label, fill=(20, 31, 46), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def motion_summary(path_report: dict) -> dict:
    motion = path_report.get("motion") or {}
    adherence = path_report.get("adherence") or {}
    expected = adherence.get("expected_visible_tracks_per_frame") or []
    observed = adherence.get("observed_expected_visible_tracks_per_frame") or []
    if expected and observed and len(expected) == len(observed):
        denominator = sum(float(x) for x in expected)
        coverage = (
            sum(float(x) for x in observed) / denominator if denominator else None
        )
    else:
        coverage_values = (
            adherence.get("coverage_of_expected_visible_tracks_per_frame") or []
        )
        coverage = (
            sum(coverage_values) / len(coverage_values) if coverage_values else None
        )
    return {
        "signed_motion_gain_pathwide": motion.get("signed_motion_gain_pathwide"),
        "motion_cosine_pathwide": motion.get("motion_cosine_pathwide"),
        "ADE_px_missing_penalty_image_diagonal": adherence.get(
            "ADE_px_missing_penalty_image_diagonal"
        ),
        "FDE_px_missing_penalty_image_diagonal": adherence.get(
            "FDE_px_missing_penalty_image_diagonal"
        ),
        "track_coverage_fraction": coverage,
        "tracks_used_pathwide": motion.get("tracks_used_pathwide"),
        "endpoint_tracks_used": motion.get("endpoint_tracks_used"),
    }


def preservation_summary(report: dict) -> dict:
    summary = report.get("temporal_summary") or {}
    return {
        "mean_ssim_gray_11x11": summary.get("mean_ssim_gray_11x11"),
        "mean_overlap_fraction": summary.get("mean_overlap_fraction"),
        "temporal_samples_excluding_reference": summary.get(
            "temporal_sample_count_excluding_reference_frame"
        ),
        "failed_homography_fits": summary.get("failed_observed_homography_fits"),
        "interpretation_limits": report.get("interpretation_limits"),
    }


def _fmt(value, digits=3, suffix="") -> str:
    return "not available" if value is None else f"{float(value):.{digits}f}{suffix}"


def _svg_gain_chart(cases: list[dict]) -> str:
    measured = [
        c
        for c in cases
        if c.get("motion", {}).get("signed_motion_gain_pathwide") is not None
    ]
    if not measured:
        return "<p>No path-adherence measurements are available yet.</p>"
    row_h, width, left = 34, 760, 210
    height = 58 + row_h * len(measured)
    scale = 340 / 1.5
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Per-case signed motion gain">',
        '<text x="10" y="20" class="chart-title">Signed pathwide motion gain (1.0 = requested displacement magnitude)</text>',
    ]
    for tick in (0, 0.5, 1.0, 1.5):
        x = left + tick * scale
        parts.append(
            f'<line x1="{x:.1f}" y1="35" x2="{x:.1f}" y2="{height - 8}" class="grid"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - 4}" text-anchor="middle" class="axis">{tick:.1f}</text>'
        )
    for i, case in enumerate(measured):
        y = 43 + i * row_h
        gain = float(case["motion"]["signed_motion_gain_pathwide"])
        bar_width = max(0, min(gain, 1.5)) * scale
        label = html.escape(f"{case['name']} · {case['mode']}")
        parts.append(
            f'<text x="{left - 8}" y="{y + 16}" text-anchor="end" class="case-label">{label}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="20" rx="3" class="bar"/>'
        )
        parts.append(
            f'<text x="{left + bar_width + 6:.1f}" y="{y + 15}" class="value">{gain:.3f}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _case_record(
    name: str,
    folder: Path,
    html_prefix: str,
    contact_dir: Path,
    provenance: dict | None = None,
) -> dict:
    run = read_json(folder / "run.json")
    request = (
        read_json(folder / "request.json")
        if (folder / "request.json").is_file()
        else {}
    )
    path_file = folder / "path_adherence.json"
    preservation_file = folder / "preservation_diagnostics.json"
    path_data = read_json(path_file) if path_file.is_file() else {}
    preservation = read_json(preservation_file) if preservation_file.is_file() else {}
    contact_name = f"{name}.png"
    make_contact_sheet(
        folder / "input.png", folder / "video.mp4", contact_dir / contact_name
    )
    contract = run.get("contract") or request.get("contract") or {}
    video_info = run.get("video") or {}
    return {
        "name": name,
        "mode": run.get("mode") or request.get("mode", "unknown"),
        "magnitude": run.get("magnitude", request.get("magnitude")),
        "seed": run.get("seed"),
        "steps": run.get("steps"),
        "contract": {
            key: contract.get(key) for key in ("width", "height", "frames", "fps")
        },
        "checkpoint_sha256": run.get("checkpoint_sha256"),
        "backbone_unchanged": run.get("backbone_unchanged"),
        "video_bytes": (folder / "video.mp4").stat().st_size,
        "video_sha256": sha256(folder / "video.mp4"),
        "input_sha256": sha256(folder / "input.png"),
        "provenance": provenance or {},
        "motion": motion_summary(path_data) if path_data else {},
        "preservation": preservation_summary(preservation) if preservation else {},
        "metric_scope": (
            path_data.get("metric_scope")
            or "2D sparse-track path adherence; see source diagnostic JSON."
        ),
        "video_href": f"{html_prefix}/video.mp4",
        "input_href": f"{html_prefix}/input.png",
        "contact_href": f"contact_sheets/{contact_name}",
        "video_info": {
            key: video_info.get(key)
            for key in ("decoded_frames", "fps", "width", "height")
        },
    }


def _render_case(case: dict, label: str) -> str:
    motion = case["motion"]
    preserve = case["preservation"]
    provenance = case.get("provenance") or {}
    source_path = provenance.get("local_example_path") or provenance.get(
        "source_relative_path", "not recorded"
    )
    source_id = provenance.get("source_id") or provenance.get("case", "not recorded")
    metarows = [
        ("Command", f"{case['mode']} · magnitude {case['magnitude']}"),
        (
            "Run",
            f"seed {case['seed']} · {case['steps']} denoising steps · {case['contract'].get('width')}×{case['contract'].get('height')} · {case['contract'].get('frames')} frames @ {case['contract'].get('fps')} fps",
        ),
        ("Signed motion gain", _fmt(motion.get("signed_motion_gain_pathwide"))),
        ("Path cosine", _fmt(motion.get("motion_cosine_pathwide"))),
        (
            "ADE / FDE",
            f"{_fmt(motion.get('ADE_px_missing_penalty_image_diagonal'), 2)} px / {_fmt(motion.get('FDE_px_missing_penalty_image_diagonal'), 2)} px",
        ),
        (
            "Track coverage",
            _fmt(
                None
                if motion.get("track_coverage_fraction") is None
                else 100 * motion["track_coverage_fraction"],
                1,
                "%",
            ),
        ),
        ("Mean overlap SSIM", _fmt(preserve.get("mean_ssim_gray_11x11"))),
        ("Overlap fraction", _fmt(preserve.get("mean_overlap_fraction"), 3)),
    ]
    rows = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in metarows
    )
    caveat = "Preservation scores use track-aligned overlapping regions. Newly exposed borders or regions may not be scored; inspect the full video and preservation diagnostic."
    return f'''<article class="case" id="{html.escape(case["name"])}">
      <div class="case-head"><div><span class="eyebrow">{html.escape(label)}</span><h2>{html.escape(case["name"].replace("_", " ").title())}</h2></div><span class="mode-tag">{html.escape(case["mode"])}</span></div>
      <div class="media-grid"><figure><video controls playsinline preload="none" poster="{html.escape(case["input_href"])}"><source src="{html.escape(case["video_href"])}" type="video/mp4">Your browser cannot play this MP4.</video><figcaption>Native RGB MP4 · controls available · playback starts only when requested</figcaption></figure>
      <figure><img loading="lazy" src="{html.escape(case["contact_href"])}" alt="Input and output frames 40, 80, and 120 for {html.escape(case["name"])}"><figcaption>Input and unannotated RGB output frames 40, 80, 120</figcaption></figure></div>
      <table><tbody>{rows}</tbody></table>
      <p class="caveat">{caveat}</p>
      <details><summary>Case files and hash provenance</summary><p>Local provenance path: <code>{html.escape(str(source_path))}</code><br>Source/case ID: <code>{html.escape(str(source_id))}</code><br>Input SHA-256: <code>{html.escape(case["input_sha256"])}</code><br>Video SHA-256: <code>{html.escape(case["video_sha256"])}</code><br>Checkpoint SHA-256: <code>{html.escape(str(case["checkpoint_sha256"]))}</code></p><p><a href="{html.escape(case["input_href"])}">Input image</a> · <a href="{html.escape(case["video_href"])}">Original video file</a></p><p>{html.escape(case["metric_scope"])}</p></details>
    </article>'''


def _stage_regressions(submission_root: Path) -> None:
    original_root = (
        submission_root.parent
        / "camera_phase1/artifacts/flow_camera_extension/step_000300"
    )
    target_root = submission_root / "examples/regression"
    target_root.mkdir(parents=True, exist_ok=True)
    wanted = {
        "video.mp4",
        "run.json",
        "request.json",
        "input.png",
        "motion_diagnostics.json",
        "motion_diagnostics.tracks.npz",
        "preservation_diagnostics.json",
        "path_adherence.json",
    }
    for name in REGRESSION_CASES:
        src, dst = original_root / name, target_root / name
        if not src.is_dir():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for source in src.iterdir():
            destination = dst / source.name
            if source.is_file() and source.name in wanted and not destination.exists():
                shutil.copyfile(source, destination)
    # The path-adherence summary for the left pan was computed from its existing tracks and request.
    left = target_root / "ball_pan_left"
    left_summary = left / "path_adherence.json"
    if (
        not left_summary.is_file()
        and (left / "motion_diagnostics.tracks.npz").is_file()
    ):
        # Generate through the packaged CPU-only summarizer, without changing source artifacts.
        import subprocess
        import sys

        script = Path(__file__).resolve().parent / "summarize_flow_adherence.py"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--tracks",
                str(left / "motion_diagnostics.tracks.npz"),
                "--request",
                str(left / "request.json"),
                "--output",
                str(left_summary),
            ],
            check=True,
        )
    case_rows = []
    for name in REGRESSION_CASES:
        dst = target_root / name
        if not (dst / "video.mp4").is_file():
            continue
        source = original_root / name
        run = read_json(dst / "run.json")
        source_has_adherence = (source / "path_adherence.json").is_file()
        files = [
            {"path": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(dst.iterdir())
            if p.is_file()
        ]
        case_rows.append(
            {
                "case": name,
                "source_relative_path": source.relative_to(
                    submission_root.parent
                ).as_posix(),
                "mode": run.get("mode"),
                "magnitude": run.get("magnitude"),
                "seed": run.get("seed"),
                "camera_residual_anchor": run.get("camera_residual_anchor"),
                "checkpoint_sha256": run.get("checkpoint_sha256"),
                "derived_reports": []
                if source_has_adherence
                else [
                    "path_adherence.json computed locally from the copied motion_diagnostics.tracks.npz and request.json with the package CPU summarizer."
                ],
                "files": files,
            }
        )
    manifest = {
        "description": "Step-300 regression cases copied from the original camera_phase1 output folders. Separate from the five preselected unseen examples.",
        "cases": case_rows,
    }
    (target_root / "regression_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build(submission_root: Path) -> dict:
    examples = submission_root / "examples"
    gallery = submission_root / "gallery"
    contact_dir = gallery / "contact_sheets"
    gallery.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)
    _stage_regressions(submission_root)
    manifest = read_json(examples / "unseen_manifest.json")
    selected = []
    pending = []
    for item in manifest["cases"]:
        name = item["case"]
        folder = examples / "unseen" / name
        missing = [
            filename
            for filename in UNSEEN_REQUIRED
            if not (folder / filename).is_file()
        ]
        if missing:
            pending.append(
                {"case": name, "mode": item["mode"], "missing_files": missing}
            )
            continue
        selected.append(
            _case_record(
                name,
                folder,
                f"../examples/unseen/{name}",
                contact_dir,
                {
                    "local_example_path": f"examples/unseen/{name}",
                    "source_id": item.get("source_id"),
                    "source_image_id": item.get("source_image_id"),
                    "source_image_sha256": item.get("image_sha256"),
                    "original_category": item.get("original_category"),
                    "original_split": item.get("original_split"),
                    "selection": item.get("selection"),
                },
            )
        )
    regression = []
    for name in REGRESSION_CASES:
        folder = examples / "regression" / name
        missing = [
            filename
            for filename in ("input.png", "video.mp4", "run.json")
            if not (folder / filename).is_file()
        ]
        if missing:
            continue
        case_provenance = next(
            (
                row
                for row in read_json(
                    examples / "regression/regression_manifest.json"
                ).get("cases", [])
                if row.get("case") == name
            ),
            {},
        )
        regression.append(
            _case_record(
                name,
                folder,
                f"../examples/regression/{name}",
                contact_dir,
                case_provenance,
            )
        )

    payload = {
        "title": "Submission 1 · Offline camera-control examples",
        "selection_scope": manifest.get("scope"),
        "unseen_examples": selected,
        "pending_unseen_examples": pending,
        "regression_controls": regression,
    }
    (gallery / "gallery_data.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    graph_stats = {
        "scope": "Per-case metrics only; five preselected inputs with one command each, not a complete mode-by-scene benchmark.",
        "unseen_cases": [
            {"case": c["name"], "mode": c["mode"], **c["motion"], **c["preservation"]}
            for c in selected
        ],
        "pending_cases": pending,
    }
    (gallery / "graph_stats.json").write_text(
        json.dumps(graph_stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    cards = "\n".join(
        _render_case(case, "Preselected unseen example") for case in selected
    )
    regression_cards = "\n".join(
        _render_case(case, "Regression reference") for case in regression
    )
    pending_html = (
        ""
        if not pending
        else '<p class="pending">Awaiting local artifacts for: '
        + ", ".join(html.escape(x["case"]) for x in pending)
        + ". Rebuild the gallery after those case folders arrive.</p>"
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Submission 1 · Offline camera-control examples</title>
<style>
:root{{color-scheme:light;--ink:#1c2533;--muted:#667085;--line:#d9dee7;--panel:#fff;--bg:#f3f5f8;--blue:#315ca8;--gold:#d39b32}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,Segoe UI,Arial,sans-serif}}
header,main{{max-width:1180px;margin:0 auto;padding:28px 22px}}header{{padding-top:38px}}h1{{font-size:clamp(28px,4vw,42px);line-height:1.12;margin:0 0 10px}}h2{{margin:0;font-size:24px}}h3{{margin-top:36px}}p{{margin:10px 0}}.lede{{max-width:900px;color:var(--muted)}}.notice{{border-left:4px solid var(--gold);background:#fff9eb;padding:14px 18px;margin:22px 0}}
.case{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:22px;margin:26px 0;box-shadow:0 3px 12px #15233a0a}}
.case-head{{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:16px}}.eyebrow{{text-transform:uppercase;letter-spacing:.1em;font-size:12px;font-weight:700;color:var(--muted)}}.mode-tag{{padding:6px 12px;background:#edf3ff;border-radius:999px;color:var(--blue);font-weight:700}}
.media-grid{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.1fr);gap:18px;align-items:start}}figure{{margin:0}}video{{display:block;width:100%;height:auto;background:#111;border-radius:8px}}img{{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:6px}}figcaption{{font-size:13px;color:var(--muted);margin-top:6px}}table{{border-collapse:collapse;width:100%;margin-top:16px}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{width:240px;color:#465064;font-weight:650}}.caveat{{color:#475467;font-size:14px;background:#f7f8fa;padding:10px 12px;border-radius:6px}}details{{font-size:13px;color:var(--muted);margin-top:10px}}summary{{cursor:pointer;font-weight:650}}code{{overflow-wrap:anywhere}}.pending{{background:#fff;border:1px dashed var(--line);padding:14px;border-radius:8px}}.chart-wrap{{padding:14px;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:auto}}svg{{min-width:700px;width:100%;max-width:900px;height:auto}}.chart-title{{font-size:14px;font-weight:700;fill:var(--ink)}}.axis,.case-label,.value{{font-size:12px;fill:#485467}}.grid{{stroke:#e5e8ee;stroke-width:1}}.bar{{fill:var(--blue)}}
@media(max-width:820px){{.media-grid{{grid-template-columns:1fr}}th{{width:150px}}.case{{padding:15px}}header,main{{padding-left:14px;padding-right:14px}}}}
</style></head><body>
<header><p class="eyebrow">Local, offline gallery</p><h1>Five preselected inputs, one command each</h1><p class="lede">This gallery presents the selected cross-image examples individually. It is a small development set, not a complete direction-by-scene benchmark. Videos are original native RGB MP4s. The video elements use browser-native controls, do not autoplay, and have no tracking overlay.</p>
<div class="notice"><strong>Read metrics with the video.</strong> Sparse 2D tracks measure apparent image-plane motion, not physical 6-DoF camera pose. Preservation scores use aligned overlap and do not certify newly revealed regions. Contact sheets show the conditioning image and output frames 40, 80, and 120.</div></header>
<main><section><h2>Per-case motion gain</h2><p class="lede">A signed gain of 1.0 matches requested displacement magnitude along the target path. The cosine is shown per case below to indicate direction alignment. No pooled score is reported.</p><div class="chart-wrap">{_svg_gain_chart(selected)}</div>{pending_html}</section>
<section><h2>Preselected examples</h2>{cards or '<p class="pending">No selected case has all required local files yet.</p>'}</section>
<section><h2>Existing regression references</h2><p class="lede">These six step-300 regression controls comprise four blue-ball pan/tilt directions and two teacher-reference food-scene zoom directions. They are separate from the five selected inputs and should not be counted as additional unseen scenes.</p>{regression_cards or '<p class="pending">Regression outputs are not yet staged.</p>'}</section>
<footer><p class="lede">All content is served locally. Metric source JSON, request metadata, and hashes are available in the case folders and gallery_data.json.</p></footer></main></body></html>"""
    (gallery / "index.html").write_text(document, encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = build(args.submission_root.resolve())
    print(
        f"Built gallery with {len(result['unseen_examples'])} selected examples and "
        f"{len(result['regression_controls'])} regression references; "
        f"{len(result['pending_unseen_examples'])} selected cases are still pending."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
