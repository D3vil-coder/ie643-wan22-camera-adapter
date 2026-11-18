"""CPU smoke test of the examples pipeline: metrics, contact sheets, summary, offline gallery."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

cv2 = pytest.importorskip("cv2")
from PIL import Image  # noqa: E402

from fpcamera.flow_geometry import VideoContract, camera_path_v2, render_source  # noqa: E402

spec = importlib.util.spec_from_file_location("generate_examples_v2", ROOT / "scripts" / "generate_examples_v2.py")
ge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ge)

CONTRACT = VideoContract(width=320, height=192, frames=25, fps=24)


def _fake_run(folder, motion="pan_right", u=0.8, delivered=0.7, with_control=True):
    fam, direction = motion.split("_")
    rng = np.random.default_rng(2)
    base = rng.integers(0, 255, size=(100, 160, 3), dtype=np.uint8)
    tex = cv2.GaussianBlur(cv2.resize(base, (1500, 900), interpolation=cv2.INTER_CUBIC), (0, 0), 2.0)
    h_req, request = camera_path_v2(fam, direction, u, CONTRACT)
    h_del, _ = camera_path_v2(fam, direction, u * delivered, CONTRACT)
    frames, _ = render_source(Image.fromarray(tex), h_del, CONTRACT, overscan=1.7, border_mode=cv2.BORDER_REFLECT_101)
    folder.mkdir(parents=True)
    writer = cv2.VideoWriter(str(folder / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24, (320, 192))
    if not writer.isOpened():
        pytest.skip("no mp4 encoder")
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    Image.fromarray(frames[0]).save(folder / "input.png")
    if with_control:
        request["strength_control"] = {"motion": motion, "strength_s": u, "strength_u": u, "label": "Pan right · test"}
    (folder / "request.json").write_text(json.dumps(request))
    (folder / "run.json").write_text(json.dumps({"mode": motion, "guide_fill_mode": "replicate", "contract": request["contract"]}))
    return folder


def test_case_motion_names():
    assert ge.case_motion("ball_pan_left") == "pan_left"
    assert ge.case_motion("living_room_zoom_in") == "zoom_in"
    assert ge.case_motion("motorcycle_pan_right") == "pan_right"
    assert {ge.case_motion(c) for c in ge.CASES} == set(__import__("fpcamera.strength", fromlist=["x"]).DIRECTIONS)
    with pytest.raises(ValueError):
        ge.case_motion("ball_static")


def test_metrics_sheet_summary_gallery(tmp_path):
    v1 = _fake_run(tmp_path / "v1", delivered=0.5, with_control=False)
    v2 = _fake_run(tmp_path / "v2", delivered=0.9)
    m1, m2 = ge.run_metrics(v1), ge.run_metrics(v2)
    assert m2["gain"] == pytest.approx(0.9, abs=0.08) and m1["gain"] == pytest.approx(0.5, abs=0.08)
    assert m2["a_obs"] == pytest.approx(0.9 * 0.8, abs=0.08)
    assert m2["n_revealed_frames"] > 0 and set(m2["border"]) == set(ge.BORDER_KEYS)
    assert (v2 / "border_quality.json").is_file() and (v2 / "measure.json").is_file()
    assert ge.run_metrics(v2)["gain"] == m2["gain"]  # cached path

    gallery = tmp_path / "gallery"
    sheet = ge.contact_sheet([("v1", v1 / "video.mp4"), ("v2 (s=1)", v2 / "video.mp4")], v2 / "input.png",
                             gallery / "contact_sheets" / "c.png", frames_at=(6, 12, 24))
    img = Image.open(sheet)
    assert img.width > 4 * 300 and img.height > 2 * 150
    entry = {"name": "ball_pan_right", "motion": "pan_right", "sheet_rel": "contact_sheets/c.png",
             "v1": {"run_dir": v1, "metrics": m1, "label": None}, "v2": {"run_dir": v2, "metrics": m2, "label": "x"}}
    sweep = {"name": "ball_pan_right", "motion": "pan_right", "sheet_rel": "contact_sheets/c.png",
             "entries": [{"s": 0.8, "run_dir": v2, "metrics": m2, "label": "Pan right · test"}]}
    meta = {"generated": "now", "steps": 5, "seed": 1}
    md = ge.summary_markdown([entry], [sweep], meta)
    assert "ball_pan_right" in md and "| v1 (Submission 1) |" in md and "| v2 (matched amplitude) |" in md
    assert "Matched before" in md and "Extended range" in md
    ge.write_template(gallery)
    data = ge.write_gallery(gallery, [entry], [sweep], meta)
    assert data["cases"][0]["v2"]["video"] == "media/ball_pan_right/v2/video.mp4"
    assert (gallery / data["cases"][0]["v1"]["video"]).is_file()
    assert (gallery / "index.html").read_text(encoding="utf8").startswith("<!doctype html>")
    loaded = json.loads((gallery / "gallery_data.json").read_text(encoding="utf8"))  # strict JSON
    assert loaded["schema"] == ge.SCHEMA and len(loaded["sweeps"][0]["entries"]) == 1
    assert (gallery / "gallery_data.js").read_text(encoding="utf8").startswith("window.GALLERY_DATA = ")


# ----------------------------------------------------------------- matched-amplitude (physical) mode
@pytest.mark.parametrize("motion,mag", [
    ("pan_left", 0.5), ("pan_right", 0.5), ("tilt_up", 0.35), ("tilt_down", 0.35),
    ("zoom_in", 1.0), ("zoom_out", 1.0),
])
def test_physical_amplitude_equals_v1_path(motion, mag):
    from fpcamera.flow_geometry import camera_path
    from fpcamera.flow_infer import build_camera_request

    kw = ge.v2_matched_kwargs(motion, mag)
    h2, req, _, u, rec, _ = build_camera_request(CONTRACT, root=ROOT, **kw)
    h1, _ = camera_path(motion, CONTRACT, magnitude=mag)
    assert np.abs(h1 - h2).max() < 1e-6
    assert rec["amplitude_source"] == "physical" and rec["strength_s"] is None
    assert req["strength_control"]["amplitude_source"] == "physical"
    assert 0 < u <= 1


def test_physical_to_u_values_and_errors():
    from fpcamera import strength as st

    assert st.physical_to_u("pan_left", amplitude_deg=2.5) == pytest.approx(2.5 / 8)
    assert st.physical_to_u("tilt_up", amplitude_deg=8.0) == pytest.approx(1.0)
    assert st.physical_to_u("zoom_in", zoom_ratio=1.12) == pytest.approx(np.log(1.12) / np.log(1.25))
    assert st.physical_to_u("zoom_out", zoom_ratio=1 / 1.12) == pytest.approx(np.log(1.12) / np.log(1.25))
    with pytest.raises(ValueError):
        st.physical_to_u("pan_left", amplitude_deg=9.0)
    with pytest.raises(ValueError):
        st.physical_to_u("zoom_in", amplitude_deg=2.0)


def test_build_camera_request_rejects_mixed_controls():
    from fpcamera.flow_infer import build_camera_request

    with pytest.raises(ValueError):
        build_camera_request(CONTRACT, root=ROOT, motion="pan_left", strength=0.5, amplitude_deg=2.5)
    with pytest.raises(ValueError):
        build_camera_request(CONTRACT, root=ROOT, amplitude_deg=2.5)


def test_v1_case_params_table_and_override(tmp_path):
    assert ge.v1_case_params("ball_pan_left") == ("pan_left", 0.5)
    assert ge.v1_case_params("flowers_tilt_up") == ("tilt_up", 0.35)
    assert ge.v1_case_params("kitchen_zoom_out") == ("zoom_out", 1.0)
    (tmp_path / "ball_pan_left").mkdir()
    (tmp_path / "ball_pan_left" / "run.json").write_text(json.dumps({"mode": "pan_left", "magnitude": 0.8}))
    assert ge.v1_case_params("ball_pan_left", tmp_path) == ("pan_left", 0.8)
    assert ge.v2_matched_kwargs("pan_left", 0.5) == {"motion": "pan_left", "amplitude_deg": 2.5}
    assert ge.v2_matched_kwargs("tilt_up", 0.35)["amplitude_deg"] == pytest.approx(1.75)
