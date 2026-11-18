"""CPU tests for v2 argument plumbing in ``fpcamera.flow_infer`` (no model, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpcamera import flow_infer as fi  # noqa: E402
from fpcamera.flow_geometry import VideoContract, camera_path, camera_path_v2  # noqa: E402
from fpcamera.runtime import file_sha256  # noqa: E402

CONTRACT = VideoContract()


# ------------------------------------------------------------------ build_camera_request
def test_v1_default_unchanged():
    path, request, mode, magnitude, record, notes = fi.build_camera_request(CONTRACT)
    expected, _ = camera_path("zoom_in", CONTRACT, magnitude=1.0)
    np.testing.assert_array_equal(path, expected)
    assert (mode, magnitude, record, notes) == ("zoom_in", 1.0, None, [])
    assert "strength_control" not in request
    path2, _, mode2, mag2, _, _ = fi.build_camera_request(CONTRACT, mode="pan_left", magnitude=0.6)
    np.testing.assert_array_equal(path2, camera_path("pan_left", CONTRACT, magnitude=0.6)[0])
    assert (mode2, mag2) == ("pan_left", 0.6)


@pytest.mark.parametrize("motion", ["pan_left", "pan_right", "tilt_up", "tilt_down", "zoom_in", "zoom_out"])
def test_motion_strength_matches_camera_path_v2(motion):
    fam, direction = motion.split("_")
    path, request, mode, u, record, _ = fi.build_camera_request(CONTRACT, motion=motion, strength=0.6, root=ROOT)
    expected, _ = camera_path_v2(fam, direction, 0.6, CONTRACT, max_yaw_deg=8.0, max_pitch_deg=8.0, max_zoom_ratio=1.25)
    np.testing.assert_allclose(path, expected, atol=1e-12)
    assert mode == motion and u == pytest.approx(0.6)  # identity calibration
    assert record["strength_s"] == 0.6 and record["strength_u"] == pytest.approx(0.6)
    assert record["calibration"]["identity"] is True
    assert request["strength_control"] == record
    assert request["homographies_first_to_current"]  # downstream tools read this


def test_calibrated_strength_changes_requested_u():
    cal = {
        "version": 1,
        "created": "t",
        "checkpoint_sha256": None,
        "max_amplitudes": {"yaw_deg": 8.0, "pitch_deg": 8.0, "zoom_ratio": 1.25},
        "directions": {"pan_right": {"s": [0, 0.5, 1], "u": [0, 0.7, 1]}},
    }
    with pytest.warns(UserWarning):  # other directions fall back to identity
        path, _, _, u, record, _ = fi.build_camera_request(
            CONTRACT, motion="pan_right", strength=0.5, calibration=cal, root=ROOT
        )
    assert u == pytest.approx(0.7)
    expected, _ = camera_path_v2("pan", "right", 0.7, CONTRACT)
    np.testing.assert_allclose(path, expected, atol=1e-12)
    assert record["amplitude"]["yaw_deg"] == pytest.approx(0.7 * 8.0)
    assert record["label"] == "Pan right · 5.6°"


def test_static_and_zero_strength_are_identity_paths():
    for kwargs in (dict(motion="static", strength=1.0), dict(motion="zoom_out", strength=0.0)):
        path, *_ = fi.build_camera_request(CONTRACT, root=ROOT, **kwargs)
        np.testing.assert_allclose(path, np.broadcast_to(np.eye(3), path.shape), atol=1e-9)


def test_argument_validation():
    with pytest.raises(ValueError):
        fi.build_camera_request(CONTRACT, motion="pan_left")  # strength missing
    with pytest.raises(ValueError):
        fi.build_camera_request(CONTRACT, strength=0.5)  # motion missing
    with pytest.raises(ValueError):
        fi.build_camera_request(CONTRACT, motion="pan_left", strength=1.2)
    with pytest.raises(ValueError):
        fi.build_camera_request(CONTRACT, motion="dolly", strength=0.5)
    with pytest.raises(ValueError):
        fi.build_camera_request(CONTRACT, motion="pan_left", strength=float("nan"))


def test_calibration_for_other_checkpoint_warns_in_notes():
    cal = json.loads((ROOT / "configs" / "strength_calibration_v2.identity.json").read_text())
    cal["checkpoint_sha256"] = "a" * 64
    *_, notes = fi.build_camera_request(
        CONTRACT, motion="pan_left", strength=0.5, calibration=cal, root=ROOT, checkpoint_sha256="b" * 64
    )
    assert any("different checkpoint" in n for n in notes)


# ------------------------------------------------------------------ fill mode / metadata
def test_fill_mode_resolution():
    assert fi.resolve_guide_fill_mode(None, {}) == ("zeros", "default", None)
    assert fi.resolve_guide_fill_mode(None, {"guide_fill_mode": "replicate"}) == ("replicate", "checkpoint", None)
    assert fi.resolve_guide_fill_mode("zeros", {"guide_fill_mode": "zeros"})[2] is None
    mode, source, note = fi.resolve_guide_fill_mode("zeros", {"guide_fill_mode": "replicate"})
    assert (mode, source) == ("zeros", "explicit") and "differs" in note
    assert fi.resolve_guide_fill_mode(None, {"identity": {"fill_mode": "replicate"}})[0] == "replicate"
    with pytest.raises(ValueError):
        fi.resolve_guide_fill_mode("mirror", {})


def _v1_checkpoint(tmp_path, **extra):
    ckpt = tmp_path / "adapter_000300.safetensors"
    ckpt.write_bytes(b"not really weights")
    meta = {
        "checkpoint_sha256": file_sha256(ckpt),
        "backbone_unchanged": True,
        "architecture": {"kind": "wan_flow_control_lora", "config": {}},
        "identity": {"video_contract": {"width": 1280, "height": 704, "frames": 121, "fps": 24}},
        "completed_updates": 300,
        "backbone_sha256_after": "z",
        **extra,
    }
    ckpt.with_suffix(".json").write_text(json.dumps(meta))
    return ckpt


def _v2_run(tmp_path, fill="replicate"):
    run = tmp_path / "v2_full"
    (run / "checkpoints").mkdir(parents=True)
    ckpt = run / "checkpoints" / "adapter_000150.safetensors"
    ckpt.write_bytes(b"v2 weights")
    manifest = {
        "step": 150,
        "adapter_file": ckpt.name,
        "adapter_sha256": file_sha256(ckpt),
        "architecture": {"kind": "wan_flow_control_lora", "config": {"hidden": 3072}},
    }
    (run / "checkpoints" / "ckpt_000150.json").write_text(json.dumps(manifest))
    config = {
        "config": {
            "fill_mode": fill,
            "video_contract": {"width": 1280, "height": 704, "frames": 121, "fps": 24},
            "expected_backbone_sha256": "abc",
            "objective": {"sigma_A": {"lower": 0.6}},
        }
    }
    (run / "config.json").write_text(json.dumps(config))
    return ckpt, run


def test_resolve_meta_v1_sidecar(tmp_path):
    ckpt = _v1_checkpoint(tmp_path)
    meta, source = fi.resolve_checkpoint_meta(ckpt)
    assert source == "sidecar" and fi.recorded_fill_mode(meta) is None
    assert fi.resolve_guide_fill_mode(None, meta)[0] == "zeros"  # v1 behaviour preserved


def test_resolve_meta_v2_manifest(tmp_path):
    ckpt, run = _v2_run(tmp_path)
    meta, source = fi.resolve_checkpoint_meta(ckpt)
    assert source == "v2_manifest"
    assert meta["checkpoint_sha256"] == file_sha256(ckpt)
    assert meta["architecture"]["kind"] == "wan_flow_control_lora"
    assert meta["completed_updates"] == 150 and meta["backbone_unchanged"] is True
    assert meta["backbone_audit"] == "start_hash_only"
    assert meta["inference_policy"]["camera_min_sigma"] == 0.6
    assert fi.resolve_guide_fill_mode(None, meta) == ("replicate", "checkpoint", None)
    (run / "run_status.json").write_text(json.dumps({"status": "complete", "backbone_unchanged": False}))
    assert fi.resolve_checkpoint_meta(ckpt)[0]["backbone_unchanged"] is False


def test_resolve_meta_missing(tmp_path):
    lone = tmp_path / "weights.safetensors"
    lone.write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        fi.resolve_checkpoint_meta(lone)


# ------------------------------------------------------------------ generate() up to request.json
class _Stop(Exception):
    pass


def _run_until_request(monkeypatch, tmp_path, ckpt, **kwargs):
    from PIL import Image

    seen = {}

    def fake_write_json(path, value):
        if Path(path).name == "request.json":
            seen["request"] = value
            raise _Stop
        seen.setdefault("other", []).append(Path(path).name)

    monkeypatch.setattr(fi, "write_json", fake_write_json)
    image = tmp_path / "in.png"
    Image.new("RGB", (64, 48), (120, 90, 60)).save(image)
    with pytest.raises(_Stop):
        fi.generate(ROOT, ckpt, image, tmp_path / "out", **kwargs)
    return seen["request"]


def test_generate_v1_default_request_is_unchanged(monkeypatch, tmp_path):
    ckpt = _v1_checkpoint(tmp_path)
    request = _run_until_request(monkeypatch, tmp_path, ckpt, mode="zoom_out", magnitude=0.5)
    expected, _ = camera_path("zoom_out", CONTRACT, magnitude=0.5)
    np.testing.assert_allclose(request["homographies_first_to_current"], expected)
    assert "strength_control" not in request


def test_generate_motion_strength_writes_v2_request(monkeypatch, tmp_path):
    ckpt, _ = _v2_run(tmp_path)
    request = _run_until_request(monkeypatch, tmp_path, ckpt, motion="tilt_up", strength=0.75)
    expected, _ = camera_path_v2("tilt", "up", 0.75, CONTRACT)
    np.testing.assert_allclose(request["homographies_first_to_current"], expected, atol=1e-12)
    ctl = request["strength_control"]
    assert ctl["strength_s"] == 0.75 and ctl["strength_u"] == pytest.approx(0.75)
    assert ctl["amplitude"]["pitch_deg"] == pytest.approx(6.0)


def test_generate_rejects_bad_v2_arguments(tmp_path):
    ckpt = _v1_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="together"):
        fi.generate(ROOT, ckpt, tmp_path / "x.png", tmp_path / "o1", motion="pan_left")
    with pytest.raises(ValueError, match="mutually exclusive"):
        fi.generate(ROOT, ckpt, tmp_path / "x.png", tmp_path / "o2", motion="pan_left", strength=0.5,
                    camera_request={"k": 1})
    with pytest.raises(ValueError, match="guide_fill_mode"):
        fi.generate(ROOT, ckpt, tmp_path / "x.png", tmp_path / "o3", guide_fill_mode="mirror")


def test_generate_warns_on_fill_mode_mismatch(monkeypatch, tmp_path):
    ckpt, _ = _v2_run(tmp_path, fill="replicate")
    with pytest.warns(UserWarning, match="differs from the checkpoint"):
        _run_until_request(monkeypatch, tmp_path, ckpt, mode="zoom_in", guide_fill_mode="zeros")


def test_generate_refuses_silent_zeros_when_run_config_missing(monkeypatch, tmp_path):
    """Only adapter+manifest restored (no <run>/config.json): the training fill mode is unknown -> hard error."""
    ckpt, run = _v2_run(tmp_path)
    (run / "config.json").unlink()
    meta, source = fi.resolve_checkpoint_meta(ckpt)
    assert source == "v2_manifest" and meta["run_config_found"] is False
    with pytest.raises(ValueError, match="config.json"):
        fi.generate(ROOT, ckpt, tmp_path / "x.png", tmp_path / "o_missing", mode="zoom_in")
    # an explicit fill mode is still accepted (reaches request.json), as is the intact run
    request = _run_until_request(monkeypatch, tmp_path, ckpt, mode="zoom_in", guide_fill_mode="replicate")
    assert request["homographies_first_to_current"]
    ckpt2, _ = _v2_run(tmp_path / "ok")
    assert fi.resolve_checkpoint_meta(ckpt2)[0]["run_config_found"] is True


# ------------------------------------------------------------------ CLI
def test_cli_flags_and_config(tmp_path):
    base = ["--project-root", ".", "--checkpoint", "c.safetensors", "--image", "i.png", "--output", "o"]
    args = fi.parse_args(base + ["--motion", "zoom_out", "--strength", "0.4", "--calibration", "cal.json"])
    assert (args.motion, args.strength, args.calibration) == ("zoom_out", 0.4, Path("cal.json"))
    assert args.guide_fill_mode is None and args.camera_residual_anchor == "auto"
    legacy = fi.parse_args(base)
    assert legacy.motion is None and legacy.mode == "zoom_in" and legacy.magnitude == 1.0
    with pytest.raises(SystemExit):
        fi.parse_args(base + ["--motion", "pan_left"])
    with pytest.raises(SystemExit):
        fi.parse_args(base + ["--strength", "0.5"])
    with pytest.raises(SystemExit):
        fi.parse_args(["--project-root", "."])  # checkpoint/image/output missing
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"_comment": "x", "motion": "pan_left", "strength": 0.3, "steps": 12, "prompt": None}))
    from_cfg = fi.parse_args(base + ["--config", str(cfg)])
    assert (from_cfg.motion, from_cfg.strength, from_cfg.steps) == ("pan_left", 0.3, 12)
    over = fi.parse_args(base + ["--config", str(cfg), "--strength", "0.9", "--steps", "5"])
    assert (over.strength, over.steps) == (0.9, 5)
    shipped = fi.parse_args(base + ["--config", str(ROOT / "configs" / "inference_v2.json")])
    assert shipped.motion in fi.MOTIONS_V2 and shipped.calibration is not None
