"""CPU tests for the strength calibration map, the isotonic fit/inversion and the gain pipeline."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fpcamera import strength as st  # noqa: E402


def _cal(**directions):
    return {
        "version": 1,
        "created": "t",
        "checkpoint_sha256": None,
        "max_amplitudes": dict(st.DEFAULT_MAX_AMPLITUDES),
        "directions": directions,
        "fit": {},
    }


# ----------------------------------------------------------------- map
def test_identity_map_and_file():
    m = st.StrengthMap(None)
    assert m.is_identity
    for d in st.DIRECTIONS:
        for s in (0.0, 0.1, 0.5, 1.0):
            assert m.s_to_u(d, s) == pytest.approx(s)
    on_disk = st.load_calibration(ROOT / "configs" / "strength_calibration_v2.identity.json")
    assert on_disk.is_identity and on_disk.sha256 and len(on_disk.sha256) == 64
    assert set(on_disk.record["directions"]) == set(st.DIRECTIONS)


def test_clamp_static_and_zero():
    m = st.StrengthMap(_cal(pan_left={"s": [0, 0.5, 1], "u": [0, 0.7, 1]}))
    assert m.s_to_u("pan_left", 0.0) == 0.0
    assert m.s_to_u("pan_left", -3) == 0.0
    assert m.s_to_u("pan_left", 7) == 1.0
    assert m.s_to_u("static", 0.8) == 0.0
    assert m.s_to_u("pan_left", 0.25) == pytest.approx(0.35)
    with pytest.raises(ValueError):
        m.s_to_u("pan_left", float("nan"))
    with pytest.raises(ValueError):
        m.s_to_u("dolly", 0.5)


def test_map_is_monotone_piecewise_linear():
    cal = _cal(zoom_out={"s": [0, 0.25, 0.5, 0.9, 1.0], "u": [0, 0.3, 0.3, 0.9, 1.0]})
    m = st.StrengthMap(cal)
    xs = np.linspace(0, 1, 201)
    us = np.array([m.s_to_u("zoom_out", x) for x in xs])
    assert np.all(np.diff(us) >= -1e-12)
    assert us.min() >= 0 and us.max() <= 1


def test_origin_anchored_and_missing_directions_fall_back_to_identity():
    cal = _cal(tilt_up={"s": [0.25, 1.0], "u": [0.0, 1.0]})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = st.StrengthMap(cal)
    assert any("no entry" in str(x.message) for x in w)
    s, u = m.knots("tilt_up")
    assert s[0] == 0 and u[0] == 0  # origin added
    assert m.s_to_u("pan_right", 0.4) == pytest.approx(0.4)
    assert "pan_right" in m.identity_directions


@pytest.mark.parametrize(
    "entry",
    [
        {"s": [0, 0.5, 1], "u": [0, 0.8, 0.7]},  # u decreasing
        {"s": [0, 0.5, 0.5, 1], "u": [0, 0.2, 0.3, 1]},  # s not strictly increasing
        {"s": [0, 1.5], "u": [0, 1]},  # s out of range
        {"s": [0, 1], "u": [0.1, 1]},  # s=0 must be static
        {"s": [0], "u": [0]},  # too few points
        {"s": [0, 1], "u": [0, 1, 1]},  # shape mismatch
        {"s": [0, float("nan")], "u": [0, 1]},
    ],
)
def test_invalid_calibrations_rejected(entry):
    with pytest.raises(ValueError):
        st.StrengthMap(_cal(pan_left=entry))


def test_version_and_unknown_direction_rejected():
    bad = _cal()
    bad["version"] = 2
    with pytest.raises(ValueError):
        st.StrengthMap(bad)
    with pytest.raises(ValueError):
        st.StrengthMap(_cal(dolly_in={"s": [0, 1], "u": [0, 1]}))


def test_load_calibration_forms(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_cal(pan_left={"s": [0, 1], "u": [0, 0.9]})))
    from_path = st.load_calibration(p)
    assert from_path.s_to_u("pan_left", 1.0) == pytest.approx(0.9) and from_path.source == str(p)
    assert st.load_calibration(from_path) is from_path
    assert st.load_calibration(json.loads(p.read_text())).sha256 is None
    assert st.load_calibration(None).is_identity


# ----------------------------------------------------------------- amplitudes / labels
def test_max_amplitudes_from_data_config():
    assert st.load_max_amplitudes(ROOT) == {"yaw_deg": 8.0, "pitch_deg": 8.0, "zoom_ratio": 1.25}


def test_amplitude_labels():
    assert st.amplitude_label("pan_right", 0.7) == "Pan right · 5.6°"
    assert st.amplitude_label("tilt_down", 1.0) == "Tilt down · 8.0°"
    assert st.amplitude_label("zoom_in", 1.0) == "Zoom in · ×1.25"
    assert st.amplitude_label("zoom_out", 1.0) == "Zoom out · ×0.80"
    assert st.amplitude_label("zoom_out", 0.0) == "Zoom out · ×1.00"
    assert st.amplitude_label("static", 0.3) == "Static"
    amp = st.physical_amplitude("pan_left", 0.5)
    assert amp["yaw_deg"] == pytest.approx(-4.0) and amp["value"] == pytest.approx(4.0)


# ----------------------------------------------------------------- isotonic
def test_pav_known_examples():
    np.testing.assert_allclose(st.isotonic_increasing([1, 3, 2, 4]), [1, 2.5, 2.5, 4])
    np.testing.assert_allclose(st.isotonic_increasing([4, 3, 2, 1]), [2.5] * 4)
    np.testing.assert_allclose(st.isotonic_increasing([1, 2, 3]), [1, 2, 3])
    # weights pull the pooled mean
    np.testing.assert_allclose(st.isotonic_increasing([3, 1], [1, 3]), [1.5, 1.5])


def test_pav_properties_random():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.normal(size=rng.integers(2, 40))
        w = rng.uniform(0.2, 3, size=len(y))
        f = st.isotonic_increasing(y, w)
        assert np.all(np.diff(f) >= -1e-12)
        assert np.average(f, weights=w) == pytest.approx(np.average(y, weights=w))
        # optimality vs. a few monotone competitors
        base = np.sum(w * (y - f) ** 2)
        for _ in range(20):
            c = np.sort(rng.normal(size=len(y)))
            assert base <= np.sum(w * (y - c) ** 2) + 1e-9


def _sweep(gain, noise, seed=0, us=(0.25, 0.5, 0.75, 1.0), n=3):
    rng = np.random.default_rng(seed)
    u = np.repeat(us, n)
    a = np.array([gain(x) * x for x in u]) + rng.normal(0, noise, size=len(u))
    return u, a


def test_fit_and_inversion_recovers_scale():
    u, a = _sweep(lambda x: 0.8, 0.01)  # adapter delivers 80 % of the request
    entry, diag = st.build_direction_entry(u, a)
    s, uo = np.array(entry["s"]), np.array(entry["u"])
    assert np.all(np.diff(uo) >= -1e-12) and uo[0] == 0
    assert diag["saturates_at"] == pytest.approx(0.8, abs=0.03)
    # reachable targets are inverted accurately: a_obs(u(s)) = s
    sm = st.StrengthMap(_cal(pan_left=entry))
    for target in (0.25, 0.5, 0.75):
        assert sm.s_to_u("pan_left", target) == pytest.approx(target / 0.8, abs=0.04)
    # unreachable targets cap at the largest tested u
    assert sm.s_to_u("pan_left", 1.0) == pytest.approx(1.0)


def test_inversion_accuracy_nonlinear_and_overshoot():
    for gain, sat in ((lambda x: x**0.5, False), (lambda x: 1.25, False)):  # a = u^1.5, a = 1.25u
        u, a = _sweep(gain, 0.005, seed=3, n=4)
        entry, diag = st.build_direction_entry(u, a)
        assert diag["saturates_at"] is None
        uk, ak = np.array(diag["u_knots"]), np.array(diag["a_obs_isotonic"])
        sm = st.StrengthMap(_cal(zoom_in=entry))
        for s in np.linspace(0.1, 1.0, 10):
            u_req = sm.s_to_u("zoom_in", s)
            realised = np.interp(u_req, uk, ak)
            assert realised == pytest.approx(s, abs=0.04)


def test_fit_handles_noisy_nonmonotone_data():
    # a dip at u=0.75 (tracker noise) must be pooled, never produce a decreasing map
    u = np.repeat([0.25, 0.5, 0.75, 1.0], 2)
    a = np.array([0.2, 0.22, 0.55, 0.5, 0.45, 0.43, 0.95, 0.9])
    entry, _ = st.build_direction_entry(u, a)
    assert np.all(np.diff(entry["u"]) >= -1e-12)
    st.StrengthMap(_cal(tilt_up=entry))  # validates


def test_fit_rejects_empty():
    with pytest.raises(ValueError):
        st.fit_amplitude_curve([0.0, float("nan")], [0.0, 1.0])


def test_calibrate_script_fit_and_verify_report():
    spec = importlib.util.spec_from_file_location("calibrate_strength", ROOT / "scripts" / "calibrate_strength.py")
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    rows = []
    for d, gain in zip(st.DIRECTIONS, (0.9, 0.9, 0.7, 0.7, 1.05, 1.05)):
        for u in (0.25, 0.5, 0.75, 1.0):
            for img in "ab":
                rows.append({"image": img, "direction": d, "level": u, "u": u, "s": u, "gain": gain, "a_obs": gain * u})
    cal, diag = cs.fit_calibration(rows, checkpoint_sha256="x" * 64)
    assert set(cal["directions"]) == set(st.DIRECTIONS)
    assert diag["tilt_up"]["saturates_at"] == pytest.approx(0.7)
    assert diag["zoom_in"]["saturates_at"] is None
    sm = st.StrengthMap(cal)
    assert sm.s_to_u("pan_left", 0.45) == pytest.approx(0.5, abs=0.02)
    vr = cs.verify_report([{"direction": "pan_left", "level": 0.5, "s": 0.5, "a_obs": 0.45}])
    assert vr["overall_mean_abs_error"] == pytest.approx(0.05)
    md = cs.markdown_report(cal, diag, rows, vr)
    assert "tilt_up" in md and "Verification" in md


# ----------------------------------------------------------------- tracker gain on a known synthetic video
def test_measure_run_gain_on_synthetic_video(tmp_path):
    """The gain pipeline must return ~the factor by which a rendered video under-delivers."""
    cv2 = pytest.importorskip("cv2")
    from fpcamera.flow_geometry import VideoContract, camera_path_v2, render_source

    contract = VideoContract(width=320, height=192, frames=25, fps=24)
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, size=(120, 200, 3), dtype=np.uint8)
    tex = cv2.GaussianBlur(cv2.resize(base, (1500, 900), interpolation=cv2.INTER_CUBIC), (0, 0), 2.0)
    from PIL import Image

    delivered, requested = 0.6, 0.9
    h_req, request = camera_path_v2("pan", "right", requested, contract)
    h_del, _ = camera_path_v2("pan", "right", requested * delivered, contract)
    frames, _ = render_source(Image.fromarray(tex), h_del, contract, overscan=1.7, border_mode=cv2.BORDER_REFLECT_101)
    run = tmp_path / "run"
    run.mkdir()
    writer = cv2.VideoWriter(str(run / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24, (320, 192))
    if not writer.isOpened():
        pytest.skip("no mp4 encoder in this OpenCV build")
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    request["strength_control"] = {"motion": "pan_right", "strength_s": requested, "strength_u": requested}
    (run / "request.json").write_text(json.dumps(request))
    spec = importlib.util.spec_from_file_location("calibrate_strength", ROOT / "scripts" / "calibrate_strength.py")
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    result = cs.measure_run_gain(run)
    assert result["gain"] == pytest.approx(delivered, abs=0.06)
    assert result["strength_u"] == pytest.approx(requested)
    assert (run / "measure.json").is_file()
    assert cs.measure_run_gain(run)["gain"] == result["gain"]  # cached
