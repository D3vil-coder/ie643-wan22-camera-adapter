import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest

from fpcamera import border_metrics as bm
from fpcamera.flow_geometry import VideoContract, camera_path

W, H = 160, 96


def zoom_out(final=0.6, frames=121, width=W, height=H):
    cx, cy = (width - 1) / 2, (height - 1) / 2
    hs = []
    for s in np.linspace(1.0, final, frames):
        hs.append([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)], [0, 0, 1]])
    hs = np.asarray(hs)
    hs[0] = np.eye(3)
    return hs


def test_pixel_mask_zoom_out_geometry():
    hs = zoom_out(0.5)
    m = bm.revealed_mask_pixels(hs, W, H, frame_indices=[0, 60, 120])
    assert m.shape == (3, H, W) and m.dtype == bool
    assert not m[0].any()  # identity: nothing revealed
    assert not m[2][H // 2, W // 2] and m[2][0, 0] and m[2][H - 1, W - 1]
    # scale 0.5: content occupies the central ~quarter; revealed fraction ~ 0.75
    assert 0.70 < m[2].mean() < 0.80
    assert m[1].mean() < m[2].mean()  # monotone growth


def test_zoom_in_reveals_nothing():
    cx, cy = (W - 1) / 2, (H - 1) / 2
    s = 1.3
    zoom = np.array([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)], [0, 0, 1]])
    hs = np.stack([np.eye(3)] + [zoom] * 4)
    assert not bm.revealed_mask_pixels(hs, W, H).any()


def test_pan_direction_convention():
    # H maps first->current: content shifts right by 20 px => left strip revealed.
    hs = np.stack([np.eye(3), np.array([[1, 0, 20.0], [0, 1, 0], [0, 0, 1]])])
    m = bm.revealed_mask_pixels(hs, W, H)[1]
    assert m[:, :20].all() and not m[:, 20:].any()


def test_latent_mask_matches_flow_control_validity():
    torch = pytest.importorskip("torch")
    from fpcamera.flow_control import build_image_condition

    contract = VideoContract()
    for mode in ("zoom_out", "pan_right", "tilt_up", "zoom_in"):
        hs, _ = camera_path(mode, contract)
        mask = bm.revealed_mask_latent(hs, contract.width, contract.height)
        assert mask.shape == (31, 44, 80)
        lat = torch.randn(1, 48, 1, 44, 80)
        cond = build_image_condition(
            lat, hs, width=contract.width, height=contract.height, frames=contract.frames
        )
        valid = cond[0, 48].numpy() > 0.5
        assert ((~mask) == valid).mean() > 0.999, mode
    hs, _ = camera_path("zoom_out", contract)
    assert bm.revealed_mask_latent(hs, 1280, 704)[-1].mean() > 0.1


def make_scene(rng, gray_border):
    rgb = rng.integers(0, 256, (H, W, 3)).astype(np.uint8)  # colourful noise
    mask = np.zeros((H, W), bool)
    mask[:, :30] = True
    mask[:10] = True
    if gray_border:
        rgb[mask] = 128
    else:
        rgb[mask] = rng.integers(0, 256, (int(mask.sum()), 3))
    return rgb, mask


def test_metrics_gray_vs_textured_border():
    gray, mask = make_scene(np.random.default_rng(0), True)
    col, _ = make_scene(np.random.default_rng(1), False)
    mg, mc = bm.frame_metrics(gray, mask), bm.frame_metrics(col, mask)
    assert mg["gray_band_R"] == 1.0 and mg["gray_band_score"] > 0.9
    assert mc["gray_band_score"] < 0.1
    assert mg["colorfulness_ratio"] < 0.05
    assert mc["colorfulness_ratio"] > 0.5
    assert mg["hf_ratio"] < 1e-6 and mc["hf_ratio"] > 0.5
    assert 0 < mg["revealed_fraction"] < 1
    assert math.isfinite(mg["seam_score"])


def test_seam_score_detects_hard_edge():
    rng = np.random.default_rng(2)
    img = np.full((H, W, 3), 100, np.int64)
    img[:, :40] = 200  # revealed strip much brighter than valid -> strong seam
    img = np.clip(img + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)
    mask = np.zeros((H, W), bool)
    mask[:, :40] = True
    smooth = np.tile(np.linspace(100, 200, W)[None, :, None], (H, 1, 3)).astype(np.uint8)
    assert bm.seam_score(img, mask) > 5 * bm.seam_score(smooth, mask)


def test_empty_revealed_region_is_nan():
    img = np.zeros((H, W, 3), np.uint8)
    m = bm.frame_metrics(img, np.zeros((H, W), bool))
    assert m["revealed_fraction"] == 0.0
    assert all(math.isnan(v) for k, v in m.items() if k != "revealed_fraction")
    assert bm.mean_metrics([m])["n_frames"] == 0
    assert math.isnan(bm.seam_score(img, np.zeros((H, W), bool)))


def test_mean_metrics_filters_small_fractions():
    base = {k: 0.0 for k in bm.METRIC_KEYS}
    a = {**base, "revealed_fraction": 0.005, "hf_R": 100.0}
    b = {**base, "revealed_fraction": 0.2, "hf_R": 2.0}
    c = {**base, "revealed_fraction": 0.4, "hf_R": 4.0}
    out = bm.mean_metrics([a, b, c])
    assert out["n_frames"] == 2 and out["hf_R"] == pytest.approx(3.0)


def test_latent_stats():
    rng = np.random.default_rng(0)
    z = rng.normal(0.5, 1.0, (48, 5, 6, 8))
    mask = np.zeros((5, 6, 8), bool)
    mask[:, :, :3] = True
    z[:, :, :, :3] = 0.0  # R collapsed to zero (H1 signature)
    s = bm.latent_stats(z, mask)
    assert s["R"]["std"] == 0 and s["R"]["mean_abs"] == 0
    assert s["valid"]["mean"] == pytest.approx(0.5, abs=0.1)
    assert s["valid"]["std"] == pytest.approx(1.0, abs=0.1)
    assert s["R"]["count"] == 4 * 6 * 3
    assert s["std_ratio_R_over_valid"] == 0
    empty = bm.latent_stats(z, np.zeros_like(mask))
    assert math.isnan(empty["R"]["mean"]) and empty["R"]["count"] == 0


def test_artifact_score_flags_gray_blur_not_texture():
    gray, mask = make_scene(np.random.default_rng(0), True)
    col, _ = make_scene(np.random.default_rng(1), False)
    mg, mc = bm.frame_metrics(gray, mask), bm.frame_metrics(col, mask)
    assert mg["artifact_score"] > 1.8  # desaturated and blurred
    assert mc["artifact_score"] < 0.3
    assert math.isfinite(mg["glow_term"])


def test_artifact_score_formula_and_clipping():
    assert bm.artifact_score(0.668, 0.386, 0.0) == pytest.approx(0.332 + 0.614)
    assert bm.artifact_score(1.233, 3.0, 0.0) == 0.0  # clipped above 1.0 gives no penalty
    assert bm.artifact_score(0.5, 0.5, 0.1) == pytest.approx(1.1)
    assert math.isnan(bm.artifact_score(float("nan"), 0.5, 0.0))


def test_glow_term_detects_bright_border():
    img = np.full((H, W, 3), 100, np.uint8)
    mask = np.zeros((H, W), bool)
    mask[:, :30] = True
    dark = bm.glow_term(np.full((H, W), 100.0), mask)
    img[mask] = 200
    glow = bm.glow_term(bm._gray(img), mask)
    assert dark == 0.0 and glow == pytest.approx(100 / 255, abs=1e-6)


def test_artifact_score_v2_symmetric():
    f = bm.artifact_score_v2
    assert f(1.0, 1.0, 0.0) == 0.0
    assert f(1.5, 1.0, 0.0) == pytest.approx(math.log(1.5))  # over-saturation penalized
    assert f(2.0, 0.5, 0.0) == pytest.approx(2 * math.log(2))
    assert f(100, 0.001, 0.1) == pytest.approx(2 * math.log(5) + 0.1)  # clipped
    assert math.isnan(f(float("nan"), 1.0, 0.0))
    assert bm.artifact_score_v1(1.5, 1.0, 0.0) == 0.0
