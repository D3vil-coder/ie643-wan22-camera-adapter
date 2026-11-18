"""Local-ring border metrics and ground-truth reveal evaluation (CPU, synthetic)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

cv2 = pytest.importorskip("cv2")
from PIL import Image  # noqa: E402

from fpcamera import border_metrics as bm  # noqa: E402
from fpcamera import reveal_eval as rv  # noqa: E402
from fpcamera.flow_cache_v2 import ClipPlan, camera_kwargs, contract_of, load_config, render_clip, source_overscan  # noqa: E402
from fpcamera.flow_geometry import camera_path_v2, render_source  # noqa: E402

W, H = 192, 128


def right_mask(frac=0.4):
    m = np.zeros((H, W), bool)
    m[:, int(W * (1 - frac)):] = True
    return m


def stripes(period=6, color=True):
    x = np.arange(W)[None, :].repeat(H, 0)
    on = (x // (period // 2)) % 2 == 0
    img = np.zeros((H, W, 3), np.uint8)
    img[on] = (200, 60, 60) if color else (200, 200, 200)
    img[~on] = (60, 60, 200) if color else (50, 50, 50)
    return img


# ------------------------------------------------------------ local ring
def test_local_ring_geometry():
    r = right_mask()
    ring = bm.local_ring_mask(r)
    assert ring.any() and not (ring & r).any()
    cols = np.where(ring.any(0))[0]
    edge = int(W * 0.6)
    assert edge - cols.max() >= 7 and edge - cols.min() <= 52


def test_stripe_continuation_scores_near_zero():
    r = right_mask()
    m = bm.frame_metrics(stripes(), r)
    assert m["artifact_score_local"] < 0.15
    assert abs(np.log(m["colorfulness_ratio_local"])) < 0.15


def test_gray_fill_scores_high():
    img = stripes()
    r = right_mask()
    img[r] = 128
    m = bm.frame_metrics(img, r)
    assert m["artifact_score_local"] > 2.5


def test_oversaturated_streak_scores_high():
    img = stripes(color=False)
    r = right_mask()
    img[r] = (255, 0, 0)
    img[r & (np.arange(H)[:, None] % 2 == 0)] = (0, 255, 0)
    m = bm.frame_metrics(img, r)
    assert m["artifact_score_local"] > 1.0


def test_local_reference_ignores_far_content():
    """Busy colourful texture far from R, plain wall near R continuing into R: local < global."""
    rng = np.random.default_rng(0)
    img = np.full((H, W, 3), 200, np.uint8)
    img[:, : W // 3] = rng.integers(0, 255, (H, W // 3, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img[:, W // 3 :] = (200, 200, 190)
    r = right_mask()
    m = bm.frame_metrics(img, r)
    assert m["artifact_score_local"] < m["artifact_score_v2"]


def test_existing_keys_kept():
    m = bm.frame_metrics(stripes(), right_mask())
    for k in ("artifact_score", "artifact_score_v1", "artifact_score_v2", "glow_term", "seam_score"):
        assert k in m
    assert "artifact_score_local" in bm.METRIC_KEYS


# ------------------------------------------------------------ GT rendering
@pytest.fixture(scope="module")
def synth():
    cfg = json.loads(json.dumps(load_config()))
    cfg["video_contract"] = {"width": 160, "height": 96, "frames": 121, "fps": 24}
    contract = contract_of(cfg)
    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8), (0, 0), 3)
    return cfg, contract, Image.fromarray(base)


def test_gt_render_matches_cache_builder(synth):
    cfg, contract, img = synth
    plan = ClipPlan("s__m0_pan_right_u0800", "s", "development", "moving", 0, "pan", "right", 0.8)
    rgb, h, overscan = rv.render_gt(img, plan, cfg, contract)
    ref_rgb, ref_h, _rec, _crop = render_clip(img, plan, cfg, contract, source_overscan(img.size, contract, cfg))
    assert np.array_equal(rgb, ref_rgb) and np.allclose(h, ref_h, atol=1e-12)
    # independent path: camera_path_v2 + render_source with the config parameters
    h2, _ = camera_path_v2("pan", "right", 0.8, contract, **camera_kwargs(cfg))
    rgb2, _ = render_source(img, h2, contract, overscan=overscan, border_mode=cv2.BORDER_REFLECT_101)
    assert np.array_equal(rgb, rgb2)
    req = rv.camera_request_for(h, contract)
    assert np.allclose(np.asarray(req["homographies_first_to_current"]), h, atol=1e-12)
    assert req["contract"]["frames"] == 121


def test_revealed_mask_consistent_with_border_metrics(synth):
    cfg, contract, img = synth
    plan = ClipPlan("s__m0_zoom_out", "s", "development", "moving", 0, "zoom", "out", 0.9)
    rgb, h, _ = rv.render_gt(img, plan, cfg, contract)
    res = rv.evaluate_video(rgb, rgb, h, [100])
    mask = bm.revealed_mask_pixels(h, contract.width, contract.height, frame_indices=[100])[0]
    assert res["frames"][100]["R_fraction"] == pytest.approx(mask.mean())
    assert mask.mean() > 0.1


def test_metric_sanity_identical_and_corrupted(synth):
    cfg, contract, img = synth
    plan = ClipPlan("s__m0_zoom_out", "s", "development", "moving", 0, "zoom", "out", 0.9)
    rgb, h, _ = rv.render_gt(img, plan, cfg, contract)

    def fake_lpips(g, t):
        return np.abs(g.astype(float) - t.astype(float)).mean(-1) / 255.0

    same = rv.evaluate_video(rgb, rgb, h, [60, 120], fake_lpips)["mean"]
    assert same["R"]["lpips"] == pytest.approx(0, abs=1e-9)
    assert same["R"]["hist_bhatt"] < 1e-6
    assert same["R"]["hf_dev"] < 1e-9
    assert same["R"]["ssim"] == pytest.approx(1.0, abs=1e-9)
    assert same["R"]["psnr"] == 99.0
    bad = rgb.copy()
    mask = bm.revealed_mask_pixels(h, contract.width, contract.height, frame_indices=[120])[0]
    bad[120][mask] = 128
    worse = rv.evaluate_video({120: bad[120]}, rgb, h, [120], fake_lpips)["mean"]
    assert worse["R"]["lpips"] > 0.01 and worse["R"]["hist_bhatt"] > 0.3 and worse["R"]["psnr"] < 40 and worse["R"]["psnr"] < same["R"]["psnr"]
    assert worse["V"]["lpips"] == pytest.approx(0, abs=1e-9)  # control untouched


# ------------------------------------------------------------ selection / aggregate
def fake_records(n=28):
    return [{"source_id": f"d{i:02d}", "split": "development"} for i in range(n)] + \
           [{"source_id": f"t{i:03d}", "split": "train"} for i in range(60)]


def test_selection_balanced_deterministic():
    cfg = load_config()
    a = rv.select_pairs(fake_records(), cfg)
    b = rv.select_pairs(fake_records(), cfg)
    assert [p.id for p in a] == [p.id for p in b]
    assert len(a) == 12
    fams = [p.family for p in a]
    assert fams.count("pan") == fams.count("tilt") == fams.count("zoom") == 4
    assert all(p.split == "development" for p in a)
    assert all(not (p.family == "zoom" and p.direction == "in") for p in a)
    assert sum(p.u >= 0.5 for p in a) >= 9


def test_aggregate_and_markdown():
    mean = {"n_frames": 3, "R": {"lpips": 0.5, "psnr": 12.0, "ssim": 0.3, "hist_bhatt": 0.2, "hf_dev": 0.4},
            "V": {"lpips": 0.1, "psnr": 30.0, "ssim": 0.9, "hist_bhatt": 0.05, "hf_dev": 0.1},
            "artifact_score_local": 0.6, "artifact_score_local_gt": 0.2}
    recs = [dict(model="v2", pair="a", family="pan", direction="left", u=0.6, mean=mean),
            dict(model="v2", pair="b", family="zoom", direction="out", u=0.9, mean=mean)]
    agg = rv.aggregate(recs)
    assert agg["v2"]["all"]["n_pairs"] == 2 and agg["v2"]["all"]["R_lpips"] == pytest.approx(0.5)
    assert set(agg["v2"]["by_family"]) == {"pan", "zoom"}
    assert "| all | v2 | 2 |" in rv.markdown(agg)


def test_contact_sheet(tmp_path, synth):
    cfg, contract, img = synth
    plan = ClipPlan("s__m0_zoom_out", "s", "development", "moving", 0, "zoom", "out", 0.9)
    rgb, h, _ = rv.render_gt(img, plan, cfg, contract)
    rows = [("GT", {t: rgb[t] for t in (40, 80, 120)}), ("m", {t: rgb[t] for t in (40, 80, 120)})]
    p = rv.contact_sheet(rows, h, (40, 80, 120), tmp_path / "s.png", tile_w=160)
    assert Path(p).is_file()


def test_parse_model():
    spec = importlib.util.spec_from_file_location("eval_dev_reveal", ROOT / "scripts" / "eval_dev_reveal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.parse_model(r"v1+rep=C:\x\ckpt.safetensors:replicate:v1")
    assert (m["name"], m["ckpt"], m["fill"], m["recipe"]) == ("v1+rep", r"C:\x\ckpt.safetensors", "replicate", "v1")
    m = mod.parse_model("v2=a/b.safetensors")
    assert m["fill"] is None and m["recipe"] == "default" and m["ckpt"] == "a/b.safetensors"
