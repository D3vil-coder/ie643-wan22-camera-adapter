"""CPU tests for the v2 data pipeline: geometry, coverage, sampling, cache I/O, sources."""

from __future__ import annotations

import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpcamera import flow_cache_v2 as fc  # noqa: E402
from fpcamera.flow_geometry import (  # noqa: E402
    V2_MOTIONS,
    VideoContract,
    assert_source_coverage,
    camera_path,
    camera_path_v2,
    coverage_slack,
    render_source,
    required_overscan,
    revealed_fraction,
)
from fpcamera.motion_sampling import MotionSampler, sample_motion  # noqa: E402

CFG = fc.load_config()
CONTRACT = fc.contract_of(CFG)
KW = fc.camera_kwargs(CFG)


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------ v1 equivalence
def test_zoom_matches_v1_at_ratio_1p12():
    u = math.log(1.12) / math.log(1.25)
    for direction in ("in", "out"):
        h2, _ = camera_path_v2("zoom", direction, u, CONTRACT)
        h1, _ = camera_path(f"zoom_{direction}", CONTRACT)
        assert np.abs(h2 - h1).max() < 1e-6


@pytest.mark.parametrize("mode", ["pan_left", "pan_right", "tilt_up", "tilt_down"])
def test_pan_tilt_match_v1_at_5_degrees(mode):
    family, direction = mode.split("_")
    h2, _ = camera_path_v2(family, direction, 5 / 8, CONTRACT)
    h1, _ = camera_path(mode, CONTRACT)  # v1 default yaw/pitch = 5 deg
    assert np.abs(h2 - h1).max() < 1e-6


def test_static_and_identity_start():
    h, rec = camera_path_v2("static", None, 0.0, CONTRACT)
    assert np.allclose(h, np.eye(3))
    for f, d in V2_MOTIONS:
        h, rec = camera_path_v2(f, d, 0.4, CONTRACT)
        assert h.shape == (121, 3, 3) and np.allclose(h[0], np.eye(3))


def test_amplitudes_and_zoom_reciprocity():
    _, r = camera_path_v2("pan", "left", 0.5, CONTRACT)
    assert r["yaw_deg"] == pytest.approx(-4.0) and r["pitch_deg"] == 0.0
    _, r = camera_path_v2("tilt", "up", 1.0, CONTRACT)
    assert r["pitch_deg"] == pytest.approx(8.0)
    _, zin = camera_path_v2("zoom", "in", 0.7, CONTRACT)
    _, zout = camera_path_v2("zoom", "out", 0.7, CONTRACT)
    assert zin["zoom_ratio"] * zout["zoom_ratio"] == pytest.approx(1.0)
    _, z = camera_path_v2("zoom", "in", 1.0, CONTRACT)
    assert z["zoom_ratio"] == pytest.approx(1.25)
    with pytest.raises(ValueError):
        camera_path_v2("pan", "left", 1.2, CONTRACT)


def test_linear_easing_is_proportional_to_strength():
    h_full, _ = camera_path_v2("pan", "right", 1.0, CONTRACT)
    h_half, _ = camera_path_v2("pan", "right", 0.5, CONTRACT)
    # tx-like entry at the end scales ~linearly for small angles; halfway frame of the
    # full path has the same yaw as the end of the half path.
    assert np.abs(h_full[60] - h_half[120]).max() < 1e-9


def test_render_source_v1_default_is_unchanged():
    import cv2

    img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    from PIL import Image

    small = VideoContract(width=320, height=192, frames=5)
    h, _ = camera_path("zoom_in", small)
    a, _ = render_source(Image.fromarray(img), h, small)
    b, _ = render_source(Image.fromarray(img), h, small, border_mode=cv2.BORDER_CONSTANT)
    assert np.array_equal(a, b)


# ------------------------------------------------------------------ coverage
def _max_paths():
    return {(f, d): camera_path_v2(f, d, 1.0, CONTRACT, **KW)[0] for f, d in V2_MOTIONS}


@pytest.mark.parametrize("aspect", [0.667, 1.0, 1.333, 1.5, 1.778, 1.818, 2.4, 3.0])
def test_per_source_overscan_covers_all_pixels_all_directions(aspect):
    size = (int(round(800 * aspect)), 800)
    overscan = fc.source_overscan(size, CONTRACT, CFG)
    for key, h in _max_paths().items():
        # every frame on a stride-8 grid, plus the extreme (last) frame at full density
        assert coverage_slack(h, size, CONTRACT, overscan, stride=8) >= 0, key
        assert coverage_slack(h[-1:], size, CONTRACT, overscan, stride=1) >= 0, key


def test_fixed_overscan_is_universal_and_minimal_is_tight():
    fixed = CFG["overscan"]["fixed"]
    worst = 0.0
    for aspect in (0.5, 1.0, 1.333, 1.5, 1.818, 2.5, 3.5):
        size = (int(round(800 * aspect)), 800)
        need = required_overscan(size, CONTRACT, list(_max_paths().values()))
        worst = max(worst, need)
        assert need <= fixed
        # tight: slightly below the requirement must fail some direction
        assert any(
            coverage_slack(h, size, CONTRACT, need - 0.01, stride=8) < 0
            for h in _max_paths().values()
        )
    assert 1.50 < worst < 1.53  # tilt on >=16:9 images dictates the universal value


def test_plan_value_1p45_is_insufficient_for_wide_images():
    size = (1440, 800)  # 16:9
    assert any(
        coverage_slack(h, size, CONTRACT, 1.45, stride=8) < 0 for h in _max_paths().values()
    )


def test_render_rejects_uncovered_clip_and_reflect_never_used():
    from PIL import Image

    img = Image.fromarray(np.zeros((480, 640, 3), np.uint8))
    h = _max_paths()[("zoom", "out")]
    with pytest.raises(ValueError, match="leaves the real source"):
        render_source(img, h, CONTRACT, overscan=1.1, assert_coverage=True)
    with pytest.raises(ValueError):
        assert_source_coverage(h, (640, 480), CONTRACT, 1.1)


def test_reflect_render_matches_constant_when_covered():
    import cv2
    from PIL import Image

    small = VideoContract(width=320, height=192, frames=9)
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
    h, _ = camera_path_v2("pan", "right", 1.0, small, **KW)
    o = required_overscan(img.size, small, [h]) + 0.02
    a, _ = render_source(img, h, small, overscan=o, border_mode=cv2.BORDER_CONSTANT)
    b, _ = render_source(
        img, h, small, overscan=o, border_mode=cv2.BORDER_REFLECT_101, assert_coverage=True
    )
    assert np.array_equal(a, b)  # no pixel ever reaches the border mode


# ---------------------------------------------------------------- revealed area
def test_revealed_fractions_at_max_strength():
    paths = _max_paths()
    frac = {k: revealed_fraction(h[-1], CONTRACT, stride=1) for k, h in paths.items()}
    assert frac[("zoom", "out")] == pytest.approx(1 - 1 / 1.25**2, abs=0.005)
    assert frac[("zoom", "in")] == pytest.approx(0.0, abs=1e-9)
    assert 0.15 < frac[("pan", "left")] < 0.18 and frac[("pan", "left")] == pytest.approx(frac[("pan", "right")], abs=1e-3)
    assert 0.22 < frac[("tilt", "up")] < 0.26 and frac[("tilt", "up")] == pytest.approx(frac[("tilt", "down")], abs=1e-3)
    # monotone in time and zero at frame 0
    series = [revealed_fraction(h, CONTRACT, stride=4) for h in paths[("zoom", "out")][::20]]
    assert series[0] == 0 and all(b >= a for a, b in zip(series, series[1:]))


# ------------------------------------------------------------------- sampling
def _entries(n_train=576, n_dev=24):
    return [(f"s{i:04d}", "train" if i < n_train else "development") for i in range(n_train + n_dev)]


def test_sampler_deterministic_and_order_independent():
    a = MotionSampler(_entries(), seed=7)
    b = MotionSampler(list(reversed(_entries())), seed=7)
    c = MotionSampler(_entries(), seed=8)
    keys = [(f"s{i:04d}", k) for i in (0, 5, 77, 599) for k in (0, 1)]
    assert [a.sample(*k) for k in keys] == [b.sample(*k) for k in keys]
    assert [a.sample(*k) for k in keys] != [c.sample(*k) for k in keys]
    assert sample_motion("x", 0, 1) == sample_motion("x", 0, 1)


def test_sampler_balance_range_and_family_diversity():
    s = MotionSampler(_entries(), seed=3, clips_per_source=2)
    rep = s.report()
    for split, n in (("train", 576), ("development", 24)):
        counts = rep[split]["direction_counts"]
        assert len(counts) == 6 and max(counts.values()) - min(counts.values()) <= 1
        assert rep[split]["u_min"] >= 0.15 and rep[split]["u_max"] <= 1.0
    # near-uniform u marginal (5 equal bins within +-15% of mean count)
    hist = np.array(rep["train"]["u_hist_5bins"])
    assert hist.min() > 0.85 * hist.mean() and hist.max() < 1.15 * hist.mean()
    # the two moving clips of a source use different families
    assert all(s.sample(e, 0).family != s.sample(e, 1).family for e, _ in _entries())
    # per-direction balance also holds within each clip slot
    for slot in (0, 1):
        from collections import Counter

        cnt = Counter(s.sample(e, slot).mode for e, sp in _entries() if sp == "train")
        assert max(cnt.values()) - min(cnt.values()) <= 1


def test_plan_clips_counts_and_ids():
    recs = [{"source_id": e, "split": sp} for e, sp in _entries(30, 6)]
    plans, _ = fc.plan_clips(recs, CFG)
    assert len(plans) == 36 * 3
    assert len({p.id for p in plans}) == len(plans)
    assert sum(p.kind == "static" for p in plans) == 36
    assert all(0.15 <= p.u <= 1.0 for p in plans if p.kind == "moving")
    plans2, _ = fc.plan_clips(recs, CFG)
    assert plans == plans2


# --------------------------------------------------------------- cache file I/O
def _fake_row(cid, sid="a"):
    return {
        "id": cid, "source_id": sid, "split": "train", "family": "zoom", "direction": "in",
        "magnitude_u": 0.5, "yaw_deg": 0.0, "pitch_deg": 0.0, "zoom_ratio": 1.1, "overscan": 1.4,
        "latent_file": f"clips/{cid}.safetensors", "reference_file": f"references/{sid}.safetensors",
    }


def test_reconcile_resume_adopt_and_rebuild(tmp_path):
    torch = pytest.importorskip("torch")
    root = tmp_path / "cache"
    fc.save_tensor_file(root / "references/a.safetensors", {"image_latent": torch.zeros(2)}, {})
    for cid in ("c1", "c2"):
        row = _fake_row(cid)
        fc.save_tensor_file(
            root / row["latent_file"], {"latents": torch.ones(3)}, {"row_json": json.dumps(row)}
        )
    # crash leftovers: a partial file and an index row only for c1
    (root / "clips/c3.safetensors.partial").write_bytes(b"junk")
    row1 = {**_fake_row("c1"), "sha256": fc.file_sha256(root / "clips/c1.safetensors")}
    fc.append_jsonl(root / "index.jsonl", row1)
    with open(root / "index.jsonl", "a") as f:
        f.write('{"id": "torn')  # torn final line
    done = fc.reconcile(root, log=lambda *_: None)
    assert set(done) == {"c1", "c2"}  # c2 adopted from embedded metadata
    assert not list(root.rglob("*.partial"))
    assert done["c2"]["sha256"] == fc.file_sha256(root / "clips/c2.safetensors")
    # tampering is detected: modified file drops out of the verified set
    p = root / "clips/c1.safetensors"
    fc.save_tensor_file(root / "clips/x.safetensors", {"latents": torch.zeros(3)}, {"row_json": "{}"})
    p.write_bytes(p.read_bytes()[:-1] + b"\x00")
    done2 = fc.reconcile(root, log=lambda *_: None)
    assert "c1" not in done2 or done2["c1"]["sha256"] != row1["sha256"]
    rows = fc.rebuild_index(root, log=lambda *_: None)
    lines = [json.loads(x) for x in (root / "index.jsonl").read_text().splitlines()]
    assert sorted(r["id"] for r in lines) == sorted(rows)


def test_disk_check_refuses_when_insufficient(tmp_path, monkeypatch):
    import collections
    import shutil

    Usage = collections.namedtuple("Usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(10**12, 10**12 - 5 * 10**9, 5 * 10**9))
    with pytest.raises(RuntimeError, match="Insufficient disk"):
        fc.check_disk(tmp_path, 10**9, CFG)
    msg, need, free = fc.check_disk(tmp_path, 1, CFG)
    assert need > 0 and "free" in msg


def test_dry_run_needs_no_torch_and_writes_previews(tmp_path):
    builder = _load_script("build_cache_v2")
    manifest = builder.make_synthetic_manifest(tmp_path / "src", 3)
    out = fc.build(
        tmp_path, CFG, cache_root=tmp_path / "cache", manifest=manifest, dry_run=True,
        preview_dir=tmp_path / "prev", preview_count=2, limit=4,
    )
    assert out["rendered"] == 4 and len(list((tmp_path / "prev").glob("*.png"))) == 2
    assert all(0 <= s["revealed_last"] < 0.4 for s in out["stats"])
    assert not (tmp_path / "cache" / "clips").exists()  # nothing written in dry-run


# ------------------------------------------------------------ source selection
def _jpeg(seed, size=(640, 480), gray=False):
    from PIL import Image

    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    img = Image.fromarray(small).resize(size, Image.BICUBIC)
    if gray:
        img = img.convert("L").convert("RGB")
        img = Image.fromarray(np.asarray(img))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def test_hashing_helpers():
    sel = _load_script("select_coco_sources_v2")
    from PIL import Image

    a = Image.open(io.BytesIO(_jpeg(1)))
    b = Image.open(io.BytesIO(_jpeg(1)))
    c = Image.open(io.BytesIO(_jpeg(2)))
    assert sel.hamming(sel.ahash(a), sel.ahash(b)) == 0
    assert sel.hamming(sel.ahash(a), sel.ahash(c)) > 5
    assert sel.channel_std(Image.open(io.BytesIO(_jpeg(1, gray=True)))) < 3.0
    assert sel.channel_std(a) > 20


def test_select_manifest_end_to_end_with_fake_coco(tmp_path):
    sel = _load_script("select_coco_sources_v2")
    cfg = json.loads(json.dumps(CFG))
    cfg["selection"].update({"include_v1_sources": False, "target_new_sources": 30, "new_dev_sources": 2,
                             "download_workers": 3})
    images, anns, caps, blobs = [], [], [], {}
    iid = 1000
    for cat_id, cat in ((1, "alpha"), (2, "beta"), (3, "gamma")):
        for k in range(8):
            iid += 1
            small = k == 7  # too small
            w, h = (300, 200) if small else (640, 480)
            name = f"{iid:012d}.jpg"
            images.append({"id": iid, "file_name": name, "width": w, "height": h, "license": 1,
                           "coco_url": f"http://x/train2017/{name}", "flickr_url": "f", "date_captured": "d"})
            anns.append({"id": iid, "image_id": iid, "category_id": cat_id, "area": w * h * 0.3, "iscrowd": 0})
            caps.append({"id": iid, "image_id": iid, "caption": f"a {cat}"})
            # image 3 of each category is gray, image 4 duplicates image 0 of the category
            blobs[name] = _jpeg(iid, gray=(k == 3)) if k != 4 else _jpeg(cat_id * 100)
            if k == 0:
                blobs[name] = _jpeg(cat_id * 100)
    inst = {"images": images, "annotations": anns, "licenses": [{"id": 1, "url": "u", "name": "n"}],
            "categories": [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}, {"id": 3, "name": "gamma"}]}
    ann_zip = tmp_path / "ann.zip"
    with zipfile.ZipFile(ann_zip, "w") as z:
        z.writestr("annotations/instances_train2017.json", json.dumps(inst))
        z.writestr("annotations/captions_train2017.json", json.dumps({"annotations": caps}))

    def fetch(url):
        return blobs[url.rsplit("/", 1)[1]]

    calls = []
    out1 = sel.build_manifest(tmp_path, cfg, seed=5, fetch=lambda u: (calls.append(u), fetch(u))[1],
                              ann_zip=ann_zip, log=lambda *_: None)
    recs = out1["records"]
    assert len(recs) == 15  # 3 categories x (8 - 1 small - 1 gray - 1 duplicate)
    assert all(r["original_image_sha256"] for r in recs)
    assert sum(r["split"] == "development" for r in recs) == 2
    cats = {r["target_category"] for r in recs}
    assert cats == {"alpha", "beta", "gamma"}
    assert not any(r["source_image_id"] % 1000 == 0 for r in recs)
    # near-duplicates (k=0 and k=4 share content) and grayscale/small were filtered
    assert out1["selection"]["rejected_counts"]["duplicate"] == 3
    assert out1["selection"]["rejected_counts"]["grayscale"] == 3
    # resumable + deterministic: a second run downloads nothing and yields the same manifest
    n_before = len(calls)
    out2 = sel.build_manifest(tmp_path, cfg, seed=5, fetch=lambda u: (calls.append(u), fetch(u))[1],
                              ann_zip=ann_zip, log=lambda *_: None)
    assert len(calls) == n_before
    assert [r["source_id"] for r in out2["records"]] == [r["source_id"] for r in recs]
    assert [r["split"] for r in out2["records"]] == [r["split"] for r in recs]


# ------------------------------------------------- full build loop, fake encoder
class _FakeEncoder:
    """Stands in for the VAE: latent[:,0] depends on frame 0 only (as the causal VAE)."""

    def __init__(self, cache_root, caption, allow_download=False):
        import torch

        self.torch = torch
        self.text = torch.zeros(5, 4096, dtype=torch.bfloat16)
        self.negative = torch.zeros(1, 4096, dtype=torch.bfloat16)
        self.model, self.revision = "fake", "fake"

    def encode(self, rgb):
        torch = self.torch
        f, h, w, _ = rgb.shape
        z = torch.zeros(48, (f - 1) // 4 + 1, h // 16, w // 16, dtype=torch.bfloat16)
        z[:, 0] = float(rgb[0].mean()) / 255
        for t in range(1, z.shape[1]):
            z[:, t] = float(rgb[(t - 1) * 4 + 1 : t * 4 + 1].mean()) / 255
        return z

    def release(self, *a):
        pass


def test_build_resume_crash_safety_and_load(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(fc, "Encoder", _FakeEncoder)
    cfg = json.loads(json.dumps(CFG))
    cfg["video_contract"] = {"width": 320, "height": 192, "frames": 9, "fps": 24}
    cfg["render"]["coverage_stride"] = 8
    builder = _load_script("build_cache_v2")
    manifest = builder.make_synthetic_manifest(tmp_path / "src", 4)
    cache = tmp_path / "cache"
    kw = dict(cache_root=cache, manifest=manifest, use_lease=False)
    out = fc.build(tmp_path, cfg, limit=5, **kw)
    assert out["built"] == 5
    out = fc.build(tmp_path, cfg, **kw)  # resume: only the remaining 7 clips
    assert out["built"] == 12 - 5 and out["total"] == 12
    rows = [json.loads(x) for x in (cache / "index.jsonl").read_text().splitlines()]
    assert len(rows) == 12 and {r["family"] for r in rows} >= {"static"}
    for key in ("id", "source_id", "split", "family", "direction", "magnitude_u", "yaw_deg",
                "pitch_deg", "zoom_ratio", "overscan", "latent_file", "reference_file", "sha256"):
        assert key in rows[0]
    item = fc.load_cached_v2(rows[1], cache, verify=True)
    assert tuple(item["latents"].shape) == (48, 3, 12, 20) and item["homographies"].shape == (9, 3, 3)
    assert item["image_latent"].shape == (48, 1, 12, 20)
    # simulate a crash after the file but before the index row, plus a torn index
    victim = rows[-1]
    idx = cache / "index.jsonl"
    idx.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows[:-1]) + '{"id": "to')
    (cache / "clips" / "zzz.safetensors.partial").write_bytes(b"x")
    out = fc.build(tmp_path, cfg, **kw)
    assert out["built"] == 0 and out["total"] == 12
    assert victim["id"] in {json.loads(x)["id"] for x in idx.read_text().splitlines()}
    # deleting a clip file makes exactly that clip rebuild
    (cache / victim["latent_file"]).unlink()
    out = fc.build(tmp_path, cfg, **kw)
    assert out["built"] == 1
    # changed config is refused instead of silently mixing clips
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["camera"]["max_yaw_deg"] = 9.0
    with pytest.raises(RuntimeError, match="differs"):
        fc.build(tmp_path, cfg2, **kw)
