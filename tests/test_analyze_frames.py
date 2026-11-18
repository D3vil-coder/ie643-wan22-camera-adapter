"""CPU tests of the per-frame drift / spike analysis helpers (``scripts/analyze_frames.py``)."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_frames as af  # noqa: E402


def test_spike_detection_ignores_period4_structure_and_trend_but_finds_isolated_spike():
    n = 121
    t = np.arange(n)
    series = 1.0 + 0.004 * t  # smooth trend
    series = series + np.where(t % 4 == 1, 0.5, 0.0)  # decoder period-4 pattern (frames 1, 5, 9, ...)
    series = series + np.random.default_rng(0).normal(0, 0.02, n)
    series[63] += 2.0  # one genuinely garbled frame
    z_glob, z_run, spikes = af.detect_spikes(series, 4.0, 0.4, rel_floor=0.15)
    assert spikes == [63]
    assert z_run[63] > 4


def test_spike_detection_no_spikes_on_smooth_series():
    t = np.arange(121)
    series = 0.8 + 0.001 * t + np.random.default_rng(1).normal(0, 0.01, 121)
    assert af.detect_spikes(series, 4.0, 0.4, rel_floor=0.15)[2] == []


def test_mc_residual_zero_for_pure_translation_and_dc_removed_ignores_brightness_step():
    rng = np.random.default_rng(2)
    base = np.clip(rng.normal(120, 30, (176, 320)).astype(np.float32), 0, 255)
    import cv2  # noqa: PLC0415

    img = cv2.GaussianBlur(base, (0, 0), 3)
    img = (img - img.min()) / (img.max() - img.min()) * 200 + 20
    big = np.kron(img, np.ones((2, 2), np.float32))  # 352 x 640 like a half-resolution flow input source
    shifted = np.roll(big, 3, axis=1)
    raw, dm = af.mc_residual(big, shifted)
    assert raw < 3.0 and dm < 3.0  # translation is compensated by the flow
    bright = np.clip(big + 12.0, 0, 255)
    raw_b, dm_b = af.mc_residual(big, bright)
    assert raw_b > 8.0  # a brightness step is a big raw residual ...
    assert dm_b < 0.5 * raw_b  # ... but not once the global offset is removed


def test_running_median_is_phase_aware():
    x = np.array([5.0 if i % 4 == 1 else 1.0 for i in range(40)])
    base = af.running_median(x)
    assert np.allclose(base[1:36:4], 5.0)
    assert np.allclose(base[2:36:4], 1.0)


def test_luma_and_lab_helpers():
    rgb = np.full((8, 8, 3), 128, np.uint8)
    assert af.luma_of(rgb).mean() == pytest.approx(128.0, abs=0.5)
    a, b = af.lab_means(rgb)
    assert abs(a) < 1 and abs(b) < 1
