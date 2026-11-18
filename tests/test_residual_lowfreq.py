"""CPU tests for the opt-in residual low-frequency filter (F1 ``mean``, F2 ``blur:K``) of the camera anchor."""
from contextlib import contextmanager
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpcamera import flow_infer as fi  # noqa: E402
from fpcamera.flow_anchor import frozen_camera_difference, lowfreq_highpass, parse_residual_lowfreq  # noqa: E402


# ------------------------------------------------------------------ spec parsing
def test_parse_specs():
    assert parse_residual_lowfreq(None) is None
    assert parse_residual_lowfreq("") is None
    assert parse_residual_lowfreq("none") is None
    assert parse_residual_lowfreq("mean") == ("mean", None)
    assert parse_residual_lowfreq("blur:6") == ("blur", 6.0)
    assert parse_residual_lowfreq(" Blur:2.5 ") == ("blur", 2.5)
    assert parse_residual_lowfreq(("blur", 4)) == ("blur", 4.0)
    for bad in ("blur", "blur:", "blur:0", "blur:-1", "blur:nan", "mean:3", "median", "blur:x"):
        with pytest.raises(ValueError):
            parse_residual_lowfreq(bad)


# ------------------------------------------------------------------ the filters themselves
def test_none_is_identity_object():
    x = torch.randn(1, 3, 5, 6, 8)
    assert lowfreq_highpass(x, None) is x


def test_mean_removes_spatial_mean_per_plane_and_channel_and_keeps_the_rest():
    torch.manual_seed(0)
    pattern = torch.randn(1, 4, 3, 10, 12)
    pattern = pattern - pattern.mean(dim=(-2, -1), keepdim=True)  # zero spatial mean
    offset = torch.randn(1, 4, 3, 1, 1) * 5  # per plane/channel constant shift
    out = lowfreq_highpass(pattern + offset, "mean")
    assert out.shape == pattern.shape
    torch.testing.assert_close(out.mean(dim=(-2, -1)), torch.zeros(1, 4, 3), atol=1e-5, rtol=0)
    torch.testing.assert_close(out, pattern, atol=1e-5, rtol=0)  # structure untouched


def test_blur_constant_goes_to_zero_and_border_is_renormalised():
    x = torch.full((1, 2, 3, 11, 17), 3.7)
    out = lowfreq_highpass(x, "blur:3")
    assert out.abs().max() < 1e-5  # includes the border cells (renormalised, no darkening at the edges)


def test_blur_keeps_high_frequencies_removes_smooth_ramp():
    h, w = 44, 80
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    checker = ((yy + xx) % 2) * 2 - 1  # 1-cell checkerboard
    ramp = (xx / w) * 4 - 2  # smooth horizontal ramp
    out_hf = lowfreq_highpass(checker[None, None], "blur:6")
    out_lf = lowfreq_highpass(ramp[None, None], "blur:6")
    assert (out_hf - checker[None, None]).abs().mean() < 0.02  # checkerboard essentially untouched
    # the ramp is removed except near the borders where the low-pass of a ramp is biased
    interior = out_lf[..., 12:-12, 20:-20]
    assert interior.abs().max() < 0.15 * ramp.abs().max()


def test_large_blur_approaches_mean_removal_for_smooth_input():
    torch.manual_seed(1)
    x = torch.randn(1, 2, 2, 20, 30) * 0.2 + torch.randn(1, 2, 2, 1, 1) * 3
    a = lowfreq_highpass(x, "mean")
    b = lowfreq_highpass(x, "blur:1000")  # sigma >> grid: uniform weights = spatial mean
    torch.testing.assert_close(a, b, atol=1e-3, rtol=0)


def test_works_on_arbitrary_rank_and_dtype_cpu():
    for shape in ((7, 4), (2, 7, 4), (1, 48, 31, 44, 80)):
        x = torch.randn(*shape)
        for spec in ("mean", "blur:3"):
            out = lowfreq_highpass(x, spec)
            assert out.shape == x.shape and out.dtype == x.dtype
            assert torch.isfinite(out).all()


# ------------------------------------------------------------------ integration with the anchor
class _Adapter:
    def __init__(self):
        self.enabled, self.cond = True, None

    def set_condition(self, c):
        self.cond = c

    @contextmanager
    def disabled(self):
        before, self.enabled = self.enabled, False
        try:
            yield self
        finally:
            self.enabled = before


class _ShiftTx(nn.Module):
    """F0 = elementwise tanh of a per-channel scaling; the 'adapter' adds ``cond`` when enabled.

    Latents are ``[1, C=4, T=3, H=5, W=6]`` like the real ones (``[1, 48, T, h, w]``), so the filter's last two dims are spatial.
    """

    def __init__(self, adapter):
        super().__init__()
        torch.manual_seed(0)
        self.scale = nn.Parameter(torch.rand(1, 4, 1, 1, 1) + 0.5)
        self.adapter = adapter

    def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict=False):
        y = torch.tanh(hidden_states * self.scale)
        if self.adapter.enabled and self.adapter.cond is not None:
            y = y + self.adapter.cond
        return (y,)


def _run(cond_requested, residual_lowfreq=None, **kwargs):
    adapter = _Adapter()
    tx = _ShiftTx(adapter).eval()
    pos, neg = torch.randn(1, 5, 6), torch.randn(1, 5, 6)
    outs = []
    with torch.no_grad(), frozen_camera_difference(
        tx, adapter, cond_requested, None, camera_gain=1.0, text_guidance=5.0, active_min_sigma=0.3,
        reference="teacher", residual_lowfreq=residual_lowfreq, **kwargs,
    ) as state:
        g = torch.Generator().manual_seed(3)
        for i in range(3):
            lat = torch.randn(1, 4, 3, 5, 6, generator=g)
            ts = torch.full((1, 3), 900.0 - 100 * i)
            p = tx(hidden_states=lat, timestep=ts, encoder_hidden_states=pos, return_dict=False)[0]
            n = tx(hidden_states=lat, timestep=ts, encoder_hidden_states=neg, return_dict=False)[0]
            outs.append(torch.stack([p, n]))
    return torch.stack(outs), state


def test_default_unchanged_and_recorded_as_none():
    cond = torch.randn(1, 4, 1, 1, 1)
    ref, st_ref = _run(cond)
    same, st = _run(cond, residual_lowfreq=None)
    torch.testing.assert_close(same, ref, rtol=0, atol=0)
    assert st["residual_lowfreq"] is None and st_ref["residual_lowfreq"] is None


def test_mean_filter_removes_a_global_shift_but_default_does_not():
    shift = torch.randn(1, 4, 1, 1, 1)  # constant over each frame's space: a pure global colour/brightness shift
    base, _ = _run(torch.zeros(1, 4, 1, 1, 1))  # no camera effect
    plain, _ = _run(shift)
    filt, st = _run(shift, residual_lowfreq="mean")
    assert (plain - base).abs().max() > 0.1  # without the filter the shift reaches the guided prediction
    # the shift is removed up to float error; the negative-text prediction (F0 only) is never touched
    torch.testing.assert_close(filt, base, atol=1e-5, rtol=0)
    assert st["residual_lowfreq"] == ["mean", None]
    assert st["active_positive_calls"] == 3
    blur, _ = _run(shift, residual_lowfreq="blur:2")
    torch.testing.assert_close(blur, base, atol=1e-5, rtol=0)  # a constant is also pure low frequency for the blur


def test_filter_leaves_non_global_residual_in_place():
    cond = torch.randn(1, 4, 3, 5, 6)  # varies over space and time
    plain, _ = _run(cond)
    filt, _ = _run(cond, residual_lowfreq="mean")
    base, _ = _run(torch.zeros(1, 4, 3, 5, 6))
    residual = plain[:, 0] - base[:, 0]
    expected = base[:, 0] + (residual - residual.mean(dim=(-2, -1), keepdim=True))
    torch.testing.assert_close(filt[:, 0], expected, atol=1e-4, rtol=0)
    assert not torch.allclose(filt[:, 0], plain[:, 0])
    torch.testing.assert_close(filt[:, 1], base[:, 1], rtol=0, atol=1e-5)  # negative text path unchanged


# ------------------------------------------------------------------ generate()/CLI plumbing
def test_cli_flag_default_off_and_parsed():
    base = ["--project-root", ".", "--checkpoint", "c", "--image", "i", "--output", "o"]
    assert fi.parse_args(base).residual_lowfreq is None
    assert fi.parse_args(base + ["--residual-lowfreq", "mean"]).residual_lowfreq == "mean"
    assert fi.parse_args(base + ["--residual-lowfreq", "blur:6"]).residual_lowfreq == "blur:6"


def test_generate_rejects_bad_or_incompatible_filter_before_any_gpu_work(tmp_path):
    with pytest.raises(ValueError):
        fi.generate(tmp_path, tmp_path / "x.safetensors", tmp_path / "i.png", tmp_path / "out", residual_lowfreq="median")
    with pytest.raises(ValueError, match="not --frozen-baseline"):
        fi.generate(tmp_path, tmp_path / "x.safetensors", tmp_path / "i.png", tmp_path / "out2",
                    residual_lowfreq="mean", frozen_baseline=True)


def test_trace_residual_diagnostic_off_by_default_and_reports_mean_share():
    shift = torch.randn(1, 4, 1, 1, 1)
    _, st_off = _run(shift)
    assert "residual_trace" not in st_off  # default: nothing recorded, run.json unchanged
    _, st = _run(shift, trace_residual=True)
    tr = st["residual_trace"]
    assert len(tr) == st["active_positive_calls"] == 3
    for rec in tr:
        assert set(rec) == {"sigma", "rms", "rms_spatial_mean", "rms_lowpass8", "channel_mean"}
        # a pure per-channel global shift: all residual energy is in the spatial mean
        assert rec["rms_spatial_mean"] == pytest.approx(rec["rms"], rel=1e-3)
        assert len(rec["channel_mean"]) == 4
    import json

    json.dumps(st)  # must stay serialisable for run.json


# ------------------------------------------------------------------ validity-weighted estimate (F4)
def _two_zone(c_valid, c_new, h=10, w=20, split=14):
    x = torch.full((1, 3, 2, h, w), float(c_valid))
    x[..., split:] = float(c_new)
    weight = torch.ones(1, 1, 2, h, w)
    weight[..., split:] = 0.0
    return x, weight


def test_weighted_mean_estimates_from_valid_cells_and_removes_it_everywhere():
    x, wgt = _two_zone(2.0, 9.0)
    out = lowfreq_highpass(x, "mean", wgt)
    assert out[..., :14].abs().max() < 1e-5  # the valid zone is its own mean -> removed completely
    torch.testing.assert_close(out[..., 14:], torch.full_like(out[..., 14:], 7.0), atol=1e-5, rtol=0)  # new content keeps its offset
    plain = lowfreq_highpass(x, "mean")
    assert plain[..., :14].abs().max() > 1.0  # the unweighted mean is polluted by the new zone


def test_weighted_blur_matches_weighted_mean_for_huge_sigma_and_none_weight_is_unchanged():
    x, wgt = _two_zone(2.0, 9.0)
    a = lowfreq_highpass(x, "mean", wgt)
    b = lowfreq_highpass(x, "blur:1000", wgt)
    torch.testing.assert_close(a, b, atol=1e-3, rtol=0)
    y = torch.randn(1, 3, 2, 10, 20)
    torch.testing.assert_close(lowfreq_highpass(y, "blur:3", None), lowfreq_highpass(y, "blur:3"))
    ones = torch.ones(1, 1, 2, 10, 20)
    torch.testing.assert_close(lowfreq_highpass(y, "blur:3", ones), lowfreq_highpass(y, "blur:3"), atol=1e-5, rtol=0)
    torch.testing.assert_close(lowfreq_highpass(y, "mean", ones), lowfreq_highpass(y, "mean"), atol=1e-5, rtol=0)


def test_weighted_filter_falls_back_when_a_plane_has_no_valid_cells():
    y = torch.randn(1, 3, 2, 10, 20)
    wgt = torch.zeros(1, 1, 2, 10, 20)  # nothing valid
    for spec in ("mean", "blur:3"):
        out = lowfreq_highpass(y, spec, wgt)
        torch.testing.assert_close(out, lowfreq_highpass(y, spec), atol=1e-5, rtol=0)
        assert torch.isfinite(out).all()


def test_anchor_with_weight_records_flag_and_generate_validates_mask_flag(tmp_path):
    shift = torch.randn(1, 4, 1, 1, 1)
    wgt = torch.ones(1, 1, 3, 5, 6)
    _, st = _run(shift, residual_lowfreq="mean", lowfreq_weight=wgt)
    assert st["residual_lowfreq_weighted"] is True
    _, st0 = _run(shift, residual_lowfreq="mean")
    assert st0["residual_lowfreq_weighted"] is False
    base = ["--project-root", ".", "--checkpoint", "c", "--image", "i", "--output", "o"]
    assert fi.parse_args(base).residual_lowfreq_mask is None
    assert fi.parse_args(base + ["--residual-lowfreq-mask", "valid"]).residual_lowfreq_mask == "valid"
    with pytest.raises(ValueError, match="needs residual_lowfreq"):
        fi.generate(tmp_path, tmp_path / "x.safetensors", tmp_path / "i.png", tmp_path / "o1", residual_lowfreq_mask="valid")
    with pytest.raises(ValueError):
        fi.generate(tmp_path, tmp_path / "x.safetensors", tmp_path / "i.png", tmp_path / "o2", residual_lowfreq="mean",
                    residual_lowfreq_mask="bogus")
