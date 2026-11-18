"""CPU tests for build_image_condition(fill_mode=...)."""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from fpcamera.flow_control import build_image_condition


def _reference_v1(image_latent, homographies, *, width, height, frames):
    """Verbatim copy of the v1 build_image_condition (before fill_mode existed)."""
    b, c, _, lh, lw = image_latent.shape
    device = image_latent.device
    matrices = torch.as_tensor(homographies, device=device, dtype=torch.float32)
    if matrices.ndim == 3:
        matrices = matrices[None]
    matrices = matrices[:, ::4]
    t = matrices.shape[1]
    yy, xx = torch.meshgrid(
        torch.linspace(0, height - 1, lh, device=device),
        torch.linspace(0, width - 1, lw, device=device),
        indexing="ij",
    )
    q = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
    mapped = torch.einsum("btij,hwj->bthwi", torch.linalg.inv(matrices), q)
    positive = mapped[..., 2] > 1e-6
    p = mapped[..., :2] / mapped[..., 2:].clamp_min(1e-6)
    denominator = torch.tensor([width - 1, height - 1], device=device)
    pn = p / denominator
    valid = positive & (pn >= 0).all(dim=-1) & (pn <= 1).all(dim=-1)
    grid = (2 * pn - 1).reshape(b * t, lh, lw, 2)
    reference = image_latent[:, :, 0].float()[:, None].expand(-1, t, -1, -1, -1)
    guide = F.grid_sample(
        reference.reshape(b * t, c, lh, lw),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    guide = guide.reshape(b, t, c, lh, lw).permute(0, 2, 1, 3, 4)
    guide = guide * valid[:, None]
    displacement = ((q[..., :2][None, None] - p) / denominator).permute(0, 4, 1, 2, 3)
    identity = (matrices - torch.eye(3, device=device)).abs().amax(dim=(-1, -2)) < 1e-7
    guide = torch.where(
        identity[:, None, :, None, None],
        image_latent.float().expand(-1, -1, t, -1, -1),
        guide,
    )
    valid = valid | identity[:, :, None, None]
    displacement = torch.where(
        identity[:, None, :, None, None], torch.zeros_like(displacement), displacement
    )
    return torch.cat((guide, valid[:, None].float(), displacement), dim=1)


W, H, FRAMES = 128, 64, 9  # latent grid 4 x 8, 3 latent frames


def zoom_out_path(final_scale=0.6, frames=FRAMES, width=W, height=H):
    cx, cy = (width - 1) / 2, (height - 1) / 2
    out = []
    for s in np.linspace(1.0, final_scale, frames):
        out.append([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)], [0, 0, 1]])
    out = np.asarray(out, dtype=np.float64)
    out[0] = np.eye(3)
    return out


def make_latent(seed=0):
    g = torch.Generator().manual_seed(seed)
    # Offset so replicated edge features are clearly nonzero.
    return torch.randn(1, 48, 1, H // 16, W // 16, generator=g) + 2.0


def test_zeros_mode_bit_identical_to_v1():
    lat, path = make_latent(), zoom_out_path()
    ref = _reference_v1(lat, path, width=W, height=H, frames=FRAMES)
    default = build_image_condition(lat, path, width=W, height=H, frames=FRAMES)
    zeros = build_image_condition(
        lat, path, width=W, height=H, frames=FRAMES, fill_mode="zeros"
    )
    assert torch.equal(default, ref)
    assert torch.equal(zeros, ref)


def test_replicate_vs_zeros():
    lat, path = make_latent(1), zoom_out_path()
    zeros = build_image_condition(lat, path, width=W, height=H, frames=FRAMES)
    rep = build_image_condition(
        lat, path, width=W, height=H, frames=FRAMES, fill_mode="replicate"
    )
    assert rep.shape == zeros.shape == (1, 51, 3, 4, 8)
    assert torch.isfinite(rep).all()
    valid = zeros[:, 48] > 0.5
    assert torch.equal(rep[:, 48], zeros[:, 48])  # validity unchanged
    assert torch.equal(rep[:, 49:], zeros[:, 49:])  # displacement unchanged
    assert (~valid).any() and valid.any()
    guide_z, guide_r = zeros[:, :48], rep[:, :48]
    vm = valid[:, None].expand_as(guide_z)
    assert torch.equal(guide_r[vm], guide_z[vm])  # identical where valid
    assert torch.count_nonzero(guide_z[~vm]) == 0  # zeros mode: exact zeros
    assert (guide_r[~vm] != 0).float().mean() > 0.99  # replicate: nonzero


def test_replicate_fill_is_nearest_edge_value():
    lat = make_latent(2)
    path = zoom_out_path(0.4)
    rep = build_image_condition(
        lat, path, width=W, height=H, frames=FRAMES, fill_mode="replicate"
    )
    # Frame 2 (last): the central region is valid; the corner cell (0,0) is
    # revealed and must equal the first-image corner cell exactly (clamped).
    assert rep[0, 48, 2, 0, 0] == 0
    assert torch.allclose(rep[0, :48, 2, 0, 0], lat[0, :, 0, 0, 0], atol=1e-5)


def test_identity_shortcut_kept_in_both_modes():
    lat = make_latent(3)
    path = zoom_out_path()
    for mode in ("zeros", "replicate"):
        c = build_image_condition(
            lat, path, width=W, height=H, frames=FRAMES, fill_mode=mode
        )
        assert torch.equal(c[:, :48, 0], lat[:, :, 0].float())
        assert (c[:, 48, 0] == 1).all()
        assert (c[:, 49:, 0] == 0).all()


def test_bad_fill_mode():
    with pytest.raises(ValueError):
        build_image_condition(
            make_latent(), zoom_out_path(), width=W, height=H, frames=FRAMES,
            fill_mode="mirror",
        )
