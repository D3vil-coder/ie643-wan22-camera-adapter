"""Border (newly revealed region) quality metrics. Pure numpy / OpenCV.

Convention (identical to ``flow_control.build_image_condition``): ``H[t]`` maps
first-view pixel coordinates to frame-t pixel coordinates (``camera_path``).
A frame-t pixel ``q`` shows content from first-view location ``p = H[t]^-1 q``.
``q`` is *revealed* (region R) iff ``p`` falls outside the first image, taken as
``0 <= x <= W-1`` and ``0 <= y <= H-1`` (pixel-centre coordinates, exactly the
normalized-in-[0,1] test used for the guide validity channel) or the point maps
behind the camera. Everything else is the *valid* region V.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

NAN = float("nan")
GRAY_CHROMA_MAX = 12.0
GRAY_LUMA_RANGE = (80.0, 176.0)
SEAM_BAND = 3
SEAM_REF = (3, 12)
HF_ERODE = 3
GLOW_RING = 24  # valid pixels within this many px of the R/valid boundary
RATIO_CLIP = 1.5
LOCAL_RING = (8, 48)  # valid pixels 8..48 px from R: the *local* reference for content-aware ratios


# ---------------------------------------------------------------- masks
def _inverse_map(homographies, xs, ys):
    """Return (x, y, positive) of H[t]^-1 applied to grid (xs, ys); shapes [T,...]."""
    h = np.asarray(homographies, dtype=np.float64)
    if h.ndim == 2:
        h = h[None]
    if h.ndim != 3 or h.shape[1:] != (3, 3):
        raise ValueError("Expected homographies [T,3,3]")
    inv = np.linalg.inv(h)
    q = np.stack((xs, ys, np.ones_like(xs)), axis=-1)  # [...,3]
    mapped = np.einsum("tij,...j->t...i", inv, q)
    positive = mapped[..., 2] > 1e-6
    w = np.clip(mapped[..., 2], 1e-6, None)
    return mapped[..., 0] / w, mapped[..., 1] / w, positive


def revealed_mask_pixels(homographies, width, height, frame_indices=None):
    """bool [T,H,W]; True where frame-t pixel is revealed (outside first image)."""
    h = np.asarray(homographies, dtype=np.float64)
    if frame_indices is not None:
        h = h[np.asarray(frame_indices, dtype=int)]
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    out = np.empty((len(h), height, width), dtype=bool)
    for i in range(len(h)):  # per frame keeps peak memory small
        px, py, pos = _inverse_map(h[i : i + 1], xs, ys)
        inside = (
            pos[0]
            & (px[0] >= 0)
            & (px[0] <= width - 1)
            & (py[0] >= 0)
            & (py[0] <= height - 1)
        )
        out[i] = ~inside
    return out


def revealed_mask_latent(
    homographies, width, height, temporal_stride=4, spatial_stride=16
):
    """bool [T_lat, h, w] revealed mask = ``1 - validity`` of flow_control.

    Mirrors ``build_image_condition``: homographies subsampled every
    ``temporal_stride`` frames, latent cell centres at linspace(0, W-1, w) x
    linspace(0, H-1, h), identity frames are fully valid.
    """
    h = np.asarray(homographies, dtype=np.float64)[::temporal_stride]
    lh, lw = height // spatial_stride, width // spatial_stride
    ys, xs = np.meshgrid(
        np.linspace(0, height - 1, lh), np.linspace(0, width - 1, lw), indexing="ij"
    )
    px, py, pos = _inverse_map(h, xs, ys)
    valid = pos & (px >= 0) & (px <= width - 1) & (py >= 0) & (py <= height - 1)
    identity = np.abs(h - np.eye(3)).max(axis=(-1, -2)) < 1e-7
    valid = valid | identity[:, None, None]
    return ~valid


# ------------------------------------------------------- frame statistics
def _rgb_f(rgb):
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("RGB frame [H,W,3] expected")
    return a.astype(np.float64)


def colorfulness(rgb, mask):
    """Hasler-Susstrunk colorfulness over the pixels in ``mask`` (NaN if empty)."""
    return _colorfulness(_rgb_f(rgb), mask)


def _colorfulness(p, mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 2:
        return NAN
    px = p[mask]
    rg = px[:, 0] - px[:, 1]
    yb = 0.5 * (px[:, 0] + px[:, 1]) - px[:, 2]
    return float(
        math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(rg.mean(), yb.mean())
    )


def _gray(rgb):
    p = _rgb_f(rgb)
    return _luma(p)


def _luma(p):
    return 0.299 * p[..., 0] + 0.587 * p[..., 1] + 0.114 * p[..., 2]


def _erode(mask, px):
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    # Image borders are treated as "kept" so they do not shrink the mask.
    return cv2.erode(
        mask.astype(np.uint8), k, borderType=cv2.BORDER_CONSTANT, borderValue=1
    ).astype(bool)


def hf_energy(rgb, mask, erode=HF_ERODE):
    """Variance of the Laplacian of luma over ``mask`` eroded by ``erode`` px.

    Erosion keeps the R/valid seam edge out of both regions' statistics.
    """
    return _hf(_gray(rgb), mask, erode)


def _hf(g, mask, erode=HF_ERODE):
    m = _erode(np.asarray(mask, dtype=bool), erode)
    if m.sum() < 2:
        return NAN
    lap = cv2.Laplacian(g, cv2.CV_64F, ksize=3)
    return float(lap[m].var())


def gray_band_fraction(rgb, mask):
    """Fraction of ``mask`` pixels with chroma(max-min) < 12 and 80 < luma < 176."""
    p = _rgb_f(rgb)
    return _gray_frac(p, _luma(p), mask)


def _gray_frac(p, luma, mask):
    mask = np.asarray(mask, dtype=bool)
    n = mask.sum()
    if n == 0:
        return NAN
    chroma = p.max(axis=-1) - p.min(axis=-1)
    gray = (
        (chroma < GRAY_CHROMA_MAX)
        & (luma > GRAY_LUMA_RANGE[0])
        & (luma < GRAY_LUMA_RANGE[1])
    )
    return float((gray & mask).sum() / n)


def seam_score(rgb, revealed):
    """Mean gradient magnitude within 3 px of the R/valid boundary divided by the
    mean gradient on the valid side 3-12 px away from it (NaN if undefined)."""
    return _seam(_gray(rgb), revealed)


def _seam(g, revealed):
    r = np.asarray(revealed, dtype=bool)
    v = ~r
    if r.sum() == 0 or v.sum() == 0:
        return NAN
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    mag = np.hypot(gx, gy)
    dv = cv2.distanceTransform(v.astype(np.uint8), cv2.DIST_L2, 3)  # valid -> nearest R
    dr = cv2.distanceTransform(r.astype(np.uint8), cv2.DIST_L2, 3)  # R -> nearest valid
    band = (v & (dv <= SEAM_BAND)) | (r & (dr <= SEAM_BAND))
    ref = v & (dv > SEAM_REF[0]) & (dv <= SEAM_REF[1])
    if band.sum() == 0 or ref.sum() == 0:
        return NAN
    denom = mag[ref].mean()
    return float(mag[band].mean() / denom) if denom > 1e-9 else NAN


def glow_term(luma, revealed):
    """|mean luma in R - mean luma of the valid ring within GLOW_RING px| / 255 (NaN if undefined)."""
    r = np.asarray(revealed, dtype=bool)
    v = ~r
    if r.sum() == 0 or v.sum() == 0:
        return NAN
    dv = cv2.distanceTransform(v.astype(np.uint8), cv2.DIST_L2, 3)
    ring = v & (dv <= GLOW_RING)
    if ring.sum() == 0:
        return NAN
    return float(abs(luma[r].mean() - luma[ring].mean()) / 255.0)


def artifact_score(color_ratio, hf_ratio, glow):
    """Border artifact score = desaturation + blur + glow (0 = natural border).

    ``max(0, 1 - color_ratio) + max(0, 1 - hf_ratio) + glow`` with both ratios
    clipped to [0, RATIO_CLIP]; ``glow`` is ``glow_term``. NaN if a ratio is NaN.
    """
    if not (math.isfinite(color_ratio) and math.isfinite(hf_ratio)):
        return NAN
    c = min(max(color_ratio, 0.0), RATIO_CLIP)
    h = min(max(hf_ratio, 0.0), RATIO_CLIP)
    g = glow if math.isfinite(glow) else 0.0
    return float(max(0.0, 1.0 - c) + max(0.0, 1.0 - h) + g)


artifact_score_v1 = artifact_score  # old one-sided score, kept for comparability


def artifact_score_v2(color_ratio, hf_ratio, glow):
    """Symmetric artifact score: |ln r_color| + |ln r_hf| + glow, ratios clipped to [0.2, 5].

    Penalizes both under- and over-saturated / over-sharp borders. NaN if a ratio is NaN.
    """
    if not (math.isfinite(color_ratio) and math.isfinite(hf_ratio)):
        return NAN
    c = min(max(color_ratio, 0.2), 5.0)
    h = min(max(hf_ratio, 0.2), 5.0)
    g = glow if math.isfinite(glow) else 0.0
    return float(abs(math.log(c)) + abs(math.log(h)) + g)


def local_ring_mask(revealed, inner=LOCAL_RING[0], outer=LOCAL_RING[1]):
    """Valid pixels whose distance to the nearest revealed pixel is in [inner, outer] px.

    Skipping the innermost ``inner`` px keeps the seam out; capping at ``outer`` keeps the
    reference *local* so content that differs across the frame (a wall here, flowers there)
    is not mistaken for an artifact. Empty if there is no R or no valid pixel.
    """
    r = np.asarray(revealed, dtype=bool)
    v = ~r
    if r.sum() == 0 or v.sum() == 0:
        return np.zeros_like(r)
    dv = cv2.distanceTransform(v.astype(np.uint8), cv2.DIST_L2, 3)
    return v & (dv >= inner) & (dv <= outer)


def glow_term_local(luma, revealed, ring=None):
    """|mean luma in R - mean luma of the local ring| / 255 (NaN if undefined)."""
    r = np.asarray(revealed, dtype=bool)
    ring = local_ring_mask(r) if ring is None else ring
    if r.sum() == 0 or ring.sum() == 0:
        return NAN
    return float(abs(luma[r].mean() - luma[ring].mean()) / 255.0)


LOCAL_EPS_COLOR = 1.0  # colorfulness units (0-255 scale)
LOCAL_EPS_HF = 1.0  # Laplacian-variance units


def _ratio_eps(a, b, eps):
    """(a+eps)/(b+eps): a flat or gray local ring (white wall) must not make the ratio undefined."""
    if not (math.isfinite(a) and math.isfinite(b)):
        return NAN
    return float((a + eps) / (b + eps))


def artifact_score_local(color_ratio_local, hf_ratio_local, glow_local):
    """Local-reference symmetric score: same formula and clipping as ``artifact_score_v2``
    but fed with ratios against the local ring (see ``local_ring_mask``)."""
    return artifact_score_v2(color_ratio_local, hf_ratio_local, glow_local)


def _ratio(a, b):
    if not (math.isfinite(a) and math.isfinite(b)) or abs(b) < 1e-12:
        return NAN
    return float(a / b)


METRIC_KEYS = (
    "revealed_fraction",
    "colorfulness_R",
    "colorfulness_valid",
    "colorfulness_ratio",
    "hf_R",
    "hf_valid",
    "hf_ratio",
    "gray_band_R",
    "gray_band_valid",
    "gray_band_score",
    "seam_score",
    "glow_term",
    "artifact_score",
    "artifact_score_v1",
    "artifact_score_v2",
    "colorfulness_ring",
    "hf_ring",
    "colorfulness_ratio_local",
    "hf_ratio_local",
    "glow_term_local",
    "artifact_score_local",
)


def frame_metrics(rgb, revealed):
    """All border metrics for one RGB frame and its revealed mask [H,W]."""
    r = np.asarray(revealed, dtype=bool)
    v = ~r
    frac = float(r.mean())
    if r.sum() == 0:
        return {k: (frac if k == "revealed_fraction" else NAN) for k in METRIC_KEYS}
    p = _rgb_f(rgb)
    g = _luma(p)
    c_r, c_v = _colorfulness(p, r), _colorfulness(p, v)
    h_r, h_v = _hf(g, r), _hf(g, v)
    g_r, g_v = _gray_frac(p, g, r), _gray_frac(p, g, v)
    c_ratio, h_ratio = _ratio(c_r, c_v), _ratio(h_r, h_v)
    glow = glow_term(g, r)
    ring = local_ring_mask(r)
    c_ring, h_ring = _colorfulness(p, ring), _hf(g, ring)
    c_ratio_l = _ratio_eps(c_r, c_ring, LOCAL_EPS_COLOR)
    h_ratio_l = _ratio_eps(h_r, h_ring, LOCAL_EPS_HF)
    glow_l = glow_term_local(g, r, ring)
    return {
        "colorfulness_ring": c_ring,
        "hf_ring": h_ring,
        "colorfulness_ratio_local": c_ratio_l,
        "hf_ratio_local": h_ratio_l,
        "glow_term_local": glow_l,
        "artifact_score_local": artifact_score_local(c_ratio_l, h_ratio_l, glow_l),
        "revealed_fraction": frac,
        "glow_term": glow,
        "artifact_score": artifact_score(c_ratio, h_ratio, glow),
        "artifact_score_v1": artifact_score(c_ratio, h_ratio, glow),
        "artifact_score_v2": artifact_score_v2(c_ratio, h_ratio, glow),
        "colorfulness_R": c_r,
        "colorfulness_valid": c_v,
        "colorfulness_ratio": c_ratio,
        "hf_R": h_r,
        "hf_valid": h_v,
        "hf_ratio": h_ratio,
        "gray_band_R": g_r,
        "gray_band_valid": g_v,
        "gray_band_score": (g_r - g_v)
        if math.isfinite(g_r) and math.isfinite(g_v)
        else NAN,
        "seam_score": _seam(g, r),
    }


def mean_metrics(per_frame, min_fraction=0.01):
    """NaN-ignoring mean of frame_metrics over frames with R fraction > min_fraction."""
    rows = [m for m in per_frame if m["revealed_fraction"] > min_fraction]
    if not rows:
        return {"n_frames": 0}
    out = {"n_frames": len(rows)}
    for k in METRIC_KEYS:
        vals = np.array([m[k] for m in rows], dtype=np.float64)
        out[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else NAN
    return out


# ------------------------------------------------------- latent statistics
def _region_stats(z, m):
    """z [48,N] values, m [N] bool."""
    if m.sum() == 0:
        return {
            "count": 0,
            "mean": NAN,
            "std": NAN,
            "mean_abs": NAN,
            "channel_mean_norm": NAN,
            "channel_mean_abs": NAN,
            "token_norm": NAN,
        }
    s = z[:, m]
    cm = s.mean(axis=1)  # per-channel mean over the region
    return {
        "count": int(m.sum()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "mean_abs": float(np.abs(s).mean()),
        "channel_mean_norm": float(np.linalg.norm(cm)),  # ||mean latent vector||_2
        "channel_mean_abs": float(np.abs(cm).mean()),
        "token_norm": float(np.linalg.norm(s, axis=0).mean()),  # mean ||z(pos)||_2
    }


def latent_stats(latent, revealed_latent, skip_first=True):
    """Region statistics of a latent [48,T,h,w] for R vs valid (latent-res mask).

    The first latent plane is the conditioning frame and is excluded by default.
    ``std`` is the std over all (channel, position) values of the region;
    ``channel_mean_norm`` is the L2 norm of the 48-d region-mean vector.
    """
    z = np.asarray(latent, dtype=np.float64)
    m = np.asarray(revealed_latent, dtype=bool)
    if z.ndim != 4 or z.shape[0] != 48 or m.shape != z.shape[1:]:
        raise ValueError("latent [48,T,h,w] and mask [T,h,w] required")
    if skip_first:
        z, m = z[:, 1:], m[1:]
    zf = z.reshape(48, -1)
    mf = m.reshape(-1)
    r, v = _region_stats(zf, mf), _region_stats(zf, ~mf)
    return {
        "R": r,
        "valid": v,
        "std_ratio_R_over_valid": _ratio(r["std"], v["std"]),
        "mean_abs_ratio_R_over_valid": _ratio(r["mean_abs"], v["mean_abs"]),
        "revealed_fraction": float(mf.mean()),
    }
