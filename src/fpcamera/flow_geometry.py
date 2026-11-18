"""Explicit virtual pan/tilt/focal-zoom geometry, independent of model weights.

OpenCV image axes: x right, y down, z forward. Orientations are camera-to-world;
positive yaw looks right and positive pitch looks up. All poses share one center.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VideoContract:
    """Output video format: size (multiples of 32), frame count (4k+1), fps and the Wan TI2V VAE/patch strides.
    Defaults are the official 1280x704, 121 frames, 24 fps, i.e. latents ``[48, 31, 44, 80]``.  Validated on creation.
    """

    width: int = 1280
    height: int = 704
    frames: int = 121
    fps: int = 24
    temporal_stride: int = 4
    spatial_stride: int = 16
    patch_stride: int = 2

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0 or self.frames < 5 or self.fps <= 0:
            raise ValueError(
                "Positive dimensions/fps and at least five frames required"
            )
        if self.width % 32 or self.height % 32 or (self.frames - 1) % 4:
            raise ValueError("Wan TI2V requires spatial multiples32 and frames=4k+1")
        if (self.temporal_stride, self.spatial_stride, self.patch_stride) != (4, 16, 2):
            raise ValueError("This contract is specific to the pinned Wan TI2V-5B")

    @property
    def latent_frames(self):
        """Number of latent frames, ``(frames-1)//4 + 1`` (31 for 121 frames)."""
        return (self.frames - 1) // self.temporal_stride + 1

    @property
    def latent_shape(self):
        """``(48, latent_frames, H/16, W/16)``."""
        return (48, self.latent_frames, self.height // 16, self.width // 16)

    def record(self):
        """JSON-able description including whether this is the official reference format."""
        return {
            **asdict(self),
            "latent_shape": list(self.latent_shape),
            "official_inference_reference": (
                self.width,
                self.height,
                self.frames,
                self.fps,
            )
            == (1280, 704, 121, 24),
            "custom_format_requires_measured_comparison": (
                self.width,
                self.height,
                self.frames,
                self.fps,
            )
            != (1280, 704, 121, 24),
        }


MODES = (
    "zoom_in",
    "zoom_out",
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "static",
)


def camera_path(
    mode,
    contract: VideoContract,
    *,
    magnitude=1.0,
    hfov_degrees=60.0,
    yaw_degrees=5.0,
    pitch_degrees=5.0,
    zoom_ratio=1.12,
    easing="linear",
):
    """Return H[t]: first RGB view -> current view; every H[0] is identity."""
    if mode not in MODES:
        raise ValueError(f"Unknown camera mode: {mode}")
    numbers = (magnitude, hfov_degrees, yaw_degrees, pitch_degrees, zoom_ratio)
    if not np.isfinite(numbers).all() or not 0 <= magnitude <= 2:
        raise ValueError("Finite camera parameters and magnitude within [0,2] required")
    if not 20 <= hfov_degrees <= 120 or not 1 < zoom_ratio <= 1.5:
        raise ValueError("Unsupported FOV or focal-zoom ratio")
    if not 0 <= yaw_degrees <= 15 or not 0 <= pitch_degrees <= 15:
        raise ValueError("Initial supported rotations are at most15 degrees")
    tau = np.linspace(0.0, 1.0, contract.frames)
    if easing == "smoothstep":
        tau = tau * tau * (3 - 2 * tau)
    elif easing != "linear":
        raise ValueError("Unknown easing")
    tau *= magnitude
    yaw = np.zeros_like(tau)
    pitch = np.zeros_like(tau)
    scale = np.ones_like(tau)
    if mode.startswith("pan_"):
        yaw = np.deg2rad(yaw_degrees) * tau * (1 if mode == "pan_right" else -1)
    if mode.startswith("tilt_"):
        pitch = np.deg2rad(pitch_degrees) * tau * (1 if mode == "tilt_up" else -1)
    if mode.startswith("zoom_"):
        scale = np.exp(math.log(zoom_ratio) * tau * (1 if mode == "zoom_in" else -1))
    f = (contract.width - 1) / (2 * math.tan(math.radians(hfov_degrees) / 2))
    cx, cy = (contract.width - 1) / 2, (contract.height - 1) / 2
    k0 = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    result = []
    for y, p, s in zip(yaw, pitch, scale):
        ry = np.array(
            [[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]
        )
        rx = np.array(
            [[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]]
        )
        kt = k0.copy()
        kt[0, 0] *= s
        kt[1, 1] *= s
        h = kt @ (ry @ rx).T @ np.linalg.inv(k0)
        result.append(h / h[2, 2])
    matrices = np.asarray(result)
    matrices[0] = np.eye(3)
    record = {
        "scope": "camera_rotation_and_focal_zoom",
        "mode": mode,
        "contract": contract.record(),
        "magnitude": float(magnitude),
        "hfov_degrees": float(hfov_degrees),
        "easing": easing,
        "K0": k0.tolist(),
        "yaw_radians": yaw.tolist(),
        "pitch_radians": pitch.tolist(),
        "focal_scales": scale.tolist(),
        "homographies_first_to_current": matrices.tolist(),
        "camera_center_translation": [0.0, 0.0, 0.0],
        "is_captured_camera_supervision": False,
    }
    return matrices, record


def project_points(homographies, xy):
    """Project pixel points ``[N,2]`` of the first view through homographies ``[T,3,3]``: returns ``[T,N,2]``.
    Raises if a point goes behind the camera.
    """
    h = np.asarray(homographies, dtype=np.float64)
    points = np.asarray(xy, dtype=np.float64)
    if h.ndim != 3 or h.shape[1:] != (3, 3) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Expected [T,3,3] homographies and [N,2] pixel points")
    q = np.einsum("tij,nj->tni", h, np.c_[points, np.ones(len(points))])
    if np.any(q[..., 2] <= 0):
        raise ValueError("A projected point goes behind the camera")
    return q[..., :2] / q[..., 2:]


def source_to_first_view(image_size, contract: VideoContract, *, overscan=1.35):
    """Initial crop mapping with real source margin reserved for future targets."""
    width, height = image_size
    if overscan < 1 or not np.isfinite(overscan):
        raise ValueError("Finite overscan>=1 required")
    crop_w = min(width, height * contract.width / contract.height) / overscan
    crop_h = crop_w * contract.height / contract.width
    x0, y0 = (width - crop_w) / 2, (height - crop_h) / 2
    return np.array(
        [
            [
                (contract.width - 1) / (crop_w - 1),
                0,
                -x0 * (contract.width - 1) / (crop_w - 1),
            ],
            [
                0,
                (contract.height - 1) / (crop_h - 1),
                -y0 * (contract.height - 1) / (crop_h - 1),
            ],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def render_source(
    image: Image.Image | str | Path,
    homographies,
    contract: VideoContract,
    *,
    overscan=1.35,
    border_mode=None,
    assert_coverage=False,
    coverage_stride=4,
):
    """Render targets only from actual source pixels; return RGB and first image.

    Never supply the overscan source or future RGB to an inference condition.

    v2 additions (defaults reproduce v1 exactly): ``border_mode`` is an OpenCV
    border flag (default ``BORDER_CONSTANT``, the v1 behaviour); with
    ``assert_coverage`` every frame is additionally checked on a dense pixel grid
    (see :func:`coverage_slack`) so no filled pixel can ever leak into a target.
    """
    import cv2

    if border_mode is None:
        border_mode = cv2.BORDER_CONSTANT

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    source = np.asarray(image.convert("RGB"))
    first_map = source_to_first_view(image.size, contract, overscan=overscan)
    h = np.asarray(homographies, dtype=np.float64)
    if h.shape != (contract.frames, 3, 3) or not np.allclose(
        h[0], np.eye(3), atol=1e-8
    ):
        raise ValueError("Target path must match contract and begin at identity")
    corners = np.array(
        [
            [0, 0],
            [contract.width - 1, 0],
            [0, contract.height - 1],
            [contract.width - 1, contract.height - 1],
        ]
    )
    if assert_coverage:
        assert_source_coverage(
            h, image.size, contract, overscan, stride=coverage_stride
        )
    rgb = []
    for matrix in h:
        source_to_output = matrix @ first_map
        mapped = project_points(np.linalg.inv(source_to_output)[None], corners)[0]
        if (
            (mapped[:, 0] < 1).any()
            or (mapped[:, 0] > source.shape[1] - 2).any()
            or (mapped[:, 1] < 1).any()
            or (mapped[:, 1] > source.shape[0] - 2).any()
        ):
            raise ValueError(
                "Camera path exceeds real source margin; reject this target"
            )
        rgb.append(
            cv2.warpPerspective(
                source,
                source_to_output,
                (contract.width, contract.height),
                flags=cv2.INTER_LINEAR,
                borderMode=border_mode,
            )
        )
    return np.stack(rgb), {
        "source_size": list(image.size),
        "overscan": overscan,
        "source_to_first_view": first_map.tolist(),
        "training_target_only_hidden_source_margin": True,
    }


# ---------------------------------------------------------------------------
# v2 additions: magnitude-parameterised motion, coverage and revealed regions.
# Nothing above this line changes behaviour; v1 callers are unaffected.
# ---------------------------------------------------------------------------

V2_FAMILIES = ("pan", "tilt", "zoom")
V2_DIRECTIONS = {"pan": ("left", "right"), "tilt": ("up", "down"), "zoom": ("in", "out")}
# Canonical direction order used for balanced sampling and reports.
V2_MOTIONS = tuple((f, d) for f in V2_FAMILIES for d in V2_DIRECTIONS[f])
V2_MAX_YAW_DEG = 8.0
V2_MAX_PITCH_DEG = 8.0
V2_MAX_ZOOM_RATIO = 1.25


def v2_mode(family, direction=None):
    """Map (family, direction) to the v1 mode string, e.g. ("pan","left")->"pan_left"."""
    if family == "static":
        return "static"
    if family not in V2_DIRECTIONS or direction not in V2_DIRECTIONS[family]:
        raise ValueError(f"Unknown v2 motion: {family!r}/{direction!r}")
    return f"{family}_{direction}"


def motion_amplitudes(
    family,
    direction,
    u,
    *,
    max_yaw_deg=V2_MAX_YAW_DEG,
    max_pitch_deg=V2_MAX_PITCH_DEG,
    max_zoom_ratio=V2_MAX_ZOOM_RATIO,
):
    """Signed end-of-clip physical amplitude for strength u in [0, 1].

    yaw/pitch = +-u*max degrees (right/up positive); zoom ratio = exp(+-u*ln(max))
    so zoom-out is the exact reciprocal of zoom-in at equal u. Static ignores u.
    """
    if not np.isfinite(u) or not 0.0 <= u <= 1.0:
        raise ValueError("Strength u must be finite and within [0,1]")
    mode = v2_mode(family, direction)
    yaw = pitch = 0.0
    zoom = 1.0
    if mode == "pan_right":
        yaw = u * max_yaw_deg
    elif mode == "pan_left":
        yaw = -u * max_yaw_deg
    elif mode == "tilt_up":
        pitch = u * max_pitch_deg
    elif mode == "tilt_down":
        pitch = -u * max_pitch_deg
    elif mode == "zoom_in":
        zoom = math.exp(u * math.log(max_zoom_ratio))
    elif mode == "zoom_out":
        zoom = math.exp(-u * math.log(max_zoom_ratio))
    return {"yaw_deg": float(yaw), "pitch_deg": float(pitch), "zoom_ratio": float(zoom)}


def camera_path_v2(
    family,
    direction,
    u,
    contract: VideoContract,
    *,
    max_yaw_deg=V2_MAX_YAW_DEG,
    max_pitch_deg=V2_MAX_PITCH_DEG,
    max_zoom_ratio=V2_MAX_ZOOM_RATIO,
    hfov_degrees=60.0,
    easing="linear",
):
    """Strength-parameterised path: same homography convention as :func:`camera_path`.

    Returns ``(H[T,3,3], record)`` with H[t] mapping the first view to frame t and
    H[0] = I. Strength u scales the *end* amplitude linearly in yaw/pitch and
    log-linearly in focal zoom, with the same linear easing as v1. Implemented by
    delegating to the v1 function (magnitude=u), so conventions cannot drift.
    """
    amp = motion_amplitudes(
        family,
        direction,
        u,
        max_yaw_deg=max_yaw_deg,
        max_pitch_deg=max_pitch_deg,
        max_zoom_ratio=max_zoom_ratio,
    )
    mode = v2_mode(family, direction)
    matrices, record = camera_path(
        mode,
        contract,
        magnitude=float(u),
        hfov_degrees=hfov_degrees,
        yaw_degrees=max_yaw_deg,
        pitch_degrees=max_pitch_deg,
        zoom_ratio=max_zoom_ratio,
        easing=easing,
    )
    record.update(
        {
            "scope": "camera_rotation_and_focal_zoom_v2",
            "family": family,
            "direction": None if family == "static" else direction,
            "strength_u": float(u),
            "max_yaw_deg": float(max_yaw_deg),
            "max_pitch_deg": float(max_pitch_deg),
            "max_zoom_ratio": float(max_zoom_ratio),
            **amp,
        }
    )
    return matrices, record


def _output_grid(contract: VideoContract, stride):
    """Output pixel grid with stride, always including the last row/column."""
    xs = np.unique(np.r_[np.arange(0, contract.width, stride), contract.width - 1])
    ys = np.unique(np.r_[np.arange(0, contract.height, stride), contract.height - 1])
    gx, gy = np.meshgrid(xs.astype(np.float64), ys.astype(np.float64))
    return np.c_[gx.ravel(), gy.ravel()]


def coverage_slack(
    homographies,
    image_size,
    contract: VideoContract,
    overscan,
    *,
    stride=4,
    edge=1.0,
):
    """Minimum slack (source pixels) over all frames and a dense output grid.

    Each grid pixel is mapped back to source coordinates through
    inv(H[t] @ source_to_first_view). Slack is the distance by which the mapped
    point is inside [edge, size-1-edge] (edge=1 matches the v1 corner check and
    keeps bilinear taps inside the image). Negative slack means a pixel would be
    filled by the border mode. Since a homography with positive w maps the convex
    output rectangle to a convex quadrilateral, the grid (which includes all four
    borders) bounds the full-frame extremum; the dense grid is belt-and-braces.
    """
    width, height = image_size
    first_map = source_to_first_view(image_size, contract, overscan=overscan)
    h = np.asarray(homographies, dtype=np.float64)
    inverse = np.linalg.inv(h @ first_map)
    points = _output_grid(contract, stride)
    homogeneous = np.c_[points, np.ones(len(points))]
    worst = np.inf
    for matrix in inverse:  # per frame keeps memory flat for stride=1
        q = homogeneous @ matrix.T
        if np.any(q[:, 2] <= 0):
            return -np.inf
        x = q[:, 0] / q[:, 2]
        y = q[:, 1] / q[:, 2]
        slack = min(
            (x - edge).min(),
            ((width - 1 - edge) - x).min(),
            (y - edge).min(),
            ((height - 1 - edge) - y).min(),
        )
        worst = min(worst, float(slack))
    return worst


def assert_source_coverage(
    homographies, image_size, contract: VideoContract, overscan, *, stride=4
):
    """Reject the clip if any output pixel of any frame would not come from the source."""
    slack = coverage_slack(homographies, image_size, contract, overscan, stride=stride)
    if not slack >= 0:
        raise ValueError(
            f"Camera path leaves the real source for image {tuple(image_size)} at "
            f"overscan {overscan:.3f} (min slack {slack:.2f} px); reject this clip"
        )
    return slack


def required_overscan(
    image_size,
    contract: VideoContract,
    homographies_list,
    *,
    lo=1.0,
    hi=4.0,
    tol=1e-4,
    stride=8,
):
    """Smallest overscan whose crop keeps every pixel of every path inside the source."""

    def ok(overscan):
        return all(
            coverage_slack(h, image_size, contract, overscan, stride=stride) >= 0
            for h in homographies_list
        )

    if not ok(hi):
        raise ValueError(f"No overscan <= {hi} covers image {tuple(image_size)}")
    if ok(lo):
        return float(lo)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return float(hi)  # conservative side of the bracket


def revealed_mask(homography, contract: VideoContract, *, stride=1):
    """Boolean [H,W] (strided) mask of output pixels outside the first view's frame.

    These are the pixels whose content is *newly revealed*: their preimage in the
    first view lies beyond the frame-0 image area (pixel extents [-0.5, W-0.5]).
    """
    h = np.asarray(homography, dtype=np.float64)
    inverse = np.linalg.inv(h)
    xs = np.arange(0, contract.width, stride, dtype=np.float64)
    ys = np.arange(0, contract.height, stride, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    q = np.stack([gx, gy, np.ones_like(gx)], -1) @ inverse.T
    if np.any(q[..., 2] <= 0):
        raise ValueError("A projected point goes behind the camera")
    x, y = q[..., 0] / q[..., 2], q[..., 1] / q[..., 2]
    return (
        (x < -0.5)
        | (x > contract.width - 0.5)
        | (y < -0.5)
        | (y > contract.height - 0.5)
    )


def revealed_fraction(homography, contract: VideoContract, *, stride=2):
    """Fraction of output pixels of this view that are newly revealed (outside the first frame), on a strided grid."""
    return float(revealed_mask(homography, contract, stride=stride).mean())
