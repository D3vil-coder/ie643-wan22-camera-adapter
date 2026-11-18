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
        return (self.frames - 1) // self.temporal_stride + 1

    @property
    def latent_shape(self):
        return (48, self.latent_frames, self.height // 16, self.width // 16)

    def record(self):
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
):
    """Render targets only from actual source pixels; return RGB and first image.

    Never supply the overscan source or future RGB to an inference condition.
    """
    import cv2

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
                borderMode=cv2.BORDER_CONSTANT,
            )
        )
    return np.stack(rgb), {
        "source_size": list(image.size),
        "overscan": overscan,
        "source_to_first_view": first_map.tolist(),
        "training_target_only_hidden_source_margin": True,
    }
