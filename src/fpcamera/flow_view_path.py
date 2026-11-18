"""Convert a user-drawn normalized view path to a Wan camera request.

Normalized x increases to the right and y increases downward. The path is a
camera-view path: displacement from its first point becomes camera-to-world yaw
and pitch, with no camera-center translation. Optional zoom values are focal
length ratios relative to the initial view.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .flow_geometry import VideoContract


def camera_view_path(
    points_normalized: Sequence[Sequence[float]],
    contract: VideoContract,
    *,
    focal_zoom_ratio_curve: Sequence[float] | None = None,
    max_yaw_degrees: float = 5.0,
    max_pitch_degrees: float = 5.0,
    hfov_degrees: float = 60.0,
) -> tuple[np.ndarray, dict]:
    """Return ``(H, request)`` with ``H[t] = K[t] R[t].T K[0]^-1``.

    ``points_normalized`` must contain at least two finite points in [0, 1]^2.
    Points and the optional per-point zoom ratios are interpolated to
    ``contract.frames`` by normalized cumulative arclength. A rightward view
    displacement produces positive C2W yaw; a downward view displacement
    produces negative C2W pitch because positive pitch in the project geometry
    looks upward. Each rotation component is bounded to five degrees.

    ``focal_zoom_ratio_curve`` contains one positive focal ratio per input
    point, starts at 1.0, and is bounded to [0.5, 1.5]. Omit it for a fixed
    focal length. For a zero-length view path, any supplied focal curve is
    interpolated uniformly over the input vertices.

    The request dictionary matches ``flow_infer.generate(camera_request=...)``:
    it includes the full video contract and ``homographies_first_to_current``.
    """
    if not isinstance(contract, VideoContract):
        raise ValueError("contract must be a VideoContract")
    points = np.asarray(points_normalized, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or points.shape[0] < 2:
        raise ValueError("Expected at least two normalized [x,y] points")
    if not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
        raise ValueError("Normalized points must be finite and lie in [0,1]")
    limits = np.asarray(
        [max_yaw_degrees, max_pitch_degrees, hfov_degrees], dtype=np.float64
    )
    if not np.isfinite(limits).all():
        raise ValueError("Camera angle and FOV limits must be finite")
    if not (0 < max_yaw_degrees <= 5 and 0 < max_pitch_degrees <= 5):
        raise ValueError("Yaw and pitch limits must be in (0, 5] degrees")
    if not 20 <= hfov_degrees <= 120:
        raise ValueError("Horizontal field of view must be in [20,120] degrees")

    if focal_zoom_ratio_curve is None:
        zoom = np.ones(points.shape[0], dtype=np.float64)
    else:
        zoom = np.asarray(focal_zoom_ratio_curve, dtype=np.float64)
        if zoom.shape != (points.shape[0],) or not np.isfinite(zoom).all():
            raise ValueError("Zoom curve must contain one finite ratio per path point")
        if (zoom < 0.5).any() or (zoom > 1.5).any():
            raise ValueError("Focal zoom ratios must lie in [0.5,1.5]")
        if not math.isclose(float(zoom[0]), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("The first focal zoom ratio must be 1.0")
        zoom[0] = 1.0

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(arc[-1])
    if total_length > 0:
        progress = arc / total_length
        # A zero-length segment has no arclength interval on which to vary its
        # focal ratio, so reject ambiguous jumps instead of inventing a timing.
        if focal_zoom_ratio_curve is not None:
            zero_segments = np.flatnonzero(segment_lengths == 0)
            if any(
                not math.isclose(
                    float(zoom[i]), float(zoom[i + 1]), rel_tol=0.0, abs_tol=1e-10
                )
                for i in zero_segments
            ):
                raise ValueError("Zoom cannot change across a zero-length path segment")
    else:
        # A stationary camera view can still have a user-supplied zoom curve.
        progress = np.linspace(0.0, 1.0, points.shape[0])
    frame_progress = np.linspace(0.0, 1.0, contract.frames)
    sampled = np.column_stack(
        [np.interp(frame_progress, progress, points[:, axis]) for axis in range(2)]
    )
    sampled_zoom = np.interp(frame_progress, progress, zoom)
    sampled[0] = points[0]
    sampled_zoom[0] = 1.0

    displacement = sampled - sampled[0]
    yaw = np.deg2rad(max_yaw_degrees) * displacement[:, 0]
    pitch = -np.deg2rad(max_pitch_degrees) * displacement[:, 1]
    yaw = np.clip(yaw, -np.deg2rad(max_yaw_degrees), np.deg2rad(max_yaw_degrees))
    pitch = np.clip(
        pitch, -np.deg2rad(max_pitch_degrees), np.deg2rad(max_pitch_degrees)
    )

    focal = (contract.width - 1) / (2 * math.tan(math.radians(hfov_degrees) / 2))
    center_x, center_y = (contract.width - 1) / 2, (contract.height - 1) / 2
    k0 = np.asarray(
        [[focal, 0.0, center_x], [0.0, focal, center_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    k0_inverse = np.linalg.inv(k0)
    homographies = np.empty((contract.frames, 3, 3), dtype=np.float64)
    for i, (yaw_i, pitch_i, zoom_i) in enumerate(zip(yaw, pitch, sampled_zoom)):
        cy, sy = math.cos(float(yaw_i)), math.sin(float(yaw_i))
        cp, sp = math.cos(float(pitch_i)), math.sin(float(pitch_i))
        rotation_y = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        rotation_x = np.asarray([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        rotation_c2w = rotation_y @ rotation_x
        kt = k0.copy()
        kt[0, 0] *= zoom_i
        kt[1, 1] *= zoom_i
        h = kt @ rotation_c2w.T @ k0_inverse
        homographies[i] = h / h[2, 2]
    homographies[0] = np.eye(3, dtype=np.float64)
    if not np.isfinite(homographies).all() or np.any(
        np.abs(np.linalg.det(homographies)) < 1e-10
    ):
        raise ValueError("Generated camera homographies are invalid")

    request = {
        "schema_version": 1,
        "scope": "user_drawn_camera_view_path",
        "contract": contract.record(),
        "coordinate_convention": "normalized image view; x-right=camera-right, y-down=camera-down; opencv camera axes",
        "input_points_normalized": points.tolist(),
        "sampled_points_normalized": sampled.tolist(),
        "arclength_progress": frame_progress.tolist(),
        "yaw_degrees": np.rad2deg(yaw).tolist(),
        "pitch_degrees": np.rad2deg(pitch).tolist(),
        "focal_zoom_ratio_curve": sampled_zoom.tolist(),
        "max_yaw_degrees": float(max_yaw_degrees),
        "max_pitch_degrees": float(max_pitch_degrees),
        "hfov_degrees": float(hfov_degrees),
        "homographies_first_to_current": homographies.tolist(),
        "camera_center_translation": [0.0, 0.0, 0.0],
        "is_captured_camera_supervision": False,
    }
    return homographies, request
