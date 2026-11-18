"""Measure image motion in a generated video with sparse optical flow.

The tracker sees decoded pixels only; optional requested homographies are
loaded only after tracking has completed. This is a diagnostic, not a quality
threshold or camera-control acceptance test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _seed_points(
    gray: np.ndarray, max_points: int = 128, min_points: int = 16
) -> np.ndarray:
    """Allocate Shi-Tomasi queries across four central-image quadrants."""
    h, w = gray.shape
    x0, x1 = int(round(0.15 * w)), int(round(0.85 * w))
    y0, y1 = int(round(0.15 * h)), int(round(0.85 * h))
    split_x, split_y = (x0 + x1) // 2, (y0 + y1) // 2
    boxes = (
        (x0, y0, split_x, split_y),
        (split_x, y0, x1, split_y),
        (x0, split_y, split_x, y1),
        (split_x, split_y, x1, y1),
    )
    selected: list[np.ndarray] = []
    per_box = max_points // 4
    for left, top, right, bottom in boxes:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[top:bottom, left:right] = 255
        found = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=per_box,
            qualityLevel=0.01,
            minDistance=8,
            mask=mask,
            blockSize=7,
            useHarrisDetector=False,
        )
        if found is not None:
            selected.extend(found[:, 0, :])
    if len(selected) < min_points:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        found = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_points,
            qualityLevel=0.001,
            minDistance=5,
            mask=mask,
            blockSize=5,
            useHarrisDetector=False,
        )
        if found is not None:
            for point in found[:, 0, :]:
                if not any(np.linalg.norm(point - prior) < 3 for prior in selected):
                    selected.append(point)
                if len(selected) >= max_points:
                    break
    if len(selected) < min_points:
        raise RuntimeError(
            f"Only {len(selected)} Shi-Tomasi points; need at least {min_points}"
        )
    return np.asarray(selected[:max_points], dtype=np.float32).reshape(-1, 2)


def _inside(points: np.ndarray, width: int, height: int) -> np.ndarray:
    return (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0)
        & (points[:, 0] <= width - 1)
        & (points[:, 1] >= 0)
        & (points[:, 1] <= height - 1)
    )


def track_video(video_path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    ok, first_bgr = cap.read()
    if not ok:
        cap.release()
        raise ValueError(f"Video contains no decodable frames: {video_path}")
    previous = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2GRAY)
    height, width = previous.shape
    initial = _seed_points(previous)
    count = len(initial)
    all_tracks = [initial.copy()]
    all_visible = [np.ones(count, dtype=bool)]
    pair_metrics: list[dict] = []
    global_map = np.eye(3, dtype=np.float64)
    center = np.array([[(width - 1) / 2], [(height - 1) / 2], [1.0]], dtype=np.float64)
    global_motion = [
        {
            "frame": 0,
            "cumulative_scale": 1.0,
            "cumulative_rotation_degrees": 0.0,
            "center_dx_px": 0.0,
            "center_dy_px": 0.0,
        }
    ]
    frame_index = 0
    prior_points = initial.copy()
    prior_visible = np.ones(count, dtype=bool)
    lk = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frame_index += 1
            current = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            next_points = np.full((count, 2), np.nan, dtype=np.float32)
            candidate_ids = np.flatnonzero(prior_visible)
            pair_residual = np.empty((0,), dtype=np.float32)
            if len(candidate_ids):
                old = prior_points[candidate_ids].reshape(-1, 1, 2)
                forward, sf, _ = cv2.calcOpticalFlowPyrLK(
                    previous, current, old, None, **lk
                )
                if forward is not None and sf is not None:
                    backward, sb, _ = cv2.calcOpticalFlowPyrLK(
                        current, previous, forward, None, **lk
                    )
                    if backward is not None and sb is not None:
                        new_xy = forward[:, 0, :]
                        fb_error = np.linalg.norm(
                            backward[:, 0, :] - old[:, 0, :], axis=1
                        )
                        good = (
                            sf[:, 0].astype(bool)
                            & sb[:, 0].astype(bool)
                            & (fb_error <= 1.5)
                            & _inside(new_xy, width, height)
                        )
                        good_ids = candidate_ids[good]
                        next_points[good_ids] = new_xy[good]
                        # Bad queries remain NaN and are explicitly counted as lost.
            visible = np.isfinite(next_points).all(axis=1)
            valid_ids = np.flatnonzero(visible & prior_visible)
            fit = None
            if len(valid_ids) >= 3:
                fit, inliers = cv2.estimateAffinePartial2D(
                    prior_points[valid_ids],
                    next_points[valid_ids],
                    method=cv2.RANSAC,
                    ransacReprojThreshold=2.0,
                    maxIters=2000,
                    confidence=0.99,
                    refineIters=10,
                )
                if fit is not None:
                    matrix = np.eye(3, dtype=np.float64)
                    matrix[:2] = fit
                    global_map = matrix @ global_map
                    inlier_mask = (
                        inliers[:, 0].astype(bool)
                        if inliers is not None
                        else np.ones(len(valid_ids), dtype=bool)
                    )
                    predicted = cv2.transform(prior_points[valid_ids][None], fit)[0]
                    pair_residual = np.linalg.norm(
                        predicted - next_points[valid_ids], axis=1
                    )[inlier_mask]
                    linear = fit[:, :2]
                    pair_scale = float(np.sqrt(max(0.0, np.linalg.det(linear))))
                    pair_rotation = float(
                        np.degrees(np.arctan2(linear[1, 0], linear[0, 0]))
                    )
                    pair_center = fit @ np.array(
                        [[(width - 1) / 2], [(height - 1) / 2], [1.0]]
                    )
                    inlier_count = int(inlier_mask.sum())
                    pair_metrics.append(
                        {
                            "frame": frame_index,
                            "valid_tracks": int(len(valid_ids)),
                            "inliers": inlier_count,
                            "inlier_coverage_of_valid": float(
                                inlier_count / len(valid_ids)
                            ),
                            "scale": pair_scale,
                            "rotation_degrees": pair_rotation,
                            "center_dx_px": float(pair_center[0, 0] - (width - 1) / 2),
                            "center_dy_px": float(pair_center[1, 0] - (height - 1) / 2),
                            "inlier_residual_rms_px": float(
                                np.sqrt(np.mean(pair_residual**2))
                            )
                            if len(pair_residual)
                            else None,
                        }
                    )
                else:
                    pair_metrics.append(
                        {
                            "frame": frame_index,
                            "valid_tracks": int(len(valid_ids)),
                            "inliers": 0,
                            "inlier_coverage_of_valid": 0.0,
                            "scale": None,
                            "rotation_degrees": None,
                            "center_dx_px": None,
                            "center_dy_px": None,
                            "inlier_residual_rms_px": None,
                        }
                    )
            else:
                pair_metrics.append(
                    {
                        "frame": frame_index,
                        "valid_tracks": int(len(valid_ids)),
                        "inliers": 0,
                        "inlier_coverage_of_valid": None,
                        "scale": None,
                        "rotation_degrees": None,
                        "center_dx_px": None,
                        "center_dy_px": None,
                        "inlier_residual_rms_px": None,
                    }
                )
            if fit is not None:
                cumulative_center = global_map @ center
                linear = global_map[:2, :2]
                global_motion.append(
                    {
                        "frame": frame_index,
                        "cumulative_scale": float(
                            np.sqrt(max(0.0, np.linalg.det(linear)))
                        ),
                        "cumulative_rotation_degrees": float(
                            np.degrees(np.arctan2(linear[1, 0], linear[0, 0]))
                        ),
                        "center_dx_px": float(cumulative_center[0, 0] - center[0, 0]),
                        "center_dy_px": float(cumulative_center[1, 0] - center[1, 0]),
                    }
                )
            else:
                global_motion.append(
                    {
                        "frame": frame_index,
                        "cumulative_scale": None,
                        "cumulative_rotation_degrees": None,
                        "center_dx_px": None,
                        "center_dy_px": None,
                    }
                )
            all_tracks.append(next_points)
            all_visible.append(visible)
            prior_points, prior_visible, previous = next_points, visible, current
    finally:
        cap.release()

    tracks = np.stack(all_tracks)
    visibility = np.stack(all_visible)
    coverage = visibility.mean(axis=1)
    usable_scales = [x["scale"] for x in pair_metrics if x["scale"] is not None]
    report = {
        "video": str(video_path.resolve()),
        "decoded_frames": int(len(tracks)),
        "fps": fps if np.isfinite(fps) and fps > 0 else None,
        "width": int(width),
        "height": int(height),
        "initial_queries": int(count),
        "initial_query_quadrants": _quadrant_counts(initial, width, height),
        "tracker": "ShiTomasi + pyramidal LK forward/backward consistency <= 1.5px + in-frame",
        "per_frame_visible_fraction": coverage.tolist(),
        "per_frame_valid_track_count": visibility.sum(axis=1).astype(int).tolist(),
        "per_frame_global_similarity": pair_metrics,
        "cumulative_global_motion": global_motion,
        "summary": {
            "final_track_coverage": float(coverage[-1]),
            "mean_track_coverage": float(coverage.mean()),
            "median_adjacent_scale": float(np.median(usable_scales))
            if usable_scales
            else None,
            "cumulative_scale": global_motion[-1]["cumulative_scale"],
            "cumulative_rotation_degrees": global_motion[-1][
                "cumulative_rotation_degrees"
            ],
            "cumulative_center_dx_px": global_motion[-1]["center_dx_px"],
            "cumulative_center_dy_px": global_motion[-1]["center_dy_px"],
            "mean_inlier_residual_rms_px": float(
                np.mean(
                    [
                        x["inlier_residual_rms_px"]
                        for x in pair_metrics
                        if x["inlier_residual_rms_px"] is not None
                    ]
                )
            )
            if any(x["inlier_residual_rms_px"] is not None for x in pair_metrics)
            else None,
        },
        "caveats": [
            "Optical flow measures apparent image motion, not physical camera calibration.",
            "No pass threshold is applied; occlusion, blur, parallax, and scene motion affect results.",
            "Global similarity is a robust 2D fit and cannot by itself prove a camera zoom.",
        ],
    }
    return report, tracks, visibility, initial


def _quadrant_counts(points: np.ndarray, width: int, height: int) -> list[int]:
    midx, midy = (width - 1) / 2, (height - 1) / 2
    right, bottom = points[:, 0] >= midx, points[:, 1] >= midy
    return [
        int((~right & ~bottom).sum()),
        int((right & ~bottom).sum()),
        int((~right & bottom).sum()),
        int((right & bottom).sum()),
    ]


def compare_request(
    report: dict, tracks: np.ndarray, visible: np.ndarray, request_path: Path
) -> dict:
    """Compare already-tracked pixels to an optional first-to-current H path."""
    request = json.loads(request_path.read_text(encoding="utf-8"))
    matrices = np.asarray(request["homographies_first_to_current"], dtype=np.float64)
    contract = request["contract"]
    width, height, frames = (
        int(contract["width"]),
        int(contract["height"]),
        int(contract["frames"]),
    )
    actual_frames, npoints, _ = tracks.shape
    if (width, height) != (report["width"], report["height"]):
        raise ValueError("Request canvas does not match decoded video")
    if matrices.shape != (frames, 3, 3) or frames != actual_frames:
        raise ValueError("Request homography count must equal decoded video frames")
    initial = tracks[0].astype(np.float64)
    homogeneous = np.concatenate((initial, np.ones((npoints, 1))), axis=1)
    expected = np.full((frames, npoints, 2), np.nan, dtype=np.float64)
    expected_visible = np.zeros((frames, npoints), dtype=bool)
    for t, h in enumerate(matrices):
        mapped = (h @ homogeneous.T).T
        good_z = mapped[:, 2] > 1e-9
        xy = np.full((npoints, 2), np.nan, dtype=np.float64)
        xy[good_z] = mapped[good_z, :2] / mapped[good_z, 2, None]
        expected[t] = xy
        expected_visible[t] = good_z & _inside(xy, width, height)
    observed_error = np.linalg.norm(tracks.astype(np.float64) - expected, axis=2)
    diagonal = float(np.hypot(width, height))
    eligible = expected_visible
    observed_eligible = eligible & visible
    missing_expected = eligible & ~visible
    penalized = np.where(
        observed_eligible, observed_error, np.where(missing_expected, diagonal, np.nan)
    )
    vis_errors = observed_error[observed_eligible]
    penalties = penalized[eligible]
    final_eligible = eligible[-1]
    final_observed = final_eligible & visible[-1]
    final_missing = final_eligible & ~visible[-1]
    fde_penalty = np.where(
        final_observed, observed_error[-1], np.where(final_missing, diagonal, np.nan)
    )
    progress = []
    for t in range(frames):
        expected_delta = expected[t] - expected[0]
        actual_delta = tracks[t].astype(np.float64) - tracks[0]
        norms = np.sum(expected_delta**2, axis=1)
        candidates = (
            expected_visible[t] & expected_visible[0] & visible[t] & (norms > 1e-8)
        )
        if candidates.any():
            vals = (
                np.sum(actual_delta[candidates] * expected_delta[candidates], axis=1)
                / norms[candidates]
            )
            progress.append(float(np.median(vals)))
        else:
            progress.append(None)
    coverage = []
    for t in range(frames):
        denominator = int(eligible[t].sum())
        coverage.append(
            float(observed_eligible[t].sum() / denominator) if denominator else None
        )
    result = {
        "request": str(request_path.resolve()),
        "comparison_after_tracking": True,
        "request_homographies_passed_to_tracker": False,
        "expected_visible_track_count_per_frame": eligible.sum(axis=1)
        .astype(int)
        .tolist(),
        "expected_visible_coverage_per_frame": coverage,
        "normalized_ADE_missing_penalty_image_diagonal": float(
            np.mean(penalties) / diagonal
        )
        if penalties.size
        else None,
        "normalized_ADE_visible_only": float(np.mean(vis_errors) / diagonal)
        if vis_errors.size
        else None,
        "normalized_FDE_missing_penalty_image_diagonal": float(
            np.nanmean(fde_penalty) / diagonal
        )
        if np.isfinite(fde_penalty).any()
        else None,
        "normalized_FDE_visible_only": float(
            np.mean(observed_error[-1, final_observed]) / diagonal
        )
        if final_observed.any()
        else None,
        "missing_expected_visible_count_over_time": missing_expected.sum(axis=1)
        .astype(int)
        .tolist(),
        "median_along_path_progress": progress,
        "expected_scale_endpoint": float(np.sqrt(np.linalg.det(matrices[-1, :2, :2]))),
        "interpretation": "Progress is a median projection of observed displacement onto requested displacement; it is undefined when visible tracked points have no expected motion.",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.request and not args.request.is_file():
        raise FileNotFoundError(args.request)
    report, tracks, visible, initial = track_video(args.video)
    if args.request:
        report["requested_path_comparison"] = compare_request(
            report, tracks, visible, args.request
        )
    track_path = args.output.with_suffix(".tracks.npz")
    if track_path.exists():
        raise FileExistsError(f"Refusing to overwrite {track_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        track_path,
        tracks_xy=tracks,
        visibility=visible,
        initial_query_xy=initial,
        frame_indices=np.arange(len(tracks), dtype=np.int32),
    )
    report["track_archive"] = str(track_path.resolve())
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "video": report["video"],
                "frames": report["decoded_frames"],
                "initial_queries": report["initial_queries"],
                "final_coverage": report["summary"]["final_track_coverage"],
                "cumulative_scale": report["summary"]["cumulative_scale"],
                "json": str(args.output.resolve()),
                "tracks": str(track_path.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
