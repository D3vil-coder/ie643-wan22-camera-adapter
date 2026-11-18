"""Measure adjacent-frame consistency after compensating for global 2D motion (CPU diagnostic).

Uses the sparse-flow tracks of ``measure_flow_video.py`` to align consecutive frames and reports the residual RGB
difference; ``--aggregate-files`` merges several per-video reports.  A diagnostic, not a quality threshold.

    python scripts/measure_temporal_consistency.py --video run/video.mp4 --tracks run/flow.tracks.npz --output run/temporal.json
    python scripts/measure_temporal_consistency.py --aggregate-files a/temporal.json b/temporal.json --output all.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


MIN_TRACKS = 16
RANSAC_THRESHOLD_PX = 3.0


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def analyze_video(video_path: Path, tracks_path: Path) -> dict[str, Any]:
    """Compare each frame with the prior frame after track fitted homography warp."""
    with np.load(tracks_path, allow_pickle=False) as archive:
        tracks = np.asarray(archive["tracks_xy"], dtype=np.float32)
        visibility = np.asarray(archive["visibility"], dtype=bool)
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int32)

    if tracks.ndim != 3 or tracks.shape[2] != 2:
        raise ValueError(f"Expected tracks_xy shaped [T,N,2], got {tracks.shape}")
    if visibility.shape != tracks.shape[:2]:
        raise ValueError(
            f"Visibility shape {visibility.shape} does not match tracks {tracks.shape}"
        )
    if frame_indices.shape != (tracks.shape[0],):
        raise ValueError("frame_indices must contain one entry per track frame")
    if not np.array_equal(frame_indices, np.arange(len(frame_indices))):
        raise ValueError("Tracks must refer to every consecutive video frame from zero")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    decoded_frames = 0
    pair_records: list[dict[str, Any]] = []
    mae_values: list[float] = []
    overlap_values: list[float] = []
    track_coverage_values: list[float] = []
    failure_counts: Counter[str] = Counter()
    previous = None

    try:
        frame_index = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            height, width = frame_rgb.shape[:2]
            if frame_index >= tracks.shape[0]:
                raise ValueError("Video has more frames than the track file")

            if previous is not None:
                pair = {
                    "from_frame": frame_index - 1,
                    "to_frame": frame_index,
                    "jointly_visible_tracks": 0,
                    "homography_inliers": 0,
                    "track_inlier_coverage": None,
                    "shared_region_fraction": None,
                    "rgb_mae_0_to_1": None,
                    "status": "failed",
                    "failure_reason": None,
                }
                jointly_visible = (
                    visibility[frame_index - 1]
                    & visibility[frame_index]
                    & np.isfinite(tracks[frame_index - 1]).all(axis=1)
                    & np.isfinite(tracks[frame_index]).all(axis=1)
                )
                points_before = tracks[frame_index - 1, jointly_visible]
                points_after = tracks[frame_index, jointly_visible]
                joint_count = int(points_before.shape[0])
                pair["jointly_visible_tracks"] = joint_count
                if joint_count < MIN_TRACKS:
                    pair["failure_reason"] = "fewer_than_16_joint_tracks"
                else:
                    homography, inlier_mask = cv2.findHomography(
                        points_before,
                        points_after,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=RANSAC_THRESHOLD_PX,
                    )
                    inlier_count = (
                        int(np.count_nonzero(inlier_mask))
                        if inlier_mask is not None
                        else 0
                    )
                    pair["homography_inliers"] = inlier_count
                    if homography is None or inlier_count < MIN_TRACKS:
                        pair["failure_reason"] = "fewer_than_16_ransac_inliers"
                    elif not np.isfinite(homography).all():
                        pair["failure_reason"] = "non_finite_homography"
                    else:
                        warped = cv2.warpPerspective(
                            previous,
                            homography,
                            (width, height),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0),
                        )
                        source_mask = np.full(previous.shape[:2], 255, dtype=np.uint8)
                        shared_mask = cv2.warpPerspective(
                            source_mask,
                            homography,
                            (width, height),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        shared = shared_mask > 0
                        shared_pixels = int(np.count_nonzero(shared))
                        if shared_pixels == 0:
                            pair["failure_reason"] = "empty_shared_region"
                        else:
                            diff = cv2.absdiff(warped, frame_rgb)
                            mae = float(np.mean(diff[shared].astype(np.float32)) / 255.0)
                            overlap = float(shared_pixels / (height * width))
                            track_coverage = float(inlier_count / joint_count)
                            pair.update(
                                {
                                    "track_inlier_coverage": track_coverage,
                                    "shared_region_fraction": overlap,
                                    "rgb_mae_0_to_1": mae,
                                    "status": "ok",
                                    "failure_reason": None,
                                }
                            )
                            mae_values.append(mae)
                            overlap_values.append(overlap)
                            track_coverage_values.append(track_coverage)

                if pair["status"] != "ok":
                    failure_counts[str(pair["failure_reason"])] += 1
                pair_records.append(pair)

            previous = frame_rgb
            decoded_frames += 1
            frame_index += 1
    finally:
        cap.release()

    if decoded_frames != tracks.shape[0]:
        raise ValueError(
            f"Decoded {decoded_frames} frames but tracks contain {tracks.shape[0]}"
        )

    successful_pairs = len(mae_values)
    total_pairs = max(0, decoded_frames - 1)
    return {
        "metric": "adjacent_rgb_mae_after_track_fitted_global_homography",
        "video": str(video_path),
        "tracks": str(tracks_path),
        "video_width": int(width),
        "video_height": int(height),
        "decoded_frames": decoded_frames,
        "expected_pairs": total_pairs,
        "successful_pairs": successful_pairs,
        "failed_pairs": total_pairs - successful_pairs,
        "failure_counts": dict(failure_counts),
        "ransac": {
            "minimum_joint_tracks": MIN_TRACKS,
            "minimum_inliers": MIN_TRACKS,
            "reprojection_threshold_pixels": RANSAC_THRESHOLD_PX,
        },
        "rgb_mae_0_to_1_successful_pairs_only": _summary(mae_values),
        "shared_region_fraction_successful_pairs_only": _summary(overlap_values),
        "track_inlier_coverage_successful_pairs_only": _summary(track_coverage_values),
        "pair_records": pair_records,
        "interpretation": (
            "Descriptive adjacent frame diagnostic after compensating for a robust "
            "global 2D homography estimated from observed tracks. It does not judge "
            "newly exposed regions or semantic drift and has no pass or fail threshold."
        ),
    }


def aggregate(paths: list[Path]) -> dict[str, Any]:
    """Pool the per-pair records of several per-video reports (RGB MAE, shared-region fraction, track inlier coverage,
    failure counts) into summary statistics.
    """
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    mae = [
        pair["rgb_mae_0_to_1"]
        for record in records
        for pair in record["pair_records"]
        if pair["status"] == "ok"
    ]
    overlap = [
        pair["shared_region_fraction"]
        for record in records
        for pair in record["pair_records"]
        if pair["status"] == "ok"
    ]
    coverage = [
        pair["track_inlier_coverage"]
        for record in records
        for pair in record["pair_records"]
        if pair["status"] == "ok"
    ]
    failure_counts: Counter[str] = Counter()
    for record in records:
        failure_counts.update(record["failure_counts"])
    total = sum(record["expected_pairs"] for record in records)
    successful = sum(record["successful_pairs"] for record in records)
    return {
        "metric": "adjacent_rgb_mae_after_track_fitted_global_homography",
        "case_count": len(records),
        "decoded_frames_per_case": [record["decoded_frames"] for record in records],
        "expected_pairs": total,
        "successful_pairs": successful,
        "failed_pairs": total - successful,
        "failure_counts": dict(failure_counts),
        "rgb_mae_0_to_1_successful_pairs_only": _summary(mae),
        "shared_region_fraction_successful_pairs_only": _summary(overlap),
        "track_inlier_coverage_successful_pairs_only": _summary(coverage),
        "cases": [
            {
                "video": record["video"],
                "tracks": record["tracks"],
                "expected_pairs": record["expected_pairs"],
                "successful_pairs": record["successful_pairs"],
                "failed_pairs": record["failed_pairs"],
                "failure_counts": record["failure_counts"],
                "rgb_mae_0_to_1_successful_pairs_only": record[
                    "rgb_mae_0_to_1_successful_pairs_only"
                ],
                "shared_region_fraction_successful_pairs_only": record[
                    "shared_region_fraction_successful_pairs_only"
                ],
                "track_inlier_coverage_successful_pairs_only": record[
                    "track_inlier_coverage_successful_pairs_only"
                ],
            }
            for record in records
        ],
        "interpretation": (
            "Descriptive pooled summary across successful adjacent pairs only. "
            "Global 2D motion is compensated using observed tracks. The diagnostic "
            "cannot judge newly exposed regions or semantic drift and has no pass "
            "or fail threshold."
        ),
    }


def main() -> None:
    """CLI entry: analyse one video (with its tracks) or aggregate existing reports."""
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--tracks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregate-files", nargs="*", type=Path)
    args = parser.parse_args()

    if args.aggregate_files is not None:
        if args.video is not None or args.tracks is not None or not args.aggregate_files:
            parser.error("Use --aggregate-files with one or more JSON files only")
        result = aggregate(args.aggregate_files)
    else:
        if args.video is None or args.tracks is None:
            parser.error("--video and --tracks are required for video analysis")
        result = analyze_video(args.video, args.tracks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "pair_records"}, indent=2))


if __name__ == "__main__":
    main()
