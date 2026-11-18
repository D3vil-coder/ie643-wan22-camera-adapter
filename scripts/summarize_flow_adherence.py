"""Summarize sparse-track adherence to a requested 2D camera path.

This CPU-only report complements measure_flow_video.py. It compares actual
tracked points with requested homographies, penalizes missing expected-visible
tracks by one image diagonal, and optionally compares a matched reference run.
It measures apparent 2D image motion, not physical camera pose or scene depth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import cv2


def _read_tracks(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        tracks = np.asarray(archive["tracks_xy"], dtype=np.float64)
        visible = np.asarray(archive["visibility"], dtype=bool)
        initial = np.asarray(archive["initial_query_xy"], dtype=np.float64)
    if tracks.ndim != 3 or tracks.shape[-1] != 2 or visible.shape != tracks.shape[:2]:
        raise ValueError(f"Invalid track archive shapes: {path}")
    if initial.shape != tracks.shape[1:]:
        raise ValueError(f"Initial query array does not match tracks: {path}")
    visible &= np.isfinite(tracks).all(axis=-1)
    return tracks, visible, initial


def _expected_path(
    initial: np.ndarray, homographies: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.concatenate((initial, np.ones((len(initial), 1))), axis=1)
    projected = np.einsum("tij,nj->tni", homographies, homogeneous)
    positive = projected[..., 2] > 1e-9
    expected = np.full(projected.shape[:2] + (2,), np.nan, dtype=np.float64)
    expected[positive] = projected[positive, :2] / projected[positive, 2, None]
    expected_visible = (
        positive
        & np.isfinite(expected).all(axis=-1)
        & (expected[..., 0] >= 0)
        & (expected[..., 0] <= width - 1)
        & (expected[..., 1] >= 0)
        & (expected[..., 1] <= height - 1)
    )
    return expected, expected_visible


def _error_metrics(
    tracks: np.ndarray,
    visible: np.ndarray,
    expected: np.ndarray,
    expected_visible: np.ndarray,
    diagonal: float,
) -> dict:
    eligible = expected_visible
    observed = eligible & visible
    missing = eligible & ~visible
    errors = np.linalg.norm(tracks - expected, axis=-1)
    penalized = np.where(observed, errors, np.where(missing, diagonal, np.nan))
    visible_errors = errors[observed]
    all_errors = penalized[eligible]
    final_eligible = eligible[-1]
    final_observed = observed[-1]
    final_penalized = np.where(
        final_observed,
        errors[-1],
        np.where(final_eligible & ~visible[-1], diagonal, np.nan),
    )
    return {
        "ADE_px_missing_penalty_image_diagonal": float(np.nanmean(all_errors))
        if all_errors.size
        else None,
        "ADE_over_image_diagonal_missing_penalty": float(
            np.nanmean(all_errors) / diagonal
        )
        if all_errors.size
        else None,
        "ADE_px_visible_only": float(np.mean(visible_errors))
        if visible_errors.size
        else None,
        "FDE_px_missing_penalty_image_diagonal": float(
            np.nanmean(final_penalized[final_eligible])
        )
        if final_eligible.any()
        else None,
        "FDE_over_image_diagonal_missing_penalty": float(
            np.nanmean(final_penalized[final_eligible]) / diagonal
        )
        if final_eligible.any()
        else None,
        "FDE_px_visible_only": float(np.mean(errors[-1, final_observed]))
        if final_observed.any()
        else None,
        "expected_visible_tracks_per_frame": eligible.sum(axis=1).astype(int).tolist(),
        "observed_expected_visible_tracks_per_frame": observed.sum(axis=1)
        .astype(int)
        .tolist(),
        "coverage_of_expected_visible_tracks_per_frame": [
            float(observed[t].sum() / eligible[t].sum()) if eligible[t].any() else None
            for t in range(len(eligible))
        ],
        "missing_expected_visible_tracks_per_frame": missing.sum(axis=1)
        .astype(int)
        .tolist(),
    }


def _global_homography_inliers(
    tracks: np.ndarray, visible: np.ndarray, expected_visible: np.ndarray
) -> tuple[np.ndarray, dict]:
    """RANSAC inliers from the observed initial-to-current sparse tracks."""
    frames, points, _ = tracks.shape
    inliers = np.zeros((frames, points), dtype=bool)
    inliers[0] = visible[0]
    per_frame = []
    failures = []
    for frame in range(1, frames):
        ids = np.flatnonzero(visible[frame] & visible[0] & expected_visible[frame])
        if len(ids) < 4:
            per_frame.append(
                {
                    "frame": frame,
                    "fit_succeeded": False,
                    "candidate_tracks": int(len(ids)),
                    "inliers": 0,
                    "inlier_fraction": None,
                }
            )
            failures.append(frame)
            continue
        matrix, mask = cv2.findHomography(
            tracks[0, ids].astype(np.float64),
            tracks[frame, ids].astype(np.float64),
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
            maxIters=2000,
            confidence=0.99,
        )
        if matrix is None or mask is None:
            per_frame.append(
                {
                    "frame": frame,
                    "fit_succeeded": False,
                    "candidate_tracks": int(len(ids)),
                    "inliers": 0,
                    "inlier_fraction": 0.0,
                }
            )
            failures.append(frame)
            continue
        chosen = mask[:, 0].astype(bool)
        inliers[frame, ids[chosen]] = True
        per_frame.append(
            {
                "frame": frame,
                "fit_succeeded": True,
                "candidate_tracks": int(len(ids)),
                "inliers": int(chosen.sum()),
                "inlier_fraction": float(chosen.mean()),
            }
        )
    successes = [
        item["inlier_fraction"]
        for item in per_frame
        if item["fit_succeeded"] and item["inlier_fraction"] is not None
    ]
    return inliers, {
        "method": "cv2.findHomography RANSAC from frame-zero points to current points, 2px threshold",
        "successful_fits": int(len(per_frame) - len(failures)),
        "failed_fit_count": int(len(failures)),
        "failed_fit_frames": failures,
        "mean_inlier_fraction_successful_fits": float(np.mean(successes))
        if successes
        else None,
        "per_frame": per_frame,
        "warning": "A global homography is a 2D image-motion proxy; parallax or scene motion can invalidate it.",
    }


def _motion_metrics(
    tracks: np.ndarray,
    visible: np.ndarray,
    expected: np.ndarray,
    expected_visible: np.ndarray,
    ransac_inliers: np.ndarray | None = None,
) -> dict:
    actual_delta = tracks - tracks[0:1]
    expected_delta = expected - expected[0:1]
    # Exclude the reference frame and zero-motion coordinates from motion
    # strength/cosine; trajectory errors still include the full sequence.
    valid = expected_visible & visible & visible[0:1]
    valid &= np.sum(expected_delta**2, axis=-1) > 1e-12
    if ransac_inliers is not None:
        valid &= ransac_inliers
    valid[0] = False
    vectors_actual = actual_delta[valid]
    vectors_expected = expected_delta[valid]
    expected_energy = float(np.sum(vectors_expected**2))
    actual_energy = float(np.sum(vectors_actual**2))
    dot = float(np.sum(vectors_actual * vectors_expected))
    if expected_energy <= 1e-12:
        gain = cosine = None
    else:
        gain = dot / expected_energy
        cosine = (
            dot / np.sqrt(expected_energy * actual_energy)
            if actual_energy > 1e-12
            else 0.0
        )

    endpoint_valid = valid[-1]
    end_actual = actual_delta[-1, endpoint_valid]
    end_expected = expected_delta[-1, endpoint_valid]
    end_target_energy = float(np.sum(end_expected**2))
    end_actual_energy = float(np.sum(end_actual**2))
    end_dot = float(np.sum(end_actual * end_expected))
    if end_target_energy <= 1e-12:
        endpoint_gain = endpoint_cosine = None
    else:
        endpoint_gain = end_dot / end_target_energy
        endpoint_cosine = (
            end_dot / np.sqrt(end_target_energy * end_actual_energy)
            if end_actual_energy > 1e-12
            else 0.0
        )
    return {
        "signed_motion_gain_pathwide": gain,
        "motion_cosine_pathwide": cosine,
        "signed_motion_gain_endpoint": endpoint_gain,
        "motion_cosine_endpoint": endpoint_cosine,
        "actual_displacement_vector_rms_px": float(
            np.sqrt(np.mean(np.sum(vectors_actual**2, axis=-1)))
        )
        if vectors_actual.size
        else None,
        "requested_displacement_vector_rms_px": float(
            np.sqrt(np.mean(np.sum(vectors_expected**2, axis=-1)))
        )
        if vectors_expected.size
        else None,
        "tracks_used_pathwide": int(valid.sum()),
        "endpoint_tracks_used": int(endpoint_valid.sum()),
        "motion_metric_excludes_reference_frame_and_zero_target_motion": True,
    }


def summarize(
    tracks_path: Path, request_path: Path, static_tracks_path: Path | None = None
) -> dict:
    tracks, visible, initial = _read_tracks(tracks_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    contract = request["contract"]
    width, height, frames = (int(contract[k]) for k in ("width", "height", "frames"))
    matrices = np.asarray(request["homographies_first_to_current"], dtype=np.float64)
    if tracks.shape[0] != frames or matrices.shape != (frames, 3, 3):
        raise ValueError("Track, request, and video contract dimensions disagree")
    finite_visible = visible & np.isfinite(tracks).all(axis=-1)
    if (
        np.any(tracks[..., 0][finite_visible] < 0)
        or np.any(tracks[..., 0][finite_visible] > width - 1)
        or np.any(tracks[..., 1][finite_visible] < 0)
        or np.any(tracks[..., 1][finite_visible] > height - 1)
    ):
        raise ValueError("Visible tracks lie outside the request canvas")
    if not np.allclose(initial, tracks[0], atol=1e-5, equal_nan=True):
        raise ValueError("Frame-zero tracks differ from archived initial queries")
    expected, expected_visible = _expected_path(initial, matrices, width, height)
    ransac_inliers, fit_report = _global_homography_inliers(
        tracks, visible, expected_visible
    )
    diagonal = float(np.hypot(width, height))
    result = {
        "schema_version": 2,
        "metric_scope": "sparse 2D tracked-point adherence to requested homographies",
        "tracks_path": str(tracks_path.resolve()),
        "request_path": str(request_path.resolve()),
        "mode": request.get("mode"),
        "magnitude": request.get("magnitude"),
        "canvas": {
            "width": width,
            "height": height,
            "frames": frames,
            "image_diagonal_px": diagonal,
            "initial_query_count": int(len(initial)),
        },
        "adherence": _error_metrics(
            tracks, visible, expected, expected_visible, diagonal
        ),
        "stationary_prediction": {
            **_error_metrics(
                np.broadcast_to(tracks[0:1], tracks.shape),
                np.broadcast_to(visible[0:1], visible.shape),
                expected,
                expected_visible,
                diagonal,
            ),
            "definition": "ideal zero-motion tracks initialized at the controlled video's own first-frame query points; not a generated model baseline",
        },
        "motion": _motion_metrics(
            tracks, visible, expected, expected_visible, ransac_inliers
        ),
        "observed_global_homography_fits": fit_report,
        "limitations": [
            "Optical flow tracks apparent image motion and cannot establish physical camera pose or 6DoF motion.",
            "A single homography is only a global 2D camera-view proxy; parallax, depth, object motion, blur, and occlusion affect it.",
            "Missing expected-visible tracks receive an image-diagonal penalty; visible-only metrics are also reported for context.",
            "Motion gain and cosine use only observed, expected-visible, nonzero-target-motion coordinates that are also global-homography RANSAC inliers; interpret them with coverage and missing-penalized ADE/FDE.",
        ],
    }
    if static_tracks_path is not None:
        static_tracks, static_visible, static_initial = _read_tracks(static_tracks_path)
        if static_tracks.shape[0] != frames:
            raise ValueError("Reference track frame count does not match the request")
        # The tracker independently seeds points from each generated first
        # frame. Match those seeds spatially, one-to-one, and compare both runs
        # on the same control-query subset.
        distances = np.linalg.norm(
            initial[:, None, :] - static_initial[None, :, :], axis=-1
        )
        candidate_pairs = [
            (float(distances[i, j]), i, j) for i, j in zip(*np.where(distances <= 2.0))
        ]
        candidate_pairs.sort()
        used_control: set[int] = set()
        used_reference: set[int] = set()
        matches = []
        for distance, i, j in candidate_pairs:
            if i not in used_control and j not in used_reference:
                matches.append((i, j, distance))
                used_control.add(i)
                used_reference.add(j)
        control_ids = np.asarray([item[0] for item in matches], dtype=int)
        reference_ids = np.asarray([item[1] for item in matches], dtype=int)
        if not len(matches):
            raise ValueError("No reference query points match within 2 pixels")
        matched_expected, matched_eligible = _expected_path(
            initial[control_ids], matrices, width, height
        )
        result["uncontrolled_reference"] = {
            "tracks_path": str(static_tracks_path.resolve()),
            "reference_type": "supplied static or no-adapter run; interpreted according to its run metadata",
            "matched_query_count": int(len(matches)),
            "control_query_count": int(len(initial)),
            "reference_query_count": int(len(static_initial)),
            "max_initial_match_distance_px": float(max(item[2] for item in matches)),
            "mean_initial_match_distance_px": float(
                np.mean([item[2] for item in matches])
            ),
            **_error_metrics(
                static_tracks[:, reference_ids],
                static_visible[:, reference_ids],
                matched_expected,
                matched_eligible,
                diagonal,
            ),
        }
        result["adherence_on_matched_reference_queries"] = _error_metrics(
            tracks[:, control_ids],
            visible[:, control_ids],
            matched_expected,
            matched_eligible,
            diagonal,
        )
        result["adherence_on_matched_reference_queries"]["matched_query_count"] = int(
            len(matches)
        )
        result["adherence_on_matched_reference_queries"][
            "max_initial_match_distance_px"
        ] = float(max(item[2] for item in matches))
        result["adherence_on_matched_reference_queries"][
            "mean_initial_match_distance_px"
        ] = float(np.mean([item[2] for item in matches]))
        result["motion_on_matched_reference_queries"] = _motion_metrics(
            tracks[:, control_ids],
            visible[:, control_ids],
            matched_expected,
            matched_eligible,
            ransac_inliers[:, control_ids],
        )
        ref_inliers, ref_fit_report = _global_homography_inliers(
            static_tracks[:, reference_ids],
            static_visible[:, reference_ids],
            matched_eligible,
        )
        result["reference_motion_on_matched_queries"] = _motion_metrics(
            static_tracks[:, reference_ids],
            static_visible[:, reference_ids],
            matched_expected,
            matched_eligible,
            ref_inliers,
        )
        result["reference_global_homography_fits_on_matched_queries"] = ref_fit_report
        control_ade = result["adherence_on_matched_reference_queries"][
            "ADE_px_missing_penalty_image_diagonal"
        ]
        ref_ade = result["uncontrolled_reference"][
            "ADE_px_missing_penalty_image_diagonal"
        ]
        control_fde = result["adherence_on_matched_reference_queries"][
            "FDE_px_missing_penalty_image_diagonal"
        ]
        ref_fde = result["uncontrolled_reference"][
            "FDE_px_missing_penalty_image_diagonal"
        ]
        result["difference_vs_uncontrolled_reference"] = {
            "ADE_error_reduction_px": ref_ade - control_ade
            if ref_ade is not None and control_ade is not None
            else None,
            "FDE_error_reduction_px": ref_fde - control_fde
            if ref_fde is not None and control_fde is not None
            else None,
            "note": "Positive values mean the controlled run is closer to the requested path than the supplied reference.",
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument(
        "--reference-tracks",
        type=Path,
        help="Optional matched no-adapter/static reference track archive",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    report = summarize(args.tracks, args.request, args.reference_tracks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "mode": report["mode"],
                "adherence": report["adherence"][
                    "ADE_px_missing_penalty_image_diagonal"
                ],
                "motion_gain": report["motion"]["signed_motion_gain_pathwide"],
                "motion_cosine": report["motion"]["motion_cosine_pathwide"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
