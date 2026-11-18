"""Measure input appearance preservation against observed video motion.

Only optical-flow tracks already computed from the generated video are used to
fit first-to-current homographies. Optional requested camera homographies are
compared afterward and never affect image alignment. This is a diagnostic;
it defines no automatic quality threshold and makes no FVD claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _ssim_gray(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float | None:
    """Standard 11x11 Gaussian SSIM (sigma 1.5), averaged over valid pixels."""
    x = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    y = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    kernel = (11, 11)
    mu_x = cv2.GaussianBlur(x, kernel, 1.5, borderType=cv2.BORDER_REFLECT)
    mu_y = cv2.GaussianBlur(y, kernel, 1.5, borderType=cv2.BORDER_REFLECT)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = (
        cv2.GaussianBlur(x * x, kernel, 1.5, borderType=cv2.BORDER_REFLECT) - mu_x2
    )
    sigma_y2 = (
        cv2.GaussianBlur(y * y, kernel, 1.5, borderType=cv2.BORDER_REFLECT) - mu_y2
    )
    sigma_xy = (
        cv2.GaussianBlur(x * y, kernel, 1.5, borderType=cv2.BORDER_REFLECT) - mu_xy
    )
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12
    )
    valid = mask.astype(bool)
    if not valid.any():
        return None
    # Only retain locations whose full SSIM window lies inside the overlap.
    interior = cv2.erode(
        valid.astype(np.uint8), np.ones((11, 11), np.uint8), iterations=1
    ).astype(bool)
    return float(np.mean(score[interior])) if interior.any() else None


def _appearance_metrics(
    source: np.ndarray, observed: np.ndarray, mask: np.ndarray
) -> dict:
    valid = mask.astype(bool)
    valid_fraction = float(valid.mean())
    if not valid.any():
        return {
            "psnr_rgb_db": None,
            "ssim_gray_11x11": None,
            "overlap_fraction": 0.0,
            "failure": "no valid warp overlap",
        }
    diff = source.astype(np.float32) - observed.astype(np.float32)
    mse = float(np.mean(np.square(diff[valid])))
    psnr = None if mse == 0 else float(10.0 * np.log10((255.0**2) / mse))
    return {
        "psnr_rgb_db": psnr,
        "psnr_is_positive_infinity": bool(mse == 0),
        "ssim_gray_11x11": _ssim_gray(source, observed, valid),
        "overlap_fraction": valid_fraction,
        "mse_rgb": mse,
        "appearance_measured": True,
        "ssim_definition": "grayscale SSIM; 11x11 Gaussian window, sigma 1.5, C1=0.01^2, C2=0.03^2",
    }


def _fit_observed_homography(
    tracks: np.ndarray, visible: np.ndarray, frame: int, width: int, height: int
) -> tuple[np.ndarray | None, dict]:
    initial = tracks[0].astype(np.float64)
    current = tracks[frame].astype(np.float64)
    usable = visible[0].astype(bool) & visible[frame].astype(bool)
    count = int(usable.sum())
    details = {
        "visible_track_count": count,
        "visible_fraction_of_queries": float(count / len(initial))
        if len(initial)
        else 0.0,
        "fit_method": "observed tracks only; cv2.findHomography RANSAC threshold 3px",
    }
    if count < 16:
        details["fit_failure"] = f"need at least 16 visible tracks; found {count}"
        return None, details
    matrix, inlier_mask = cv2.findHomography(
        initial[usable].astype(np.float32),
        current[usable].astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=3000,
        confidence=0.995,
    )
    if (
        matrix is None
        or not np.isfinite(matrix).all()
        or abs(np.linalg.det(matrix)) < 1e-10
    ):
        details["fit_failure"] = "RANSAC did not return a finite nonsingular homography"
        return None, details
    inliers = (
        inlier_mask.reshape(-1).astype(bool)
        if inlier_mask is not None
        else np.ones(count, bool)
    )
    n_inliers = int(inliers.sum())
    if n_inliers < 16:
        details["fit_failure"] = f"need at least 16 RANSAC inliers; found {n_inliers}"
        return None, details
    src = initial[usable][inliers]
    dst = current[usable][inliers]
    mapped = cv2.perspectiveTransform(src.astype(np.float32)[None], matrix)[0]
    residuals = np.linalg.norm(mapped - dst.astype(np.float32), axis=1)
    details.update(
        {
            "ransac_inlier_count": n_inliers,
            "ransac_inlier_fraction_of_visible": float(n_inliers / count),
            "inlier_residual_median_px": float(np.median(residuals)),
            "inlier_residual_rms_px": float(np.sqrt(np.mean(residuals**2))),
        }
    )
    return matrix.astype(np.float64), details


def _requested_track_metrics(
    request_path: Path, tracks: np.ndarray, visible: np.ndarray, width: int, height: int
) -> dict:
    """Report requested-vs-observed tracks after observed-fit image alignment."""
    request = json.loads(request_path.read_text(encoding="utf-8"))
    contract = request.get("contract", {})
    matrices = np.asarray(
        request.get("homographies_first_to_current"), dtype=np.float64
    )
    frames, npoints, _ = tracks.shape
    if any(
        contract.get(key) != expected
        for key, expected in (("width", width), ("height", height), ("frames", frames))
    ):
        raise ValueError("Request contract dimensions/frames do not match the video")
    if matrices.shape != (frames, 3, 3) or not np.isfinite(matrices).all():
        raise ValueError(
            "Request must contain one finite 3x3 homography per video frame"
        )
    initial = tracks[0].astype(np.float64)
    homogeneous = np.column_stack((initial, np.ones(npoints)))
    diagonal = float(np.hypot(width, height))
    per_frame = []
    all_penalized, visible_errors = [], []
    for frame, matrix in enumerate(matrices):
        mapped = (matrix @ homogeneous.T).T
        in_front = mapped[:, 2] > 1e-9
        expected = np.full((npoints, 2), np.nan, dtype=np.float64)
        expected[in_front] = mapped[in_front, :2] / mapped[in_front, 2, None]
        expected_visible = (
            in_front
            & np.isfinite(expected).all(axis=1)
            & (expected[:, 0] >= 0)
            & (expected[:, 0] < width)
            & (expected[:, 1] >= 0)
            & (expected[:, 1] < height)
        )
        observed_mask = expected_visible & visible[frame].astype(bool)
        missing = expected_visible & ~visible[frame].astype(bool)
        errors = np.linalg.norm(tracks[frame].astype(np.float64) - expected, axis=1)
        if observed_mask.any():
            visible_errors.extend(errors[observed_mask].tolist())
        values = errors[observed_mask].tolist() + [diagonal] * int(missing.sum())
        if values:
            all_penalized.extend(values)
        per_frame.append(
            {
                "frame": frame,
                "expected_visible_queries": int(expected_visible.sum()),
                "observed_expected_visible_queries": int(observed_mask.sum()),
                "missing_expected_visible_queries": int(missing.sum()),
                "expected_visible_coverage": float(
                    observed_mask.sum() / expected_visible.sum()
                )
                if expected_visible.any()
                else None,
                "visible_track_error_median_px": float(np.median(errors[observed_mask]))
                if observed_mask.any()
                else None,
            }
        )
    return {
        "label": "requested homographies versus observed tracks; separate from appearance warp fit",
        "request_homographies_used_to_fit_observed_warp": False,
        "normalized_ADE_visible_only": float(np.mean(visible_errors) / diagonal)
        if visible_errors
        else None,
        "normalized_ADE_missing_expected_penalty_image_diagonal": float(
            np.mean(all_penalized) / diagonal
        )
        if all_penalized
        else None,
        "per_frame": per_frame,
    }


def measure(
    input_path: Path,
    video_path: Path,
    tracks_path: Path,
    samples: int = 13,
    request_path: Path | None = None,
) -> dict:
    # OpenCV's Windows filename API can reject valid non-ASCII paths. Python
    # reads the exact path; OpenCV decodes the same bytes without path handling.
    input_bgr = cv2.imdecode(
        np.frombuffer(Path(input_path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if input_bgr is None:
        raise OSError(f"Could not decode input image: {input_path}")
    archive = np.load(tracks_path, allow_pickle=False)
    try:
        tracks = np.asarray(archive["tracks_xy"], dtype=np.float32)
        visible = np.asarray(archive["visibility"], dtype=bool)
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Track archive is missing required array {error}") from error
    if tracks.ndim != 3 or tracks.shape[-1] != 2 or visible.shape != tracks.shape[:2]:
        raise ValueError(
            f"Unexpected tracks/visibility shapes: {tracks.shape}, {visible.shape}"
        )
    if frame_indices.shape != (len(tracks),) or not np.array_equal(
        frame_indices, np.arange(len(tracks))
    ):
        raise ValueError("Track archive frame_indices must be consecutive from zero")
    nframes, npoints, _ = tracks.shape
    if nframes < 1 or npoints < 16:
        raise ValueError(
            "Track archive needs at least one frame and 16 initial queries"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_h, input_w = input_bgr.shape[:2]
    sampled_frames = sorted(
        set(
            np.linspace(0, nframes - 1, min(samples, nframes))
            .round()
            .astype(int)
            .tolist()
        )
    )
    sample_set = set(sampled_frames)
    per_frame = []
    first_video_frame = None
    width = height = None
    decoded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if decoded >= nframes:
                raise ValueError(
                    "Video has more decoded frames than the supplied track archive"
                )
            height, width = frame.shape[:2]
            if (width, height) != (input_w, input_h):
                raise ValueError(
                    f"Input image is {input_w}x{input_h}, video is {width}x{height}; provide the exact inference input.png"
                )
            if first_video_frame is None:
                first_video_frame = frame.copy()
            if decoded in sample_set:
                matrix, fit = _fit_observed_homography(
                    tracks, visible, decoded, width, height
                )
                item = {
                    "frame": decoded,
                    "time_seconds": decoded / fps if fps > 0 else None,
                    "track_fit": fit,
                }
                if matrix is not None:
                    warped = cv2.warpPerspective(
                        input_bgr,
                        matrix,
                        (width, height),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )
                    source_mask = np.full((input_h, input_w), 255, dtype=np.uint8)
                    overlap = cv2.warpPerspective(
                        source_mask,
                        matrix,
                        (width, height),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    item["input_warp_to_observed_view"] = _appearance_metrics(
                        warped, frame, overlap
                    )
                else:
                    item["input_warp_to_observed_view"] = {
                        "failure": fit.get("fit_failure")
                    }
                per_frame.append(item)
            decoded += 1
    finally:
        cap.release()
    if decoded != nframes:
        raise ValueError(
            f"Video decoded {decoded} frames but track archive contains {nframes}"
        )
    assert first_video_frame is not None and width is not None and height is not None
    first_mask = np.full((height, width), 255, dtype=np.uint8)
    first_comparison = _appearance_metrics(input_bgr, first_video_frame, first_mask)
    # Frame zero is an identity/no-motion reference, not an observed temporal
    # homography fit. Keep its unaligned metric in source_vs_video_first_frame,
    # but do not let it make an otherwise all-failed temporal fit look usable.
    temporal_frames = [x for x in per_frame if x["frame"] > 0]
    usable = [
        x["input_warp_to_observed_view"]
        for x in temporal_frames
        if x["input_warp_to_observed_view"].get("appearance_measured")
    ]
    finite_psnr = [x["psnr_rgb_db"] for x in usable if x.get("psnr_rgb_db") is not None]
    ssims = [
        x["ssim_gray_11x11"] for x in usable if x.get("ssim_gray_11x11") is not None
    ]
    report = {
        "schema_version": 1,
        "metric": "observed_track_aligned_input_appearance",
        "input_image": str(input_path.resolve()),
        "video": str(video_path.resolve()),
        "track_archive": str(tracks_path.resolve()),
        "dimensions": {"width": width, "height": height},
        "video_fps": fps if np.isfinite(fps) and fps > 0 else None,
        "video_declared_frame_count": declared_frames,
        "decoded_frames": decoded,
        "initial_query_count": npoints,
        "sampled_frame_indices": sampled_frames,
        "source_vs_video_first_frame": {
            "label": "unaligned input.png versus decoded first video frame",
            **first_comparison,
        },
        "temporal_samples_aligned_by_observed_tracks": per_frame,
        "temporal_summary": {
            "sample_count": len(per_frame),
            "temporal_sample_count_excluding_reference_frame": len(temporal_frames),
            "successful_observed_homography_fits": sum(
                x["input_warp_to_observed_view"].get("appearance_measured", False)
                for x in temporal_frames
            ),
            "failed_observed_homography_fits": [
                {"frame": x["frame"], **x["track_fit"]}
                for x in temporal_frames
                if not x["input_warp_to_observed_view"].get(
                    "appearance_measured", False
                )
            ],
            "mean_psnr_rgb_db_finite_only": float(np.mean(finite_psnr))
            if finite_psnr
            else None,
            "median_psnr_rgb_db_finite_only": float(np.median(finite_psnr))
            if finite_psnr
            else None,
            "min_psnr_rgb_db_finite_only": float(np.min(finite_psnr))
            if finite_psnr
            else None,
            "perfect_pixel_match_count_psnr_positive_infinity": sum(
                x.get("psnr_is_positive_infinity", False) for x in usable
            ),
            "mean_ssim_gray_11x11": float(np.mean(ssims)) if ssims else None,
            "mean_overlap_fraction": float(
                np.mean([x["overlap_fraction"] for x in usable])
            )
            if usable
            else None,
        },
        "fitting_protocol": {
            "observed_fit_source": "tracks_xy and visibility from independent video tracking archive",
            "minimum_visible_tracks": 16,
            "minimum_ransac_inliers": 16,
            "ransac_reprojection_threshold_px": 3.0,
            "requested_homographies_used_for_alignment": False,
            "warp_direction": "input image to observed video view (first-frame to current-frame)",
            "overlap_mask": "warped source support only; black padding excluded",
        },
        "interpretation_limits": [
            "This diagnostic measures appearance after a robust 2D homography fit to observed tracks; it is not a perceptual or semantic score.",
            "Parallax, moving objects, nonrigid content, blur, tracking errors, and poor track coverage can bias the fitted warp and image scores.",
            "The first-frame comparison is unaligned and is reported separately from temporal aligned comparisons.",
            "No automatic quality pass/fail threshold is applied; no FVD or distribution-level claim is made.",
        ],
    }
    if request_path is not None:
        report["requested_path_track_comparison"] = _requested_track_metrics(
            request_path, tracks, visible, width, height
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="Exact inference input.png"
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument(
        "--tracks",
        required=True,
        type=Path,
        help="Independent .tracks.npz from measure_flow_video.py",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--request",
        type=Path,
        help="Optional request JSON; compared with observed tracks after alignment",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=13,
        help="Uniform temporal sample count (default 13; minimum 1)",
    )
    args = parser.parse_args()
    for path in (args.input, args.video, args.tracks):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.request is not None and not args.request.is_file():
        raise FileNotFoundError(args.request)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.samples < 1:
        parser.error("--samples must be at least one")
    report = measure(args.input, args.video, args.tracks, args.samples, args.request)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "measured",
                "sampled_frames": report["sampled_frame_indices"],
                "mean_aligned_psnr_db_finite_only": report["temporal_summary"][
                    "mean_psnr_rgb_db_finite_only"
                ],
                "mean_aligned_ssim": report["temporal_summary"]["mean_ssim_gray_11x11"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
