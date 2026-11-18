"""Input preservation and temporal proxies, separate from 3D pose adherence.

SSIM/PSNR, frozen DINO feature similarity and flow-compensated pixel error do
not establish perceptual quality by themselves. Full acceptance also requires
independent pose success and observed-depth parallax; human review is optional.
Thresholds are predeclared engineering gates, not universally validated norms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
from pathlib import Path
from typing import Any

import numpy as np

from .runtime import file_sha256

DINO_MODEL = "facebook/dinov2-small"
# Verified from the official public HF model API on 2026-10-24.
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"


@dataclass(frozen=True)
class PreservationThresholds:
    min_dino_mean_cosine: float = 0.80
    min_first_frame_ssim: float = 0.75
    max_flow_compensated_temporal_l1: float = 0.06
    min_flow_valid_fraction: float = 0.50
    min_parallax_confidence: float = 0.50


def _rgb(value: Any, name: str = "image") -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 3 or result.shape[-1] != 3 or result.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 RGB [H,W,3]")
    return result


def first_frame_preservation(reference: Any, generated: Any) -> dict[str, Any]:
    """RGB SSIM (11x11 Gaussian sigma=1.5) and PSNR, data range 1."""
    import cv2

    first, second = _rgb(reference), _rgb(generated)
    if first.shape != second.shape or min(first.shape[:2]) < 11:
        raise ValueError("SSIM needs matching RGB canvases with dimensions >=11")
    x, y = first.astype(np.float64) / 255.0, second.astype(np.float64) / 255.0
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    var_x = np.maximum(0.0, cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x)
    var_y = np.maximum(0.0, cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y)
    covariance = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    mse = float(np.mean((x - y) ** 2))
    return {
        "status": "ok",
        "ssim": float(np.clip(ssim[5:-5, 5:-5].mean(), -1, 1)),
        "psnr_db": float(-10 * np.log10(mse)) if mse > 0 else None,
        "psnr_infinite": mse == 0,
        "mse_data_range_1": mse,
        "method": "RGB_Gaussian11_sigma1.5_SSIM_valid_window_and_PSNR",
        "interpretation": "First decoded frame versus supplied preprocessed input; VAE/image preservation proxy",
    }


def _flow_pair(
    previous: np.ndarray, following: np.ndarray, max_dimension: int
) -> dict[str, Any]:
    import cv2

    height, width = previous.shape[:2]
    factor = min(1.0, max_dimension / max(height, width))
    target = (max(16, round(width * factor)), max(16, round(height * factor)))
    if min(height, width) < 16:
        raise ValueError("DIS optical flow requires at least a 16x16 frame")
    a = (
        cv2.resize(previous, target, interpolation=cv2.INTER_AREA)
        if factor < 1
        else previous
    )
    b = (
        cv2.resize(following, target, interpolation=cv2.INTER_AREA)
        if factor < 1
        else following
    )
    gray_a, gray_b = (
        cv2.cvtColor(a, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(b, cv2.COLOR_RGB2GRAY),
    )
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    forward = dis.calc(np.ascontiguousarray(gray_a), np.ascontiguousarray(gray_b), None)
    backward = dis.calc(
        np.ascontiguousarray(gray_b), np.ascontiguousarray(gray_a), None
    )
    if not np.isfinite(forward).all() or not np.isfinite(backward).all():
        raise RuntimeError("DIS produced nonfinite flow")
    h, w = gray_a.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    qx, qy = xx + forward[..., 0], yy + forward[..., 1]
    sampled_backward = cv2.remap(
        backward, qx, qy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    fb_error = np.linalg.norm(forward + sampled_backward, axis=-1)
    flow_magnitude = np.linalg.norm(forward, axis=-1)
    threshold = 1.0 + 0.05 * flow_magnitude
    # A forward/backward inconsistency mask is an occlusion/reliability proxy;
    # it is not ground-truth visibility or calibrated confidence.
    valid = (
        (qx >= 0) & (qx < w - 1) & (qy >= 0) & (qy < h - 1) & (fb_error <= threshold)
    )
    coverage = float(valid.mean())
    if not valid.any():
        raise RuntimeError("No forward/backward-consistent in-bounds flow pixels")
    warped = cv2.remap(
        b.astype(np.float32) / 255.0,
        qx,
        qy,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    residual = np.abs(a.astype(np.float32) / 255.0 - warped).mean(axis=-1)
    return {
        "status": "ok",
        "l1": float(residual[valid].mean()),
        "valid_fraction": coverage,
        "uncompensated_l1": float(
            np.abs(a.astype(np.float32) - b.astype(np.float32)).mean() / 255
        ),
        "mean_flow_pixels": float(flow_magnitude[valid].mean()),
        "mean_fb_error_pixels": float(fb_error[valid].mean()),
        "flow_canvas_hw": [h, w],
    }


def temporal_flow_consistency(
    frames: Any, *, max_dimension: int = 416
) -> dict[str, Any]:
    """All adjacent pairs with F/B masking; failed pairs receive L1=1, coverage=0."""
    video = np.asarray(frames)
    if (
        video.ndim != 4
        or video.shape[-1] != 3
        or video.dtype != np.uint8
        or len(video) < 2
    ):
        raise ValueError("Temporal metric requires >=2 uint8 RGB frames [T,H,W,3]")
    if not isinstance(max_dimension, int) or max_dimension < 16:
        raise ValueError("max_dimension must be an integer >=16")
    pairs = []
    for index in range(len(video) - 1):
        try:
            row = _flow_pair(video[index], video[index + 1], max_dimension)
        except Exception as exc:
            row = {
                "status": "failed",
                "l1": 1.0,
                "valid_fraction": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        pairs.append({"from_frame": index, "to_frame": index + 1, **row})
    failures = sum(row["status"] != "ok" for row in pairs)
    return {
        "status": "ok" if not failures else "partial_or_failed",
        "evidence": "CPU_optical_flow_pixel_proxy",
        "method": "OpenCV_DIS_MEDIUM_bidirectional",
        "max_dimension": max_dimension,
        "expected_pairs": len(video) - 1,
        "successful_pairs": len(pairs) - failures,
        "failed_pairs": failures,
        "mean_l1": float(np.mean([row["l1"] for row in pairs])),
        "mean_valid_fraction": float(np.mean([row["valid_fraction"] for row in pairs])),
        "failure_penalty_l1": 1.0,
        "pairs": pairs,
        "occlusion_mask": "in_bounds and ||forward+sampled_backward|| <= 1 + 0.05*||forward|| pixels",
        "interpretation": "Motion-compensated temporal pixel consistency; static/blurred videos can score well, so pose and observed parallax remain required",
    }


class DINOv2SemanticScorer:
    """Lazy, frozen official HF DINOv2-small; no weights load at construction."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        batch_size: int = 8,
        local_files_only: bool = False,
    ):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("DINO batch_size must be positive")
        self.device, self.batch_size, self.local_files_only = (
            device,
            batch_size,
            local_files_only,
        )
        self.model = self.processor = None

    def _load(self):
        if self.model is None:
            from transformers import AutoImageProcessor, AutoModel

            self.processor = AutoImageProcessor.from_pretrained(
                DINO_MODEL,
                revision=DINO_REVISION,
                use_fast=False,
                local_files_only=self.local_files_only,
            )
            self.model = AutoModel.from_pretrained(
                DINO_MODEL,
                revision=DINO_REVISION,
                local_files_only=self.local_files_only,
                use_safetensors=True,
                trust_remote_code=False,
            )
            self.model.eval().requires_grad_(False).to(self.device)

    def score(
        self, reference: Any, frames: Any, *, frame_stride: int = 4
    ) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        reference = _rgb(reference)
        video = np.asarray(frames)
        if (
            video.ndim != 4
            or video.shape[-1] != 3
            or video.dtype != np.uint8
            or not len(video)
        ):
            raise ValueError("DINO scoring requires uint8 RGB video")
        if not isinstance(frame_stride, int) or frame_stride < 1:
            raise ValueError("frame_stride must be positive")
        indices = sorted(set(range(0, len(video), frame_stride)) | {len(video) - 1})
        self._load()
        images = [reference] + [video[index] for index in indices]
        embeddings = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                inputs = self.processor(
                    images=images[start : start + self.batch_size], return_tensors="pt"
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                features = self.model(**inputs).last_hidden_state[:, 0].float()
                if not torch.isfinite(features).all():
                    raise RuntimeError("DINO returned nonfinite CLS features")
                embeddings.append(F.normalize(features, dim=-1).cpu())
        features = torch.cat(embeddings)
        cosines = torch.sum(features[1:] * features[:1], dim=-1).clamp(-1, 1).tolist()
        return {
            "status": "ok",
            "evidence": "frozen_image_semantic_feature_similarity",
            "model": DINO_MODEL,
            "revision": DINO_REVISION,
            "frozen": True,
            "feature": "L2_normalized_last_hidden_state_CLS",
            "processor": "official_pinned_BitImageProcessor_shortest256_center224",
            "frame_indices": indices,
            "expected_sampled_frames": len(indices),
            "scored_frames": len(cosines),
            "cosines": cosines,
            "mean_cosine": float(np.mean(cosines)),
            "min_cosine": float(np.min(cosines)),
            "interpretation": "Input-to-frame semantic similarity; not a trained perceptual video-quality predictor",
        }

    def offload(self):
        model, self.model = self.model, None
        self.processor = None
        if model is not None:
            model.to("cpu")
        del model
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _decode_video(
    path: str | Path, expected_frames: int | None
) -> tuple[np.ndarray, dict[str, Any]]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open generated video")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or abs(fps - 24.0) > 0.01:
            raise RuntimeError(
                "Visual evaluation requires the declared 24 fps timeline"
            )
        frames = []
        while True:
            okay, frame = capture.read()
            if not okay:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames or expected_frames is not None and len(frames) != expected_frames:
        raise RuntimeError(f"Decoded {len(frames)} frames; expected {expected_frames}")
    return np.stack(frames), {
        "video_sha256": file_sha256(path),
        "decoded_frames": len(frames),
        "fps": fps,
        "decoder": "OpenCV",
    }


def evaluate_visuals(
    input_image: Any,
    video: Any,
    *,
    semantic: bool = True,
    dino_scorer: Any = None,
    device: str = "cuda",
    expected_frames: int | None = 97,
    frame_stride: int = 4,
    flow_max_dimension: int = 416,
) -> dict[str, Any]:
    """Compute actual decoded-video metrics; unavailable components stay failures."""
    from PIL import Image

    result: dict[str, Any] = {
        "schema_version": 1,
        "evidence": "preservation_and_temporal_proxies",
        "threshold_status": "predeclared_engineering_not_universally_validated",
        "thresholds": asdict(PreservationThresholds()),
    }
    try:
        reference = (
            np.asarray(Image.open(input_image).convert("RGB"))
            if isinstance(input_image, (str, Path))
            else _rgb(input_image)
        )
        if isinstance(video, (str, Path)):
            frames, metadata = _decode_video(video, expected_frames)
        else:
            frames, metadata = (
                np.asarray(video),
                {"source": "caller_supplied_RGB_frames"},
            )
            if (
                frames.ndim != 4
                or not len(frames)
                or frames.dtype != np.uint8
                or frames.shape[-1] != 3
                or expected_frames is not None
                and len(frames) != expected_frames
            ):
                raise ValueError("Generated RGB frame contract failed")
        result["video"] = metadata
    except Exception as exc:
        return {**result, "status": "failed", "failure": f"{type(exc).__name__}: {exc}"}
    try:
        result["first_frame"] = first_frame_preservation(reference, frames[0])
    except Exception as exc:
        result["first_frame"] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    result["temporal"] = temporal_flow_consistency(
        frames, max_dimension=flow_max_dimension
    )
    scorer = dino_scorer
    if semantic:
        try:
            scorer = scorer or DINOv2SemanticScorer(device=device)
            result["semantic"] = scorer.score(
                reference, frames, frame_stride=frame_stride
            )
        except Exception as exc:
            result["semantic"] = {
                "status": "failed",
                "model": DINO_MODEL,
                "revision": DINO_REVISION,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            if scorer is not None and hasattr(scorer, "offload"):
                try:
                    scorer.offload()
                except Exception as cleanup_error:
                    result.setdefault("semantic", {})["cleanup_error"] = (
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
    else:
        result["semantic"] = {
            "status": "not_run",
            "model": DINO_MODEL,
            "revision": DINO_REVISION,
            "mean_cosine": None,
        }
    result["status"] = (
        "ok"
        if all(
            result[key]["status"] == "ok"
            for key in ("first_frame", "temporal", "semantic")
        )
        else "partial_or_failed"
    )
    return result


def joint_acceptance(
    pose_metrics: dict[str, Any],
    visual_metrics: dict[str, Any],
    *,
    manual_review: dict[str, Any] | None = None,
    parallax: dict[str, Any] | None = None,
    thresholds: PreservationThresholds | None = None,
) -> dict[str, Any]:
    """Full acceptance is separate from evaluate.py's pose-only joint_success.

    If provided, human review needs approved=True, reviewer, matching video_sha256 and true
    camera_is_3d/fidelity_acceptable/temporal_quality_acceptable. Observed video
    parallax needs parallax_consistent=True, source='generated_video_tracks',
    matching video_sha256 and declared confidence >= threshold. Requested or
    analytic-reference geometry cannot meet this evidence requirement.
    """
    limits = thresholds or PreservationThresholds()
    first, temporal, semantic = (
        visual_metrics.get(key, {}) for key in ("first_frame", "temporal", "semantic")
    )

    def finite(value):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and bool(np.isfinite(value))
        )

    mean_cosine, ssim, temporal_l1, coverage = (
        semantic.get("mean_cosine"),
        first.get("ssim"),
        temporal.get("mean_l1"),
        temporal.get("mean_valid_fraction"),
    )
    video_hash = visual_metrics.get("video", {}).get("video_sha256")
    review, depth_tracks = manual_review or {}, parallax or {}
    checks = {
        "pose_success": pose_metrics.get("joint_success") is True,
        "pose_video_identity": bool(video_hash)
        and (
            pose_metrics.get("video_sha256")
            or pose_metrics.get("estimator", {})
            .get("provenance", {})
            .get("video_sha256")
        )
        == video_hash,
        "first_frame_preservation": first.get("status") == "ok"
        and finite(ssim)
        and ssim >= limits.min_first_frame_ssim,
        "semantic_preservation": semantic.get("status") == "ok"
        and finite(mean_cosine)
        and mean_cosine >= limits.min_dino_mean_cosine
        and semantic.get("scored_frames", 0)
        == semantic.get("expected_sampled_frames", -1),
        "temporal_consistency": temporal.get("status") == "ok"
        and temporal.get("failed_pairs", 1) == 0
        and temporal.get("successful_pairs", 0) == temporal.get("expected_pairs", -1)
        and finite(temporal_l1)
        and temporal_l1 <= limits.max_flow_compensated_temporal_l1,
        "temporal_coverage": finite(coverage)
        and coverage >= limits.min_flow_valid_fraction,
        "provided_review_valid": not manual_review
        or (
            bool(video_hash)
            and review.get("video_sha256") == video_hash
            and review.get("approved") is True
            and bool(review.get("reviewer"))
            and all(
                review.get(key) is True
                for key in (
                    "camera_is_3d",
                    "fidelity_acceptable",
                    "temporal_quality_acceptable",
                )
            )
        ),
        "observed_depth_parallax": bool(video_hash)
        and depth_tracks.get("video_sha256") == video_hash
        and depth_tracks.get("source") == "generated_video_tracks"
        and depth_tracks.get("parallax_consistent") is True
        and finite(depth_tracks.get("confidence"))
        and depth_tracks["confidence"] >= limits.min_parallax_confidence,
    }
    return {
        "schema_version": 1,
        "accepted": bool(all(checks.values())),
        "pose_success": checks["pose_success"],
        "checks": checks,
        "thresholds": asdict(limits),
        "threshold_status": "predeclared_engineering_not_universally_validated",
        "scope": "automated_camera_pose_preservation_temporal_and_observed_parallax",
        "qualitative_review_status": "provided"
        if manual_review
        else "not_provided_optional",
        "video_sha256": video_hash,
        "unmet_requirements": [name for name, okay in checks.items() if not okay],
    }
