"""Decode two frozen-cache clips with Wan VAE and compare to exact RGB targets.

Read-only with respect to caches, training state, checkpoints, and model weights.
The script requires the already-cached pinned Wan VAE and never downloads it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image


MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
REVISION = "b8fff7315c768468a5333511427288870b2e9635"
SOURCE_ID = "coco2017val_000000109916_ann710699"
EXPECTED_TARGET_CACHE_SHA256 = {
    "zoom_in": "028f3d95af8218a4631d76c638b8f6c0a9ac286034f3531dca58e2191785463a",
    "static": "8508623987d7ad2896e207aa679aebe6a7714cf86e666ea62517352de481e910",
}
SAMPLE_FRAMES = (0, 40, 80, 120)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_png(path: Path, pixels: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(path.stem + ".partial" + path.suffix)
    Image.fromarray(pixels, mode="RGB").save(temp, format="PNG")
    os.replace(temp, path)


def _frame_metrics(
    target: np.ndarray, decoded_float: np.ndarray, decoded_u8: np.ndarray
) -> dict:
    expected = target.astype(np.float32) / 255.0
    error = decoded_float.astype(np.float32) - expected
    mse = float(np.mean(error * error))
    return {
        "psnr_rgb_db": None if mse == 0 else float(10 * np.log10(1.0 / mse)),
        "mse_rgb_0_1": mse,
        "target_rgb_mean_0_255": float(target.mean()),
        "target_rgb_std_0_255": float(target.astype(np.float32).std()),
        "decoded_rgb_mean_0_255": float(decoded_u8.astype(np.float32).mean()),
        "decoded_rgb_std_0_255": float(decoded_u8.astype(np.float32).std()),
        "decoded_rgb_channel_std_0_255": decoded_u8.astype(np.float32)
        .std(axis=(0, 1))
        .tolist(),
    }


def verify(root: Path) -> dict:
    root = root.resolve()
    index_path = root / "data/flow_poc/cache_1280x704_f121_v1/index.jsonl"
    manifest_path = root / "data/flow_poc/pilot_sources.json"
    output_dir = root / "artifacts/flow_cache_roundtrip_audit"
    report_path = output_dir / "audit.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed audit: {report_path}")
    if not index_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Required frozen cache index or source manifest is missing"
        )

    from fpcamera.flow_cache import NORMALIZATION, load_cached
    from fpcamera.flow_geometry import render_source
    from fpcamera.runtime import gpu_lease
    from fpcamera.wan import require_pinned_runtime, versions

    index_sha = sha256(index_path)
    manifest_sha = sha256(manifest_path)
    rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_records = json.loads(manifest_path.read_text(encoding="utf-8"))["records"]
    manifest_by_id = {record["source_id"]: record for record in source_records}
    source_record = manifest_by_id.get(SOURCE_ID)
    if source_record is None:
        raise ValueError(f"Source id {SOURCE_ID} is absent from pilot_sources.json")

    selected = {}
    for mode in ("zoom_in", "static"):
        candidates = [
            row
            for row in rows
            if row.get("source_id") == SOURCE_ID
            and row.get("split") == "development"
            and row.get("mode") == mode
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one dev cache row for {SOURCE_ID}/{mode}; found {len(candidates)}"
            )
        row = candidates[0]
        expected_hash = EXPECTED_TARGET_CACHE_SHA256[mode]
        if row.get("sha256") != expected_hash:
            raise ValueError(
                f"Frozen target-cache SHA256 guard failed for {mode}: {row.get('sha256')}"
            )
        if row.get("source_sha256") != source_record.get("original_image_sha256"):
            raise ValueError(
                f"Source image SHA256 differs between cache index and source manifest ({mode})"
            )
        selected[mode] = row

    output_files = [report_path]
    for mode in selected:
        for frame in SAMPLE_FRAMES:
            output_files.extend(
                (
                    output_dir / f"{mode}_cache_vae_audit_target_{frame:03d}.png",
                    output_dir / f"{mode}_cache_vae_audit_decoded_{frame:03d}.png",
                )
            )
    existing = [str(path) for path in output_files if path.exists()]
    if existing:
        raise FileExistsError("Refusing existing audit outputs: " + ", ".join(existing))

    source_path = root / "data/flow_poc/sources" / source_record["source_image"]
    if sha256(source_path) != source_record["original_image_sha256"]:
        raise ValueError("Original source photo no longer matches source manifest")

    cache_data = {}
    targets = {}
    cache_metadata = {}
    for mode, row in selected.items():
        item = load_cached(row, root)
        if not __import__("torch").equal(item["latents"][:, :1], item["image_latent"]):
            raise ValueError(
                f"First-frame latent differs from image-only reference for {mode}"
            )
        cache_path = root / row["cache_path"]
        from safetensors import safe_open

        with safe_open(str(cache_path), framework="pt", device="cpu") as archive:
            metadata = json.loads(archive.metadata()["metadata_json"])
        if metadata.get("model") != MODEL or metadata.get("revision") != REVISION:
            raise ValueError(f"Cache model/revision metadata mismatch for {mode}")
        if metadata.get("normalization") != NORMALIZATION:
            raise ValueError(f"Cache normalization metadata mismatch for {mode}")
        if (
            metadata.get("synthetic_camera_video") is not True
            or metadata.get("future_rgb_used_for_condition") is not False
        ):
            raise ValueError(
                f"Cache does not carry the expected no-future-RGB conditioning audit for {mode}"
            )
        crop = metadata.get("crop") or {}
        overscan = float(crop.get("overscan", 0))
        if overscan != 1.35:
            raise ValueError(
                f"Expected the recorded 1.35 source overscan for {mode}; got {overscan}"
            )
        cached_camera = np.asarray(item["homographies"], dtype=np.float64)
        metadata_camera = np.asarray(
            metadata["camera"]["homographies_first_to_current"], dtype=np.float64
        )
        # JSON records original float64 matrices; the saved conditioning tensor
        # deliberately uses float32. Require that exact declared cast, rather
        # than a tolerance smaller than representable float32 translations.
        if not np.array_equal(
            cached_camera.astype(np.float32), metadata_camera.astype(np.float32)
        ):
            raise ValueError(
                f"Cached homographies and immutable source metadata disagree for {mode}"
            )
        pixels, _crop = render_source(
            source_path, metadata_camera, item["contract"], overscan=overscan
        )
        if (
            pixels.shape
            != (
                item["contract"].frames,
                item["contract"].height,
                item["contract"].width,
                3,
            )
            or pixels.dtype != np.uint8
        ):
            raise ValueError(
                f"Rendered source targets do not match the RGB contract for {mode}"
            )
        cache_data[mode] = item
        targets[mode] = pixels
        cache_metadata[mode] = metadata

    import torch

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    runtime = require_pinned_runtime()
    from diffusers import AutoencoderKLWan

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "verified_cache_vae_roundtrip_diagnostic",
        "purpose": "Compare cached training targets with FP32 Wan VAE decode of saved normalized latents; not a generated-video evaluation.",
        "model": MODEL,
        "revision": REVISION,
        "runtime_versions": runtime,
        "cache_index_path": str(index_path),
        "cache_index_sha256": index_sha,
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": manifest_sha,
        "source_id": SOURCE_ID,
        "source_image_path": str(source_path),
        "source_image_sha256": source_record["original_image_sha256"],
        "normalization": NORMALIZATION,
        "vae_dtype": "float32",
        "model_loading": "AutoencoderKLWan only; no transformer or T5 loaded",
        "offline_weights_only": True,
        "frames_checked": list(SAMPLE_FRAMES),
        "cases": {},
        "decode_postprocess": "Wan pipeline inverse normalization z*latents_std+latents_mean; VAE decode; clamp(video/2+0.5,0,1); RGB NTHWC float; saved PNG uses round(x*255).",
    }
    started = time.monotonic()
    with gpu_lease(root):
        vae = (
            AutoencoderKLWan.from_pretrained(
                MODEL,
                subfolder="vae",
                revision=REVISION,
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            .to("cuda")
            .eval()
        )
        vae.requires_grad_(False)
        vae.disable_tiling()
        if any(parameter.dtype != torch.float32 for parameter in vae.parameters()):
            raise RuntimeError("Loaded VAE is not fully FP32")
        means = torch.tensor(
            vae.config.latents_mean, device="cuda", dtype=torch.float32
        ).view(1, -1, 1, 1, 1)
        stds = torch.tensor(
            vae.config.latents_std, device="cuda", dtype=torch.float32
        ).view(1, -1, 1, 1, 1)
        for mode, row in selected.items():
            item = cache_data[mode]
            contract = item["contract"]
            normalized = (
                item["latents"].unsqueeze(0).to(device="cuda", dtype=torch.float32)
            )
            unnormalized = normalized * stds + means
            with (
                torch.inference_mode(),
                torch.autocast(device_type="cuda", enabled=False),
            ):
                decoded = vae.decode(unnormalized, return_dict=False)[0]
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
            if decoded.ndim != 5 or tuple(decoded.shape) != (
                1,
                3,
                contract.frames,
                contract.height,
                contract.width,
            ):
                raise RuntimeError(
                    f"VAE decoded shape mismatch for {mode}: {tuple(decoded.shape)}"
                )
            decoded_rgb = decoded[0].permute(1, 2, 3, 0).float().cpu().numpy()
            decoded_u8 = np.uint8(np.rint(np.clip(decoded_rgb, 0, 1) * 255))
            if (
                decoded_rgb.shape != targets[mode].shape
                or decoded_u8.shape != targets[mode].shape
            ):
                raise RuntimeError(
                    f"Decoded RGB dimensions differ from cached targets for {mode}"
                )
            frames_report = {}
            for frame in SAMPLE_FRAMES:
                target_path = (
                    output_dir / f"{mode}_cache_vae_audit_target_{frame:03d}.png"
                )
                decoded_path = (
                    output_dir / f"{mode}_cache_vae_audit_decoded_{frame:03d}.png"
                )
                _atomic_png(target_path, targets[mode][frame])
                _atomic_png(decoded_path, decoded_u8[frame])
                frames_report[str(frame)] = {
                    **_frame_metrics(
                        targets[mode][frame], decoded_rgb[frame], decoded_u8[frame]
                    ),
                    "target_png": str(target_path),
                    "decoded_png": str(decoded_path),
                }
            report["cases"][mode] = {
                "source_id": row["source_id"],
                "split": row["split"],
                "cache_record_id": row["record_id"],
                "cache_path": row["cache_path"],
                "cache_sha256": sha256(root / row["cache_path"]),
                "expected_cache_sha256": EXPECTED_TARGET_CACHE_SHA256[mode],
                "reference_cache_sha256": row["reference_sha256"],
                "video_contract": contract.record(),
                "decoded_frame_count": int(decoded_rgb.shape[0]),
                "decoded_dimensions": [
                    int(decoded_rgb.shape[2]),
                    int(decoded_rgb.shape[1]),
                ],
                "target_dimensions": [contract.width, contract.height],
                "camera_mode": mode,
                "source_overscan": 1.35,
                "frames": frames_report,
            }
            del normalized, unnormalized, decoded, decoded_rgb, decoded_u8
            torch.cuda.empty_cache()
        del vae
        torch.cuda.empty_cache()
    report["seconds"] = time.monotonic() - started
    report["report_path"] = str(report_path)
    temp_report = report_path.with_suffix(".partial")
    temp_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temp_report, report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    report = verify(args.project_root)
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": list(report["cases"]),
                "report": report["report_path"],
                "seconds": report["seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
