"""Immutable synthetic camera-motion caches at an explicit Wan I2V contract."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file, save_file

from .flow_geometry import VideoContract, camera_path, render_source
from .runtime import file_sha256, write_json, gpu_lease
from .wan import MODEL, REVISION, load_encoders, encode_text

NORMALIZATION = "native_fp32_vae_mode_normalize_then_bf16"


# # Native VAE encoding and cache-file integrity helpers.
def encode_rgb(vae, rgb):
    """Use the native VAE path for one image or any explicitly valid4k+1 video."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError("Expected uint8[F,H,W,3] RGB")
    frames, height, width, _ = rgb.shape
    if (frames - 1) % 4 or height % 32 or width % 32:
        raise ValueError("RGB and Wan VAE contracts disagree")
    device = next(vae.parameters()).device
    x = (
        torch.from_numpy(rgb.copy())
        .permute(3, 0, 1, 2)[None]
        .to(device=device, dtype=torch.float32)
    )
    x = x / 127.5 - 1.0
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        z = vae.encode(x).latent_dist.mode().float()
        mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        z = (z - mean) / std
    if (
        z.shape != (1, 48, (frames - 1) // 4 + 1, height // 16, width // 16)
        or not torch.isfinite(z).all()
    ):
        raise RuntimeError("Invalid native encoded latent")
    return z[0].to(device="cpu", dtype=torch.bfloat16).contiguous()


def _save_tensor_file(path, tensors, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_suffix(".partial")
    save_file(
        tensors,
        str(temp),
        metadata={
            "flow_cache_schema": "1",
            "metadata_json": json.dumps(metadata, sort_keys=True),
        },
    )
    os.replace(temp, path)


def load_cached(row, root):
    root = Path(root)
    path = root / row["cache_path"]
    reference = root / row["reference_path"]
    if (
        file_sha256(path) != row["sha256"]
        or file_sha256(reference) != row["reference_sha256"]
    ):
        raise RuntimeError("Flow cache or source reference changed")
    data = load_file(str(path), device="cpu")
    refs = load_file(str(reference), device="cpu")
    contract = VideoContract(**row["video_contract"])
    if tuple(data["latents"].shape) != contract.latent_shape or tuple(
        refs["image_latent"].shape
    ) != (48, 1, contract.height // 16, contract.width // 16):
        raise ValueError("Cached tensors have the wrong contract")
    if data["homographies"].shape != (contract.frames, 3, 3):
        raise ValueError("Camera path timing mismatch")
    if any(not torch.isfinite(x).all() for x in (*data.values(), *refs.values())):
        raise ValueError("Non-finite flow training cache")
    return {**data, **refs, "row": row, "contract": contract}


# # Index-driven cache preparation for one explicit video contract.
def prepare(
    root,
    *,
    frames=121,
    max_new_clips=24,
    modes=("zoom_in", "zoom_out", "static"),
    source_limit=None,
    width=1280,
    height=704,
):
    root = Path(root)
    contract = VideoContract(width=width, height=height, frames=frames)
    manifest_path = root / "data/flow_poc/pilot_sources.json"
    manifest = json.loads(manifest_path.read_text())
    records = manifest["records"]
    if source_limit is not None:
        records = [
            x
            for split in ("train", "development")
            for x in [r for r in records if r["split"] == split][:source_limit]
        ]
    cache_root = root / f"data/flow_poc/cache_{width}x{height}_f{frames}_v1"
    cache_root.mkdir(parents=True, exist_ok=True)
    index = cache_root / "index.jsonl"
    rows = (
        [json.loads(x) for x in index.read_text().splitlines() if x.strip()]
        if index.exists()
        else []
    )
    existing = {x["record_id"]: x for x in rows}
    prepared = 0
    started = time.monotonic()
    source_hashes = {
        name: file_sha256(Path(__file__).with_name(name))
        for name in ("flow_cache.py", "flow_geometry.py", "wan.py")
    }
    with gpu_lease(root):
        pipe = load_encoders(log_path=cache_root / "encoder.log", local_files_only=True)
        captions = {r["training_caption"] for r in records}
        text_features = {c: encode_text(pipe, c) for c in captions}
        negative_text = encode_text(pipe, "")
        pipe.text_encoder.to("cpu")
        pipe.text_encoder = None
        gc.collect()
        torch.cuda.empty_cache()
        for source in records:
            image_path = root / "data/flow_poc/sources" / source["source_image"]
            if file_sha256(image_path) != source["original_image_sha256"]:
                raise ValueError("Source photograph digest mismatch")
            reference_path = (
                cache_root / "references" / f"{source['source_image_id']}.safetensors"
            )
            for mode in modes:
                path, camera = camera_path(mode, contract)
                identity = {
                    "source_sha256": source["original_image_sha256"],
                    "video_contract": asdict(contract),
                    "mode": mode,
                    "camera": camera,
                    "model": MODEL,
                    "revision": REVISION,
                    "source_hashes": source_hashes,
                    "normalization": NORMALIZATION,
                    "caption": source["training_caption"],
                }
                key = hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode()
                ).hexdigest()
                if key in existing:
                    load_cached(existing[key], root)
                    continue
                if prepared >= max_new_clips:
                    return {
                        "status": "encoding_stage_complete",
                        "new_clips": prepared,
                        "total_clips": len(rows),
                        "index": str(index),
                        "seconds": time.monotonic() - started,
                    }
                pixels, crop = render_source(image_path, path, contract)
                first_rgb = pixels[0]
                if not reference_path.exists():
                    first = encode_rgb(pipe.vae, first_rgb[None])
                    _save_tensor_file(
                        reference_path,
                        {
                            "image_latent": first,
                            "text_embeddings": text_features[
                                source["training_caption"]
                            ],
                            "negative_text_embeddings": negative_text,
                        },
                        {
                            "source_sha256": source["original_image_sha256"],
                            "normalization": NORMALIZATION,
                            "model": MODEL,
                            "revision": REVISION,
                            "caption": source["training_caption"],
                        },
                    )
                    Image.fromarray(first_rgb).save(reference_path.with_suffix(".png"))
                else:
                    first = load_file(str(reference_path), device="cpu")["image_latent"]
                z = encode_rgb(pipe.vae, pixels)
                if not torch.equal(z[:, :1], first):
                    raise RuntimeError(
                        "Video first latent differs from image-only I2V condition"
                    )
                cache_path = cache_root / "videos" / f"{key}.safetensors"
                meta = {
                    **identity,
                    "source_id": source["source_id"],
                    "split": source["split"],
                    "crop": crop,
                    "synthetic_camera_video": True,
                    "future_rgb_used_for_condition": False,
                }
                if not cache_path.exists():
                    _save_tensor_file(
                        cache_path,
                        {"latents": z, "homographies": torch.from_numpy(path).float()},
                        meta,
                    )
                row = {
                    "record_id": key,
                    "source_id": source["source_id"],
                    "source_image_id": source["source_image_id"],
                    "source_sha256": source["original_image_sha256"],
                    "split": source["split"],
                    "mode": mode,
                    "video_contract": asdict(contract),
                    "cache_path": cache_path.relative_to(root).as_posix(),
                    "reference_path": reference_path.relative_to(root).as_posix(),
                    "sha256": file_sha256(cache_path),
                    "reference_sha256": file_sha256(reference_path),
                    "source_hashes": source_hashes,
                    "normalization": NORMALIZATION,
                    "caption": source["training_caption"],
                }
                load_cached(row, root)
                with index.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                rows.append(row)
                existing[key] = row
                prepared += 1
                write_json(
                    cache_root / "progress.json",
                    {
                        "new_clips": prepared,
                        "total_clips": len(rows),
                        "last_mode": mode,
                        "last_source": source["source_id"],
                        "seconds": time.monotonic() - started,
                    },
                )
                print(
                    json.dumps(
                        {
                            "encoded": prepared,
                            "total": len(rows),
                            "mode": mode,
                            "source": source["source_id"],
                        }
                    ),
                    flush=True,
                )
                del pixels, z
                gc.collect()
                torch.cuda.empty_cache()
    return {
        "status": "all_requested_sources_encoded",
        "new_clips": prepared,
        "total_clips": len(rows),
        "index": str(index),
        "seconds": time.monotonic() - started,
    }


# # Command-line entry point for cache preparation.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--max-new-clips", type=int, default=24)
    parser.add_argument("--source-limit", type=int)
    parser.add_argument("--modes", nargs="+", default=["zoom_in", "zoom_out", "static"])
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.project_root,
                frames=args.frames,
                max_new_clips=args.max_new_clips,
                modes=args.modes,
                source_limit=args.source_limit,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
