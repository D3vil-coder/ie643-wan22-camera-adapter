"""Pinned, frozen Wan TI2V loading and faithful conditioning encoders.

This module has no dependency on any earlier project adapter or checkpoint.
Heavy libraries are imported lazily after notebook-safe log/progress setup.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import os
from pathlib import Path

import numpy as np

MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
REVISION = "b8fff7315c768468a5333511427288870b2e9635"
MAX_TEXT_LENGTH = 512
TEXT_WIDTH = 4096


def versions() -> dict:
    """Installed versions of the pinned runtime libraries (None if a package is missing)."""
    result = {}
    for name in (
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "numpy",
        "Pillow",
        "huggingface-hub",
        "marimo",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def require_pinned_runtime() -> dict:
    """Raise unless diffusers 0.36.0, transformers 4.57.6 and torch 2.11.x are installed (the exact versions the
    adapter was trained and verified with); returns the version dict.
    """
    current = versions()
    errors = []
    for name, required in (("diffusers", "0.36.0"), ("transformers", "4.57.6")):
        if current[name] != required:
            errors.append(f"{name}={current[name]!r}, required {required}")
    if not str(current["torch"]).split("+")[0].startswith("2.11."):
        errors.append(f"torch={current['torch']!r}, required 2.11.x")
    if errors:
        raise RuntimeError("Pinned Wan runtime mismatch: " + "; ".join(errors))
    return current


def _route_logs(destination: str | Path) -> Path:
    """Detach cached notebook stream writers, retaining library logs on disk."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for prefix in (
        "diffusers",
        "transformers",
        "huggingface_hub",
        "accelerate",
        "moge",
        "vggt",
    ):
        parent = logging.getLogger(prefix)
        names = [prefix] + [
            name
            for name in list(logging.Logger.manager.loggerDict)
            if name.startswith(prefix + ".")
        ]
        for name in names:
            logger = logging.getLogger(name)
            for handler in list(logger.handlers):
                if isinstance(handler, logging.StreamHandler):
                    logger.removeHandler(handler)
                    if getattr(handler, "_fpcamera_owned", False):
                        handler.close()
            if name != prefix:
                logger.propagate = True
        handler = logging.FileHandler(destination, encoding="utf-8")
        handler._fpcamera_owned = True
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        parent.addHandler(handler)
        parent.propagate = False
    return destination


def configure_library_output(
    log_path: str | Path = "artifacts/runtime/library.log",
) -> Path:
    """Silence progress bars and route library logs to a file *before* heavy imports (notebook safe); returns the log path."""
    # Set environment flags before any heavy module import or model load.
    # environment flags must be set before any heavy library import so notebooks do not get flooded by progress bars.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    destination = _route_logs(log_path)
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    from transformers.utils import logging as transformer_logging

    transformer_logging.disable_progress_bar()
    from diffusers.utils import logging as diffusion_logging

    diffusion_logging.disable_progress_bar()
    # Imports may install default handlers; replace those before loading models.
    _route_logs(destination)
    return destination


def freeze_module(module):
    """Put a module in eval mode and disable all gradients."""
    module.eval()
    module.requires_grad_(False)
    return module


def load_encoders(device="cuda", log_path=None, local_files_only=False):
    """Load the exact frozen native VAE/T5 path without unused Wan weights."""
    configure_library_output(log_path or "artifacts/runtime/encoder_library.log")
    require_pinned_runtime()
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for frozen encoder preparation")
    vae = AutoencoderKLWan.from_pretrained(
        MODEL,
        subfolder="vae",
        revision=REVISION,
        torch_dtype=torch.float32,
        local_files_only=local_files_only,
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL,
        revision=REVISION,
        vae=vae,
        transformer=None,
        image_encoder=None,
        image_processor=None,
        expand_timesteps=True,
        torch_dtype=torch.bfloat16,
        local_files_only=local_files_only,
    )
    for module in (pipe.vae, pipe.text_encoder):
        freeze_module(module)
    if (
        pipe.transformer is not None
        or pipe.vae_scale_factor_spatial != 16
        or pipe.vae_scale_factor_temporal != 4
    ):
        raise RuntimeError("Unexpected encoder-only TI2V configuration")
    pipe.vae.disable_tiling()
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    if any(p.dtype != torch.float32 for p in pipe.vae.parameters()):
        raise RuntimeError("Frozen native VAE must remain FP32")
    pipe._fpcamera_provenance = {
        "model": MODEL,
        "revision": REVISION,
        "versions": versions(),
        "expand_timesteps": True,
        "vae_dtype": "float32",
        "encoder_only": True,
        "transformer_weights_loaded": False,
    }
    return pipe


def load_pipeline(
    device="cuda",
    flow_shift: float = 3.0,
    log_path=None,
    strict_versions: bool = True,
    local_files_only: bool = False,
):
    """Load the pinned native TI2V path, retaining original FP32 exceptions.

    No pipeline-wide dtype cast is performed after loading: diffusers' original
    FP32 module exceptions survive, and the separately loaded VAE stays FP32.
    """
    configure_library_output(log_path or "artifacts/runtime/library.log")
    if strict_versions:
        require_pinned_runtime()
    import torch
    from diffusers import (
        AutoencoderKLWan,
        UniPCMultistepScheduler,
        WanImageToVideoPipeline,
    )

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; stopped before loading frozen Wan weights"
        )
    if not np.isfinite(flow_shift) or flow_shift <= 0:
        raise ValueError("flow_shift must be finite and positive")
    vae = AutoencoderKLWan.from_pretrained(
        MODEL,
        subfolder="vae",
        revision=REVISION,
        torch_dtype=torch.float32,
        local_files_only=local_files_only,
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL,
        revision=REVISION,
        vae=vae,
        torch_dtype=torch.bfloat16,
        image_encoder=None,
        image_processor=None,
        expand_timesteps=True,
        local_files_only=local_files_only,
    )
    for module in (pipe.vae, pipe.text_encoder, pipe.transformer):
        freeze_module(module)
    if getattr(pipe, "transformer_2", None) is not None:
        raise RuntimeError("Pinned TI2V 5B must have a single transformer")
    if pipe.transformer.config.in_channels != 48 or not pipe.config.expand_timesteps:
        raise RuntimeError("Unexpected Wan TI2V architecture or conditioning path")
    if pipe.vae_scale_factor_spatial != 16 or pipe.vae_scale_factor_temporal != 4:
        raise RuntimeError("Unexpected Wan VAE compression factors")
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=float(flow_shift)
    )
    pipe.vae.disable_tiling()
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)  # Device only: preserve original FP32 modules and VAE.
    if any(p.dtype != torch.float32 for p in pipe.vae.parameters()):
        raise RuntimeError("Frozen Wan VAE must remain FP32")
    pipe._fpcamera_provenance = {
        "model": MODEL,
        "revision": REVISION,
        "versions": versions(),
        "flow_shift": float(flow_shift),
        "expand_timesteps": True,
        "image_encoder": None,
        "image_processor": None,
        "vae_dtype": "float32",
        "weight_loading_dtype": "bfloat16_with_original_fp32_exceptions",
    }
    return pipe


def hash_parameters(parameters, *, chunk_bytes: int = 64 << 20) -> str:
    """Hash an iterable of named tensors or tensors without staging whole models."""
    import torch

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    for index, entry in enumerate(parameters):
        name, tensor = entry if isinstance(entry, tuple) else (str(index), entry)
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}\n".encode())
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
        for start in range(0, raw.numel(), chunk_bytes):
            digest.update(raw[start : start + chunk_bytes].cpu().numpy().tobytes())
    return digest.hexdigest()


def hash_module(
    module,
    parameters=None,
    *,
    only_frozen: bool = False,
    include_buffers: bool = False,
    chunk_bytes: int = 64 << 20,
) -> str:
    """sha256 over a module's parameters (chunked, no whole-model staging); used to prove the frozen backbone
    never changed during training or inference.
    """
    def values():
        """Yield the (ordered) parameters to hash, optionally only frozen ones and/or buffers.  Hashing by index, not name, is
        deliberate: adapter wrappers rename modules but keep the same ordered original Parameter objects.
        """
        # Index identity is deliberate: adapter wrappers may change module names
        # while retaining the exact same ordered original Parameter objects.
        original = module.parameters() if parameters is None else parameters
        for parameter in original:
            if not only_frozen or not parameter.requires_grad:
                yield parameter
        if include_buffers:
            yield from module.buffers()

    return hash_parameters(values(), chunk_bytes=chunk_bytes)


frozen_digest = hash_parameters


def _device(module):
    return next(module.parameters()).device


def _encode_rgb(pipe, rgb: np.ndarray):
    """Encode 1 or 97 uint8 RGB frames with the frozen native VAE (FP32 posterior mode, normalised in FP32 then cast
    to bf16; see ``flow_cache.encode_rgb`` for the v2 cache path).  Returns ``[48,T,h,w]`` on the CPU.
    """
    import torch

    value = np.asarray(rgb)
    if value.ndim != 4 or value.shape[-1] != 3 or value.dtype != np.uint8:
        raise ValueError("RGB input must be uint8 [F,H,W,3]")
    frames, height, width, _ = value.shape
    if frames not in (1, 97) or height % 32 or width % 32:
        raise ValueError(
            "Expected 1 or 97 RGB frames and canvas dimensions divisible by 32"
        )
    device = _device(pipe.vae)
    video = (
        torch.from_numpy(value.copy())
        .permute(3, 0, 1, 2)[None]
        .to(device=device, dtype=torch.float32)
    )
    video = video / 127.5 - 1.0
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        raw = pipe.vae.encode(video).latent_dist.mode()
    # The pinned native pipeline passes dtype=float32 to prepare_latents.
    # Normalize in FP32, then cast the model input/cache to BF16. Casting the
    # posterior before normalization introduces a second rounding error.
    raw = raw.float()
    mean = torch.tensor(
        pipe.vae.config.latents_mean, device=device, dtype=raw.dtype
    ).view(1, -1, 1, 1, 1)
    inverse_std = 1.0 / torch.tensor(
        pipe.vae.config.latents_std, device=device, dtype=raw.dtype
    ).view(1, -1, 1, 1, 1)
    result = (raw - mean) * inverse_std
    expected = (1, 48, (frames - 1) // 4 + 1, height // 16, width // 16)
    if tuple(result.shape) != expected or not torch.isfinite(result).all():
        raise RuntimeError(
            f"Invalid Wan latent shape/values: {tuple(result.shape)}, expected {expected}"
        )
    return result[0].detach().to(dtype=torch.bfloat16, device="cpu").contiguous()


def encode_video(pipe, rgb_frames):
    """Encode a 97-frame uint8 RGB clip (the legacy training contract) to normalised latents ``[48,25,h,w]``."""
    if np.asarray(rgb_frames).shape[0] != 97:
        raise ValueError("Training video must contain exactly 97 selected RGB frames")
    return _encode_rgb(pipe, rgb_frames)


def encode_first_image(pipe, first_rgb):
    """Encode only the first selected photo; no future-image conditioning."""
    image = np.asarray(first_rgb)
    if image.ndim != 3:
        raise ValueError("First image must be RGB [H,W,3]")
    return _encode_rgb(pipe, image[None])


def encode_text(pipe, caption: str):
    """Store only valid T5 rows, after checking exact native zero padding."""
    import torch

    if not isinstance(caption, str):
        raise ValueError(
            "Caption must be a string; the empty string is valid CFG conditioning"
        )
    method = pipe._get_t5_prompt_embeds
    globals_ = getattr(method, "__func__", method).__globals__
    if "prompt_clean" not in globals_:
        raise RuntimeError(
            "Cannot identify the pinned pipeline's exact prompt cleaning function"
        )
    cleaned = globals_["prompt_clean"](caption)
    tokens = pipe.tokenizer(
        [cleaned],
        padding="max_length",
        max_length=MAX_TEXT_LENGTH,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    length = int(tokens.attention_mask.gt(0).sum())
    if not 1 <= length <= MAX_TEXT_LENGTH:
        raise RuntimeError("Invalid T5 attention-mask length")
    with torch.no_grad():
        embeds = method(
            prompt=caption,
            num_videos_per_prompt=1,
            max_sequence_length=MAX_TEXT_LENGTH,
            device=_device(pipe.text_encoder),
            dtype=torch.bfloat16,
        )
    if tuple(embeds.shape) != (1, MAX_TEXT_LENGTH, TEXT_WIDTH):
        raise RuntimeError(f"Unexpected T5 embedding shape: {tuple(embeds.shape)}")
    if (
        torch.count_nonzero(embeds[:, length:]).item()
        or not torch.isfinite(embeds).all()
    ):
        raise RuntimeError("Pinned T5 zero-padding/finite-output contract failed")
    return (
        embeds[0, :length].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    )


def pad_text_embeddings(embeddings, max_length: int = MAX_TEXT_LENGTH):
    """Reconstruct the native exact-zero padded [512,4096] T5 tensor."""
    import torch

    if (
        embeddings.ndim != 2
        or embeddings.shape[1] != TEXT_WIDTH
        or not 1 <= embeddings.shape[0] <= max_length
    ):
        raise ValueError("Expected unpadded T5 [valid_tokens,4096] embeddings")
    if max_length != MAX_TEXT_LENGTH:
        raise ValueError("Pinned Wan uses exactly 512 text tokens")
    result = embeddings.new_zeros((max_length, TEXT_WIDTH))
    result[: embeddings.shape[0]] = embeddings
    return result
