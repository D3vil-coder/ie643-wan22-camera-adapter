"""Single-image camera-controlled generation with the trained v2 adapter (the inference entry point).

Pipeline position: after training and before measurement.  ``generate`` takes one image plus a
camera request (v2: ``motion`` + ``strength`` s, or a physical amplitude; v1: ``mode`` +
``magnitude``; or a custom homography path) and produces ``video.mp4`` and a ``run.json`` ledger.
Steps inside ``generate``:

1. Resolve and verify the checkpoint (sha256 + unchanged-backbone audit; v1 sidecar or v2
   trainer manifest, ``resolve_checkpoint_meta``) and the guide fill mode (``replicate`` for v2).
2. Build the camera path: s --calibration--> u --``camera_path_v2``--> homographies
   (``build_camera_request``; see ``strength.py``), then the 51-channel guide for the requested
   motion and, when needed, for a static path (``flow_control.build_image_condition``).
3. Attach ``FlowControlLoRA`` and run the native Wan pipeline with the transformer forward wrapped
   by ``flow_anchor.frozen_camera_difference`` (teacher-anchored residual, the v2 default) or
   ``flow_guidance.camera_prediction_guidance`` (legacy).  The adapter is only active while the
   noise level sigma >= ``camera_min_sigma`` (0.6 for v2).
4. Write ``video.mp4`` (decode contract verified), key frames, ``run.json`` with all settings,
   timings and hashes; verify the frozen backbone hash did not change.

Command line (needs the GPU lease; the frozen base lives in the HF cache, offline)::

    python -m fpcamera.flow_infer --project-root . --checkpoint adapter.safetensors 
        --image photo.jpg --output outputs/run1 --motion pan_left --strength 0.6
        --calibration configs/strength_calibration_v2.json

or ``--config configs/inference_v2.json`` for the recommended defaults.  Python API:
``load_inference_pipeline()`` once, then ``generate(root, ckpt, image, out, pipe=pipe, ...)``
repeatedly (used by ``notebooks/camera_studio_v2.py`` and the batch scripts).
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import re
import time
import warnings

import numpy as np
from PIL import Image, ImageOps
import torch
from safetensors.torch import load_file

from .flow_geometry import VideoContract, camera_path, camera_path_v2
from .flow_control import (
    FILL_MODES,
    FlowControlConfig,
    FlowControlLoRA,
    build_image_condition,
)
from .flow_guidance import camera_prediction_guidance
from .flow_anchor import frozen_camera_difference, parse_residual_lowfreq
from .speed_options import resolve_attn_backend, set_block_compile, transformer_attention
from .runtime import gpu_lease, file_sha256, write_json
from .strength import (
    DIRECTIONS as MOTIONS_V2,
    amplitude_label,
    load_calibration,
    load_max_amplitudes,
    physical_amplitude,
    physical_to_u,
    split_motion,
)
from .wan import load_pipeline, hash_module, encode_first_image


# # Decoded video contract verification.
def _verify_video(path, contract):
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape != (contract.height, contract.width, 3):
            raise RuntimeError("Wrong decoded dimensions")
        count += 1
    cap.release()
    if count != contract.frames or abs(fps - contract.fps) > 0.01:
        raise RuntimeError("Video decode contract mismatch")
    return {
        "decoded_frames": count,
        "fps": fps,
        "width": contract.width,
        "height": contract.height,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


# # Diagnostics helpers (debug_dir): intermediate x0-hat and final-latent capture.
X0_CONVENTION = (
    "Wan2.2 flow matching, x_t = (1-sigma)*x0 + sigma*eps, model output v = eps - x0, "
    "hence x0_hat = x_t - sigma*v. This is exactly what diffusers 0.36 "
    "UniPCMultistepScheduler.convert_model_output computes for "
    "prediction_type='flow_prediction' (x0_pred = sample - sigmas[step_index]*model_output), "
    "with sigmas = flow-shifted schedule and timestep = 1000*sigma. v here is the final "
    "model_output handed to scheduler.step (after text CFG and camera-residual anchoring)."
)


def load_inference_pipeline(flow_shift=5.0, log_path=None):
    """Load the pinned frozen Wan pipeline once; pass it to ``generate(pipe=...)``.

    The caller owns the GPU lease (``runtime.gpu_lease``) while the pipeline is
    in use; ``generate`` does not take the lease again when ``pipe`` is given.
    """
    return load_pipeline(
        flow_shift=flow_shift,
        log_path=log_path or "artifacts/runtime/library.log",
        local_files_only=True,
    )


def _decode_latent_frames(pipe, z, indices):
    """Decode a normalized latent [48,T,h,w] with the native VAE; return uint8 frames."""
    device = next(pipe.vae.parameters()).device
    dtype = pipe.vae.dtype
    mean = torch.tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1)
    z = z[None].to(device=device, dtype=dtype)
    z = z * std.to(device, dtype) + mean.to(device, dtype)  # inverse of (x-mean)/std
    video = pipe.vae.decode(z, return_dict=False)[0]  # [1,3,F,H,W] in [-1,1]
    frames = []
    for index in indices:
        f = video[0, :, index].float().clamp(-1, 1).add(1).mul(127.5).round()
        frames.append(f.permute(1, 2, 0).to(torch.uint8).cpu().numpy())
    del video
    return frames


def _write_debug(pipe, debug_dir, state, first, condition, debug_sigmas, decode):
    """Persist captured latents/validity and optionally decode x0-hat frames."""
    from safetensors.torch import save_file

    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    image_latent = first[0, :, 0].detach().float().cpu()  # [48,h,w]
    tensors = {
        "image_latent": image_latent.contiguous(),
        "validity": condition[0, 48].detach().float().cpu().contiguous(),
    }
    final = state["last_latents"]
    if final is None:
        raise RuntimeError("No denoising callback fired; no final latent to save")
    final = final[0].detach().float().cpu().clone()
    final[:, 0] = image_latent  # native pipeline restores the first plane before decode
    tensors["final_latent"] = final.contiguous()
    meta = {"x0_convention": X0_CONVENTION, "x0_hat": {}, "missing_sigmas": []}
    captured = {}
    for target in debug_sigmas:
        item = state["x0"].get(target)
        if item is None:
            meta["missing_sigmas"].append(target)
            continue
        x0 = item["x0_hat"].clone()
        x0[:, 0] = image_latent
        key = f"x0_hat_sigma_{target:.2f}"
        tensors[key] = x0.contiguous()
        captured[target] = x0
        meta["x0_hat"][key] = {
            "step": item["step"],
            "sigma": item["sigma"],
            "timestep": item["timestep"],
            "requested_sigma": target,
        }
    save_file(tensors, str(debug_dir / "debug_latents.safetensors"))
    meta["files"] = {"latents": "debug_latents.safetensors", "pngs": []}
    if decode:
        indices = (40, 80, 120)
        with torch.inference_mode():
            for target, x0 in captured.items():
                frames = _decode_latent_frames(pipe, x0, indices)
                for index, frame in zip(indices, frames):
                    name = f"x0hat_sigma_{target:.2f}_frame_{index:03d}.png"
                    Image.fromarray(frame).save(debug_dir / name)
                    meta["files"]["pngs"].append(name)
    write_json(debug_dir / "debug_meta.json", meta)
    return meta


# # Checkpoint metadata (v1 sidecar or v2 trainer manifest) and v2 request building.
V2_OBJECTIVE = "hybrid_teacher_relative_plus_conditional_flow_matching_v2"


def resolve_checkpoint_meta(checkpoint):
    """Return ``(meta, source)`` describing an adapter checkpoint.

    * ``source == "sidecar"``: v1 layout, ``<adapter>.json`` next to the weights with
      ``checkpoint_sha256`` etc. (also accepted for any exporter writing that schema).
    * ``source == "v2_manifest"``: the v2 trainer layout
      ``<run>/checkpoints/adapter_NNNNNN.safetensors`` + ``ckpt_NNNNNN.json`` (manifest
      with ``adapter_sha256`` and ``architecture``) + ``<run>/config.json`` (resolved
      training config incl. ``fill_mode``) [+ ``<run>/run_status.json``]. A v1-shaped
      metadata dict is synthesised from them.
    """
    checkpoint = Path(checkpoint)
    sidecar = checkpoint.with_suffix(".json")
    if sidecar.is_file():
        meta = json.loads(sidecar.read_text())
        if "checkpoint_sha256" in meta:
            return meta, "sidecar"
    found = re.fullmatch(r"adapter_(\d+)", checkpoint.stem)
    manifest = checkpoint.with_name(f"ckpt_{found.group(1)}.json") if found else None
    if manifest is not None and manifest.is_file():
        return _meta_from_v2_manifest(checkpoint, manifest), "v2_manifest"
    raise FileNotFoundError(
        f"No usable metadata for {checkpoint}: expected {sidecar.name} (v1) or "
        "ckpt_NNNNNN.json next to adapter_NNNNNN.safetensors (v2 trainer)"
    )


def _meta_from_v2_manifest(checkpoint, manifest_path):
    """Synthesise a v1-shaped metadata dict (checkpoint digest, architecture, video contract, fill mode,
    backbone audit, inference policy ``camera_min_sigma`` / ``camera_residual_anchor='teacher'``) from the v2 trainer's
    ``ckpt_NNNNNN.json`` + the run's ``config.json`` / ``run_status.json``.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    run_dir = (
        checkpoint.parent.parent
        if checkpoint.parent.name == "checkpoints"
        else checkpoint.parent
    )
    cfg, status = {}, {}
    if (run_dir / "config.json").is_file():
        cfg = json.loads((run_dir / "config.json").read_text()).get("config", {})
    if (run_dir / "run_status.json").is_file():
        status = json.loads((run_dir / "run_status.json").read_text())
    contract = {"width": 1280, "height": 704, "frames": 121, "fps": 24}
    contract.update(cfg.get("video_contract") or {})
    sigma_a = (cfg.get("objective") or {}).get("sigma_A") or {}
    return {
        "checkpoint_sha256": manifest.get("adapter_sha256"),
        "architecture": manifest.get("architecture"),
        "completed_updates": manifest.get("step"),
        "identity": {"video_contract": contract, "objective": V2_OBJECTIVE},
        # The trainer verifies the backbone hash at every launch (expected_backbone_sha256)
        # and writes run_status.json (backbone_unchanged) when a launch ends; mid-run
        # checkpoints have no status file yet, then only the start-of-launch hash is known.
        "backbone_unchanged": status.get("backbone_unchanged", True),
        "backbone_audit": "run_status.json"
        if "backbone_unchanged" in status
        else "start_hash_only",
        "backbone_sha256_after": cfg.get("expected_backbone_sha256"),
        "guide_fill_mode": cfg.get("fill_mode"),
        # False when <run>/config.json is absent (e.g. only adapter+manifest were restored): the training
        # fill mode is then unknown and must not silently default to zeros (train/inference mismatch).
        "run_config_found": bool(cfg),
        "inference_policy": {
            "camera_min_sigma": float(sigma_a.get("lower", 0.6)),
            "camera_residual_anchor": "teacher",
        },
        "trainer": "flow_train_v2",
        "run_dir": str(run_dir),
    }


def recorded_fill_mode(meta):
    """Guide fill mode the checkpoint was trained with, or None if not recorded (v1)."""
    for holder in (meta, meta.get("identity") or {}, meta.get("training_config") or {}):
        for key in ("guide_fill_mode", "fill_mode"):
            if holder.get(key) in FILL_MODES:
                return holder[key]
    return None


def resolve_guide_fill_mode(explicit, meta):
    """Return ``(mode, source, warning_or_None)``: explicit > checkpoint record > zeros."""
    recorded = recorded_fill_mode(meta)
    if explicit is not None:
        if explicit not in FILL_MODES:
            raise ValueError(f"guide_fill_mode must be one of {FILL_MODES}")
        note = None
        if recorded is not None and recorded != explicit:
            note = (
                f"guide_fill_mode={explicit!r} differs from the checkpoint's training fill "
                f"mode {recorded!r}; expect a train/inference mismatch"
            )
        return explicit, "explicit", note
    if recorded is not None:
        return recorded, "checkpoint", None
    return "zeros", "default", None


def build_camera_request(
    contract,
    *,
    mode="zoom_in",
    magnitude=1.0,
    motion=None,
    strength=None,
    calibration=None,
    max_amplitudes=None,
    root=None,
    checkpoint_sha256=None,
    amplitude_deg=None,
    zoom_ratio=None,
):
    """Return ``(homographies, request, mode, magnitude, strength_record, notes)``.

    v1 (``motion is None``): ``camera_path(mode, contract, magnitude=magnitude)``.
    v2: UI strength ``strength`` (s in [0,1]) -> requested strength u via the calibration
    map -> ``camera_path_v2(family, direction, u, ...)`` with the trained maximum
    amplitudes (``configs/data_v2.json``). s, u, the physical amplitude and the
    calibration file digest are returned in ``strength_record``.
    Physical mode: ``amplitude_deg`` (pan/tilt) or ``zoom_ratio`` (zoom) instead of
    ``strength`` maps to u exactly (no calibration; ``amplitude_source: "physical"``).
    """
    physical = amplitude_deg is not None or zoom_ratio is not None
    if physical and strength is not None:
        raise ValueError("strength and a physical amplitude are mutually exclusive")
    if (motion is None) != (strength is None and not physical):
        raise ValueError("motion and strength (or amplitude_deg/zoom_ratio) must be given together")
    if motion is None:
        path, request = camera_path(mode, contract, magnitude=magnitude)
        return path, request, mode, magnitude, None, []
    import math

    if motion not in MOTIONS_V2 + ("static",):
        raise ValueError(
            f"motion must be one of {MOTIONS_V2 + ('static',)}, got {motion!r}"
        )
    if not physical:
        strength = float(strength)
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be finite and within [0,1]")
    notes = []
    smap = load_calibration(calibration, root=root)
    amax = dict(max_amplitudes) if max_amplitudes else load_max_amplitudes(root)
    if not smap.is_identity:
        for key, value in smap.max_amplitudes.items():
            if key in amax and abs(float(value) - float(amax[key])) > 1e-9:
                notes.append(
                    f"calibration was fitted with max {key}={value} but inference uses {amax[key]}"
                )
    cal_sha = smap.record.get("checkpoint_sha256")
    if cal_sha and checkpoint_sha256 and cal_sha != checkpoint_sha256:
        notes.append(
            "calibration was fitted for a different checkpoint than the one in use"
        )
    if physical:
        u = physical_to_u(motion, amplitude_deg, zoom_ratio, amax)
    else:
        u = smap.s_to_u(motion, strength)
    family, direction = split_motion(motion)
    path, request = camera_path_v2(
        family,
        direction,
        u,
        contract,
        max_yaw_deg=amax["yaw_deg"],
        max_pitch_deg=amax["pitch_deg"],
        max_zoom_ratio=amax["zoom_ratio"],
    )
    record = {
        "motion": motion,
        "amplitude_source": "physical" if physical else "calibrated",
        "strength_s": None if physical else strength,
        "strength_u": float(u),
        "amplitude": physical_amplitude(motion, u, amax),
        "label": amplitude_label(motion, u, amax),
        "max_amplitudes": amax,
        "calibration": None if physical else {
            "path": smap.source,
            "sha256": smap.sha256,
            "identity": smap.is_identity,
            "checkpoint_sha256": cal_sha,
        },
    }
    request["strength_control"] = record
    return path, request, motion, float(u), record, notes


def _report_progress(progress_callback, step, total):
    """Forward ``(step, total)`` to the optional UI hook of :func:`generate`.

    ``step`` counts completed denoising steps (1-based), ``total`` is the requested number of
    steps. ``None`` means no hook. An exception raised by the hook is deliberately *not*
    swallowed: it aborts the generation (``generate`` still restores the transformer in its
    ``finally``), which is how a UI implements Cancel.
    """
    if progress_callback is not None:
        progress_callback(int(step), int(total))


# # Single-image generation, adapter setup, denoising, and output capture.
def generate(
    root,
    checkpoint,
    image_path,
    output,
    *,
    mode="zoom_in",
    prompt=None,
    seed=42,
    steps=50,
    guidance=5.0,
    flow_shift=5.0,
    magnitude=1.0,
    frozen_baseline=False,
    camera_request=None,
    camera_guidance=1.0,
    component="full",
    camera_min_sigma=None,
    camera_residual_anchor="auto",
    guide_fill_mode=None,
    debug_dir=None,
    debug_sigmas=(0.6,),
    debug_decode=True,
    pipe=None,
    verify_backbone=True,
    motion=None,
    strength=None,
    calibration=None,
    max_amplitudes=None,
    amplitude_deg=None,
    zoom_ratio=None,
    batch_forwards=True,
    attn_backend="auto",
    compile_blocks=False,
    progress_callback=None,
    residual_lowfreq=None,
    trace_residual=False,
    residual_lowfreq_mask=None,
):
    """Generate one video. New optional arguments (defaults reproduce v1 exactly):

    residual_lowfreq_mask  None (default) | ``"valid"``: with ``residual_lowfreq``, estimate the low-frequency part of the
                      residual from the cells of the guide that still show the first image (validity channel) and
                      subtract it everywhere, so the revealed region cannot bias the estimate. Recorded in run.json.
    trace_residual    diagnostic (default False): record per-step statistics of the adapter residual in run.json
                      ``camera_guidance_execution.residual_trace`` (see ``flow_anchor.residual_stats``).
    residual_lowfreq  None (default: the camera residual is used unchanged) | ``"mean"`` | ``"blur:K"``: high-pass
                      the adapter's residual (adapter-on minus teacher) in latent space at every adapter-active
                      step, per latent frame and channel (see ``flow_anchor.lowfreq_highpass``). ``"mean"``
                      removes its spatial mean (the adapter cannot shift a frame's global colour/brightness),
                      ``"blur:K"`` removes a Gaussian low-pass of K latent cells (1 latent cell = 16 px).
                      Needs the teacher or frozen residual anchor. Recorded in run.json.

    progress_callback optional ``f(step, total)`` called in the calling thread after every denoising step
                      (``step`` = completed steps, 1..``total``; ``total`` = ``steps``). Default None = no
                      call. For live UI progress; raising from it aborts the run (see ``_report_progress``).
                      It is not called for the VAE decode / mp4 write that follow the last step.

    attn_backend      "auto" (default: torch's SDPA dispatch, untouched) | "cudnn" | "flash" | "efficient":
                      SDPA kernel priority for every transformer forward (not the VAE decode). Speed only,
                      kernel-level (not bit-identical); recorded in run.json ``speed_options``.
    compile_blocks    regional torch.compile of the 30 transformer blocks (mode=default, dynamic=False), applied
                      after the adapter is attached and undone when this call ends. Pays off only for long runs
                      (compile is minutes); not bit-identical. Recorded in run.json ``speed_options``.

    guide_fill_mode   "zeros" (v1) | "replicate": guide fill of revealed regions.
                      None (default): the fill mode recorded by the checkpoint (v2
                      trainer config.json), else "zeros" (v1). An explicit value
                      overrides and warns if it differs from the recorded one.
    motion, strength  v2 single-slider control: ``motion`` in pan_left, pan_right,
                      tilt_up, tilt_down, zoom_in, zoom_out, static and ``strength``
                      s in [0,1]. s -> u through ``calibration`` (path, dict or None =
                      identity; see ``fpcamera.strength``), then
                      ``camera_path_v2(..., u)`` with the trained maxima of
                      ``configs/data_v2.json`` (override: ``max_amplitudes`` with keys
                      yaw_deg, pitch_deg, zoom_ratio). s, u, the physical amplitude and
                      the calibration sha256 go to request.json and run.json. Both must
                      be given together; they replace ``mode``/``magnitude``.
    amplitude_deg, zoom_ratio
                      physical amplitude instead of ``strength``: pan/tilt degrees or
                      zoom ratio mapped to u exactly (u = deg/8, ln r/ln 1.25),
                      bypassing the calibration; recorded as
                      ``amplitude_source: "physical"``. Needs ``motion``.
    debug_dir         if set, write ``debug_latents.safetensors`` (final denoised
                      latent before VAE decode, x0-hat at the first denoising step
                      with sigma <= each of ``debug_sigmas``, image latent and the
                      guide validity channel), ``debug_meta.json`` and PNGs of the
                      decoded x0-hat (frames 40/80/120) when ``debug_decode``.
    pipe              an already-loaded pipeline (``load_inference_pipeline``). The
                      caller then holds the GPU lease; the model is not reloaded.
    verify_backbone   hash the 5B backbone before/after (slow). Disable on
                      repeated calls that share ``pipe``.
    """
    import imageio.v2 as imageio

    root, checkpoint, output = Path(root), Path(checkpoint), Path(output)
    import math

    attn_spec = resolve_attn_backend(attn_backend)  # validates (and warns on a torch without the backend) early

    if not math.isfinite(float(camera_guidance)) or float(camera_guidance) < 0:
        raise ValueError("Camera guidance must be finite and nonnegative")
    if camera_min_sigma is not None and (
        not math.isfinite(float(camera_min_sigma))
        or not 0 <= float(camera_min_sigma) <= 1
    ):
        raise ValueError("Minimum camera sigma must be finite and in [0,1]")
    if frozen_baseline and float(camera_guidance) != 1.0:
        raise ValueError(
            "Frozen baseline must use the native camera-guidance passthrough"
        )
    if component not in ("full", "guide_only", "attention_only", "lora_only"):
        raise ValueError("Unknown camera-adapter component ablation")
    residual_lowfreq = parse_residual_lowfreq(residual_lowfreq)
    if residual_lowfreq_mask not in (None, "valid"):
        raise ValueError("residual_lowfreq_mask must be None or 'valid'")
    if residual_lowfreq_mask is not None and residual_lowfreq is None:
        raise ValueError("residual_lowfreq_mask needs residual_lowfreq")
    if residual_lowfreq is not None and frozen_baseline:
        raise ValueError("A residual low-frequency filter needs the adapter (not --frozen-baseline)")
    if guide_fill_mode is not None and guide_fill_mode not in FILL_MODES:
        raise ValueError(f"guide_fill_mode must be one of {FILL_MODES}")
    physical = amplitude_deg is not None or zoom_ratio is not None
    if physical and strength is not None:
        raise ValueError("strength and a physical amplitude are mutually exclusive")
    if (motion is None) != (strength is None and not physical):
        raise ValueError("motion and strength (or amplitude_deg/zoom_ratio) must be given together")
    if motion is not None and camera_request is not None:
        raise ValueError("motion/strength and camera_request are mutually exclusive")
    debug_sigmas = tuple(float(x) for x in (debug_sigmas or ()))
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in debug_sigmas):
        raise ValueError("debug_sigmas must be finite values in [0,1]")
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
    if frozen_baseline and component != "full":
        raise ValueError(
            "Frozen baseline and partial-adapter ablations are distinct experiments"
        )
    if (
        frozen_baseline
        and camera_min_sigma is not None
        and float(camera_min_sigma) != 0.0
    ):
        raise ValueError("Frozen baseline does not use a camera activation schedule")
    # outputs are write-once: refuse to overwrite an existing run (run.json is the completion marker).
    if (output / "run.json").exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    meta, meta_source = resolve_checkpoint_meta(checkpoint)
    if (
        guide_fill_mode is None
        and meta_source == "v2_manifest"
        and not meta.get("run_config_found", True)
    ):
        raise ValueError(
            f"{checkpoint}: the v2 run's config.json (training fill_mode) was not found next to the "
            "checkpoints; refusing to fall back to zeros. Restore config.json "
            "(scripts/restore_from_hf.py --run NAME --checkpoints ...) or pass guide_fill_mode explicitly."
        )
    guide_fill_mode, fill_source, fill_note = resolve_guide_fill_mode(
        guide_fill_mode, meta
    )
    notes = []
    if fill_note:
        warnings.warn(fill_note)
        notes.append(fill_note)
    # anchor policy: baseline -> none; otherwise whatever the checkpoint recorded (v2 trainer: 'teacher'); legacy
    # v1 checkpoints that used the shared-static-input objective -> 'frozen'; anything else -> 'none'.
    if camera_residual_anchor == "auto":
        policy_anchor = meta.get("inference_policy", {}).get("camera_residual_anchor")
        camera_residual_anchor = (
            "none"
            if frozen_baseline
            else policy_anchor
            if policy_anchor in ("none", "frozen", "teacher")
            else "frozen"
            if meta.get("identity", {}).get("objective")
            == "teacher_relative_camera_velocity_on_shared_static_noisy_input"
            else "none"
        )
    if camera_residual_anchor not in ("none", "frozen", "teacher"):
        raise ValueError("Unknown camera residual anchor")
    if frozen_baseline and camera_residual_anchor != "none":
        raise ValueError("Frozen baseline has no learned residual anchor")
    if residual_lowfreq is not None and camera_residual_anchor not in ("frozen", "teacher"):
        raise ValueError("A residual low-frequency filter needs --camera-residual-anchor teacher or frozen")
    if camera_min_sigma is None:
        # camera_min_sigma: below this noise level the adapter is switched off and only the frozen model runs.  v2
        # default = the lower bound of the pass-A noise window (0.6), i.e. the adapter is only used at noise levels it
        # was trained for; low-noise steps (fine detail) stay pure Wan.
        camera_min_sigma = (
            0.0
            if frozen_baseline
            else float(meta.get("inference_policy", {}).get("camera_min_sigma", 0.0))
        )
    if (
        not math.isfinite(float(camera_min_sigma))
        or not 0 <= float(camera_min_sigma) <= 1
    ):
        raise ValueError("Checkpoint camera activation policy is invalid")
    if (
        file_sha256(checkpoint) != meta["checkpoint_sha256"]
        or meta.get("backbone_unchanged") is not True
    ):
        raise ValueError(
            "Checkpoint lacks matching digest and unchanged-backbone audit"
        )
    if meta["architecture"]["kind"] != "wan_flow_control_lora":
        raise ValueError("Wrong adapter family")
    contract = VideoContract(**meta["identity"]["video_contract"])
    image = ImageOps.fit(
        Image.open(image_path).convert("RGB"),
        (contract.width, contract.height),
        method=Image.Resampling.LANCZOS,
    )
    image.save(output / "input.png")
    strength_record = None
    if camera_request is None:
        path, request, mode, magnitude, strength_record, request_notes = (
            build_camera_request(
                contract,
                mode=mode,
                magnitude=magnitude,
                motion=motion,
                strength=strength,
                calibration=calibration,
                max_amplitudes=max_amplitudes,
                amplitude_deg=amplitude_deg,
                zoom_ratio=zoom_ratio,
                root=root,
                checkpoint_sha256=meta["checkpoint_sha256"],
            )
        )
        for note in request_notes:
            warnings.warn(note)
        notes.extend(request_notes)
    else:
        request = (
            json.loads(Path(camera_request).read_text())
            if isinstance(camera_request, (str, Path))
            else dict(camera_request)
        )
        path = np.asarray(request["homographies_first_to_current"], dtype=np.float64)
        if (
            path.shape != (contract.frames, 3, 3)
            or not np.isfinite(path).all()
            or not np.allclose(path[0], np.eye(3), atol=1e-6)
            or np.any(np.abs(np.linalg.det(path)) < 1e-8)
        ):
            raise ValueError("Invalid custom camera homographies")
        supplied = request.get("contract", {})
        if any(
            supplied.get(k) != getattr(contract, k)
            for k in ("width", "height", "frames", "fps")
        ):
            raise ValueError("Custom camera path and trained video contract disagree")
        mode = "drawn_camera_view_path"
    write_json(output / "request.json", request)
    prompt = (
        prompt
        if prompt is not None
        else "A photograph of a scene. One continuous camera shot."
    )
    with nullcontext() if pipe is not None else gpu_lease(root):
        if pipe is None:
            pipe = load_pipeline(
                flow_shift=flow_shift,
                log_path=output / "library.log",
                local_files_only=True,
            )
        elif float(pipe._fpcamera_provenance.get("flow_shift", -1)) != float(
            flow_shift
        ):
            raise ValueError("Supplied pipeline was loaded with a different flow_shift")
        if float(camera_min_sigma) > 0.0 and (
            pipe.scheduler.config.prediction_type != "flow_prediction"
            or not pipe.scheduler.config.use_flow_sigmas
            or pipe.scheduler.config.num_train_timesteps != 1000
        ):
            raise ValueError(
                "Camera sigma gating requires the verified native1000-step flow scheduler"
            )
        original = tuple(pipe.transformer.parameters())
        before = (
            hash_module(pipe.transformer, parameters=original)
            if verify_backbone
            else None
        )
        if (
            verify_backbone
            and meta.get("backbone_sha256_after") is not None
            and before != meta["backbone_sha256_after"]
        ):
            raise ValueError("Inference base differs from training")
        first = encode_first_image(pipe, np.asarray(image))[None].to("cuda")
        condition = build_image_condition(
            first,
            path,
            width=contract.width,
            height=contract.height,
            frames=contract.frames,
            fill_mode=guide_fill_mode,
        )
        adapter = FlowControlLoRA(
            pipe.transformer, FlowControlConfig(**meta["architecture"]["config"])
        )
        adapter.load_adapter_state_dict(load_file(str(checkpoint), device="cpu"))
        adapter.set_condition(condition)
        if component == "guide_only":
            for wrapper in adapter._wrappers.values():
                wrapper.enabled = False
        elif component == "attention_only":
            for path_name, wrapper in adapter._wrappers.items():
                wrapper.enabled = ".attn1." in path_name
        elif component == "lora_only":
            adapter.enabled = False
        active_adapter_parameters = (
            sum(p.numel() for p in adapter.guide_projection.parameters())
            if adapter.enabled
            else 0
        )
        active_adapter_parameters += sum(
            w.lora_A.numel() + w.lora_B.numel()
            for w in adapter._wrappers.values()
            if w.enabled
        )
        # the static guide is only needed when the residual reference is the adapted static view ('frozen' anchor) or
        # with camera_guidance != 1; the 'teacher' anchor reuses the frozen positive prediction instead (one forward less).
        static_condition = None
        if camera_residual_anchor == "frozen" or (
            camera_residual_anchor == "none" and float(camera_guidance) != 1.0
        ):
            static_path, _ = camera_path("static", contract)
            static_condition = build_image_condition(
                first,
                static_path,
                width=contract.width,
                height=contract.height,
                frames=contract.frames,
                fill_mode=guide_fill_mode,
            )
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        camera_schedule = []
        guidance_stats = None
        debug_record = None
        debug_state = {
            "last_latents": None,
            "x0": {},
            "pending": sorted(debug_sigmas, reverse=True),
        }
        original_step = pipe.scheduler.step

        def debug_step(model_output, timestep, sample, *args, **kwargs):
            """Wrapper around ``scheduler.step`` that records x0-hat = x_t - sigma*v at the requested noise levels."""
            # Wrap scheduler.step: it receives exactly the final guided/anchored
            # prediction v and the current x_t. step_index is incremented by step().
            result = original_step(model_output, timestep, sample, *args, **kwargs)
            index = pipe.scheduler.step_index - 1
            sigma = float(pipe.scheduler.sigmas[index])
            while debug_state["pending"] and sigma <= debug_state["pending"][0] + 1e-9:
                target = debug_state["pending"].pop(0)
                x0 = sample.float() - sigma * model_output.float()
                debug_state["x0"][target] = {
                    "x0_hat": x0[0].detach().cpu(),
                    "step": int(index) + 1,
                    "sigma": sigma,
                    "timestep": float(timestep),
                }
            return result

        def callback(_pipe, step, timestep, values):
            """Per-step callback: asserts the native CFG call count matches the anchoring protocol, records the camera
            sigma schedule, writes ``progress.json`` and forwards ``(step, total)`` to the UI progress hook.
            """
            if (
                camera_residual_anchor in ("frozen", "teacher")
                and guidance_stats["native_calls"]
                != (step + 1) * guidance_stats["native_calls_per_step"]
            ):
                raise RuntimeError(
                    "Native CFG call count disagrees with frozen camera anchoring protocol"
                )
            if debug_dir is not None:
                debug_state["last_latents"] = values["latents"]
            noise_level = float(timestep) / 1000.0
            active = not frozen_baseline and noise_level >= float(camera_min_sigma)
            camera_schedule.append(
                {"step": step + 1, "sigma": noise_level, "adapter_active": active}
            )
            if step % 10 == 0 or step == steps - 1:
                write_json(
                    output / "progress.json",
                    {
                        "step": step + 1,
                        "total": steps,
                        "seconds": time.perf_counter() - started,
                        "sigma": noise_level,
                        "camera_adapter_active": active,
                    },
                )
            _report_progress(progress_callback, step + 1, steps)
            return values

        if debug_dir is not None:
            pipe.scheduler.step = debug_step
        try:
            if compile_blocks:  # after the LoRA wrappers exist; undone in the finally below
                set_block_compile(pipe.transformer, True)
            # teacher-anchored residual (v2 default): every denoising step runs the FROZEN model for text CFG, and the
            # adapter only contributes camera_gain * (F_adapter(requested guide) - F_frozen), exactly the quantity pass A
            # was trained to produce.  This keeps text/CFG behaviour native and limits the adapter's influence to the camera.
            camera_context = (
                frozen_camera_difference(
                    pipe.transformer,
                    adapter,
                    condition,
                    static_condition,
                    camera_gain=camera_guidance,
                    text_guidance=guidance,
                    active_min_sigma=camera_min_sigma,
                    reference="teacher"
                    if camera_residual_anchor == "teacher"
                    else "static",
                    batch_forwards=batch_forwards,
                    residual_lowfreq=residual_lowfreq,
                    trace_residual=trace_residual,
                    lowfreq_weight=condition[:, 48:49] if residual_lowfreq_mask == "valid" else None,
                )
                if camera_residual_anchor in ("frozen", "teacher")
                else camera_prediction_guidance(
                    pipe.transformer,
                    adapter,
                    condition,
                    static_condition,
                    camera_guidance,
                    camera_min_sigma,
                )
            )
            with (
                adapter.disabled() if frozen_baseline else nullcontext(),
                torch.inference_mode(),
                # entered before camera_context: its wrapper captures transformer.forward at entry
                transformer_attention(pipe.transformer, attn_spec),
                camera_context as guidance_stats,
            ):
                result = pipe(
                    image=image,
                    prompt=prompt,
                    negative_prompt="",
                    height=contract.height,
                    width=contract.width,
                    num_frames=contract.frames,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                    output_type="np",
                    callback_on_step_end=callback,
                    callback_on_step_end_tensor_inputs=["latents"],
                )
            pixels = np.asarray(result.frames[0])
            if not np.isfinite(pixels).all():
                raise RuntimeError("Nonfinite generated video")
            if pixels.dtype != np.uint8:
                pixels = np.uint8(np.rint(np.clip(pixels, 0, 1) * 255))
            imageio.mimwrite(
                output / "video.mp4",
                pixels,
                fps=contract.fps,
                codec="libx264",
                quality=8,
                ffmpeg_params=["-movflags", "+faststart"],
            )
            video = _verify_video(output / "video.mp4", contract)
            for index in sorted(
                {0, contract.frames // 3, 2 * contract.frames // 3, contract.frames - 1}
            ):
                Image.fromarray(pixels[index]).save(output / f"frame_{index:03d}.png")
            if debug_dir is not None:
                debug_started = time.perf_counter()
                debug_record = _write_debug(
                    pipe,
                    debug_dir,
                    debug_state,
                    first,
                    condition,
                    debug_sigmas,
                    debug_decode,
                )
                debug_record["seconds"] = time.perf_counter() - debug_started
        finally:
            if debug_dir is not None and "step" in pipe.scheduler.__dict__:
                del pipe.scheduler.step  # restore the class method
            if compile_blocks:
                set_block_compile(pipe.transformer, False)
            adapter.close()
        after = (
            hash_module(pipe.transformer, parameters=original)
            if verify_backbone
            else None
        )
        if verify_backbone and before != after:
            raise RuntimeError("Frozen original weights changed during inference")
        record = {
            "status": "generated_video_requires_motion_and_preservation_evaluation",
            "adapter_enabled": not frozen_baseline,
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "adapter_component": component if not frozen_baseline else "disabled",
            "active_adapter_parameters": active_adapter_parameters
            if not frozen_baseline
            else 0,
            "completed_updates": meta["completed_updates"],
            "contract": contract.record(),
            "mode": mode,
            "magnitude": magnitude,
            "prompt": prompt,
            "negative_prompt": "",
            "seed": seed,
            "steps": steps,
            "guidance": guidance,
            "flow_shift": flow_shift,
            "camera_guidance": float(camera_guidance),
            "camera_residual_anchor": camera_residual_anchor,
            "camera_guidance_execution": guidance_stats,
            "camera_min_sigma": float(camera_min_sigma),
            "residual_lowfreq": None if residual_lowfreq is None else list(residual_lowfreq),
            "residual_lowfreq_mask": residual_lowfreq_mask,
            "camera_sigma_schedule": camera_schedule,
            "camera_active_denoising_steps": sum(
                r["adapter_active"] for r in camera_schedule
            ),
            "camera_guidance_reference": (
                "frozen_positive_prediction"
                if camera_residual_anchor == "teacher"
                else (
                    "same_image_static_view"
                    if camera_residual_anchor == "frozen"
                    or float(camera_guidance) != 1.0
                    else "native_passthrough"
                )
            ),
            "seconds": time.perf_counter() - started,
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "video": video,
            "backbone_unchanged": True if verify_backbone else None,
            "backbone_verified": bool(verify_backbone),
            "backbone_sha256": after,
            "guide_fill_mode": guide_fill_mode,
            "guide_fill_mode_source": fill_source,
            "checkpoint_meta_source": meta_source,
            "strength_control": strength_record,
            "speed_options": {"attn_backend": attn_spec.record(), "compile_blocks": bool(compile_blocks),
                              "batch_forwards": bool(batch_forwards)},
            "notes": notes,
            "debug": {
                "debug_dir": str(debug_dir),
                "debug_sigmas": list(debug_sigmas),
                "debug_decode": bool(debug_decode),
                "x0_convention": X0_CONVENTION,
                "capture": debug_record,
            }
            if debug_dir is not None
            else None,
            "input_sha256": file_sha256(output / "input.png"),
            "wan": pipe._fpcamera_provenance,
        }
        record["inference_source_sha256"] = {
            name: file_sha256(Path(__file__).with_name(name))
            for name in (
                "flow_infer.py",
                "flow_guidance.py",
                "flow_anchor.py",
                "flow_control.py",
                "flow_geometry.py",
                "strength.py",
                "wan.py",
            )
        }
        write_json(output / "run.json", record)
        return record


def build_parser():
    """Argument parser of the CLI (see ``generate`` for the meaning of each option); ``--config`` supplies JSON defaults."""
    parser = argparse.ArgumentParser(
        description="Generate one 5 s camera-controlled video from a single image."
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="JSON of defaults using these option names with '_' (e.g. configs/inference_v2.json); "
                        "explicit flags win")
    for name in ("project-root", "checkpoint", "image", "output"):
        parser.add_argument("--" + name, type=Path, default=None)
    parser.add_argument("--mode", default="zoom_in", help="v1 preset (ignored with --motion)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--magnitude", type=float, default=1.0, help="v1 magnitude in [0,2]")
    parser.add_argument("--motion", choices=MOTIONS_V2 + ("static",), default=None,
                        help="v2 motion; requires --strength")
    parser.add_argument("--strength", type=float, default=None,
                        help="v2 strength s in [0,1] (1 = trained maximum; mapped through --calibration)")
    parser.add_argument("--amplitude-deg", type=float, default=None,
                        help="v2 physical pan/tilt amplitude in degrees (u = deg/8; bypasses calibration)")
    parser.add_argument("--zoom-ratio", type=float, default=None,
                        help="v2 physical zoom ratio, e.g. 1.12 (u = ln r/ln 1.25; bypasses calibration)")
    parser.add_argument("--calibration", type=Path, default=None,
                        help="strength calibration JSON (default: identity map u = s)")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--camera-request", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--frozen-baseline", action="store_true")
    parser.add_argument("--camera-guidance", type=float, default=1.0)
    parser.add_argument(
        "--batch-forwards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="frozen text-cond+uncond predictions as one batch-2 forward in frozen/teacher anchoring "
        "(same math, fewer launches; --no-batch-forwards restores sequential batch-1 calls)",
    )
    parser.add_argument(
        "--attn-backend",
        choices=("auto", "cudnn", "flash", "efficient"),
        default="auto",
        help="SDPA kernel priority for the transformer forwards: auto = torch default (unchanged); cudnn = cuDNN, "
        "then flash, then memory-efficient; flash / efficient = that kernel only. Kernel-level, not bit-identical",
    )
    parser.add_argument(
        "--compile-blocks",
        action="store_true",
        help="regional torch.compile of the transformer blocks for this run (minutes of compile; worthwhile only for "
        "long or repeated runs; not bit-identical)",
    )
    parser.add_argument(
        "--residual-lowfreq",
        default=None,
        metavar="mean|blur:K",
        help="opt-in (default off): high-pass the adapter residual in latent space per frame and channel: 'mean' removes "
        "its spatial mean (no global colour/brightness shift by the adapter), 'blur:K' removes a Gaussian low-pass of "
        "K latent cells (16 px each). Needs the teacher/frozen residual anchor",
    )
    parser.add_argument(
        "--residual-lowfreq-mask",
        choices=("valid",),
        default=None,
        help="with --residual-lowfreq: estimate the low-frequency part from the cells that still show the first image "
        "(guide validity channel) and subtract it everywhere (default: estimate over the whole frame)",
    )
    parser.add_argument(
        "--camera-residual-anchor",
        choices=("auto", "none", "frozen", "teacher"),
        default="auto",
    )
    parser.add_argument(
        "--camera-min-sigma",
        type=float,
        default=None,
        help="Override checkpoint camera-noise window; legacy checkpoints default to all steps",
    )
    parser.add_argument(
        "--component",
        choices=("full", "guide_only", "attention_only", "lora_only"),
        default="full",
    )
    parser.add_argument(
        "--guide-fill-mode",
        choices=FILL_MODES,
        default=None,
        help="Guide fill of revealed regions. Default: the mode recorded by the checkpoint "
        "(v2 trainer), else zeros (v1). Explicit value overrides and warns on mismatch",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Write final/x0-hat latents, validity and decoded x0-hat PNGs here",
    )
    parser.add_argument("--debug-sigmas", type=float, nargs="+", default=[0.6])
    return parser


def parse_args(argv=None):
    """Parse CLI arguments, applying ``--config`` JSON defaults first (explicit flags win) and enforcing that
    ``--motion`` is given with exactly one of ``--strength`` / ``--amplitude-deg`` / ``--zoom-ratio``.
    """
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.config is not None:
        defaults = json.loads(pre.config.read_text(encoding="utf8"))
        known = {a.dest for a in parser._actions}
        parser.set_defaults(
            **{
                k.replace("-", "_"): v
                for k, v in defaults.items()
                if k.replace("-", "_") in known and v is not None
            }
        )
    args = parser.parse_args(argv)
    for name in ("project_root", "checkpoint", "image", "output"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    physical = args.amplitude_deg is not None or args.zoom_ratio is not None
    if physical and args.strength is not None:
        parser.error("--strength and --amplitude-deg/--zoom-ratio are mutually exclusive")
    if (args.motion is None) != (args.strength is None and not physical):
        parser.error("--motion must be given with exactly one of --strength, --amplitude-deg, --zoom-ratio")
    return args


def main(argv=None):
    """CLI entry point: parse arguments, call ``generate`` and print a one-line JSON status."""
    args = parse_args(argv)
    result = generate(
        args.project_root,
        args.checkpoint,
        args.image,
        args.output,
        mode=args.mode,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
        frozen_baseline=args.frozen_baseline,
        magnitude=args.magnitude,
        prompt=args.prompt,
        camera_request=args.camera_request,
        camera_guidance=args.camera_guidance,
        component=args.component,
        camera_min_sigma=args.camera_min_sigma,
        camera_residual_anchor=args.camera_residual_anchor,
        guide_fill_mode=args.guide_fill_mode,
        debug_dir=args.debug_dir,
        debug_sigmas=tuple(args.debug_sigmas),
        motion=args.motion,
        strength=args.strength,
        calibration=args.calibration,
        amplitude_deg=args.amplitude_deg,
        zoom_ratio=args.zoom_ratio,
        batch_forwards=args.batch_forwards,
        attn_backend=args.attn_backend,
        compile_blocks=args.compile_blocks,
        residual_lowfreq=args.residual_lowfreq,
        residual_lowfreq_mask=args.residual_lowfreq_mask,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "seconds": result["seconds"],
                "backbone_unchanged": result["backbone_unchanged"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
