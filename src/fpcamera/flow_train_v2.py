r"""v2 hybrid trainer for the frozen Wan2.2-TI2V-5B camera adapter (``FlowControlLoRA``).

Objective (per microbatch = one source, one requested camera motion)
--------------------------------------------------------------------
Notation: ``z_s`` static clip latent, ``z_m`` requested (moving) clip latent,
``first`` the clean first-image latent plane, ``R = 1 - valid`` the revealed
(newly exposed) mask at latent resolution computed from the *requested*
homographies (channel 48 of ``build_image_condition``), first plane excluded.

Velocity parameterisation.  Wan uses rectified flow, ``x_s = (1-s) z + s eps``
and the network predicts the velocity ``v = eps - z``; the scheduler step is
``x_{s'} = x_s + (s' - s) v``.  Check against v1's target: for the teacher
``v_T = (x_s - z_s)/s`` (what v1 assumes) we have
``(x_s - z_s)/s = ((1-s) z_s + s eps - z_s)/s = eps - z_s``, i.e. the same ``v``.
So v1's ``teacher + (z_s - z_m)/s`` and pass B's ``eps - z_m`` live in the same
output space; they differ only in what they ask the network to do.

Pass A (teacher-relative, identical to v1, keeps its sign convention)::

    x_A     = (1-s_A) z_s + s_A eps_A          (first plane := clean first latent)
    target_A = teacher(x_A) + (z_s - z_m)/s_A   (teacher = adapter disabled, no grad)
    w_A     = (1-R) + alpha_R * R               (alpha_R = 0.1)
    L_A     = sum(w_A (pred_A - target_A)^2) / sum(w_A)   (w broadcast over 48 ch)

Pass B (ordinary conditional flow matching on the moving clip, generative)::

    x_B      = (1-s_B) z_m + s_B eps_B
    target_B = eps_B - z_m
    w_B      = lambda_B (1 + beta R)            (lambda_B = 1, beta = 4)
    L_B      = sum(w_B (pred_B - target_B)^2) / sum(w_B)

Static requests (prob ``p_static``) run pass A only (identity regularisation;
configurable via ``static_pass_b``).  Total per microbatch ``L_A + L_B``; the
update averages the ``effective_batch`` microbatches (backward is done per pass
and gradients accumulate to save memory).

Why B fixes the gray borders: in revealed regions the content is not a function
of the inputs, so pass A's target (deterministic, independent of the noise
sample eps) has the MSE-optimal answer "the conditional mean" -> gray.  Pass B's
target contains ``eps``: the network sees ``x_B`` (noisy, containing real
content) and must regress ``eps - z_m``; at the noise levels where holes are
decided this forces sample-specific, textured content rather than the mean.

Everything stateful (data order, per-update randomness) is a deterministic
function of ``(seed, update_index, microbatch)`` plus a saved sampler state, so
an interrupted run resumes bit-identically (see ``checkpointing.py``).
"""
from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, field
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import threading
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from . import flow_control
from . import speed_options as so
from .checkpointing import CheckpointManager, atomic_write_json, sha256_file
from .session_deadline import Ema, UPLOAD_GRACE_SECONDS, should_stop

NORMALIZATION = "native_fp32_vae_mode_normalize_then_bf16"
EXIT_COMPLETE = 0
EXIT_INTERRUPTED = 75  # checkpoint saved; relaunch the same command to continue
COMPILE_ALLOWANCE_SECONDS = 420.0  # extra time the deadline rule reserves for the first (compiling) update of a launch


# =============================================================================
# 1. Target / weight math (pure, CPU-testable)
# =============================================================================
def token_timestep(clean, sigma):
    """TI2V image-conditioning time is zero for the complete first latent plane (as v1)."""
    b, _, t, h, w = clean.shape
    mask = torch.ones(b, t, h // 2, w // 2, device=clean.device)
    mask[:, 0] = 0
    return mask.flatten(1) * sigma[:, None] * 1000


def camera_velocity_target(teacher_future, static_future, requested_future, sigma):
    """Teacher-relative camera edit; all inputs are future latents only (ported from v1).

    If teacher=(x-static)/sigma, this is exactly (x-requested)/sigma.
    """
    if not (teacher_future.shape == static_future.shape == requested_future.shape):
        raise ValueError("Teacher/static/requested future latent shapes disagree")
    scale = sigma.float().reshape(-1, 1, 1, 1, 1)
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Positive finite noise levels required")
    delta = (static_future.detach().float() - requested_future.detach().float()) / scale
    return teacher_future.detach().float() + delta, delta


def sample_truncated_normal(generator, mean, std, lower, upper, n=1):
    """Inverse-CDF truncated normal driven by an explicit CPU generator (deterministic)."""
    # sigma sampling: draw u ~ U(0,1), rescale it into [cdf(a), cdf(b)] and invert the normal CDF, so the
    # result is an exact truncated normal.  Using an explicit CPU generator keeps the draw deterministic and
    # independent of any global RNG (needed for bit-exact resume).
    a, b = (lower - mean) / std, (upper - mean) / std

    def cdf(x):
        """Standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    lo, hi = cdf(a), cdf(b)
    u = torch.rand(n, generator=generator, dtype=torch.float64)
    p = lo + (hi - lo) * u
    x = mean + std * math.sqrt(2) * torch.erfinv((2 * p - 1).clamp(-1 + 1e-12, 1 - 1e-12))
    return x.clamp(lower, upper).float()


def revealed_mask(condition):
    """R = 1 - validity at latent resolution, future planes only: [B,1,T-1,H,W] in {0,1}."""
    if condition.ndim != 5 or condition.shape[1] != 51:
        raise ValueError("Expected guide [B,51,T,H,W]")
    valid = condition[:, 48:49, 1:].float().clamp(0, 1)
    return 1.0 - valid


def pass_a_weight(R, alpha_R):
    """Pass-A voxel weight ``(1-R) + alpha_R*R``: valid voxels weigh 1, revealed voxels only alpha_R (0.1),
    because the teacher-relative target is uninformative there (it would only teach 'gray').
    """
    return (1.0 - R) + alpha_R * R


def pass_b_weight(R, lambda_B, beta):
    """Pass-B voxel weight ``lambda_B*(1 + beta*R)``: revealed voxels weigh 1+beta (5x), so the generative
    flow-matching loss concentrates on filling newly exposed regions.
    """
    return lambda_B * (1.0 + beta * R)


def weighted_mse(pred, target, weight):
    """sum(w * (pred-target)^2) / sum(w); ``weight`` [B,1,T,H,W] is broadcast over channels."""
    if pred.shape != target.shape:
        raise ValueError("prediction/target shape mismatch")
    w = weight.float().expand_as(pred)
    se = (pred.float() - target.float()).square()
    return (w * se).sum() / w.sum()


# =============================================================================
# 2. Hyper-parameters and config
# =============================================================================
@dataclass(frozen=True)
class HybridHParams:
    """Frozen bundle of objective + optimiser hyper-parameters (built from the config by
    ``hparams_from_config``).  ``sigma_a`` / ``sigma_b`` are ``(mean, std, lower, upper)`` of the truncated
    normal over the noise level s for pass A / pass B; ``p_static`` is the probability that a microbatch
    requests the static clip (identity regularisation); ``seed`` drives every deterministic draw.
    """

    p_static: float = 0.1
    p_text_dropout: float = 0.1
    alpha_R: float = 0.1
    lambda_B: float = 1.0
    beta: float = 4.0
    static_pass_b: bool = False
    sigma_a: tuple = (0.8, 0.075, 0.6, 1.0)  # mean, std, lower, upper
    sigma_b: tuple = (0.7, 0.15, 0.2, 1.0)
    effective_batch: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup: int = 25
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    seed: int = 20261116
    total_updates: int = 1500


DEFAULT_CONFIG = {
    "name": "v2",
    "project_root": None,
    "seed": 20261116,
    "updates": 1500,
    "effective_batch": 2,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup": 25,
    "min_lr_ratio": 0.1,
    "grad_clip": 1.0,
    "objective": {
        "p_static": 0.1,
        "p_text_dropout": 0.1,
        "alpha_R": 0.1,
        "lambda_B": 1.0,
        "beta": 4.0,
        "static_pass_b": False,
        "sigma_A": {"mean": 0.8, "std": 0.075, "lower": 0.6, "upper": 1.0},
        "sigma_B": {"mean": 0.7, "std": 0.15, "lower": 0.2, "upper": 1.0},
    },
    "fill_mode": "replicate",
    "architecture": {"hidden": 3072, "rank": 16, "alpha": 16.0, "layers": 16},
    "video_contract": {"width": 1280, "height": 704, "frames": 121, "fps": 24},
    "checkpointing": "full",  # legacy; superseded by activation_checkpointing when set
    "activation_checkpointing": None,  # {"mode": "all"|"none"|"first_k"|"every_n", "k": K, "n": N}; not in run identity
    "checkpoint": {"every_updates": 25, "every_seconds": 600, "keep_last": 3, "keep_every": 250},
    "max_peak_gib": 80,
    "expected_backbone_sha256": None,
    "verify_backbone_hash": True,
    "split": "train",
    "families": None,
    "caches": [],
    "init_from": None,
}


def _deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def normalize_config(config: dict) -> dict:
    """Merge a user config over ``DEFAULT_CONFIG`` and validate it (noise windows, batch/updates,
    fill mode, checkpointing mode).  Returns the fully resolved dict that is stored in ``config.json``.
    """
    cfg = _deep_merge(DEFAULT_CONFIG, config)
    for key in ("sigma_A", "sigma_B"):
        s = cfg["objective"][key]
        if not (0 < s["lower"] <= s["mean"] <= s["upper"] <= 1 and s["std"] > 0 and math.isfinite(s["std"])):
            raise ValueError(f"Invalid noise distribution {key}: {s}")
    if cfg["effective_batch"] < 1 or cfg["updates"] < 1:
        raise ValueError("effective_batch and updates must be >= 1")
    if cfg["fill_mode"] not in ("zeros", "replicate"):
        raise ValueError("fill_mode must be zeros or replicate")
    if cfg["checkpointing"] not in ("full", "none"):
        raise ValueError("checkpointing must be full or none")
    if cfg.get("activation_checkpointing"):
        resolve_checkpointed_blocks(cfg["activation_checkpointing"], 1)
    return cfg


def hparams_from_config(cfg: dict) -> HybridHParams:
    """Extract the ``HybridHParams`` (objective + optimiser fields) from a normalised config."""
    o = cfg["objective"]
    sa, sb = o["sigma_A"], o["sigma_B"]
    return HybridHParams(
        p_static=float(o["p_static"]),
        p_text_dropout=float(o["p_text_dropout"]),
        alpha_R=float(o["alpha_R"]),
        lambda_B=float(o["lambda_B"]),
        beta=float(o["beta"]),
        static_pass_b=bool(o["static_pass_b"]),
        sigma_a=(sa["mean"], sa["std"], sa["lower"], sa["upper"]),
        sigma_b=(sb["mean"], sb["std"], sb["lower"], sb["upper"]),
        effective_batch=int(cfg["effective_batch"]),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        warmup=int(cfg["warmup"]),
        min_lr_ratio=float(cfg["min_lr_ratio"]),
        grad_clip=float(cfg["grad_clip"]),
        seed=int(cfg["seed"]),
        total_updates=int(cfg["updates"]),
    )


# ---- activation checkpointing (speed/memory only; NOT part of the run identity) -----------------------------
def resolve_checkpointed_blocks(spec, n_blocks):
    """Indices of transformer blocks to checkpoint.

    spec: {"mode": "all"|"none"|"first_k"|"every_n", "k": int, "n": int}.  Pure speed/memory trade-off: the
    forward/backward math is identical (recomputation is deterministic up to kernel nondeterminism).
    """
    mode = spec.get("mode", "all")
    if mode == "all":
        return list(range(n_blocks))
    if mode == "none":
        return []
    if mode == "first_k":
        k = int(spec["k"])
        if k < 0:
            raise ValueError("first_k needs k >= 0")
        return list(range(min(k, n_blocks)))
    if mode == "every_n":
        n = int(spec.get("n", 1))
        if n < 1:
            raise ValueError("every_n needs n >= 1")
        return list(range(0, n_blocks, n))
    raise ValueError(f"Unknown activation_checkpointing mode {mode!r}")


def checkpointing_spec(cfg, ckpt_blocks=None):
    """Effective spec: --ckpt-blocks K wins, then config 'activation_checkpointing', then legacy 'checkpointing'."""
    if ckpt_blocks is not None:
        return {"mode": "first_k", "k": int(ckpt_blocks)}
    spec = cfg.get("activation_checkpointing")
    if spec:
        resolve_checkpointed_blocks(spec, 1)  # validate
        return dict(spec)
    return {"mode": "all" if cfg["checkpointing"] == "full" else "none"}


def set_block_checkpointing(model, indices):
    """Checkpoint exactly the given ``model.blocks`` (idempotent; replaces any earlier setting).

    Wraps each selected block's forward in non-reentrant torch.utils.checkpoint (what diffusers uses for
    enable_gradient_checkpointing), active only when grad is enabled and the block is in training mode.
    """
    from torch.utils.checkpoint import checkpoint

    wanted = set(indices)
    for i, block in enumerate(model.blocks):
        orig = block.__dict__.get("_unckpt_forward")
        if i in wanted and orig is None:
            fwd = block.forward

            def wrapped(*args, _fwd=fwd, _block=block, **kwargs):
                """Forward of a checkpointed block: recompute in backward when grad is enabled and the block is training."""
                if torch.is_grad_enabled() and _block.training:
                    return checkpoint(_fwd, *args, use_reentrant=False, **kwargs)
                return _fwd(*args, **kwargs)

            block.__dict__["_unckpt_forward"] = fwd
            block.forward = wrapped
        elif i not in wanted and orig is not None:
            block.forward = orig
            del block.__dict__["_unckpt_forward"]
    return sorted(wanted)


def apply_activation_checkpointing(model, spec):
    """'all' keeps the diffusers-native path (unchanged behaviour); other settings use per-block wrapping."""
    if spec.get("mode", "all") == "all":
        set_block_checkpointing(model, [])
        model.enable_gradient_checkpointing()
        return list(range(len(model.blocks)))
    if getattr(model, "gradient_checkpointing", False):
        model.disable_gradient_checkpointing()
    return set_block_checkpointing(model, resolve_checkpointed_blocks(spec, len(model.blocks)))


DEFAULT_PROBE_BLOCKS = (None, 20, 15, 10, 0)  # None = all blocks


def configure_blocks(model, k_eff, compile_on=False, compile_kwargs=None):
    """Probe helper: exactly the first ``k_eff`` blocks checkpointed (per-block path, not diffusers' native flag),
    with block compilation on or off.  Order matters: unwrap, (un)compile, re-wrap (see ``so.set_block_compile``)."""
    if getattr(model, "gradient_checkpointing", False):
        model.disable_gradient_checkpointing()
    set_block_checkpointing(model, [])
    so.set_block_compile(model, compile_on, **(compile_kwargs or {}))
    set_block_checkpointing(model, range(k_eff))


def memory_probe(trainer, model, settings=DEFAULT_PROBE_BLOCKS, limit_gib=85.0, progress=print, compile_blocks=False,
                 compile_kwargs=None):
    """One accumulate-only update (both passes, effective_batch micro-batches) per setting."""
    n_blocks = len(model.blocks)
    rows, oom = [], False
    for k in settings:
        k_eff = n_blocks if k is None else min(int(k), n_blocks)
        if oom:
            rows.append({"ckpt_blocks": k_eff, "status": "skipped_after_oom"})
            continue
        configure_blocks(model, k_eff, compile_blocks, compile_kwargs)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        try:
            trainer.run_update(accumulate_only=True)
            torch.cuda.synchronize()
            row = {"ckpt_blocks": k_eff, "status": "ok", "peak_gib": peak_gib("cuda"),
                   "seconds": time.monotonic() - t0}
        except torch.cuda.OutOfMemoryError:
            trainer.inside_update = False
            trainer.optimizer.zero_grad(set_to_none=True)
            oom = True
            row = {"ckpt_blocks": k_eff, "status": "oom"}
        rows.append(row)
        progress(json.dumps({"event": "memory_probe_row", **row}))
    ok = [r for r in rows if r["status"] == "ok" and r["peak_gib"] <= limit_gib]
    best = min(ok, key=lambda r: r["seconds"]) if ok else None
    # the first row includes CUDA warm-up cost, so seconds are indicative: rerun if two rows are within ~3%
    return {"event": "memory_probe", "limit_gib": limit_gib, "rows": rows,
            "recommended_ckpt_blocks": best["ckpt_blocks"] if best else None,
            "recommended_flag": f"--ckpt-blocks {best['ckpt_blocks']}" if best else None}


def _probe_microbatches(trainer, max_draws=200):
    """``effective_batch`` micro-batches drawn from the sampler ONCE and reused by every probe row (moving requests only,
    so both passes run).  Check-only never saves sampler state, so advancing it here is harmless."""
    mbs, draws = [], 0
    while len(mbs) < trainer.hp.effective_batch:
        draws += 1
        if draws > max_draws:
            raise RuntimeError("speed probe could not draw moving-request micro-batches from the sampler")
        plan = trainer.sampler.next_plan()
        if plan.is_static_request and not trainer.hp.static_pass_b:
            continue
        mbs.append(trainer.loader(plan))
    return mbs


def _adapter_is_zero(adapter):
    """True when every LoRA-B and the guide projection weight are exactly zero (an untrained adapter: the LoRA-A
    gradients vanish, so the grad-norm equivalence check is weak).  Unknown structure -> False."""
    try:
        vals = [p.detach().abs().max() for n, p in adapter.named_parameters()
                if n.endswith("B") or n.endswith("guide_projection.weight")]
        return bool(vals) and all(float(v) == 0.0 for v in vals)
    except Exception:
        return False


def _cuda_sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _cuda_free(device):
    import gc

    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def _probe_values(rec):
    micro = rec["micro"]
    return {"loss_a": _mean([m["loss_a"] for m in micro]), "loss_b": _mean([m["loss_b"] for m in micro]),
            "grad_norm": rec["grad_norm"]}


def _run_probe_row(trainer, model, p, spec, mbs, cfg, compile_kwargs, state, clock):
    """Execute one grid row; returns the fields to merge into the row.  Never raises on OOM or a kernel/compile error."""
    device = trainer.device
    status, error, out = "ok", None, {}
    try:
        if p["compile"] and state["compiled_attn"] not in (None, p["attn_backend"]):
            so.reset_dynamo()  # the attention kernel is baked into the traced graph
        configure_blocks(model, p["ckpt_blocks"], p["compile"], compile_kwargs)
        if p["compile"]:
            state["compiled_attn"] = p["attn_backend"]
        trainer.attn_backend = spec
        _cuda_free(device)
        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = clock()
        warm = trainer.run_update(accumulate_only=True, microbatches=mbs)
        _cuda_sync(device)
        warm_s = clock() - t0
        warm_peak = peak_gib(device)
        secs, peak, last = [], 0.0, warm
        for _ in range(cfg.timed_updates):
            if torch.device(device).type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = clock()
            last = trainer.run_update(accumulate_only=True, microbatches=mbs)
            _cuda_sync(device)
            secs.append(clock() - t0)
            peak = max(peak, peak_gib(device))
        seconds = sum(secs) / len(secs)
        vals, warm_vals = _probe_values(last), _probe_values(warm)
        noise = [d for d in (so.rel_diff(warm_vals[m], vals[m]) for m in so.METRICS) if d is not None]
        out = {"warmup_seconds": warm_s, "seconds": seconds, "peak_gib": max(peak, warm_peak), **vals,
               "repeat_rel_diff": max(noise) if noise else None}  # warm-up vs timed update: run-to-run noise floor
        if p["compile"]:
            out["compile_overhead_seconds"] = max(warm_s - seconds, 0.0)
            out["dynamo"] = so.dynamo_stats()
    except torch.cuda.OutOfMemoryError:
        status = "oom"
    except Exception as e:  # a failing kernel / compile must not abort the probe; the row records why
        status, error = "error", f"{type(e).__name__}: {str(e)[:400]}"
    if status != "ok":
        trainer.inside_update = False
        trainer.optimizer.zero_grad(set_to_none=True)
        out = {}
    _cuda_free(device)
    return {"status": status, **({"error": error} if error else {}), **out}


def speed_probe(trainer, model, cfg, progress=print, compile_kwargs=None, clock=time.monotonic):
    """Time one full update (every micro-batch, both passes) per (checkpointed blocks, attention backend, compile) row on
    ONE fixed batch with identical noise and sigma (the trainer step is not advanced), after a discarded warm-up update.
    loss_A, loss_B and the global grad norm of each row are compared with the baseline row (K=20, auto, no compile):
    a relative difference above ``cfg.equiv_tol`` marks the row NOT lossless.  A row stops cleanly on OOM (and rows that
    could only use more memory are skipped); no new row starts after ``cfg.budget_minutes``."""
    plan = so.plan_probe_rows(cfg, len(model.blocks))
    mbs = _probe_microbatches(trainer)
    notes = []
    if _adapter_is_zero(trainer.adapter):
        notes.append("adapter is zero-initialised (no --init-from): LoRA-A gradients vanish, so the grad-norm "
                     "equivalence check is weak; pass --init-from <trained adapter .safetensors>")
        progress(json.dumps({"event": "speed_probe_warning", "message": notes[-1]}))
    start = clock()
    rows, base, oom_at, state = [], None, {}, {"compiled_attn": None}
    for p in plan:
        row = dict(p)
        spec = so.resolve_attn_backend(p["attn_backend"])
        row["attn_effective"] = spec.effective
        key = (p["attn_backend"], p["compile"])
        if not p["baseline"] and clock() - start > cfg.budget_minutes * 60:
            row["status"] = "skipped_budget"
        elif key in oom_at and p["ckpt_blocks"] <= oom_at[key]:
            row["status"] = "skipped_after_oom"
        else:
            row.update(_run_probe_row(trainer, model, p, spec, mbs, cfg, compile_kwargs, state, clock))
            if row["status"] == "oom":
                oom_at[key] = max(p["ckpt_blocks"], oom_at.get(key, -1))
        if row["status"] == "ok":
            if p["baseline"]:
                base = row
                row.update({"rel_diff": {m: 0.0 for m in so.METRICS}, "max_rel_diff": 0.0, "lossless": True,
                            "speedup": 1.0})
            elif base is not None:
                row.update(so.equivalence(row, base, cfg.equiv_tol))
                row["speedup"] = base["seconds"] / row["seconds"] if row["seconds"] > 0 else None
                gain = base["seconds"] - row["seconds"]
                if p["compile"] and gain > 0:
                    row["compile_breakeven_updates"] = row["compile_overhead_seconds"] / gain
            else:
                row.update({"lossless": None, "note": "no baseline row: equivalence unchecked"})
        rows.append(row)
        progress(json.dumps({"event": "speed_probe_row", **row}, default=str))
    best = so.recommend(rows, cfg.limit_gib)
    keep = ("ckpt_blocks", "attn_effective", "compile", "seconds", "peak_gib", "speedup", "max_rel_diff",
            "compile_overhead_seconds", "compile_breakeven_updates")
    report = {"event": "speed_probe", "limit_gib": cfg.limit_gib, "equiv_tol": cfg.equiv_tol,
              "budget_minutes": cfg.budget_minutes, "elapsed_minutes": (clock() - start) / 60.0, "rows": rows,
              "baseline_noise_rel_diff": base.get("repeat_rel_diff") if base else None,
              "adapter_zero_init": bool(notes), "notes": notes,
              "recommended": {k: best[k] for k in keep if k in best} if best else None,
              "recommended_flags": so.flags_for(best) if best else None}
    for line in so.format_table(report):
        progress(line)
    return report


def derive_seed(seed, *parts) -> int:
    """Stable 63-bit seed from ``(seed, *parts)`` via SHA-256.  Every per-update random draw (sigma, noise)
    is seeded from ``(seed, step, micro, tag)`` so a resumed run reproduces the same draws as an uninterrupted
    one, independent of any global RNG state.
    """
    h = hashlib.sha256(repr((int(seed),) + tuple(parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") & (2**63 - 1)


# =============================================================================
# 3. Data: cache rows (v2 + v1 adapter), sampler, loader
# =============================================================================
@dataclass(frozen=True)
class ClipRef:
    """Pointer to one cached clip (latents + homographies file and its per-source reference file).
    ``fmt`` is ``'v2'`` or ``'v1'`` (v1 rows have no magnitude: ``magnitude_u`` is None).
    """

    cache_idx: int
    clip_id: str
    source_id: str
    split: str
    family: str  # static | pan | tilt | zoom
    direction: str | None
    magnitude_u: float | None  # None for v1 rows (single fixed magnitude)
    latent_path: str
    reference_path: str
    sha256: str | None
    reference_sha256: str | None
    fmt: str  # "v2" | "v1"

    @property
    def is_static(self):
        """True for the source's static (no camera motion) clip."""
        return self.family == "static"


_V1_MODES = {"static": ("static", None), "zoom_in": ("zoom", "in"), "zoom_out": ("zoom", "out")}


def _resolve(rel, index_path: Path, project_root):
    p = Path(rel)
    if p.is_absolute():
        return p
    cands = [index_path.parent / p]
    if project_root:
        cands.append(Path(project_root) / p)
    cands += [a / p for a in index_path.resolve().parents]
    for c in cands:
        if c.exists():
            return c
    return cands[0]


def normalize_row(row: dict, cache_idx: int, index_path: Path, project_root=None) -> ClipRef:
    """Map a v2 or v1 ``index.jsonl`` row onto the common ``ClipRef`` schema."""
    if row.get("normalization", NORMALIZATION) != NORMALIZATION:
        raise ValueError("Wrong VAE cache normalization")
    if "latent_file" in row:  # v2
        return ClipRef(
            cache_idx, str(row["id"]), str(row["source_id"]), row["split"], row["family"],
            row.get("direction"), row.get("magnitude_u"),
            str(_resolve(row["latent_file"], index_path, project_root)),
            str(_resolve(row["reference_file"], index_path, project_root)),
            row.get("sha256"), row.get("reference_sha256"), "v2",
        )
    if "mode" in row and "cache_path" in row:  # v1 (static / zoom_in / zoom_out)
        if row["mode"] not in _V1_MODES:
            raise ValueError(f"Unknown v1 mode {row['mode']!r}")
        family, direction = _V1_MODES[row["mode"]]
        return ClipRef(
            cache_idx, str(row["record_id"]), str(row["source_id"]), row["split"], family, direction, None,
            str(_resolve(row["cache_path"], index_path, project_root)),
            str(_resolve(row["reference_path"], index_path, project_root)),
            row.get("sha256"), row.get("reference_sha256"), "v1",
        )
    raise ValueError("Unrecognised cache row schema (neither v2 nor v1)")


@dataclass
class SourceEntry:
    """One training source: its single static clip and the list of moving clips (other families/directions/u)."""

    source_key: str
    static: ClipRef
    moving: list


@dataclass
class CacheTable:
    """All usable sources of one cache index for one split, plus the sampling weight of the cache and the
    sha256 of its ``index.jsonl`` (part of the run identity).  ``dropped`` lists sources without a moving clip.
    """

    cache_idx: int
    index_path: str
    index_sha256: str
    weight: float
    sources: dict  # source_id -> SourceEntry, sorted by id
    dropped: list = field(default_factory=list)

    @property
    def source_ids(self):
        """Source ids in sorted (deterministic) order."""
        return list(self.sources)


def load_cache_table(index_path, cache_idx=0, weight=1.0, *, split="train", families=None,
                     project_root=None, check_files=True) -> CacheTable:
    """Read an ``index.jsonl`` (v2 or v1 rows) into a ``CacheTable`` for ``split``.

    Each source must have exactly one static row; moving rows can be restricted with ``families``.  With
    ``check_files`` every referenced latent/reference file must exist.  Sources with no moving row left are
    dropped (and listed).  Raises on an empty result.
    """
    index_path = Path(index_path)
    rows = [json.loads(l) for l in index_path.read_text(encoding="utf8").splitlines() if l.strip()]
    refs = [normalize_row(r, cache_idx, index_path, project_root) for r in rows]
    refs = [r for r in refs if r.split == split]
    if not refs:
        raise ValueError(f"No {split!r} rows in {index_path}")
    if check_files:
        missing = [r.clip_id for r in refs if not Path(r.latent_path).is_file() or not Path(r.reference_path).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} cache files missing, e.g. {missing[:3]}")
    by_source: dict = {}
    for r in refs:
        by_source.setdefault(r.source_id, []).append(r)
    sources, dropped = {}, []
    for sid in sorted(by_source):
        group = by_source[sid]
        statics = [r for r in group if r.is_static]
        if len(statics) != 1:
            raise ValueError(f"Source {sid}: expected exactly one static row, found {len(statics)}")
        moving = [r for r in group if not r.is_static and (families is None or r.family in families)]
        if not moving:
            dropped.append(sid)
            continue
        sources[sid] = SourceEntry(sid, statics[0], sorted(moving, key=lambda r: r.clip_id))
    if not sources:
        raise ValueError("No usable sources (need static + at least one moving row)")
    return CacheTable(cache_idx, str(index_path), sha256_file(index_path), float(weight), sources, dropped)


@dataclass(frozen=True)
class Plan:
    """One microbatch decision from the sampler: which source, which clip is requested, whether the
    request is static, and whether text conditioning is dropped (CFG-style dropout to the negative embedding).
    """

    cache_idx: int
    source_id: str
    static_ref: ClipRef
    request_ref: ClipRef
    is_static_request: bool
    text_dropout: bool


class SourceSampler:
    """Weighted multi-cache source sampler with an exactly resumable state.

    Each cache keeps its own seeded permutation + cursor (re-shuffled per epoch);
    the cache choice, static-vs-moving choice, moving-row choice and text dropout
    all come from one Python ``Random`` whose state is part of the saved state.
    """

    def __init__(self, tables, *, seed, p_static, p_text_dropout):
        self.tables = list(tables)
        self.p_static, self.p_text_dropout = float(p_static), float(p_text_dropout)
        self.weights = [t.weight for t in self.tables]
        if not self.tables or min(self.weights) <= 0:
            raise ValueError("Need at least one cache with positive weight")
        self.rng = random.Random(derive_seed(seed, "sampler"))
        self.perm = {i: [] for i in range(len(self.tables))}
        self.epoch = {i: 0 for i in range(len(self.tables))}
        self.draws = 0

    def _next_source(self, i):
        if not self.perm[i]:
            ids = self.tables[i].source_ids
            self.rng.shuffle(ids)
            self.perm[i] = ids
            self.epoch[i] += 1
        return self.perm[i].pop()  # pop from the end of the seeded permutation

    def next_plan(self) -> Plan:
        """Draw the next microbatch plan.  The order of RNG draws is fixed (cache, source, static?, moving row,
        text dropout) and all come from one saved ``Random``, which is what makes resume exact.
        """
        i = self.rng.choices(range(len(self.tables)), weights=self.weights)[0]
        entry = self.tables[i].sources[self._next_source(i)]
        static_req = self.rng.random() < self.p_static
        moving = entry.moving[self.rng.randrange(len(entry.moving))]
        dropout = self.rng.random() < self.p_text_dropout
        self.draws += 1
        return Plan(i, entry.source_key, entry.static, entry.static if static_req else moving, static_req, dropout)

    def state_dict(self):
        """JSON/pickle-able sampler state (RNG state, per-cache permutations and epochs, draw count)."""
        return {"rng": self.rng.getstate(), "perm": {k: list(v) for k, v in self.perm.items()},
                "epoch": dict(self.epoch), "draws": self.draws}

    def load_state_dict(self, s):
        """Restore the sampler state saved by ``state_dict``."""
        self.rng.setstate(s["rng"])
        self.perm = {int(k): list(v) for k, v in s["perm"].items()}
        self.epoch = {int(k): v for k, v in s["epoch"].items()}
        self.draws = s["draws"]


@dataclass
class MicroBatch:
    """Device tensors for one microbatch (see field comments).  ``condition`` is the guide for the *requested*
    motion; ``meta`` carries ids and the request (family, direction, u) for logging.
    """

    static: torch.Tensor      # [1,48,T,H,W] clean static latents z_s
    requested: torch.Tensor   # [1,48,T,H,W] clean requested latents z_m
    first: torch.Tensor       # [1,48,1,H,W] clean first-image latent
    text: torch.Tensor        # conditioning actually used (negative if dropped)
    condition: torch.Tensor   # [1,51,T,H,W] guide for the REQUESTED motion
    meta: dict


def make_condition_fn(fill_mode, contract):
    """Wrap ``flow_control.build_image_condition`` so ``fill_mode`` is passed when supported."""

    def fn(first, homographies):
        """Build the guide ``[1,51,T,H,W]`` for a first-image latent and homographies with the configured fill mode."""
        builder = flow_control.build_image_condition
        kwargs = dict(width=contract["width"], height=contract["height"], frames=contract["frames"])
        if "fill_mode" in inspect.signature(builder).parameters:
            kwargs["fill_mode"] = fill_mode
        elif fill_mode != "zeros":
            raise RuntimeError(
                "flow_control.build_image_condition has no fill_mode argument; "
                f"cannot honour fill_mode={fill_mode!r}"
            )
        return builder(first, homographies, **kwargs)

    return fn


def _load_clip_files(ref: ClipRef, contract: dict, verified: set):
    from safetensors.torch import load_file as _lf

    if ref.latent_path not in verified:
        if ref.sha256 and sha256_file(ref.latent_path) != ref.sha256:
            raise RuntimeError(f"Cache file changed: {ref.latent_path}")
        if ref.reference_sha256 and sha256_file(ref.reference_path) != ref.reference_sha256:
            raise RuntimeError(f"Reference file changed: {ref.reference_path}")
    data = _lf(ref.latent_path, device="cpu")
    refs = _lf(ref.reference_path, device="cpu")
    lat_shape = (48, (contract["frames"] - 1) // 4 + 1, contract["height"] // 16, contract["width"] // 16)
    if tuple(data["latents"].shape) != lat_shape:
        raise ValueError(f"{ref.clip_id}: latents {tuple(data['latents'].shape)} != {lat_shape}")
    if tuple(data["homographies"].shape) != (contract["frames"], 3, 3):
        raise ValueError(f"{ref.clip_id}: homography timing mismatch")
    if ref.latent_path not in verified:
        for x in (*data.values(), *refs.values()):
            if not torch.isfinite(x).all():
                raise ValueError(f"{ref.clip_id}: non-finite cache tensor")
        verified.add(ref.latent_path)
    return data, refs


class MicrobatchLoader:
    """Turns a ``Plan`` into device tensors (works for v1 and v2 caches)."""

    def __init__(self, contract, fill_mode, device="cuda", dtype=torch.bfloat16, text_pad=None, condition_fn=None):
        self.contract = contract
        self.device, self.dtype = torch.device(device), dtype
        if text_pad is None:
            from .wan import pad_text_embeddings as text_pad
        self.text_pad = text_pad
        self.condition_fn = condition_fn or make_condition_fn(fill_mode, contract)
        self.verified: set = set()

    def _dev(self, x):
        return x.to(device=self.device, dtype=self.dtype)

    def __call__(self, plan: Plan) -> MicroBatch:
        sdata, refs = _load_clip_files(plan.static_ref, self.contract, self.verified)
        static = self._dev(sdata["latents"])[None]
        first = self._dev(refs["image_latent"])[None]
        if not torch.equal(static[:, :, :1], first):
            raise ValueError("Static clip first plane differs from the image latent")
        if plan.is_static_request:
            requested, hom = static, sdata["homographies"]
        else:
            rdata, _ = _load_clip_files(plan.request_ref, self.contract, self.verified)
            requested, hom = self._dev(rdata["latents"])[None], rdata["homographies"]
            if not torch.equal(requested[:, :, :1], first):
                raise ValueError("Requested clip first plane differs from the image latent")
        key = "negative_text_embeddings" if plan.text_dropout else "text_embeddings"
        text = self._dev(self.text_pad(refs[key]))[None]
        condition = self.condition_fn(first, hom[None].to(self.device))
        r = plan.request_ref
        meta = {"source_id": plan.source_id, "cache_idx": plan.cache_idx, "clip_id": r.clip_id,
                "family": r.family, "direction": r.direction, "u": r.magnitude_u,
                "static_request": plan.is_static_request, "text_dropout": plan.text_dropout}
        return MicroBatch(static, requested, first, text, condition, meta)


# =============================================================================
# 4. The core step (dependency-injected: any model/adapter with the v1 call surface)
# =============================================================================
def _autocast(device, enabled=None):
    on = device.type == "cuda" if enabled is None else enabled
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=on)


def _noisy_input(clean, first, sigma, noise):
    s = sigma[:, None, None, None, None]
    # rectified-flow forward process x_s = (1-s) z + s eps; plane 0 below is then reset to the clean
    # image latent, matching Wan TI2V's image conditioning (those tokens also get timestep 0, see token_timestep).
    x = (1 - s) * clean + s * noise
    return torch.cat((first, x[:, :, 1:]), dim=2)  # plane 0 := clean image latent (anchor)


def _gen(device, seed):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def _call(model, x, timesteps, text):
    return model(hidden_states=x, timestep=timesteps, encoder_hidden_states=text, return_dict=False)[0]


def build_pass_a(model, adapter, mb: MicroBatch, hp: HybridHParams, sigma, noise, R, autocast=None):
    """Noisy static input, teacher forward (no grad, adapter off), v1 target and weight."""
    device = mb.static.device
    noisy = _noisy_input(mb.static, mb.first, sigma, noise)
    timesteps = token_timestep(mb.static, sigma)
    adapter.set_condition(mb.condition)
    with torch.no_grad(), adapter.disabled(), _autocast(device, autocast):
        teacher = _call(model, noisy, timesteps, mb.text)
    teacher_future = teacher[:, :, 1:].detach().float()
    target, delta = camera_velocity_target(teacher_future, mb.static[:, :, 1:], mb.requested[:, :, 1:], sigma)
    return {"noisy": noisy, "timesteps": timesteps, "target": target, "delta": delta,
            "weight": pass_a_weight(R, hp.alpha_R), "teacher_future": teacher_future}


def build_pass_b(mb: MicroBatch, hp: HybridHParams, sigma, noise, R):
    """Plain conditional flow matching on the requested clip: x=(1-s)z_m+s eps, target=eps-z_m."""
    noisy = _noisy_input(mb.requested, mb.first, sigma, noise)
    timesteps = token_timestep(mb.requested, sigma)
    target = (noise.float() - mb.requested.float())[:, :, 1:]
    return {"noisy": noisy, "timesteps": timesteps, "target": target,
            "weight": pass_b_weight(R, hp.lambda_B, hp.beta)}


def _finite(loss, name):
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite {name}")
    return loss


def run_microbatch(model, adapter, mb: MicroBatch, hp: HybridHParams, *, step, micro, loss_scale, autocast=None):
    """Forward + backward for one microbatch (A then B); gradients accumulate in the adapter."""
    device = mb.static.device
    R = revealed_mask(mb.condition)
    # sigma_A and sigma_B come from one CPU generator seeded by (seed, update, microbatch): pass A concentrates at
    # high noise (0.8 +- 0.075, [0.6, 1]) where the camera-driven layout is decided -- matching the inference window
    # camera_min_sigma = sigma_A.lower = 0.6 -- while pass B is broader (0.7 +- 0.15, [0.2, 1]) because texture of
    # revealed regions is decided at lower noise too.  Seeding by (step, micro) makes a resumed run identical.
    sig_gen = torch.Generator().manual_seed(derive_seed(hp.seed, step, micro, "sigma"))
    sigma_a = sample_truncated_normal(sig_gen, *hp.sigma_a).to(device)
    sigma_b = sample_truncated_normal(sig_gen, *hp.sigma_b).to(device)
    noise_a = torch.randn(mb.static.shape, generator=_gen(device, derive_seed(hp.seed, step, micro, "nA")),
                          device=device, dtype=mb.static.dtype)
    out = {"sigma_a": float(sigma_a), "sigma_b": None, "loss_a": None, "loss_b": None,
           "r_fraction": float(R.mean()), **mb.meta}

    # ---- pass A (teacher-relative) -----------------------------------------
    a = build_pass_a(model, adapter, mb, hp, sigma_a, noise_a, R, autocast)
    adapter.set_condition(mb.condition)
    with _autocast(device, autocast):
        pred = _call(model, a["noisy"], a["timesteps"], mb.text)
    loss_a = _finite(weighted_mse(pred[:, :, 1:], a["target"], a["weight"]), "pass-A loss")
    # backward per pass (loss_scale = 1/effective_batch): only one adapted graph is alive at a time, and the
    # gradients of the microbatches and of A and B simply accumulate in the adapter's .grad.
    (loss_a * loss_scale).backward()
    out["loss_a"] = float(loss_a.detach())
    # diagnostic only: how much of the requested edit delta = (z_s - z_m)/s the adapter actually adds on top of
    # the teacher (1.0 = exactly the requested edit, 0 = adapter does nothing).  Logged as edit_target_projection.
    energy = a["delta"].square().sum()
    resid = pred[:, :, 1:].detach().float() - a["teacher_future"]
    out["edit_target_projection"] = float((resid * a["delta"]).sum() / energy) if energy > 0 else None
    del a, pred, loss_a, resid

    # ---- pass B (generative flow matching) ----------------------------------
    # pass B needs a *moving* clip: for a static request z_m == z_s, R == 0 and pass A is already the identity
    # regulariser, so B is skipped unless static_pass_b is set.
    if (not mb.meta["static_request"]) or hp.static_pass_b:
        noise_b = torch.randn(mb.requested.shape, generator=_gen(device, derive_seed(hp.seed, step, micro, "nB")),
                              device=device, dtype=mb.requested.dtype)
        b = build_pass_b(mb, hp, sigma_b, noise_b, R)
        adapter.set_condition(mb.condition)
        with _autocast(device, autocast):
            pred = _call(model, b["noisy"], b["timesteps"], mb.text)
        loss_b = _finite(weighted_mse(pred[:, :, 1:], b["target"], b["weight"]), "pass-B loss")
        (loss_b * loss_scale).backward()
        out["loss_b"] = float(loss_b.detach())
        out["sigma_b"] = float(sigma_b)
        del b, pred, loss_b
    return out


def capture_rng():
    """Snapshot the Python / NumPy / torch (CPU + CUDA) RNG states for the resume checkpoint."""
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(s):
    """Restore the RNG states captured by ``capture_rng`` (CUDA state is skipped if the device count changed;
    per-update seeds make that harmless).
    """
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(s["cuda"])
        except Exception:  # different device count; per-update seeds make this non-critical
            pass


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


class HybridTrainer:
    """One-update-at-a-time trainer; model/adapter/loader/sampler are injected."""

    def __init__(self, model, adapter, optimizer, hp: HybridHParams, sampler, loader, *, device,
                 frozen_params=(), identity=None, step=0, autocast=None, attn_backend="auto"):
        self.model, self.adapter, self.optimizer = model, adapter, optimizer
        self.hp, self.sampler, self.loader = hp, sampler, loader
        self.device = torch.device(device)
        self.params = [p for p in adapter.parameters() if p.requires_grad]
        self.frozen_params = tuple(frozen_params)
        self.identity = identity or {}
        self.step = step
        self.autocast = autocast
        # opt-in SDPA kernel priority (speed only); "auto" = no context at all = the unchanged default path
        self.attn_backend = so.resolve_attn_backend(attn_backend)
        self.inside_update = False
        self.elapsed_before = 0.0
        self._t0 = time.monotonic()

    def lr_at(self, step):
        """Learning rate at update ``step``: linear warm-up over ``hp.warmup`` updates times a cosine decay from 1
        to ``min_lr_ratio`` over ``total_updates``.
        """
        hp = self.hp
        # linear warm-up x cosine decay (1 -> min_lr_ratio over total_updates); a function of the update index only,
        # so resume needs no scheduler state beyond `step`.
        warm = min(1.0, (step + 1) / max(1, hp.warmup))
        cos = hp.min_lr_ratio + (1 - hp.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * step / hp.total_updates))
        return hp.learning_rate * warm * cos

    def elapsed(self):
        """Wall-clock seconds of training including earlier launches of the same run (restored from the checkpoint)."""
        return self.elapsed_before + time.monotonic() - self._t0

    def run_update(self, accumulate_only=False, microbatches=None):
        """``microbatches``: pre-loaded micro-batches (speed probe only; the sampler is then not touched)."""
        hp = self.hp
        t0 = time.monotonic()
        lr = self.lr_at(self.step)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        self.inside_update = True
        micros = []
        # the attention context spans forward AND backward, so checkpoint recomputation uses the same kernel
        with so.attention_backend(self.attn_backend):
            for m in range(hp.effective_batch):
                mb = microbatches[m] if microbatches is not None else self.loader(self.sampler.next_plan())
                micros.append(run_microbatch(self.model, self.adapter, mb, hp, step=self.step, micro=m,
                                             loss_scale=1.0 / hp.effective_batch, autocast=self.autocast))
                del mb
        if any(p.grad is not None for p in self.frozen_params):
            raise RuntimeError("Frozen original acquired a gradient")
        norm = torch.nn.utils.clip_grad_norm_(self.params, hp.grad_clip)
        if not torch.isfinite(norm):
            raise RuntimeError("Non-finite adapter gradient")
        if accumulate_only:  # --check-only
            self.optimizer.zero_grad(set_to_none=True)
            self.inside_update = False
            return {"micro": micros, "grad_norm": float(norm), "update_seconds": time.monotonic() - t0}
        self.optimizer.step()
        self.step += 1
        self.inside_update = False
        la, lb = _mean([x["loss_a"] for x in micros]), _mean([x["loss_b"] for x in micros])
        return {
            "step": self.step, "loss_a": la, "loss_b": lb, "loss": (la or 0) + (lb or 0),
            "r_fraction": _mean([x["r_fraction"] for x in micros]),
            "sigma_a": [x["sigma_a"] for x in micros], "sigma_b": [x["sigma_b"] for x in micros],
            "family": [x["family"] for x in micros], "u": [x["u"] for x in micros],
            "grad_norm": float(norm), "lr": lr, "update_seconds": time.monotonic() - t0,
            "elapsed_seconds": self.elapsed(), "peak_gib": peak_gib(self.device), "micro": micros,
        }

    # ---- resume state ---------------------------------------------------------
    def state_dict(self):
        """Everything needed to resume bit-identically: update counter, optimizer, sampler (RNG, permutations,
        cursor), global RNGs, run identity and elapsed time.
        """
        return {"step": self.step, "optimizer": self.optimizer.state_dict(), "sampler": self.sampler.state_dict(),
                "rng": capture_rng(), "lr_state": {"step": self.step, "total": self.hp.total_updates},
                "identity": self.identity, "elapsed_seconds": self.elapsed()}

    def load_state_dict(self, s):
        """Restore the state saved by ``state_dict`` (optimizer, sampler, RNG, step, elapsed time)."""
        self.optimizer.load_state_dict(s["optimizer"])
        self.sampler.load_state_dict(s["sampler"])
        restore_rng(s["rng"])
        self.step = int(s["step"])
        self.elapsed_before = float(s.get("elapsed_seconds", 0.0))
        self._t0 = time.monotonic()

    def save_checkpoint(self, manager: CheckpointManager, reason="periodic", extra=None):
        """Write adapter weights + resume state through the ``CheckpointManager`` (atomic, sha256 manifest).
        ``reason`` is recorded in the manifest: periodic | deadline | emergency | final.
        """
        return manager.save(self.step, self.adapter.adapter_state_dict(), self.state_dict(),
                            extra={"architecture": self.adapter.architecture(), **(extra or {})}, reason=reason)


def peak_gib(device):
    """Peak CUDA memory allocated so far in GiB (0.0 on CPU)."""
    return torch.cuda.max_memory_allocated() / 2**30 if torch.device(device).type == "cuda" else 0.0


# =============================================================================
# 5. Warm start, logging, provenance, lease
# =============================================================================
def load_warm_start(adapter, path) -> dict:
    """Load a v1 adapter safetensors with a strict key/shape check and a report."""
    path = Path(path)
    state = load_file(str(path), device="cpu")
    expected = adapter.adapter_state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    bad_shape = sorted(k for k in set(expected) & set(state) if tuple(state[k].shape) != tuple(expected[k].shape))
    report = {"path": str(path), "sha256": sha256_file(path), "n_tensors": len(state),
              "missing": missing, "unexpected": unexpected, "shape_mismatch": bad_shape}
    if missing or unexpected or bad_shape:
        raise ValueError("Warm-start checkpoint does not match the adapter: " + json.dumps(report)[:2000])
    try:
        with safe_open(str(path), "pt") as f:
            arch = json.loads((f.metadata() or {}).get("architecture_json", "null"))
        report["architecture_matches"] = (arch is None) or arch.get("config") == adapter.architecture()["config"]
    except Exception:
        report["architecture_matches"] = None
    adapter.load_adapter_state_dict(state)
    return report


class ProgressLog:
    """Append-only ``progress.jsonl`` (fsynced per line) plus the latest record in ``progress.json``."""

    def __init__(self, run_dir):
        self.jsonl = Path(run_dir) / "progress.jsonl"
        self.latest = Path(run_dir) / "progress.json"
        Path(run_dir).mkdir(parents=True, exist_ok=True)

    def write(self, record):
        """Append one update record (fsynced) and atomically rewrite ``progress.json``."""
        slim = {k: v for k, v in record.items()}
        with open(self.jsonl, "a", encoding="utf8") as f:
            f.write(json.dumps(slim, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        atomic_write_json(self.latest, slim)

    def truncate_to(self, step):
        """On resume drop log lines from updates that were lost with the crash."""
        if not self.jsonl.exists():
            return 0
        keep, orphan = [], []
        for line in self.jsonl.read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            try:
                s = json.loads(line).get("step", 0)
            except Exception:
                s = 10**12
            (keep if s <= step else orphan).append(line)
        if orphan:
            with open(self.jsonl.with_name("progress_orphaned.jsonl"), "a", encoding="utf8") as f:
                f.write("\n".join(orphan) + "\n")
            tmp = self.jsonl.with_name("progress.jsonl.tmp")
            tmp.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf8")
            os.replace(tmp, self.jsonl)
        return len(orphan)


def source_hashes(src_dir=None):
    """sha256 of every ``src/fpcamera/*.py`` file, stored in ``config.json`` for provenance."""
    d = Path(src_dir) if src_dir else Path(__file__).parent
    return {p.name: sha256_file(p) for p in sorted(d.glob("*.py"))}


def git_info(repo_root):
    """``{'commit': HEAD sha, 'dirty': bool}`` of the repo, or None fields if git is unavailable."""
    def run(*a):
        """Run a git command in the repo, returning its stripped stdout or None on any failure."""
        try:
            return subprocess.run(["git", *a], cwd=repo_root, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return None
    commit = run("rev-parse", "HEAD") or None
    dirty = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def write_provenance(run_dir, cfg, extra):
    """``config.json`` is written on the first launch only; every launch appends ``launches.jsonl``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(cfg.get("project_root") or Path(__file__).resolve().parents[2])
    prov = {"config": cfg, "git": git_info(repo), "source_sha256": source_hashes(),
            "launched_unix_time": time.time(), **extra}
    if not (run_dir / "config.json").exists():
        atomic_write_json(run_dir / "config.json", prov)
    with open(run_dir / "launches.jsonl", "a", encoding="utf8") as f:
        f.write(json.dumps({k: prov[k] for k in ("git", "launched_unix_time")} | extra, default=str) + "\n")
    return prov


def run_identity(cfg, tables):
    """The part of the config that defines *what is being trained* (seed, schedule, objective, architecture,
    fill mode, video contract, split/families, cache index hashes + weights).

    Stored in every checkpoint; a resume with a different identity is refused unless explicitly allowed
    (``check_identity_change``).  Speed-only options (activation checkpointing, attention backend, compile) and
    checkpoint cadence are deliberately NOT part of it, so they can change between launches.
    """
    keys = ("seed", "updates", "effective_batch", "learning_rate", "weight_decay", "warmup", "min_lr_ratio",
            "grad_clip", "objective", "fill_mode", "architecture", "video_contract", "families", "split")
    ident = {k: cfg[k] for k in keys}
    ident["caches"] = [{"index_sha256": t.index_sha256, "weight": t.weight} for t in tables]
    return json.loads(json.dumps(ident, sort_keys=True))


def parse_identity_keys(value):
    """``'caches,seed'`` / iterable -> frozenset of keys (empty for None)."""
    if not value:
        return frozenset()
    items = value.split(",") if isinstance(value, str) else value
    return frozenset(k.strip() for k in items if k and k.strip())


def check_identity_change(saved, new, *, allow_all=False, allow_keys=()):
    """Compare run identities. Returns ``{key: {"old":..., "new":...}}`` for the differing keys when every one of
    them is permitted (``allow_all`` or listed in ``allow_keys``); raises ValueError otherwise."""
    allow_keys = parse_identity_keys(allow_keys)
    diff = sorted(k for k in set(saved) | set(new) if saved.get(k) != new.get(k))
    blocked = [k for k in diff if not allow_all and k not in allow_keys]
    if blocked:
        raise ValueError(f"Run identity changed since the checkpoint (keys: {diff}; not allowed: {blocked}); "
                         "use --allow-identity-change KEY[,KEY] (targeted), --allow-config-change (all) or a new --run-dir")
    return {k: {"old": saved.get(k), "new": new.get(k)} for k in diff}


def record_identity_change(run_dir, changes, allow_all):
    """Append an ``identity_change_accepted`` event (with old/new values) to ``launches.jsonl`` for audit."""
    with open(Path(run_dir) / "launches.jsonl", "a", encoding="utf8") as f:
        f.write(json.dumps({"event": "identity_change_accepted", "launched_unix_time": time.time(),
                            "allow_all": bool(allow_all), "changes": changes}, default=str) + chr(10))


def _pid_alive(pid):
    if os.name == "nt":
        return True  # cannot probe safely; use --break-lock
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextlib.contextmanager
def gpu_lease(root, break_lock=False):
    """Same lock file as ``runtime.gpu_lease`` but a stale lock (dead pid) is replaced.

    After a dropped session the lock file can survive; auto-resume must not be blocked by it.
    """
    lock = Path(root) / "artifacts" / "gpu.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                pid = int(lock.read_text().strip() or 0)
            except Exception:
                pid = 0
            if attempt == 0 and (break_lock or pid == 0 or pid == os.getpid() or not _pid_alive(pid)):
                lock.unlink(missing_ok=True)
                continue
            raise RuntimeError(f"Another job (pid {pid}) holds {lock}; use --break-lock if it is stale")
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


class StopRequest:
    """First SIGTERM/SIGINT: finish the current update, checkpoint, exit. Second: hard interrupt."""

    def __init__(self):
        self.requested = False
        self.signal = None

    def install(self):
        """Install SIGTERM/SIGINT handlers (main thread only)."""
        if threading.current_thread() is not threading.main_thread():
            return

        def handler(signum, _frame):
            """First signal: ask the loop to stop after the current update; second signal: raise KeyboardInterrupt."""
            if self.requested:
                raise KeyboardInterrupt
            self.requested, self.signal = True, signum

        for s in (signal.SIGTERM, signal.SIGINT):
            signal.signal(s, handler)


# =============================================================================
# 6. Full run (GPU; not unit-tested)
# =============================================================================
def log_sdpa_backends(log=print):
    """Report (never change) the SDPA kernels torch may dispatch to; diffusers Wan attention calls
    F.scaled_dot_product_attention via dispatch_attention_fn (native backend)."""
    try:
        b = torch.backends.cuda
        info = {"event": "sdpa_backends", "flash": b.flash_sdp_enabled(), "mem_efficient": b.mem_efficient_sdp_enabled(),
                "math": b.math_sdp_enabled(),
                "cudnn": b.cudnn_sdp_enabled() if hasattr(b, "cudnn_sdp_enabled") else None,
                "tf32_matmul": bool(b.matmul.allow_tf32), "torch": torch.__version__}
        log(json.dumps(info))
        return info
    except Exception as e:  # logging must never break a run
        log(json.dumps({"event": "sdpa_backends", "error": str(e)}))


def _load_transformer(log_path):
    from diffusers import WanTransformer3DModel
    from .wan import MODEL, REVISION, configure_library_output, require_pinned_runtime

    configure_library_output(log_path)
    versions = require_pinned_runtime()
    model = WanTransformer3DModel.from_pretrained(
        MODEL, subfolder="transformer", revision=REVISION, torch_dtype=torch.bfloat16, local_files_only=True)
    model.requires_grad_(False)
    model.eval()
    model.to("cuda")
    log_sdpa_backends()
    if model.config.in_channels != 48 or tuple(model.config.patch_size) != (1, 2, 2):
        raise RuntimeError("Unexpected TI2V-5B architecture")
    return model, versions


def train_until(trainer, manager, plog, *, stop_step, stop, deadline=None, clock=time.time, progress=print,
                max_peak_gib=1e9, est_update=45.0, est_ckpt=60.0, grace=UPLOAD_GRACE_SECONDS,
                first_update_compiles=False, compile_allowance=COMPILE_ALLOWANCE_SECONDS):
    """Update loop with periodic checkpoints and the deadline stop rule.

    Before each update: if ``now + est_update + est_ckpt + grace > deadline`` stop (the caller writes the
    final checkpoint).  ``est_update`` is an EMA of measured s/update, ``est_ckpt`` the last measured
    checkpoint write time.  Returns ``{"mem_stop", "deadline_stop"}``.

    ``first_update_compiles`` (--compile-blocks): the first update of this launch includes torch.compile time; it is
    logged separately (``compile_first_update``, then ``compile_overhead_estimate`` once a steady update is known) and kept
    out of the s/update EMA, so the deadline rule is not thrown off by a one-off compile.  Before that first update the
    rule assumes ``compile_allowance`` extra seconds, so a launch does not start a compiling update it cannot finish.
    """
    ema, ckpt_s = Ema(est_update), float(est_ckpt)
    mem_stop = deadline_stop = False
    n_done, first_dt = 0, None
    while trainer.step < stop_step and not stop.requested:
        est = ema.value + (compile_allowance if first_update_compiles and n_done == 0 else 0.0)
        if should_stop(clock(), deadline, est, ckpt_s, grace):
            deadline_stop = True
            progress(f"DEADLINE_STOP step={trainer.step} est_update_s={est:.1f} est_ckpt_s={ckpt_s:.1f} "
                     f"seconds_left={deadline - clock():.0f}")
            break
        t0 = clock()
        rec = trainer.run_update()
        dt = clock() - t0
        if first_update_compiles and n_done == 0:
            first_dt = dt
            rec["includes_compile"] = True  # its update_seconds in progress.jsonl is not a steady-state step time
            progress(json.dumps({"event": "compile_first_update", "seconds": round(dt, 1),
                                 "note": "includes torch.compile; excluded from the deadline s/update estimate"}))
        else:
            ema.update(dt)
            if first_update_compiles and n_done == 1:
                progress(json.dumps({"event": "compile_overhead_estimate", "first_update_seconds": round(first_dt, 1),
                                     "steady_update_seconds": round(dt, 1),
                                     "compile_seconds_estimate": round(max(first_dt - dt, 0.0), 1)}))
        n_done += 1
        plog.write(rec)
        progress(json.dumps({k: v for k, v in rec.items() if k != "micro"}, default=str))
        if rec["peak_gib"] > float(max_peak_gib):
            progress(json.dumps({"event": "peak_memory_budget_exceeded", "peak_gib": rec["peak_gib"]}))
            stop.requested = mem_stop = True
        if manager.due(trainer.step):
            t1 = clock()
            trainer.save_checkpoint(manager, "periodic")
            ckpt_s = max(clock() - t1, 1e-3)
    return {"mem_stop": mem_stop, "deadline_stop": deadline_stop}


def run_training(config, *, run_dir, caches=None, init_from=None, stop_after=None, resume=True,
                 check_only=False, break_lock=False, allow_config_change=False, allow_identity_change=(), progress=print, stop=None,
                 deadline=None, ckpt_blocks=None, probe=False, probe_limit_gib=85.0, attn_backend="auto",
                 compile_blocks=False, speed_probe_config=None):
    """Returns a status dict; ``status`` is ``complete``, ``interrupted`` or ``deadline`` (clean stop, exit 0).

    Speed-only options, none part of the run identity: ``ckpt_blocks`` (activation checkpointing), ``attn_backend``
    ("auto" | "cudnn" | "flash" | "efficient": SDPA kernel priority) and ``compile_blocks`` (regional torch.compile);
    ``probe`` = memory probe, ``speed_probe_config`` (a ``speed_options.SpeedProbeConfig``) = speed probe, both with
    ``check_only``.  Defaults reproduce the previous behaviour exactly."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    from .wan import hash_module

    cfg = normalize_config(config)
    root = Path(cfg["project_root"] or Path(__file__).resolve().parents[2]).resolve()
    run_dir = Path(run_dir)
    cache_specs = caches or cfg["caches"]
    if not cache_specs:
        raise ValueError("No caches given (config 'caches' or --cache)")
    tables = []
    for i, spec in enumerate(cache_specs):
        idx, w = (spec["index"], spec.get("weight", 1.0)) if isinstance(spec, dict) else spec
        idx = Path(idx) if Path(idx).is_absolute() else root / idx
        tables.append(load_cache_table(idx, i, w, split=cfg["split"], families=cfg["families"], project_root=root))
    hp = hparams_from_config(cfg)
    identity = run_identity(cfg, tables)
    n_sources = {t.index_path: len(t.sources) for t in tables}
    progress(json.dumps({"event": "tables_loaded", "sources": n_sources, "dropped": [len(t.dropped) for t in tables]}))
    stop = stop or StopRequest()
    if not check_only:
        write_provenance(run_dir, cfg, {"init_from": str(init_from) if init_from else None, "caches": n_sources,
                                        "check_only": False, "resume_enabled": resume})
        stop.install()
    init_from = init_from or cfg.get("init_from")
    if init_from and not Path(init_from).is_absolute() and not Path(init_from).exists():
        init_from = root / init_from
    plog = ProgressLog(run_dir) if not check_only else None
    manager = None if check_only else CheckpointManager(run_dir, **{
        k: cfg["checkpoint"][k] for k in ("every_updates", "every_seconds", "keep_last", "keep_every")})
    status = {"status": "failed", "step": 0}
    with gpu_lease(root, break_lock):
        model, versions = _load_transformer(
            (root / "artifacts/runtime" if check_only else run_dir) / "library.log")
        original = tuple(model.parameters())
        before = hash_module(model, parameters=original) if cfg["verify_backbone_hash"] else None
        if cfg["expected_backbone_sha256"] and before and before != cfg["expected_backbone_sha256"]:
            raise RuntimeError("Frozen backbone differs from the verified baseline")
        contract = cfg["video_contract"]
        loader = MicrobatchLoader(contract, cfg["fill_mode"], device="cuda")
        sampler = SourceSampler(tables, seed=hp.seed, p_static=hp.p_static, p_text_dropout=hp.p_text_dropout)

        # attach-is-identity probe (as v1), on the first static clip
        probe_entry = next(iter(tables[0].sources.values()))
        pclip = loader(Plan(0, probe_entry.source_key, probe_entry.static, probe_entry.static, True, False))
        sig = torch.tensor([0.8], device="cuda")
        pnoise = torch.randn(pclip.static.shape, generator=_gen(pclip.static.device, hp.seed + 177),
                             device="cuda", dtype=pclip.static.dtype)
        px = _noisy_input(pclip.static, pclip.first, sig, pnoise)
        pt = token_timestep(pclip.static, sig)
        with torch.no_grad(), _autocast(pclip.static.device):
            frozen = _call(model, px, pt, pclip.text)
        adapter = flow_control.FlowControlLoRA(model, flow_control.FlowControlConfig(**cfg["architecture"]))
        adapter.set_condition(pclip.condition)
        with torch.no_grad(), _autocast(pclip.static.device):
            zero = _call(model, px, pt, pclip.text)
        # attach-is-identity probe: the adapter is zero-initialised (guide projection and every LoRA-B are 0), so attaching it must not change
        # the frozen prediction AT ALL: compare against the prediction taken before the adapter existed (exactly 0 diff).
        if float((frozen - zero).abs().max()) != 0:
            raise RuntimeError("Initial adapter changed the frozen Wan prediction")
        del pclip, px, pt, pnoise, frozen, zero

        params = [p for p in adapter.parameters() if p.requires_grad]
        # safety net: the optimizer must only ever see adapter parameters, never the frozen Wan weights.
        if {id(p) for p in params} & {id(p) for p in original}:
            raise RuntimeError("Original parameters entered the optimizer")
        optimizer = torch.optim.AdamW(params, lr=hp.learning_rate, weight_decay=hp.weight_decay, fused=True)
        attn_spec = so.resolve_attn_backend(attn_backend)
        compile_on = bool(compile_blocks)
        if compile_on:
            so.set_block_compile(model, True)  # before checkpointing: the compiled forward goes inside the wrapper
        ck_spec = checkpointing_spec(cfg, ckpt_blocks)
        ck_blocks = apply_activation_checkpointing(model, ck_spec)
        progress(json.dumps({"event": "activation_checkpointing", "spec": ck_spec, "blocks": len(ck_blocks)}))
        progress(json.dumps({"event": "speed_options", "attn_backend": attn_spec.record(),
                             "compile_blocks": compile_on, "compile": {"mode": "default", "dynamic": False}
                             if compile_on else None}))
        if attn_spec.active:
            so.log_attn_backend(attn_spec, progress)
        model.train()
        trainer = HybridTrainer(model, adapter, optimizer, hp, sampler, loader, device="cuda",
                                frozen_params=original, identity=identity, attn_backend=attn_spec)

        resumed_from = warm = None
        if manager is not None:
            if not resume:
                archived = manager.archive_existing()
                if archived:
                    progress(json.dumps({"event": "archived_previous_state", "to": str(archived)}))
            loaded = manager.load_latest() if resume else None
            if loaded is not None:
                saved_id = loaded.resume_state.get("identity", {})
                changes = check_identity_change(saved_id, identity, allow_all=allow_config_change,
                                                allow_keys=allow_identity_change)
                if changes:
                    progress(json.dumps({"event": "identity_change_accepted", "changes": changes}, default=str))
                    record_identity_change(run_dir, changes, allow_config_change)
                adapter.load_adapter_state_dict(loaded.adapter_state)
                trainer.load_state_dict(loaded.resume_state)
                # resume: log lines of updates beyond the checkpoint step (lost with a crash) are moved to progress_orphaned.jsonl
                # so progress.jsonl stays consistent with the model state.
                dropped = plog.truncate_to(trainer.step)
                resumed_from = {"step": loaded.step, "skipped_corrupt": loaded.skipped, "orphaned_log_lines": dropped}
                progress(json.dumps({"event": "resumed", **resumed_from}))
            elif init_from:
                warm = load_warm_start(adapter, init_from)
                progress(json.dumps({"event": "warm_start", **warm}))
        elif init_from:
            warm = load_warm_start(adapter, init_from)

        if check_only and probe:
            report = memory_probe(trainer, model, limit_gib=probe_limit_gib, progress=progress,
                                  compile_blocks=compile_on)
            progress(json.dumps(report))
            return {"status": "check_only", "step": trainer.step, **report}
        if check_only and speed_probe_config is not None:
            report = speed_probe(trainer, model, speed_probe_config, progress=progress)
            progress(json.dumps(report, default=str))
            return {"status": "check_only", "step": trainer.step, **report}
        if check_only:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.monotonic()
            rec = trainer.run_update(accumulate_only=True)
            report = {"event": "check_only", "peak_gib": peak_gib("cuda"), "seconds": time.monotonic() - t0,
                      "grad_norm": rec["grad_norm"], "micro": rec["micro"], "warm_start": warm,
                      "versions": versions, "backbone_sha256": before}
            progress(json.dumps(report, default=str))
            return {"status": "check_only", "step": trainer.step, **report}

        torch.cuda.reset_peak_memory_stats()
        stop_step = min(hp.total_updates, int(stop_after)) if stop_after else hp.total_updates
        # deadline handling: train_until stops *before* an update that would not leave time for a checkpoint plus the
        # upload grace; the code below then writes the final/deadline checkpoint, run_status.json and exits cleanly
        # (exit 0), so the next session resumes from latest.json with identical data order and RNG.
        interrupted = mem_stop = deadline_stop = failed = False
        failed = True
        try:
            res = train_until(trainer, manager, plog, stop_step=stop_step, stop=stop, deadline=deadline,
                              progress=progress, max_peak_gib=cfg["max_peak_gib"],
                              first_update_compiles=compile_on)
            mem_stop, deadline_stop = res["mem_stop"], res["deadline_stop"]
            interrupted = stop.requested and trainer.step < stop_step and not mem_stop
            if manager.last_saved_step != trainer.step:
                trainer.save_checkpoint(manager, "deadline" if deadline_stop else
                                        "emergency" if (interrupted or mem_stop) else "final")
            failed = False
        except KeyboardInterrupt:
            interrupted, failed = True, False
            if not trainer.inside_update and manager.last_saved_step != trainer.step:
                trainer.save_checkpoint(manager, "emergency")
        finally:
            after = hash_module(model, parameters=original) if cfg["verify_backbone_hash"] else None
            name = ("failed" if failed else "deadline" if deadline_stop else "interrupted" if interrupted
                    else "memory_budget_exceeded" if mem_stop else "complete")
            status = {"status": name, "step": trainer.step,
                      "stop_step": stop_step, "backbone_unchanged": before == after,
                      "resumed_from": resumed_from, "warm_start": warm,
                      "stopped_reason": "deadline" if deadline_stop else name,
                      "deadline_epoch": deadline}
            atomic_write_json(run_dir / "run_status.json", status)
            if deadline_stop:
                progress(f"DEADLINE_STOP step={trainer.step} checkpoint_saved=true")
            if before != after:
                raise RuntimeError("Original frozen backbone changed")
    return status
