"""Inference-only frozen-base anchoring of a learned camera-condition difference.

Pinned Wan 0.36 evaluates positive text before negative text, in separate calls.
The wrapper checks that order using stable text tensor storage identities and
exposes call counters for a per-denoising-step assertion in the caller.
"""

from __future__ import annotations

from contextlib import contextmanager
import math

import torch


def parse_residual_lowfreq(spec):
    """Normalise a residual low-frequency filter spec.

    ``None`` / ``""`` / ``"none"`` -> ``None`` (default, the residual is used unchanged);
    ``"mean"`` -> ``("mean", None)``: remove the per-plane, per-channel spatial mean of the residual (F1);
    ``"blur:K"`` -> ``("blur", K)``: remove a Gaussian low-pass of standard deviation ``K`` latent cells (F2).
    Tuples are returned unchanged (after validation).
    """
    if spec is None:
        return None
    if isinstance(spec, (tuple, list)):
        kind, arg = spec
    else:
        text = str(spec).strip().lower()
        if text in ("", "none", "off"):
            return None
        kind, _, arg = text.partition(":")
        arg = arg or None
    if kind == "mean":
        if arg is not None:
            raise ValueError("residual low-frequency filter 'mean' takes no argument")
        return ("mean", None)
    if kind == "blur":
        try:
            sigma = float(arg)
        except (TypeError, ValueError):
            raise ValueError("residual low-frequency filter must be 'mean' or 'blur:K' with K > 0") from None
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("residual low-frequency blur sigma must be finite and > 0")
        return ("blur", sigma)
    raise ValueError(f"Unknown residual low-frequency filter {spec!r} (use 'none', 'mean' or 'blur:K')")


def _gaussian_matrix(n, sigma, like):
    """Row-normalised Gaussian smoothing matrix ``[n, n]``; row i = weights of cell i (zero padding, renormalised)."""
    idx = torch.arange(n, device=like.device, dtype=torch.float32)
    w = torch.exp(-((idx[:, None] - idx[None, :]) ** 2) / (2.0 * float(sigma) ** 2))
    return w / w.sum(dim=1, keepdim=True)


MIN_WEIGHT_FRACTION = 0.05  # a plane with less valid weight than this (fraction of cells) falls back to the unweighted filter


def lowfreq_highpass(residual, spec, weight=None):
    """Remove the low spatial frequencies of ``residual`` (``[..., H, W]``, per plane and channel).

    ``spec`` as in ``parse_residual_lowfreq``. ``("mean", None)`` subtracts the spatial mean over the last two
    dims; ``("blur", K)`` subtracts a Gaussian low-pass (sigma K cells, border-renormalised so a constant maps to
    itself). ``None`` returns the input. The result has zero spatial mean for ``mean`` and keeps all structure
    finer than about K cells for ``blur``. Works for any tensor with >= 2 dims.

    ``weight`` (optional, broadcastable to ``residual``, values in [0, 1], e.g. the guide's validity channel): the
    low-frequency part is *estimated* from the weighted cells only (weighted mean / normalised convolution), so
    content that is genuinely new (revealed region, weight 0) does not pollute the estimate, and it is subtracted
    everywhere. Planes or cells with almost no weight fall back to the unweighted estimate.
    """
    spec = parse_residual_lowfreq(spec)
    if spec is None:
        return residual
    kind, arg = spec
    if weight is not None:
        w = weight.to(device=residual.device, dtype=residual.dtype).expand(*residual.shape[:-2], *weight.shape[-2:])
        total = w.sum(dim=(-2, -1), keepdim=True)
        enough = total >= MIN_WEIGHT_FRACTION * residual.shape[-2] * residual.shape[-1]
        if kind == "mean":
            weighted = (w * residual).sum(dim=(-2, -1), keepdim=True) / total.clamp_min(1e-6)
            low = torch.where(enough, weighted, residual.mean(dim=(-2, -1), keepdim=True))
            return residual - low
        height, width = residual.shape[-2], residual.shape[-1]
        gh = _gaussian_matrix(height, arg, residual).to(residual.dtype)
        gw = _gaussian_matrix(width, arg, residual).to(residual.dtype)

        def smooth(x):
            return torch.matmul(torch.matmul(gh, x), gw.transpose(0, 1))

        den = smooth(w)
        weighted = smooth(w * residual) / den.clamp_min(1e-6)
        plain = smooth(residual)
        low = torch.where(enough & (den > 0.02), weighted, plain)
        return residual - low
    if kind == "mean":
        return residual - residual.mean(dim=(-2, -1), keepdim=True)
    height, width = residual.shape[-2], residual.shape[-1]
    gh, gw = _gaussian_matrix(height, arg, residual), _gaussian_matrix(width, arg, residual)
    low = torch.matmul(torch.matmul(gh.to(residual.dtype), residual), gw.to(residual.dtype).transpose(0, 1))
    return residual - low


def residual_stats(residual, sigma):
    """Diagnostic statistics of a camera residual ``[1, C, T, H, W]`` (frames 1.. only; frame 0 is the clean image plane)."""
    r = residual[..., 1:, :, :] if residual.ndim >= 5 and residual.shape[-3] > 1 else residual
    mean_sp = r.mean(dim=(-2, -1), keepdim=True)
    low8 = r - lowfreq_highpass(r, ("blur", 8.0))
    chan_mean = r.mean(dim=tuple(d for d in range(r.ndim) if d != r.ndim - 4)) if r.ndim >= 4 else r.mean().reshape(1)
    return {
        "sigma": float(sigma),
        "rms": float(r.pow(2).mean().sqrt()),
        "rms_spatial_mean": float(mean_sp.pow(2).mean().sqrt()),
        "rms_lowpass8": float(low8.pow(2).mean().sqrt()),
        "channel_mean": [round(float(v), 5) for v in chan_mean.flatten()],
    }


def _sigma(timestep):
    value = timestep.amax().item() if hasattr(timestep, "amax") else float(timestep)
    value = float(value) / 1000.0
    if not math.isfinite(value):
        raise ValueError("Non-finite native flow timestep")
    return value


@contextmanager
def frozen_camera_difference(
    transformer,
    adapter,
    requested,
    static,
    *,
    camera_gain=1.0,
    text_guidance=5.0,
    active_min_sigma=0.6,
    reference="static",
    batch_forwards=True,
    residual_lowfreq=None,
    trace_residual=False,
    lowfreq_weight=None,
):
    """Native text CFG(F0) + camera_gain * (Fphi(text,cam)-Fphi(text,static)).

    The positive prediction returned to native text CFG includes a factor
    camera_gain/text_guidance. Native CFG multiplies it back, giving the exact
    independent camera_gain above. Negative-text predictions use F0 only.
    Four original forwards per active CFG step, two per inactive step. This is
    a pinned separate-pass CFG protocol, not compatible with concatenated CFG.
    With reference='teacher', subtract the already computed frozen positive
    prediction instead; this uses three forwards per active CFG step and
    matches the teacher-relative requested-condition training target directly.

    batch_forwards (lossless speed option, default on): once the negative-text tensor has been seen (from
    step 2 on), the frozen positive and frozen negative predictions of a step are computed in ONE batch-2
    forward (adapter disabled, so no per-sample adapter handling is needed) at the positive call; the
    negative call then returns its cached half after verifying that its latents/timestep/text are exactly
    the ones that were batched, else it recomputes normally.  The adapted passes stay batch-1.  Same math;
    results differ from sequential calls only at kernel-nondeterminism level.

    residual_lowfreq (opt-in, default None = unchanged behaviour): high-pass filter of the camera residual
    (adapter-on minus reference) before it is added, in latent space, per plane and channel:
    ``"mean"`` removes the spatial mean (the adapter then cannot shift a frame's global colour/brightness),
    ``"blur:K"`` removes a Gaussian low-pass of K latent cells. See ``lowfreq_highpass``.

    lowfreq_weight (opt-in, needs residual_lowfreq): weights in [0, 1] broadcastable to the residual (the guide's
    validity channel ``condition[:, 48:49]``, ``[1, 1, T, h, w]``): the low-frequency part is estimated from valid cells
    only and removed everywhere (see ``lowfreq_highpass``).

    trace_residual (diagnostic, default False = no extra work): append per active step to
    ``state["residual_trace"]`` the noise level and statistics of the raw residual (before any filter) over the
    latent frames 1.. : its rms, the rms of its per-frame/per-channel spatial mean (the part ``mean`` would remove),
    the rms of its Gaussian low-pass of 8 cells, and the per-channel mean over time and space (48 numbers).
    """
    camera_gain = float(camera_gain)
    text_guidance = float(text_guidance)
    active_min_sigma = float(active_min_sigma)
    if reference not in ("static", "teacher"):
        raise ValueError("Unknown camera residual reference")
    if not math.isfinite(camera_gain) or camera_gain < 0:
        raise ValueError("Invalid camera gain")
    if not math.isfinite(text_guidance) or text_guidance < 1:
        raise ValueError("Text guidance must be >=1")
    if not math.isfinite(active_min_sigma) or not 0 <= active_min_sigma <= 1:
        raise ValueError("Invalid noise window")
    lowfreq = parse_residual_lowfreq(residual_lowfreq)
    cfg = text_guidance > 1
    state = {
        "residual_lowfreq": None if lowfreq is None else list(lowfreq),
        "residual_lowfreq_weighted": lowfreq_weight is not None and lowfreq is not None,
        "native_calls": 0,
        "original_forward_calls": 0,
        "batched_forward_calls": 0,
        "batch_forwards": bool(batch_forwards),
        "active_positive_calls": 0,
        "native_calls_per_step": 2 if cfg else 1,
        "formula": (
            "CFG_text(F0) + camera_gain * (Fphi_positive(camera) - Fphi_positive(static))"
            if reference == "static"
            else "CFG_text(F0) + camera_gain * (Fphi_positive(camera) - F0_positive)"
        ),
        "reference_prediction": "adapted_static"
        if reference == "static"
        else "frozen_positive",
        "combination_precision": "FP32 residual addition, cast to native prediction dtype before text CFG",
    }
    forward = transformer.forward
    had_forward = "forward" in transformer.__dict__
    saved_forward = transformer.__dict__.get("forward")
    positive_ptr = None
    negative_ptr = None
    pending_sigma = None
    negative_text = None  # reference to the negative text tensor (stable storage is asserted below)
    cached_negative = None  # (hidden_states, timestep, text, prediction) from a batched positive call

    def batched_frozen(args, kwargs):
        """Frozen positive+negative in one batch-2 forward; returns (positive_result, negative_result)."""
        h, t, txt = kwargs["hidden_states"], kwargs["timestep"], kwargs["encoder_hidden_states"]
        merged = dict(kwargs)
        merged["hidden_states"] = torch.cat((h, h), 0)
        merged["timestep"] = torch.cat((t, t), 0)
        merged["encoder_hidden_states"] = torch.cat((txt, negative_text), 0)
        state["batched_forward_calls"] += 1
        out = original(*args, **merged)[0]
        if out.shape[0] != 2:
            raise RuntimeError("Batched frozen forward returned an unexpected batch")
        return (out[:1],), (out[1:],)

    def can_batch(kwargs):
        """True when this positive call can run frozen-positive + frozen-negative as one batch-2 forward (CFG on,
        negative text already seen, batch-1 inputs of matching shape).
        """
        h, t = kwargs.get("hidden_states"), kwargs.get("timestep")
        return (
            batch_forwards
            and cfg
            and negative_text is not None
            and hasattr(h, "shape")
            and hasattr(t, "shape")
            and h.shape[0] == 1
            and t.shape[0] == 1
            and "hidden_states" in kwargs
            and kwargs["encoder_hidden_states"].shape == negative_text.shape
        )

    def original(*args, **kwargs):
        """Run the transformer's native forward (counting calls); enforces the ``return_dict=False`` tuple contract."""
        state["original_forward_calls"] += 1
        result = forward(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 1:
            raise TypeError("Expected native return_dict=False one-tensor prediction")
        return result

    def anchored(*args, **kwargs):
        """Replacement ``transformer.forward``.  Identifies positive vs negative text calls (native separate-pass CFG
        order is asserted via text-tensor storage), computes the frozen baseline F0 for every call and, for positive
        calls inside the active sigma window, adds ``camera_gain/text_guidance * (F_requested - F_reference)``.
        """
        nonlocal positive_ptr, negative_ptr, pending_sigma, negative_text, cached_negative
        if "timestep" not in kwargs or "encoder_hidden_states" not in kwargs:
            raise ValueError("Pinned keyword timestep and text conditioning required")
        sigma = _sigma(kwargs["timestep"])
        positive = (not cfg) or state["native_calls"] % 2 == 0
        text = kwargs["encoder_hidden_states"]
        if not hasattr(text, "data_ptr"):
            raise TypeError("Native text tensor required")
        pointer = text.data_ptr()
        if positive:
            if positive_ptr is None:
                positive_ptr = pointer
            if pointer != positive_ptr:
                raise RuntimeError("Positive text storage/order changed")
            pending_sigma = sigma
        else:
            if negative_ptr is None:
                negative_ptr = pointer
            if (
                pointer != negative_ptr
                or pointer == positive_ptr
                or sigma != pending_sigma
            ):
                raise RuntimeError("Native separate-pass CFG order changed")
            negative_text = text
        state["native_calls"] += 1
        try:
            if not positive and cached_negative is not None:
                h, t, txt, result = cached_negative
                cached_negative = None
                if (
                    txt is text
                    and torch.equal(h, kwargs.get("hidden_states"))
                    and torch.equal(t, kwargs.get("timestep"))
                ):
                    return result
            baseline = None
            if positive and can_batch(kwargs):
                with adapter.disabled():
                    baseline, negative_result = batched_frozen(args, kwargs)
                cached_negative = (
                    kwargs["hidden_states"],
                    kwargs["timestep"],
                    negative_text,
                    negative_result,
                )
            if baseline is None:
                with adapter.disabled():
                    baseline = original(*args, **kwargs)
            # negative-text calls and low-noise steps (sigma < camera_min_sigma) return the frozen baseline F0 untouched: the adapter
            # only ever influences the positive prediction at the noise levels it was trained for.
            if not positive or sigma < active_min_sigma or camera_gain == 0:
                return baseline
            if reference == "static":
                adapter.set_condition(static)
                null = original(*args, **kwargs)
            else:
                # 'teacher' reference: the frozen positive prediction already computed above is the 'no camera' reference, which is
                # exactly what pass A's target (teacher + edit) was built from, so training and inference use the same residual.
                null = baseline
            adapter.set_condition(requested)
            controlled = original(*args, **kwargs)
            # the native pipeline multiplies the positive prediction by text_guidance inside CFG (F_neg + g*(F_pos - F_neg)); dividing the
            # residual by g here makes the net camera contribution exactly camera_gain * (F_requested - F_reference), independent of g.
            factor = camera_gain / text_guidance if cfg else camera_gain
            residual = controlled[0].float() - null[0].float()
            if trace_residual:
                state["residual_trace"].append(residual_stats(residual, sigma))
            if lowfreq is not None:
                residual = lowfreq_highpass(residual, lowfreq, lowfreq_weight)
            combined = (baseline[0].float() + factor * residual).to(
                dtype=baseline[0].dtype
            )
            state["active_positive_calls"] += 1
            return (combined,)
        finally:
            adapter.set_condition(requested)

    if trace_residual:
        state["residual_trace"] = []
    transformer.forward = anchored
    try:
        yield state
    finally:
        if had_forward:
            transformer.forward = saved_forward
        else:
            del transformer.forward
        adapter.set_condition(requested)
