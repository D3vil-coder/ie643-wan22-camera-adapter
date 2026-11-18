"""Inference-only frozen-base anchoring of a learned camera-condition difference.

Pinned Wan 0.36 evaluates positive text before negative text, in separate calls.
The wrapper checks that order using stable text tensor storage identities and
exposes call counters for a per-denoising-step assertion in the caller.
"""

from __future__ import annotations

from contextlib import contextmanager
import math

import torch


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
    cfg = text_guidance > 1
    state = {
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
            combined = (baseline[0].float() + factor * residual).to(
                dtype=baseline[0].dtype
            )
            state["active_positive_calls"] += 1
            return (combined,)
        finally:
            adapter.set_condition(requested)

    transformer.forward = anchored
    try:
        yield state
    finally:
        if had_forward:
            transformer.forward = saved_forward
        else:
            del transformer.forward
        adapter.set_condition(requested)
