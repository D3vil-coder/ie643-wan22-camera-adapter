"""Inference-only frozen-base anchoring of a learned camera-condition difference.

Pinned Wan 0.36 evaluates positive text before negative text, in separate calls.
The wrapper checks that order using stable text tensor storage identities and
exposes call counters for a per-denoising-step assertion in the caller.
"""

from __future__ import annotations

from contextlib import contextmanager
import math


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

    def original(*args, **kwargs):
        state["original_forward_calls"] += 1
        result = forward(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 1:
            raise TypeError("Expected native return_dict=False one-tensor prediction")
        return result

    def anchored(*args, **kwargs):
        nonlocal positive_ptr, negative_ptr, pending_sigma
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
        state["native_calls"] += 1
        try:
            with adapter.disabled():
                baseline = original(*args, **kwargs)
            if not positive or sigma < active_min_sigma or camera_gain == 0:
                return baseline
            if reference == "static":
                adapter.set_condition(static)
                null = original(*args, **kwargs)
            else:
                null = baseline
            adapter.set_condition(requested)
            controlled = original(*args, **kwargs)
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
