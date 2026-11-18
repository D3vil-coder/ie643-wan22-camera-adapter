"""Optional inference-only camera guidance against a same-image static hint.

The original transformer parameters are never modified. Gain one with no
sigma threshold is an exact passthrough, including the number of model calls.
An optional minimum-sigma gate disables the adapter below that noise level.
Text CFG stays in the native pipeline.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import numbers


@contextmanager
def camera_prediction_guidance(transformer, adapter, requested, static, gain=1.,
                               active_min_sigma=0.):
    """For each text condition return v_static + gain*(v_requested-v_static).

    With a positive active_min_sigma, native timestep values are converted to
    noise coordinates by max(timestep)/1000. Below that threshold the original
    forward runs once inside adapter.disabled(); otherwise the existing gain
    behavior applies. For gain one and threshold zero this is an exact
    passthrough. Non-unit gains use a static view as reference, so shared LoRA
    corrections remain in both predictions. This requires the native
    inference return_dict=False tuple contract and must not run concurrently
    with another forward on the same transformer instance.
    """
    gain = float(gain)
    if not math.isfinite(gain) or gain < 0:
        raise ValueError('Camera guidance must be finite and nonnegative')
    active_min_sigma = float(active_min_sigma)
    if not math.isfinite(active_min_sigma) or not 0. <= active_min_sigma <= 1.:
        raise ValueError('Minimum active sigma must be finite and between zero and one')
    if gain == 1. and active_min_sigma == 0.:
        yield
        return
    forward = transformer.forward
    had_instance_forward = 'forward' in transformer.__dict__
    original_instance_forward = transformer.__dict__.get('forward')

    def guided_forward(*args, **kwargs):
        try:
            if active_min_sigma > 0.:
                if 'timestep' not in kwargs:
                    raise ValueError('A timestep keyword is required when minimum active sigma is enabled')
                timestep = kwargs['timestep']
                if isinstance(timestep, numbers.Real):
                    maximum_timestep = float(timestep)
                else:
                    amax = getattr(timestep, 'amax', None)
                    if not callable(amax):
                        raise TypeError('Timestep must be numeric or provide amax().item()')
                    maximum_timestep = amax()
                    item = getattr(maximum_timestep, 'item', None)
                    if callable(item):
                        maximum_timestep = item()
                    if not isinstance(maximum_timestep, numbers.Real):
                        raise TypeError('Timestep amax() must produce a numeric scalar')
                    maximum_timestep = float(maximum_timestep)
                noise_level = maximum_timestep / 1000.
                if not math.isfinite(noise_level):
                    raise ValueError('Timestep must be finite')
                if noise_level < active_min_sigma:
                    with adapter.disabled():
                        return forward(*args, **kwargs)
            if gain == 1.:
                return forward(*args, **kwargs)
            adapter.set_condition(static)
            null_prediction = forward(*args, **kwargs)
            if gain == 0.:
                return null_prediction
            adapter.set_condition(requested)
            prediction = forward(*args, **kwargs)
            if (not isinstance(prediction, tuple) or len(prediction) != 1 or
                    not isinstance(null_prediction, tuple) or len(null_prediction) != 1):
                raise TypeError('Camera guidance requires a one-tensor tuple prediction')
            return (null_prediction[0] + gain * (prediction[0] - null_prediction[0]),)
        finally:
            adapter.set_condition(requested)

    transformer.forward = guided_forward
    try:
        yield
    finally:
        if had_instance_forward:
            transformer.forward = original_instance_forward
        else:
            del transformer.forward
        adapter.set_condition(requested)
