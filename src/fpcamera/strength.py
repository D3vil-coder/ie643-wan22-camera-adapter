"""Strength calibration for the v2 single-slider camera control.

The UI exposes one strength ``s`` in [0, 1] per motion direction (``s = 1`` is
the trained maximum: 8 degrees of pan/tilt, or a 1.25x focal zoom). The model
is trained on a requested strength ``u`` (same units) but does not realise it
exactly, so a per-direction monotone map ``s -> u`` is fitted from measured
motion (``scripts/calibrate_strength.py``) and applied by ``flow_infer`` and the
interface. With no calibration the map is the identity (``u = s``).

Calibration file schema (``configs/strength_calibration_v2*.json``)::

    {
      "version": 1,
      "created": "...",
      "checkpoint_sha256": "..." | null,
      "max_amplitudes": {"yaw_deg": 8.0, "pitch_deg": 8.0, "zoom_ratio": 1.25},
      "directions": {"pan_left": {"s": [0, ..., 1], "u": [0, ..., 1]}, ... 6 keys},
      "fit": { ...diagnostics, free-form... }
    }

Everything here is pure numpy/stdlib so it runs (and is tested) on CPU.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import warnings

import numpy as np

from .flow_geometry import (
    V2_MAX_PITCH_DEG,
    V2_MAX_YAW_DEG,
    V2_MAX_ZOOM_RATIO,
    motion_amplitudes,
)

CALIBRATION_VERSION = 1
DIRECTIONS = ("pan_left", "pan_right", "tilt_up", "tilt_down", "zoom_in", "zoom_out")
MOTION_LABELS = {
    "pan_left": "Pan left",
    "pan_right": "Pan right",
    "tilt_up": "Tilt up",
    "tilt_down": "Tilt down",
    "zoom_in": "Zoom in",
    "zoom_out": "Zoom out",
    "static": "Static",
}
DEFAULT_MAX_AMPLITUDES = {
    "yaw_deg": V2_MAX_YAW_DEG,
    "pitch_deg": V2_MAX_PITCH_DEG,
    "zoom_ratio": V2_MAX_ZOOM_RATIO,
}
_EPS = 1e-9
# A maximum observed amplitude within this distance of 1 counts as "reaches full strength"
# (tracker noise is a few percent; flagging a 0.99 maximum as saturation would be spurious).
SATURATION_TOL = 0.02


def split_motion(motion: str):
    """``"pan_left"`` -> ``("pan", "left")``; ``"static"`` -> ``("static", None)``."""
    if motion == "static":
        return "static", None
    if motion not in DIRECTIONS:
        raise ValueError(f"motion must be one of {DIRECTIONS + ('static',)}, got {motion!r}")
    family, direction = motion.split("_")
    return family, direction


def file_sha256(path) -> str:
    """Streaming sha256 of a file (used to record which calibration file produced a generation)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- amplitudes
def load_max_amplitudes(root=None, config="configs/data_v2.json") -> dict:
    """Trained maximum amplitudes from ``configs/data_v2.json`` (``camera`` section)."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    path = Path(config) if Path(config).is_absolute() else root / config
    try:
        cam = json.loads(path.read_text(encoding="utf8"))["camera"]
        return {
            "yaw_deg": float(cam["max_yaw_deg"]),
            "pitch_deg": float(cam["max_pitch_deg"]),
            "zoom_ratio": float(cam["max_zoom_ratio"]),
        }
    except (OSError, KeyError, ValueError):
        warnings.warn(f"could not read camera maxima from {path}; using built-in defaults")
        return dict(DEFAULT_MAX_AMPLITUDES)


def physical_amplitude(motion: str, u: float, max_amplitudes=None) -> dict:
    """Signed end-of-clip physical amplitude of ``motion`` at strength ``u``.

    Returns ``{"yaw_deg", "pitch_deg", "zoom_ratio", "value", "unit"}`` where
    ``value``/``unit`` is the single number a UI shows (degrees, or zoom ratio).
    """
    amax = {**DEFAULT_MAX_AMPLITUDES, **(max_amplitudes or {})}
    family, direction = split_motion(motion)
    amp = motion_amplitudes(
        family,
        direction,
        float(u),
        max_yaw_deg=amax["yaw_deg"],
        max_pitch_deg=amax["pitch_deg"],
        max_zoom_ratio=amax["zoom_ratio"],
    )
    if family == "pan":
        value, unit = abs(amp["yaw_deg"]), "deg"
    elif family == "tilt":
        value, unit = abs(amp["pitch_deg"]), "deg"
    elif family == "zoom":
        value, unit = amp["zoom_ratio"], "x"
    else:
        value, unit = 0.0, ""
    return {**amp, "value": float(value), "unit": unit}


def physical_to_u(motion: str, amplitude_deg=None, zoom_ratio=None, max_amplitudes=None) -> float:
    """Strength ``u`` that realises a physical amplitude exactly (no calibration).

    pan/tilt: ``u = deg / max_deg``; zoom: ``u = ln(ratio) / ln(max_ratio)`` where
    ``ratio`` is the magnitude of the zoom (a zoom-out of x0.893 and a zoom-in of
    x1.12 both give ``ratio = 1.12``; ratios below 1 are taken as reciprocals).
    Raises if the amplitude exceeds the trained maximum.
    """
    amax = {**DEFAULT_MAX_AMPLITUDES, **(max_amplitudes or {})}
    family, _ = split_motion(motion)
    if family == "static":
        return 0.0
    if family == "zoom":
        if zoom_ratio is None or amplitude_deg is not None:
            raise ValueError("zoom motions need --zoom-ratio (not --amplitude-deg)")
        r = float(zoom_ratio)
        if not math.isfinite(r) or r <= 0:
            raise ValueError("zoom ratio must be positive and finite")
        u = abs(math.log(r)) / math.log(amax["zoom_ratio"])
    else:
        if amplitude_deg is None or zoom_ratio is not None:
            raise ValueError("pan/tilt motions need --amplitude-deg (not --zoom-ratio)")
        a = abs(float(amplitude_deg))
        if not math.isfinite(a):
            raise ValueError("amplitude must be finite")
        u = a / amax["yaw_deg" if family == "pan" else "pitch_deg"]
    if u > 1.0 + 1e-9:
        raise ValueError(f"physical amplitude exceeds the trained maximum (u = {u:.3f} > 1)")
    return float(min(1.0, u))


def amplitude_label(motion: str, u: float, max_amplitudes=None) -> str:
    """Human-readable label, e.g. ``"Pan right · 5.6°"`` or ``"Zoom out · ×0.86"``."""
    if motion == "static":
        return "Static"
    amp = physical_amplitude(motion, u, max_amplitudes)
    name = MOTION_LABELS[motion]
    if amp["unit"] == "deg":
        return f"{name} · {amp['value']:.1f}°"
    return f"{name} · ×{amp['value']:.2f}"


# --------------------------------------------------------------------------- calibration
def identity_calibration(max_amplitudes=None, created=None) -> dict:
    """The no-op calibration (``u = s``) as a schema-conformant dict."""
    return {
        "version": CALIBRATION_VERSION,
        "created": created or "2026-11-16",
        "checkpoint_sha256": None,
        "max_amplitudes": {**DEFAULT_MAX_AMPLITUDES, **(max_amplitudes or {})},
        "directions": {d: {"s": [0.0, 1.0], "u": [0.0, 1.0]} for d in DIRECTIONS},
        "fit": {
            "method": "identity",
            "note": "No measured calibration: u = s. Replace by scripts/calibrate_strength.py output.",
        },
    }


def _validate_direction(name, entry):
    try:
        s = np.asarray(entry["s"], dtype=np.float64)
        u = np.asarray(entry["u"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(f"calibration direction {name!r} needs numeric 's' and 'u' lists") from err
    if s.ndim != 1 or s.shape != u.shape or len(s) < 2:
        raise ValueError(f"calibration {name!r}: 's' and 'u' must be equal-length lists of >= 2 points")
    if not (np.isfinite(s).all() and np.isfinite(u).all()):
        raise ValueError(f"calibration {name!r}: non-finite values")
    if s.min() < -_EPS or s.max() > 1 + _EPS or u.min() < -_EPS or u.max() > 1 + _EPS:
        raise ValueError(f"calibration {name!r}: s and u must lie in [0, 1]")
    if np.any(np.diff(s) <= 0):
        raise ValueError(f"calibration {name!r}: 's' must be strictly increasing")
    if np.any(np.diff(u) < -_EPS):
        raise ValueError(f"calibration {name!r}: 'u' must be non-decreasing (monotone map)")
    s, u = np.clip(s, 0.0, 1.0), np.clip(u, 0.0, 1.0)
    if abs(s[0]) > _EPS:  # anchor the origin: s = 0 is static
        s, u = np.r_[0.0, s], np.r_[0.0, u]
    if abs(u[0]) > _EPS:
        raise ValueError(f"calibration {name!r}: s = 0 must map to u = 0")
    u[0] = 0.0
    return s, u


class StrengthMap:
    """Validated per-direction monotone piecewise-linear map ``s -> u``."""

    def __init__(self, calibration: dict | None = None, *, source=None, sha256=None):
        self.record = identity_calibration() if calibration is None else dict(calibration)
        self.source = None if source is None else str(source)
        self.sha256 = sha256
        if self.record.get("version") != CALIBRATION_VERSION:
            raise ValueError(
                f"unsupported calibration version {self.record.get('version')!r} "
                f"(expected {CALIBRATION_VERSION})"
            )
        dirs = self.record.get("directions")
        if not isinstance(dirs, dict):
            raise ValueError("calibration needs a 'directions' object")
        unknown = sorted(set(dirs) - set(DIRECTIONS))
        if unknown:
            raise ValueError(f"unknown calibration directions: {unknown}")
        self.max_amplitudes = {**DEFAULT_MAX_AMPLITUDES, **(self.record.get("max_amplitudes") or {})}
        self._knots = {}
        self.identity_directions = []
        for d in DIRECTIONS:
            if d in dirs:
                self._knots[d] = _validate_direction(d, dirs[d])
            else:
                self._knots[d] = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
                self.identity_directions.append(d)
        if self.identity_directions and calibration is not None:
            warnings.warn(f"calibration has no entry for {self.identity_directions}; using u = s there")

    @property
    def is_identity(self) -> bool:
        """True if every direction is the identity map u = s (no calibration applied)."""
        return all(
            len(s) == 2 and np.allclose(s, [0, 1]) and np.allclose(u, [0, 1])
            for s, u in self._knots.values()
        )

    def knots(self, direction):
        """Copies of the (s, u) knot arrays of one direction."""
        s, u = self._knots[direction]
        return s.copy(), u.copy()

    def s_to_u(self, direction: str, s: float) -> float:
        """Requested strength ``u`` for UI strength ``s``; clamped to [0, 1]; ``s = 0 -> 0``."""
        if direction == "static":
            return 0.0
        if direction not in self._knots:
            raise ValueError(f"unknown motion {direction!r}")
        s = float(s)
        if not math.isfinite(s):
            raise ValueError("strength must be finite")
        s = min(1.0, max(0.0, s))
        if s <= 0.0:
            return 0.0
        ks, ku = self._knots[direction]
        return float(min(1.0, max(0.0, np.interp(s, ks, ku))))


def load_calibration(calibration=None, *, root=None) -> StrengthMap:
    """Load ``calibration`` (path, dict, ``StrengthMap`` or ``None`` = identity)."""
    if isinstance(calibration, StrengthMap):
        return calibration
    if calibration is None:
        return StrengthMap(None)
    if isinstance(calibration, dict):
        return StrengthMap(calibration)
    path = Path(calibration)
    if not path.is_absolute() and root is not None and not path.exists():
        path = Path(root) / path
    return StrengthMap(
        json.loads(path.read_text(encoding="utf8")), source=path, sha256=file_sha256(path)
    )


# --------------------------------------------------------------------------- fitting
def isotonic_increasing(y, w=None):
    """Weighted least-squares non-decreasing fit by pool-adjacent-violators (PAV).

    ``y`` must already be ordered by the independent variable. Returns an array
    ``f`` with ``f[i] <= f[i+1]`` minimising ``sum w (y - f)^2``.
    """
    y = np.asarray(y, dtype=np.float64)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64)
    if y.ndim != 1 or w.shape != y.shape or np.any(w <= 0):
        raise ValueError("y and positive weights of equal length required")
    mean, weight, count = [], [], []  # block stack
    for yi, wi in zip(y, w):
        mean.append(yi)
        weight.append(wi)
        count.append(1)
        while len(mean) > 1 and mean[-2] > mean[-1]:
            wsum = weight[-2] + weight[-1]
            mean[-2] = (mean[-2] * weight[-2] + mean[-1] * weight[-1]) / wsum
            weight[-2] = wsum
            count[-2] += count[-1]
            mean.pop(), weight.pop(), count.pop()
    return np.repeat(mean, count)


def fit_amplitude_curve(u, a, w=None):
    """Isotonic ``a_obs(u)`` through the origin from raw (u, a) samples.

    Samples at equal ``u`` are first merged to their weighted mean (so repeated
    images share one knot), then PAV is run on the origin-anchored sequence. The
    origin ``(0, 0)`` is a fixed knot (static request -> no motion).
    Returns ``(u_knots, a_knots)``, strictly increasing ``u``, non-decreasing ``a``.
    """
    u = np.asarray(u, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    w = np.ones_like(u) if w is None else np.asarray(w, dtype=np.float64)
    keep = np.isfinite(u) & np.isfinite(a) & (u > _EPS)
    u, a, w = u[keep], a[keep], w[keep]
    if len(u) == 0:
        raise ValueError("no finite positive-u samples to fit")
    uk = np.unique(np.round(u, 12))
    ak = np.array([np.average(a[np.isclose(u, x)], weights=w[np.isclose(u, x)]) for x in uk])
    wk = np.array([w[np.isclose(u, x)].sum() for x in uk])
    uk, ak, wk = np.r_[0.0, uk], np.r_[0.0, ak], np.r_[wk.sum() * 1e3, wk]  # origin ~fixed
    # pool-adjacent-violators makes a_obs(u) monotone: measurement noise must not create a non-invertible map.
    fit = isotonic_increasing(ak, wk)
    fit[0] = 0.0
    return uk, fit


def invert_amplitude_curve(u_knots, a_knots, s_grid=None):
    """Invert a monotone ``a_obs(u)`` to the calibration map ``u(s)``.

    For each target observed amplitude ``s`` returns the *smallest* ``u`` with
    ``a_obs(u) = s`` (flat stretches of the isotonic fit have no unique inverse;
    the smallest u is the least extrapolating choice). If the curve never reaches
    ``s`` (max observed amplitude < 1) the request is capped at the largest tested
    ``u``; ``saturates_at`` reports that maximum observed amplitude when it is more
    than ``SATURATION_TOL`` below 1, else None.

    Returns ``(s_grid, u_of_s, saturates_at_or_None)``.
    """
    uk = np.asarray(u_knots, dtype=np.float64)
    ak = np.asarray(a_knots, dtype=np.float64)
    if np.any(np.diff(uk) <= 0) or np.any(np.diff(ak) < -_EPS):
        raise ValueError("amplitude curve must be strictly increasing in u and non-decreasing in a")
    s_grid = np.linspace(0.0, 1.0, 21) if s_grid is None else np.asarray(s_grid, dtype=np.float64)
    # calibration inversion: measurement gives a_obs(u) (how much motion the model actually produces for requested
    # u).  The UI wants the opposite, u(s): the request that yields observed amplitude s.  So invert the isotonic curve;
    # on flat stretches take the smallest u (least extrapolation) and cap at the largest tested u if the curve never
    # reaches s (saturation, reported separately).
    u_max = float(min(1.0, uk[-1]))
    a_max = float(ak[-1])
    # first occurrence of each distinct a value (smallest u on flat stretches)
    a_unique, first = np.unique(np.round(ak, 12), return_index=True)
    u_unique = uk[first]
    out = []
    for s in s_grid:
        if s <= 0:
            out.append(0.0)
        elif s > a_max + _EPS:  # unreachable observed amplitude: cap at the largest tested u
            out.append(u_max)
        else:
            out.append(float(np.interp(s, a_unique, u_unique)))
    # force u(s) to be non-decreasing and inside the tested range, so a stronger slider never asks for a weaker request.
    u_of_s = np.clip(np.maximum.accumulate(np.asarray(out)), 0.0, u_max)
    saturates = a_max if a_max < 1.0 - SATURATION_TOL else None
    return s_grid, u_of_s, saturates


def build_direction_entry(u, a, w=None, s_grid=None):
    """Fit + invert for one direction; returns ``(entry, diagnostics)``."""
    uk, ak = fit_amplitude_curve(u, a, w)
    sg, ug, sat = invert_amplitude_curve(uk, ak, s_grid)
    entry = {"s": [round(float(x), 6) for x in sg], "u": [round(float(x), 6) for x in ug]}
    diag = {
        "u_knots": [float(x) for x in uk],
        "a_obs_isotonic": [float(x) for x in ak],
        "a_obs_max": float(ak[-1]),
        "saturates_at": sat,
        "mean_abs_error_before": float(np.mean(np.abs(ak[1:] - uk[1:]))),
    }
    return entry, diag


def new_calibration(directions: dict, *, checkpoint_sha256=None, max_amplitudes=None, fit=None, created=None):
    """Assemble a calibration dict (validated by round-tripping through ``StrengthMap``)."""
    import datetime

    cal = {
        "version": CALIBRATION_VERSION,
        "created": created or datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "checkpoint_sha256": checkpoint_sha256,
        "max_amplitudes": {**DEFAULT_MAX_AMPLITUDES, **(max_amplitudes or {})},
        "directions": directions,
        "fit": fit or {},
    }
    StrengthMap(cal)  # raises if invalid
    return cal
