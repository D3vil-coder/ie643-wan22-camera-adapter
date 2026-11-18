"""Pan / tilt direction from a hand-drawn stroke (pure Python: no numpy, no torch, no model).

The v2 camera adapter takes ONE motion (``pan_left``, ``pan_right``, ``tilt_up``, ``tilt_down``,
``zoom_in``, ``zoom_out``) plus a strength; it does not follow an arbitrary curve. The studio
notebook (``notebooks/camera_studio_v2.py``) therefore lets the user *draw* a trajectory on the
photograph and reduces it to the nearest of the four pan/tilt directions with
:func:`detect_direction`.

Convention (identical to the v1 trajectory UI, ``notebooks/flow_camera_interface.py``,
``classify_camera_path``): the stroke is a **camera-view path**. Drag in the direction the camera
should move / look:

* drag right  -> ``pan_right``  (the camera turns right, the scene slides left in the frame)
* drag left   -> ``pan_left``
* drag down   -> ``tilt_down``  (canvas y grows downward, so a positive ``dy`` is a downward drag)
* drag up     -> ``tilt_up``

Method. The direction is the **net displacement** (last point minus first point), which is robust
to a wobbly path and to hand jitter at the end of the stroke. Points are normalised to the image
(x and y both in [0, 1]); because the frame is 16:9 a normalised ``dx`` is *longer on screen* than
the same normalised ``dy``, so x is rescaled by the aspect ratio before the two components are
compared. The dominant axis picks pan (horizontal) or tilt (vertical); its sign picks the side.

Quality figures returned with the direction:

* ``confidence`` = ``dominant / (dominant + secondary)`` of the displacement, in [0.5, 1]:
  1.0 for a perfectly axis-aligned stroke, 0.5 for an exact 45 degree diagonal (a coin flip
  between two axes). Below ``ambiguity_threshold`` (default 0.62, i.e. within about 12 degrees of a
  diagonal) the stroke is flagged ``"ambiguous"``.
* ``length`` of the net displacement in units of the image height; shorter than ``min_length``
  (default 0.08, about 56 px of a 704 px high frame) is ``"too_short"`` and yields no motion.
* ``straightness`` = net displacement / path length. A curvy stroke (< 0.5) only adds a warning:
  the direction still comes from start to end.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

# Frame the model sees: 1280 x 704. Normalised x is multiplied by this before comparing with y.
DEFAULT_ASPECT = 1280.0 / 704.0
# Net displacement must be at least this long, in units of the image height.
MIN_LENGTH = 0.08
# dominant / (dominant + secondary) below this is "near the 45 degree diagonal".
AMBIGUITY_THRESHOLD = 0.62
# net displacement / path length below this adds a "curvy stroke" warning.
STRAIGHTNESS_WARN = 0.5

_ARROWS = {"pan_left": "←", "pan_right": "→", "tilt_up": "↑", "tilt_down": "↓"}
_LABELS = {"pan_left": "Pan left", "pan_right": "Pan right", "tilt_up": "Tilt up", "tilt_down": "Tilt down"}
# Where the arrow points on screen, as a unit vector in canvas coordinates (x right, y down).
_SCREEN_VECTORS = {"pan_left": (-1, 0), "pan_right": (1, 0), "tilt_up": (0, -1), "tilt_down": (0, 1)}


@dataclass(frozen=True)
class StrokeDirection:
    """Result of :func:`detect_direction`.

    ``status`` is one of ``"empty"`` (fewer than two usable points), ``"too_short"`` (no motion
    returned), ``"ambiguous"`` (a best-guess ``motion`` is returned together with a warning) and
    ``"ok"``. ``motion`` is a ``fpcamera.strength.DIRECTIONS`` key or ``None``.
    """

    status: str
    motion: str | None
    dx: float  # net displacement, x in units of image height (aspect corrected), canvas axes (right +)
    dy: float  # net displacement, y in units of image height, canvas axes (down +)
    length: float  # |(dx, dy)|, image heights
    path_length: float  # summed segment lengths, image heights
    straightness: float  # length / path_length in [0, 1] (1 for a straight stroke, 0 if no path)
    confidence: float  # dominant / (dominant + secondary), 0.5..1; 0.0 when there is no displacement
    angle_deg: float  # stroke angle, counter-clockwise from "right" (90 = up); nan without displacement
    off_axis_deg: float  # angle between the stroke and the chosen axis, 0..45; nan without displacement
    warnings: tuple[str, ...]

    @property
    def usable(self) -> bool:
        """True when a motion was chosen (``ok`` or ``ambiguous``)."""
        return self.motion is not None

    @property
    def label(self) -> str:
        """``"Pan right"`` etc., or ``""`` without a motion."""
        return _LABELS.get(self.motion, "")

    @property
    def arrow(self) -> str:
        """Unicode arrow of the detected direction, or ``""``."""
        return _ARROWS.get(self.motion, "")

    @property
    def screen_vector(self) -> tuple[int, int]:
        """Unit vector of the detected direction in canvas axes (x right, y down); ``(0, 0)`` if none."""
        return _SCREEN_VECTORS.get(self.motion, (0, 0))

    @property
    def message(self) -> str:
        """One human-readable sentence for the UI."""
        if self.status == "empty":
            return "Draw a stroke on the image to choose the pan or tilt direction."
        if self.status == "too_short":
            return "The stroke is too short. Drag a longer line in the direction the camera should move."
        text = f"{self.label} (confidence {self.confidence:.0%})"
        return text + ("".join(" " + w for w in self.warnings) if self.warnings else "")


def _clean_points(points) -> list[tuple[float, float]]:
    """Validate ``points`` ([[x, y], ...]) and drop non-finite entries."""
    if points is None:
        return []
    cleaned = []
    for p in points:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError, KeyError) as error:
            raise ValueError(f"each point must be a pair of numbers, got {p!r}") from error
        if math.isfinite(x) and math.isfinite(y):
            cleaned.append((x, y))
    return cleaned


def detect_direction(
    points: Sequence[Sequence[float]] | None,
    *,
    aspect: float = DEFAULT_ASPECT,
    min_length: float = MIN_LENGTH,
    ambiguity_threshold: float = AMBIGUITY_THRESHOLD,
) -> StrokeDirection:
    """Reduce a stroke to ``pan_left`` / ``pan_right`` / ``tilt_up`` / ``tilt_down``.

    ``points``: the stroke in drawing order, each ``[x, y]`` normalised to the image (x right, y
    down, both 0..1). ``aspect`` is image width / height (x is scaled by it so that the two axes
    are compared in the same on-screen units). See the module docstring for the convention and
    the meaning of the returned figures.
    """
    if not (aspect > 0 and math.isfinite(aspect)):
        raise ValueError("aspect must be positive and finite")
    pts = _clean_points(points)
    if len(pts) < 2:
        return StrokeDirection("empty", None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.nan, math.nan, ())
    scaled = [(x * aspect, y) for x, y in pts]
    dx = scaled[-1][0] - scaled[0][0]
    dy = scaled[-1][1] - scaled[0][1]
    length = math.hypot(dx, dy)
    path_length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(scaled, scaled[1:]))
    straightness = min(1.0, length / path_length) if path_length > 0 else 0.0
    if length < min_length:
        return StrokeDirection("too_short", None, dx, dy, length, path_length, straightness, 0.0, math.nan, math.nan, ())

    ax, ay = abs(dx), abs(dy)
    horizontal = ax >= ay  # an exact diagonal falls to pan; it is flagged ambiguous below anyway
    dominant, secondary = (ax, ay) if horizontal else (ay, ax)
    confidence = dominant / (dominant + secondary)
    if horizontal:
        motion = "pan_right" if dx > 0 else "pan_left"
    else:
        motion = "tilt_down" if dy > 0 else "tilt_up"  # canvas y grows downward
    angle_deg = math.degrees(math.atan2(-dy, dx)) % 360.0
    off_axis_deg = math.degrees(math.atan2(secondary, dominant))

    warnings = []
    status = "ok"
    if confidence < ambiguity_threshold:
        status = "ambiguous"
        other = ("tilt " + ("down" if dy > 0 else "up")) if horizontal else ("pan " + ("right" if dx > 0 else "left"))
        warnings.append(f"The stroke is close to a diagonal (off-axis by {off_axis_deg:.0f} degrees); it could also be {other}. "
                        "Redraw it more horizontally or more vertically.")
    if straightness < STRAIGHTNESS_WARN:
        warnings.append("The stroke is curvy; only its start and end points set the direction.")
    return StrokeDirection(status, motion, dx, dy, length, path_length, straightness, confidence, angle_deg,
                           off_axis_deg, tuple(warnings))
