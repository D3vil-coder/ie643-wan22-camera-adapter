"""CPU tests for ``fpcamera.trajectory_direction`` (stroke -> pan/tilt direction; no model, no GPU)."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpcamera import trajectory_direction as td  # noqa: E402
from fpcamera.strength import DIRECTIONS  # noqa: E402


def line(x0, y0, x1, y1, n=12):
    """Straight stroke of ``n`` points from (x0, y0) to (x1, y1) in normalised image coordinates."""
    return [[x0 + (x1 - x0) * i / (n - 1), y0 + (y1 - y0) * i / (n - 1)] for i in range(n)]


# ------------------------------------------------------------------ the four cardinal directions
@pytest.mark.parametrize(
    "stroke, motion, arrow",
    [
        (line(0.3, 0.5, 0.7, 0.5), "pan_right", "→"),
        (line(0.7, 0.5, 0.3, 0.5), "pan_left", "←"),
        (line(0.5, 0.8, 0.5, 0.2), "tilt_up", "↑"),  # canvas y grows downward: up = decreasing y
        (line(0.5, 0.2, 0.5, 0.8), "tilt_down", "↓"),
    ],
)
def test_cardinal_strokes(stroke, motion, arrow):
    result = td.detect_direction(stroke)
    assert result.status == "ok" and result.motion == motion and result.arrow == arrow
    assert result.confidence == pytest.approx(1.0)
    assert result.straightness == pytest.approx(1.0)
    assert result.warnings == () and result.usable
    assert motion in DIRECTIONS  # the strings are valid flow_infer / strength motions


def test_v1_convention_right_is_pan_right_down_is_tilt_down():
    # v1 (classify_camera_path): dx > 0 -> "pan right", dy > 0 -> "tilt down". Same here.
    assert td.detect_direction(line(0.2, 0.5, 0.4, 0.5)).motion == "pan_right"
    assert td.detect_direction(line(0.5, 0.2, 0.5, 0.4)).motion == "tilt_down"


def test_screen_vector_matches_motion():
    assert td.detect_direction(line(0.3, 0.5, 0.7, 0.5)).screen_vector == (1, 0)
    assert td.detect_direction(line(0.5, 0.8, 0.5, 0.2)).screen_vector == (0, -1)
    assert td.detect_direction(None).screen_vector == (0, 0)


def test_reversing_the_stroke_flips_the_side():
    forward = line(0.2, 0.4, 0.8, 0.45)
    assert td.detect_direction(forward).motion == "pan_right"
    assert td.detect_direction(forward[::-1]).motion == "pan_left"


# ------------------------------------------------------------------ dominant axis, aspect ratio, confidence
def test_dominant_axis_wins_on_oblique_strokes():
    # 20 degrees above the horizontal in on-screen units (aspect-correct), still a pan.
    dx = 0.30
    dy_norm = dx * td.DEFAULT_ASPECT * math.tan(math.radians(20))  # normalised dy for 20 degrees on screen
    result = td.detect_direction(line(0.3, 0.6, 0.3 + dx, 0.6 - dy_norm))
    assert result.motion == "pan_right" and result.status == "ok"
    assert result.angle_deg == pytest.approx(20.0, abs=1e-6)
    assert result.off_axis_deg == pytest.approx(20.0, abs=1e-6)
    assert result.confidence == pytest.approx(1 / (1 + math.tan(math.radians(20))))


def test_aspect_ratio_is_applied_before_comparing_axes():
    # Equal NORMALISED displacements are not equal on a 16:9 screen: the horizontal one is 1.82x longer.
    assert td.detect_direction(line(0.3, 0.3, 0.45, 0.45)).motion == "pan_right"
    # With a square canvas the same stroke is an exact diagonal -> ambiguous.
    assert td.detect_direction(line(0.3, 0.3, 0.45, 0.45), aspect=1.0).status == "ambiguous"
    # A normalised dy that is larger than dx but shorter on screen stays a pan (dx*1.82 > dy).
    assert td.detect_direction(line(0.3, 0.3, 0.45, 0.5)).motion == "pan_right"
    # ... while a clearly taller one is a tilt.
    assert td.detect_direction(line(0.3, 0.2, 0.4, 0.8)).motion == "tilt_down"


def test_confidence_decreases_towards_the_diagonal():
    values = []
    for degrees in (0, 15, 30, 40):
        dy = 0.3 * td.DEFAULT_ASPECT * math.tan(math.radians(degrees))
        values.append(td.detect_direction(line(0.3, 0.3, 0.6, 0.3 + dy)).confidence)
    assert values == sorted(values, reverse=True) and values[0] == pytest.approx(1.0)
    assert 0.5 < values[-1] < 0.6


# ------------------------------------------------------------------ warnings
def test_too_short_stroke_has_no_motion():
    result = td.detect_direction(line(0.5, 0.5, 0.52, 0.5))
    assert result.status == "too_short" and result.motion is None and not result.usable
    assert "too short" in result.message
    assert result.label == "" and result.arrow == ""


def test_min_length_is_in_image_height_units():
    just_over = td.MIN_LENGTH * 1.05 / td.DEFAULT_ASPECT  # horizontal normalised length
    just_under = td.MIN_LENGTH * 0.95 / td.DEFAULT_ASPECT
    assert td.detect_direction(line(0.5, 0.5, 0.5 + just_over, 0.5)).status == "ok"
    assert td.detect_direction(line(0.5, 0.5, 0.5 + just_under, 0.5)).status == "too_short"
    assert td.detect_direction(line(0.5, 0.5, 0.52, 0.5), min_length=0.01).status == "ok"


def test_exact_diagonal_is_ambiguous_but_still_returns_a_best_guess():
    stroke = line(0.3, 0.3, 0.3 + 0.2, 0.3 + 0.2 * td.DEFAULT_ASPECT)  # 45 degrees on screen
    result = td.detect_direction(stroke)
    assert result.status == "ambiguous" and result.usable
    assert result.confidence == pytest.approx(0.5)
    assert result.off_axis_deg == pytest.approx(45.0)
    assert "diagonal" in result.message and "tilt" in result.message  # names the alternative axis


def test_ambiguity_threshold_is_configurable():
    dy = 0.3 * td.DEFAULT_ASPECT * math.tan(math.radians(25))
    stroke = line(0.3, 0.3, 0.6, 0.3 + dy)  # confidence ~0.68
    assert td.detect_direction(stroke).status == "ok"
    assert td.detect_direction(stroke, ambiguity_threshold=0.8).status == "ambiguous"


def test_curvy_stroke_warns_but_uses_start_to_end():
    # A wide loop that ends to the right of where it started.
    arc = [[0.2, 0.5], [0.3, 0.1], [0.5, 0.05], [0.7, 0.1], [0.8, 0.5], [0.7, 0.9], [0.5, 0.95], [0.3, 0.9], [0.45, 0.52]]
    result = td.detect_direction(arc)
    assert result.straightness < td.STRAIGHTNESS_WARN
    assert any("curvy" in w for w in result.warnings)
    assert result.motion == "pan_right"  # net displacement 0.25 right, almost nothing vertical


def test_hook_at_the_end_does_not_flip_a_long_pan():
    stroke = line(0.2, 0.5, 0.8, 0.5) + [[0.81, 0.52], [0.8, 0.53]]
    assert td.detect_direction(stroke).motion == "pan_right"


# ------------------------------------------------------------------ degenerate input
@pytest.mark.parametrize("points", [None, [], [[0.5, 0.5]], ((0.1, 0.1),)])
def test_empty_or_single_point(points):
    result = td.detect_direction(points)
    assert result.status == "empty" and result.motion is None and not result.usable
    assert "Draw a stroke" in result.message


def test_non_finite_points_are_dropped():
    stroke = [[float("nan"), 0.5], *line(0.3, 0.5, 0.7, 0.5), [0.5, float("inf")]]
    assert td.detect_direction(stroke).motion == "pan_right"
    assert td.detect_direction([[float("nan"), 0.1], [0.2, 0.2]]).status == "empty"


def test_malformed_points_raise():
    with pytest.raises(ValueError):
        td.detect_direction([[0.1, 0.1], ["a", 0.2]])
    with pytest.raises(ValueError):
        td.detect_direction([[0.1, 0.1], [0.2]])
    with pytest.raises(ValueError):
        td.detect_direction(line(0.2, 0.5, 0.7, 0.5), aspect=0.0)


def test_identical_points_are_too_short():
    result = td.detect_direction([[0.4, 0.4]] * 5)
    assert result.status == "too_short" and result.straightness == 0.0 and math.isnan(result.angle_deg)


def test_accepts_tuples_and_list_subclasses():
    class Row(list):
        pass

    assert td.detect_direction([(0.2, 0.5), (0.8, 0.5)]).motion == "pan_right"
    assert td.detect_direction([Row([0.5, 0.8]), Row([0.5, 0.2])]).motion == "tilt_up"
