"""CPU tests for the optional ``progress_callback`` hook of ``fpcamera.flow_infer.generate``.

``generate`` itself needs a GPU, so the wiring is checked three ways: the signature, the helper
that forwards ``(step, total)``, and an AST check that the per-step diffusers callback calls the
helper with the 1-based step and the requested number of steps.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpcamera import flow_infer as fi  # noqa: E402


def test_generate_accepts_optional_progress_callback_defaulting_to_none():
    parameter = inspect.signature(fi.generate).parameters["progress_callback"]
    assert parameter.default is None and parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_no_callback_is_a_no_op():
    assert fi._report_progress(None, 3, 50) is None


def test_forwards_integer_step_and_total():
    calls = []
    fi._report_progress(lambda step, total: calls.append((step, total)), 7.0, 50.0)
    assert calls == [(7, 50)]
    assert all(type(v) is int for v in calls[0])


def test_every_step_is_reported_once_in_order():
    calls = []
    total = 12
    for step in range(total):  # mirrors callback_on_step_end(i = 0..total-1) -> step + 1
        fi._report_progress(lambda s, t: calls.append((s, t)), step + 1, total)
    assert calls == [(i, total) for i in range(1, total + 1)]


def test_exception_from_the_callback_propagates_so_a_ui_can_cancel():
    class Cancelled(Exception):
        pass

    def hook(step, total):
        if step == 3:
            raise Cancelled

    with pytest.raises(Cancelled):
        for step in range(5):
            fi._report_progress(hook, step + 1, 5)


def test_step_end_callback_reports_one_based_step_and_total():
    """The nested ``callback`` of ``generate`` must call the helper with ``(progress_callback, step + 1, steps)``."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fi.generate)))
    generate_def = tree.body[0]
    callback_def = next(
        node for node in ast.walk(generate_def) if isinstance(node, ast.FunctionDef) and node.name == "callback"
    )
    calls = [
        node
        for node in ast.walk(callback_def)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_report_progress"
    ]
    assert len(calls) == 1
    first, step_arg, total_arg = calls[0].args
    assert ast.unparse(first) == "progress_callback"
    assert ast.unparse(step_arg) == "step + 1"
    assert ast.unparse(total_arg) == "steps"
