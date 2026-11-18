"""Session deadlines for Molab sandboxes that are hard-killed ~3 h after they start.

All helpers are pure / injectable (clock and /proc readers) so they are unit-testable.

Effective deadline = min(constraints) - margin, where the constraints are
``deadline_epoch`` (absolute), ``max_minutes`` (from *now*, i.e. process start) and
``session_hours`` (from the sandbox start).  Long jobs then use ``should_stop`` before each
work item: stop if ``now + est_item + est_finish + grace > deadline``.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Optional

DEFAULT_MARGIN_MINUTES = 15.0
UPLOAD_GRACE_SECONDS = 90.0


def _read(path):
    with open(path, "r", encoding="utf8") as f:
        return f.read()


def sandbox_start_epoch(*, read: Callable[[str], str] = _read, env=None, now: Callable[[], float] = time.time,
                        clk_tck: Optional[int] = None) -> float:
    """Best-effort start time of the sandbox (unix seconds).

    1. ``/proc/1`` start time (``btime`` from /proc/stat + starttime ticks from /proc/1/stat);
    2. ``MOLAB_SESSION_START`` env var (unix seconds);
    3. ``now()`` (this process' start - the least conservative fallback).
    """
    env = os.environ if env is None else env
    try:
        btime = next(int(l.split()[1]) for l in read("/proc/stat").splitlines() if l.startswith("btime "))
        stat = read("/proc/1/stat")
        fields = stat[stat.rindex(")") + 2:].split()  # after "(comm)"; field 3 is index 0
        # the sandbox lifetime (about 3 h on Molab) is counted from PID 1's start, not from this process's start.
        start_ticks = int(fields[19])  # starttime is field 22 overall
        if clk_tck is None:
            clk_tck = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        return btime + start_ticks / float(clk_tck)
    except Exception:
        pass
    try:
        return float(env["MOLAB_SESSION_START"])
    except Exception:
        return float(now())


def resolve_deadline(deadline_epoch=None, max_minutes=None, session_hours=None,
                     margin_minutes=DEFAULT_MARGIN_MINUTES, *, now: Callable[[], float] = time.time,
                     start_epoch: Optional[float] = None, **start_kwargs) -> Optional[float]:
    """Absolute stop epoch (margin already subtracted), or None if no constraint was given."""
    cands = []
    if deadline_epoch is not None:
        cands.append(float(deadline_epoch))
    if max_minutes is not None:
        cands.append(float(now()) + 60.0 * float(max_minutes))
    if session_hours is not None:
        start = sandbox_start_epoch(now=now, **start_kwargs) if start_epoch is None else start_epoch
        cands.append(start + 3600.0 * float(session_hours))
    if not cands:
        return None
    return min(cands) - 60.0 * float(margin_minutes)


def should_stop(now, deadline, est_item, est_finish=0.0, grace=UPLOAD_GRACE_SECONDS) -> bool:
    """True if one more item (+ the final write + upload grace) would not fit before ``deadline``."""
    # deadline rule: stop BEFORE an item that would not leave room for its own duration, the final checkpoint write
    # and the upload grace, so the session ends with everything saved and uploaded rather than being killed mid-write.
    return deadline is not None and now + est_item + est_finish + grace > deadline


class Ema:
    """Exponential moving average (alpha 0.3) used to estimate seconds per update / per item; the first sample
    replaces the initial guess.
    """

    def __init__(self, initial, alpha=0.3):
        self.value, self.alpha, self.n = float(initial), alpha, 0

    def update(self, x):
        """Fold in a new measurement and return the new average."""
        self.value = float(x) if self.n == 0 else self.alpha * float(x) + (1 - self.alpha) * self.value
        self.n += 1
        return self.value


def add_deadline_args(ap):
    """Add ``--deadline-epoch / --max-minutes / --session-hours / --deadline-margin-minutes`` to an argparse parser
    (shared by training, cache building, calibration and example generation).
    """
    g = ap.add_argument_group("session deadline (Molab chunking)")
    g.add_argument("--deadline-epoch", type=float, help="absolute unix time to stop by (margin is subtracted)")
    g.add_argument("--max-minutes", type=float, help="stop within this many minutes from now")
    g.add_argument("--session-hours", type=float,
                   help="sandbox lifetime in hours (e.g. 3.0); measured from the sandbox start")
    g.add_argument("--deadline-margin-minutes", type=float, default=DEFAULT_MARGIN_MINUTES,
                   help="safety margin subtracted from the deadline (default 15)")
    return g


def deadline_from_args(args) -> Optional[float]:
    """Resolve the parsed deadline flags to an absolute stop epoch (margin subtracted) or None."""
    return resolve_deadline(args.deadline_epoch, args.max_minutes, args.session_hours,
                            args.deadline_margin_minutes)


def log_deadline_stop(print_fn=print, **fields):
    """Print the greppable ``DEADLINE_STOP key=value ...`` line that tooling (molab_recover) looks for."""
    items = " ".join(f"{k}={v}" for k, v in fields.items())
    print_fn(f"DEADLINE_STOP {items}".strip())
