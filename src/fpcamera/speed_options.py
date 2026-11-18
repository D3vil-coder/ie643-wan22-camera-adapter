"""Opt-in, kernel-level speed options shared by training and inference (all OFF by default).

* ``--attn-backend {auto,cudnn,flash,efficient}``: run the transformer's scaled-dot-product attention under
  ``torch.nn.attention.sdpa_kernel`` with an explicit backend priority.  ``auto`` (default) does nothing at all.
* ``--compile-blocks``: regional ``torch.compile`` of every transformer block (``mode="default"``, ``dynamic=False``).
* ``--speed-probe``: grid of the above (plus the number of checkpointed blocks) timed on one fixed batch, with a numerical
  equivalence check against the baseline row (helpers here; the GPU loop lives in ``flow_train_v2.speed_probe``).

None of this changes the training math, losses, sampling, RNG streams, checkpoint format or run identity.  It is the same
standing as SDPA versus a math kernel: the arithmetic is the same up to kernel-level floating-point differences, so
results are NOT bit-identical to the default path.  The probe measures how far apart they are.

Everything below is importable and testable on a CPU-only machine.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import json
import math
import warnings

import torch

ATTN_BACKENDS = ("auto", "cudnn", "flash", "efficient")
EQUIV_TOL = 1e-2  # relative; the bf16 kernel-noise level.  Anything above is reported as NOT lossless.
BASELINE_ROW = (20, "auto", False)  # (checkpointed blocks, attention backend, compile) = the S5 production setting
DYNAMO_CACHE_LIMIT = 64  # one compiled variant per (adapter on/off, grad on/off, batch shape, ...) per block structure


# =============================================================================
# 1. Attention backend
# =============================================================================
def _sdpa_api():
    """``(SDPBackend, sdpa_kernel)`` or ``(None, None)`` on a torch without them (own function so tests can patch it)."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return SDPBackend, sdpa_kernel
    except Exception:  # pragma: no cover - very old torch
        return None, None


# Priority order handed to sdpa_kernel(..., set_priority=True).  cudnn falls back to flash then memory-efficient when
# cuDNN cannot take the inputs; flash / efficient are strict (they fail loudly instead of silently timing another kernel).
_PRIORITY = {
    "cudnn": ("CUDNN_ATTENTION", "FLASH_ATTENTION", "EFFICIENT_ATTENTION"),
    "flash": ("FLASH_ATTENTION",),
    "efficient": ("EFFICIENT_ATTENTION",),
}


@dataclass(frozen=True)
class AttnBackendSpec:
    """Resolved attention-kernel preference: ``requested`` vs ``effective`` (falls back with a note when a kernel
    is not available), and the SDPA backend priority list (empty = leave torch's default dispatch untouched).
    """

    requested: str
    effective: str
    priority: tuple = ()  # SDPBackend member names in priority order; empty = leave torch's default dispatch alone
    note: str | None = None

    @property
    def active(self) -> bool:
        """True if a non-default kernel priority is in force."""
        return bool(self.priority)

    def record(self) -> dict:
        """JSON-able record stored in run.json / progress logs."""
        return {"requested": self.requested, "effective": self.effective, "priority": list(self.priority),
                "note": self.note, "torch": torch.__version__}


def resolve_attn_backend(name="auto", warn=warnings.warn) -> AttnBackendSpec:
    """Validate ``name`` and map it to a priority list.  Falls back to ``auto`` (with a warning) when this torch lacks
    ``sdpa_kernel`` or the requested backend enum (e.g. ``SDPBackend.CUDNN_ATTENTION`` before torch 2.5)."""
    if isinstance(name, AttnBackendSpec):  # already resolved
        return name
    name = "auto" if name is None else str(name)
    if name not in ATTN_BACKENDS:
        raise ValueError(f"attn backend must be one of {ATTN_BACKENDS}, got {name!r}")
    if name == "auto":
        return AttnBackendSpec("auto", "auto")
    enum, kernel = _sdpa_api()
    if enum is None or kernel is None:
        note = f"attn backend {name!r}: torch.nn.attention.sdpa_kernel is missing in torch {torch.__version__}; using auto"
    else:
        missing = [n for n in _PRIORITY[name] if not hasattr(enum, n)]
        if not missing:
            return AttnBackendSpec(name, name, _PRIORITY[name])
        note = (f"attn backend {name!r}: torch {torch.__version__} has no SDPBackend.{'/'.join(missing)}; "
                "using auto (torch's default dispatch)")
    warn(note)
    return AttnBackendSpec(name, "auto", (), note)


@contextlib.contextmanager
def attention_backend(spec="auto"):
    """Run the body with the given backend priority; ``auto`` (or a fallback to it) changes nothing.

    The context spans whatever is run inside it, including a ``backward()`` that recomputes checkpointed blocks, so the
    recomputation uses the same kernel as the forward.  Backend flags are process-global (not per-thread safe).
    """
    if not isinstance(spec, AttnBackendSpec):
        spec = resolve_attn_backend(spec)
    if not spec.active:
        yield spec
        return
    enum, kernel = _sdpa_api()
    backends = [getattr(enum, n) for n in spec.priority]
    try:
        cm = kernel(backends, set_priority=True)
    except TypeError:  # torch without set_priority: the restriction to the listed kernels still applies
        cm = kernel(backends)
    with cm:
        yield spec


@contextlib.contextmanager
def transformer_attention(transformer, spec="auto"):
    """Make every ``transformer.forward`` call run under ``attention_backend(spec)`` (inference: the diffusers pipeline
    and the camera-guidance wrappers all call the transformer, while the VAE decode must stay on torch's default).

    Install this BEFORE ``frozen_camera_difference`` / ``camera_prediction_guidance``: they wrap whatever
    ``transformer.forward`` is when they are entered, and restore it on exit.
    """
    if not isinstance(spec, AttnBackendSpec):
        spec = resolve_attn_backend(spec)
    if not spec.active:
        yield spec
        return
    had = "forward" in transformer.__dict__
    saved = transformer.__dict__.get("forward")
    inner = transformer.forward

    def forward(*args, **kwargs):
        """Run the wrapped transformer forward inside the chosen SDPA backend context."""
        with attention_backend(spec):
            return inner(*args, **kwargs)

    transformer.forward = forward
    try:
        yield spec
    finally:
        if had:
            transformer.forward = saved
        else:
            del transformer.forward


def sdpa_flags() -> dict:
    """Snapshot of the process-global SDPA switches (used to log what a context really enabled)."""
    b = torch.backends.cuda
    out = {}
    for key, fn in (("flash", "flash_sdp_enabled"), ("mem_efficient", "mem_efficient_sdp_enabled"),
                    ("math", "math_sdp_enabled"), ("cudnn", "cudnn_sdp_enabled")):
        out[key] = getattr(b, fn)() if hasattr(b, fn) else None
    return out


def log_attn_backend(spec, log=print) -> dict:
    """One ``attn_backend`` event: what was requested, what is effective, and the flags seen inside the context."""
    with attention_backend(spec) as s:
        info = {"event": "attn_backend", **s.record(), "sdpa_flags_inside": sdpa_flags()}
    log(json.dumps(info))
    return info


# =============================================================================
# 2. Regional torch.compile of the transformer blocks
# =============================================================================
def _torch_compile(fn, **kwargs):
    """Indirection so tests can swap the compiler (no C++ toolchain needed on a CPU CI box)."""
    return torch.compile(fn, **kwargs)


def configure_dynamo(cache_limit: int = DYNAMO_CACHE_LIMIT) -> dict:
    """Raise Dynamo's recompile limits (never lower them).  A block is traced once per structure and variant (adapter
    on/off, grad on/off, batch shape); the default limit of 8 is too small to be safe across all of them."""
    changed = {}
    try:
        import torch._dynamo.config as cfg
    except Exception:  # pragma: no cover
        return changed
    for name, target in (("cache_size_limit", cache_limit), ("recompile_limit", cache_limit),
                         ("accumulated_cache_size_limit", 16 * cache_limit),
                         ("accumulated_recompile_limit", 16 * cache_limit)):
        try:
            cur = getattr(cfg, name)
            if isinstance(cur, int) and cur < target:
                setattr(cfg, name, target)
                changed[name] = target
        except Exception:
            continue
    return changed


def reset_dynamo():
    """Drop all compiled artifacts (needed after changing the attention backend: it is baked in at trace time)."""
    try:
        torch._dynamo.reset()
    except Exception:  # pragma: no cover
        pass


def dynamo_stats() -> dict:
    """Best-effort graph/graph-break counters (None where this torch does not expose them)."""
    try:
        from torch._dynamo.utils import counters

        return {"unique_graphs": int(counters["stats"].get("unique_graphs", 0)),
                "graph_breaks": int(sum(counters["graph_break"].values()))}
    except Exception:
        return {"unique_graphs": None, "graph_breaks": None}


def set_block_compile(model, enabled=True, indices=None, *, mode="default", dynamic=False, backend=None) -> list:
    """Compile (or restore) ``model.blocks[i].forward``; idempotent, reversible, returns the compiled indices.

    Each block's own forward is compiled (regional compilation: all blocks of one structure share graphs).  The compiled
    callable is installed as the block's ``forward``, so it sits INSIDE any activation-checkpoint wrapper added later by
    ``flow_train_v2.set_block_checkpointing`` (and inside diffusers' native ``enable_gradient_checkpointing`` path, which
    calls ``block.__call__``): checkpointing stays eager and recomputes by calling the compiled forward again.
    Install compilation BEFORE checkpointing; the reverse order is rejected.

    ``backend`` is only for tests (``"eager"`` / ``"aot_eager"`` need no C++ compiler); production uses Inductor.
    """
    wanted = set()
    if enabled:
        wanted = set(range(len(model.blocks))) if indices is None else {int(i) for i in indices}
        configure_dynamo()
    for i, block in enumerate(model.blocks):
        if "_unckpt_forward" in block.__dict__:
            raise RuntimeError("remove the activation-checkpoint wrappers (set_block_checkpointing(model, [])) "
                               "before changing block compilation")
        raw = block.__dict__.get("_uncompiled_forward")
        if i in wanted and raw is None:
            kwargs = {"backend": backend} if backend else {"mode": mode}
            kwargs["dynamic"] = dynamic
            block.__dict__["_forward_was_instance_attr"] = "forward" in block.__dict__
            block.__dict__["_uncompiled_forward"] = block.forward
            block.forward = _torch_compile(block.forward, **kwargs)
        elif i not in wanted and raw is not None:
            if block.__dict__.pop("_forward_was_instance_attr", False):
                block.forward = raw
            else:
                del block.__dict__["forward"]
            del block.__dict__["_uncompiled_forward"]
    return sorted(i for i, b in enumerate(model.blocks) if "_uncompiled_forward" in b.__dict__)


# =============================================================================
# 3. Speed probe: grid, equivalence, recommendation (pure; the GPU loop is flow_train_v2.speed_probe)
# =============================================================================
def parse_int_list(text) -> tuple:
    """Parse ``'20,15,10'`` into a tuple of non-negative ints (CLI helper)."""
    try:
        vals = tuple(int(x) for x in str(text).replace(";", ",").split(",") if x.strip())
    except ValueError as e:
        raise ValueError(f"expected a comma-separated list of integers, got {text!r}") from e
    if not vals or any(v < 0 for v in vals):
        raise ValueError(f"expected non-negative integers, got {text!r}")
    return vals


def parse_attn_list(text) -> tuple:
    """Parse a comma list of attention backend names (validated against ``ATTN_BACKENDS``)."""
    vals = tuple(x.strip() for x in str(text).split(",") if x.strip())
    if not vals or any(v not in ATTN_BACKENDS for v in vals):
        raise ValueError(f"attn backends must be a comma list from {ATTN_BACKENDS}, got {text!r}")
    return vals


def parse_onoff_list(text) -> tuple:
    """Parse a comma list of on/off (also 0/1, true/false, yes/no) into booleans."""
    table = {"off": False, "on": True, "0": False, "1": True, "false": False, "true": True, "no": False, "yes": True}
    try:
        vals = tuple(table[x.strip().lower()] for x in str(text).split(",") if x.strip())
    except KeyError as e:
        raise ValueError(f"compile settings must be a comma list of off/on, got {text!r}") from e
    if not vals:
        raise ValueError("empty compile list")
    return vals


@dataclass(frozen=True)
class SpeedProbeConfig:
    """Settings of the training speed probe: which checkpointed-block counts, attention backends and compile
    modes to time, the memory limit, a time budget and the tolerance for calling a row 'lossless'.
    """

    ckpt_blocks: tuple = (20, 15)
    attn_backends: tuple = ("auto", "cudnn")
    compile: tuple = (False, True)
    limit_gib: float = 85.0
    budget_minutes: float = 20.0  # no new row starts after this much probe time (the baseline always runs)
    equiv_tol: float = EQUIV_TOL
    timed_updates: int = 1  # measured updates per row after the discarded warm-up update

    def __post_init__(self):
        if self.timed_updates < 1:
            raise ValueError("timed_updates must be >= 1")
        if not (self.equiv_tol > 0 and math.isfinite(self.equiv_tol)):
            raise ValueError("equiv_tol must be positive and finite")


def plan_probe_rows(cfg: SpeedProbeConfig, n_blocks: int) -> list:
    """Row plan: the baseline (K=20, auto, no compile) first, then all non-compile rows, then the compile rows.

    Cheap rows first so that a time budget cuts the expensive ones; compile rows are ordered attention-major so a
    Dynamo reset (the attention backend is baked into the traced graph) is needed once, not per row.
    """
    def row(k, attn, comp, baseline=False):
        """One probe grid row dict (checkpointed blocks, attention backend, compile flag, baseline marker)."""
        return {"ckpt_blocks": min(int(k), n_blocks), "attn_backend": attn, "compile": bool(comp), "baseline": baseline}

    base = row(*BASELINE_ROW, baseline=True)
    seen = {(base["ckpt_blocks"], base["attn_backend"], base["compile"])}
    rows = [base]
    for comp in (False, True):
        if comp not in cfg.compile:
            continue
        for attn in cfg.attn_backends:
            for k in cfg.ckpt_blocks:
                r = row(k, attn, comp)
                key = (r["ckpt_blocks"], r["attn_backend"], r["compile"])
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
    return rows


def rel_diff(value, reference):
    """|value - reference| / |reference| (None if either is missing)."""
    if value is None or reference is None:
        return None
    return abs(float(value) - float(reference)) / max(abs(float(reference)), 1e-12)


METRICS = ("loss_a", "loss_b", "grad_norm")


def equivalence(row: dict, base: dict, tol: float = EQUIV_TOL) -> dict:
    """Relative difference of loss_A, loss_B and the global grad norm against the baseline row (same batch, noise, sigma).
    ``lossless`` is False when any exceeds ``tol`` or a value is non-finite."""
    diffs = {m: rel_diff(row.get(m), base.get(m)) for m in METRICS}
    present = [d for d in diffs.values() if d is not None]
    worst = max(present) if present else None
    ok = bool(present) and all(math.isfinite(d) and d <= tol for d in present)
    return {"rel_diff": diffs, "max_rel_diff": worst, "lossless": ok}


def recommend(rows: list, limit_gib: float) -> dict | None:
    """Fastest ok row within the memory limit that passes the equivalence check (the baseline always passes)."""
    ok = [r for r in rows if r.get("status") == "ok" and r.get("peak_gib") is not None and r["peak_gib"] <= limit_gib
          and (r.get("baseline") or r.get("lossless"))]
    return min(ok, key=lambda r: r["seconds"]) if ok else None


def flags_for(row: dict) -> str:
    """Command-line flags (``--ckpt-blocks K --attn-backend X --compile-blocks``) that reproduce a probe row."""
    parts = [f"--ckpt-blocks {row['ckpt_blocks']}"]
    if row.get("attn_effective", row.get("attn_backend")) != "auto":
        parts.append(f"--attn-backend {row['attn_effective']}")
    if row.get("compile"):
        parts.append("--compile-blocks")
    return " ".join(parts)


def format_table(report: dict) -> list:
    """Human-readable lines for the probe result (printed before the JSON event)."""
    head = f"{'K':>3} {'attn':<9} {'compile':<7} {'status':<14} {'s/update':>9} {'speedup':>8} {'peakGiB':>8} " \
           f"{'compile_s':>9} {'maxRelDiff':>10}  equivalence"
    lines = [head, "-" * len(head)]
    for r in report["rows"]:
        def f(key, fmt):
            """Format a numeric field of a probe row (``-`` if missing)."""
            v = r.get(key)
            return format(v, fmt) if isinstance(v, (int, float)) else "-"

        verdict = "baseline" if r.get("baseline") else ("ok" if r.get("lossless") else
                                                       "NOT lossless" if r.get("lossless") is False else "-")
        lines.append(f"{r['ckpt_blocks']:>3} {r['attn_backend']:<9} {'on' if r['compile'] else 'off':<7} "
                     f"{r['status']:<14} {f('seconds', '9.2f')} {f('speedup', '7.3f')}x {f('peak_gib', '8.1f')} "
                     f"{f('compile_overhead_seconds', '9.1f')} {f('max_rel_diff', '10.2e')}  {verdict}")
    rec = report.get("recommended")
    lines.append("")
    lines.append(f"recommended: {report['recommended_flags']}" if rec else "recommended: none (no row fit and passed)")
    return lines
