"""Crash-safe checkpointing for long, interruptible (Molab) training runs.

Layout inside ``run_dir``::

    latest.json                      pointer to the newest verified-at-write checkpoint
    checkpoints/adapter_000025.safetensors
    checkpoints/resume_000025.pt     optimizer / RNG / sampler / counters
    checkpoints/ckpt_000025.json     manifest with sha256 of the two files above

Write protocol (every file): write ``<name>.tmp`` -> flush + fsync -> ``os.replace``.
Order: adapter, resume, manifest, and only then ``latest.json``.  A kill at any
point therefore leaves either the previous complete checkpoint or the new one;
partially written files are never named like a checkpoint.  On load, ``latest.json``
is tried first, then every manifest in descending step order, each verified
against its sha256 values, so a corrupt latest checkpoint falls back to the
previous one.

Retention: the newest ``keep_last`` checkpoints plus every checkpoint whose step
is a multiple of ``keep_every``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Callable

import torch
from safetensors.torch import load_file, save_file

_MANIFEST_RE = re.compile(r"^ckpt_(\d{6,})\.json$")


def sha256_file(path) -> str:
    """Streaming sha256 of a file (8 MiB blocks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fsync_path(path: Path) -> None:
    # O_RDWR: Windows refuses fsync on a read-only descriptor.
    fd = os.open(str(path), os.O_RDWR | getattr(os, "O_BINARY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    # Directory fsync is unsupported on Windows; best effort elsewhere.
    if os.name == "nt":
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def atomic_write(path, writer: Callable[[Path], None]) -> Path:
    """``writer(tmp_path)`` must create the file; we fsync and atomically rename it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    writer(tmp)
    _fsync_path(tmp)
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return path


def atomic_write_json(path, value) -> Path:
    """Write JSON with the same tmp + fsync + ``os.replace`` protocol as every other checkpoint file."""
    def w(tmp: Path):
        with open(tmp, "w", encoding="utf8") as f:
            json.dump(value, f, indent=2, default=str)
            f.flush()
    return atomic_write(path, w)


@dataclass
class LoadedCheckpoint:
    """A verified checkpoint returned by ``CheckpointManager.load_latest``: step, adapter tensors, resume state
    (optimizer / RNG / sampler), manifest, and ``skipped`` = newer candidates rejected as corrupt (with the reason).
    """

    step: int
    adapter_state: dict
    resume_state: dict
    manifest: dict
    skipped: list = field(default_factory=list)  # [(manifest_name, reason)] tried before this one


class CheckpointManager:
    """Crash-safe periodic checkpoints in ``<run_dir>/checkpoints`` (see module docstring for the layout and
    write order).  ``due`` says when to save (every N updates or T seconds), ``save`` writes adapter + resume state +
    manifest + ``latest.json`` atomically and prunes, ``load_latest`` returns the newest checkpoint whose sha256 values
    verify (falling back to older ones), ``archive_existing`` supports ``--no-resume``.
    """

    def __init__(
        self,
        run_dir,
        *,
        every_updates: int = 25,
        every_seconds: float = 600.0,
        keep_last: int = 3,
        keep_every: int = 250,
        clock: Callable[[], float] = time.monotonic,
    ):
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        self.run_dir = Path(run_dir)
        self.dir = self.run_dir / "checkpoints"
        self.every_updates = int(every_updates)
        self.every_seconds = float(every_seconds)
        self.keep_last = int(keep_last)
        self.keep_every = int(keep_every)
        self.clock = clock
        self.dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_tmp()
        self.last_saved_step = None
        self.last_save_time = self.clock()

    # -- helpers -----------------------------------------------------------------
    def _cleanup_tmp(self):
        for p in list(self.dir.glob("*.tmp")) + list(self.run_dir.glob("*.tmp")):
            try:
                p.unlink()
            except OSError:
                pass

    @property
    def latest_path(self) -> Path:
        """Path of the ``latest.json`` pointer (written last, so it only ever names complete checkpoints)."""
        return self.run_dir / "latest.json"

    def _names(self, step: int):
        return (
            f"adapter_{step:06d}.safetensors",
            f"resume_{step:06d}.pt",
            f"ckpt_{step:06d}.json",
        )

    def manifest_steps(self):
        """Sorted update numbers that have a ``ckpt_NNNNNN.json`` manifest."""
        steps = []
        for p in self.dir.iterdir():
            m = _MANIFEST_RE.match(p.name)
            if m:
                steps.append(int(m.group(1)))
        return sorted(steps)

    # -- policy ------------------------------------------------------------------
    def due(self, step: int) -> bool:
        """True if a periodic checkpoint should be written now: every ``every_updates`` updates, or once
        ``every_seconds`` have passed since the last save (whichever comes first); never twice for the same step.
        """
        if self.last_saved_step == step:
            return False
        if self.every_updates > 0 and step % self.every_updates == 0:
            return True
        return self.every_seconds > 0 and (self.clock() - self.last_save_time) >= self.every_seconds

    # -- save --------------------------------------------------------------------
    def save(self, step: int, adapter_state: dict, resume_state: dict, extra=None, reason="periodic"):
        """Write checkpoint ``step``: adapter safetensors (FP32), resume state (torch.save), manifest with both sha256
        values, then ``latest.json``; finally prune old checkpoints.  ``reason`` (periodic/deadline/emergency/final) and
        ``extra`` (e.g. architecture) go into the manifest.  Returns the manifest.
        """
        adapter_name, resume_name, manifest_name = self._names(step)
        adapter_path = self.dir / adapter_name
        resume_path = self.dir / resume_name
        tensors = {k: v.detach().cpu().float().contiguous() for k, v in adapter_state.items()}
        atomic_write(
            adapter_path,
            lambda tmp: save_file(tensors, str(tmp), metadata={"step": str(step), "reason": reason}),
        )
        atomic_write(resume_path, lambda tmp: torch.save(resume_state, str(tmp)))
        manifest = {
            "step": int(step),
            "reason": reason,
            "adapter_file": adapter_name,
            "adapter_sha256": sha256_file(adapter_path),
            "resume_file": resume_name,
            "resume_sha256": sha256_file(resume_path),
            "saved_unix_time": time.time(),
            **(extra or {}),
        }
        atomic_write_json(self.dir / manifest_name, manifest)
        pointer = {
            "step": int(step),
            "manifest": manifest_name,
            "adapter_file": adapter_name,
            "adapter_sha256": manifest["adapter_sha256"],
            "resume_file": resume_name,
            "resume_sha256": manifest["resume_sha256"],
            "reason": reason,
        }
        # latest.json is written last on purpose: if the process dies earlier, the pointer still names the previous
        # complete checkpoint, so resume never sees a half-written one.
        atomic_write_json(self.latest_path, pointer)  # written last
        self.last_saved_step = int(step)
        self.last_save_time = self.clock()
        self.prune(protect=step)
        return manifest

    def prune(self, protect: int | None = None):
        """Delete checkpoints that are neither among the newest ``keep_last`` nor a multiple of ``keep_every``
        (nor ``protect``).  The manifest is removed first so a half-deleted checkpoint is never listed.
        """
        steps = self.manifest_steps()
        keep = set(steps[-self.keep_last:])
        if self.keep_every > 0:
            keep |= {s for s in steps if s % self.keep_every == 0}
        if protect is not None:
            keep.add(protect)
        removed = []
        for s in steps:
            if s in keep:
                continue
            adapter_name, resume_name, manifest_name = self._names(s)
            for name in (manifest_name, adapter_name, resume_name):  # manifest first
                try:
                    (self.dir / name).unlink()
                except FileNotFoundError:
                    pass
            removed.append(s)
        return removed

    # -- load --------------------------------------------------------------------
    def _candidates(self):
        seen, order = set(), []
        if self.latest_path.exists():
            try:
                ptr = json.loads(self.latest_path.read_text(encoding="utf8"))
                name = ptr["manifest"]
                order.append((name, ptr))
                seen.add(name)
            except Exception:
                order.append(("latest.json", None))
        for s in reversed(self.manifest_steps()):
            name = self._names(s)[2]
            if name not in seen:
                order.append((name, None))
                seen.add(name)
        return order

    def has_checkpoint_state(self) -> bool:
        """True if a ``latest.json`` or any manifest exists (i.e. a resume is possible)."""
        return self.latest_path.exists() or bool(self.manifest_steps())

    def _try_load(self, name: str, pointer):
        if name == "latest.json":
            raise ValueError("unreadable latest.json")
        manifest = json.loads((self.dir / name).read_text(encoding="utf8"))
        adapter_path = self.dir / manifest["adapter_file"]
        resume_path = self.dir / manifest["resume_file"]
        for path, key in ((adapter_path, "adapter_sha256"), (resume_path, "resume_sha256")):
            if not path.is_file():
                raise FileNotFoundError(path.name)
            # integrity check on load: a truncated or bit-rotted file makes load_latest fall back to the previous checkpoint
            # (recorded in LoadedCheckpoint.skipped) instead of resuming from garbage.
            if sha256_file(path) != manifest[key]:
                raise ValueError(f"sha256 mismatch for {path.name}")
            if pointer is not None and pointer.get(key) != manifest[key]:
                raise ValueError(f"latest.json disagrees with manifest for {path.name}")
        adapter_state = load_file(str(adapter_path), device="cpu")
        resume_state = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        if int(resume_state.get("step", -1)) != int(manifest["step"]):
            raise ValueError("resume step disagrees with manifest")
        return LoadedCheckpoint(int(manifest["step"]), adapter_state, resume_state, manifest)

    def load_latest(self) -> LoadedCheckpoint | None:
        """Return the newest verifiable ``LoadedCheckpoint`` or None if the run is fresh.  Tries ``latest.json`` first,
        then every manifest newest-first; raises if state exists but nothing verifies.
        """
        skipped = []
        for name, pointer in self._candidates():
            try:
                loaded = self._try_load(name, pointer)
            except Exception as exc:  # corrupt / partial / missing -> fall back
                skipped.append((name, f"{type(exc).__name__}: {exc}"))
                continue
            loaded.skipped = skipped
            self.last_saved_step = loaded.step
            self.last_save_time = self.clock()
            return loaded
        if skipped:
            raise RuntimeError(
                "Checkpoint state exists but no checkpoint verifies: " + json.dumps(skipped)
            )
        return None

    def archive_existing(self) -> Path | None:
        """Used by --no-resume: move (never delete) prior checkpoint state aside."""
        if not self.has_checkpoint_state():
            return None
        dest = self.run_dir / f"old_{time.strftime('%Y%m%d_%H%M%S')}"
        dest.mkdir()
        if self.latest_path.exists():
            os.replace(self.latest_path, dest / "latest.json")
        os.replace(self.dir, dest / "checkpoints")
        for name in ("progress.jsonl", "progress.json"):
            p = self.run_dir / name
            if p.exists():
                os.replace(p, dest / name)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.last_saved_step = None
        return dest
