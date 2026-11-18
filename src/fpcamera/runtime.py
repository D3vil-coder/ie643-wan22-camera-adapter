"""Shared GPU lease and atomic run ledgers."""

from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf8"
    )
    os.replace(temp, path)


@contextlib.contextmanager
def gpu_lease(root=None):
    root = Path(
        root or os.environ.get("FPCAMERA_ROOT", Path(__file__).resolve().parents[2])
    )
    lock = root / "artifacts" / "gpu.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"A camera training, encoding or generation job already holds {lock}"
        )
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)
