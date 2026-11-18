"""Machine-specific settings for the operational tools, read from the environment (no laptop path is hard-coded).

Two things differ from one machine to the next:

  FPCAMERA_KEYS_ENV          path of a ``KEY=value`` file that holds the Hugging Face tokens (``HF_WRITE_TOKEN``, ...).
                             Default: ``<repo root>/keys.env`` (git-ignored). The file never lives in git and its
                             values are never printed.
  FPCAMERA_CHECKPOINTS_DIR   folder that holds the pulled run folders, e.g. ``<dir>/v2_full/adapter_000415.safetensors``.
                             Default: ``<repo root>/artifacts/checkpoints/v2`` (``artifacts/`` is git-ignored).

Used by ``tools/molab_recover.py``, ``tools/pull_milestones.py`` and ``tools/publish_public_hf.py``.  Every function takes
an ``environ`` mapping and a ``root`` so it can be tested without touching the real environment.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS_ENV_VAR = "FPCAMERA_KEYS_ENV"
CHECKPOINTS_DIR_VAR = "FPCAMERA_CHECKPOINTS_DIR"
DEFAULT_KEYS_NAME = "keys.env"
DEFAULT_CHECKPOINTS_REL = Path("artifacts") / "checkpoints" / "v2"


class KeysEnvNotFound(FileNotFoundError):
    """No keys file could be located; the message says how to point the tools at one (it never contains a secret)."""


def keys_env_path(environ=None, root: Path = ROOT) -> Path:
    """Locate the ``KEY=value`` file with the Hugging Face tokens.

    ``$FPCAMERA_KEYS_ENV`` wins (it must exist); otherwise ``<root>/keys.env``. Raises :class:`KeysEnvNotFound` with
    instructions when neither is a file.
    """
    environ = os.environ if environ is None else environ
    explicit = (environ.get(KEYS_ENV_VAR) or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path
        raise KeysEnvNotFound(f"{KEYS_ENV_VAR} is set to {path}, but that is not a file. Point it at a KEY=value file "
                              f"that contains the Hugging Face token (for example HF_WRITE_TOKEN=...).")
    default = Path(root) / DEFAULT_KEYS_NAME
    if default.is_file():
        return default
    raise KeysEnvNotFound(
        f"no keys file found: {KEYS_ENV_VAR} is not set and {default} does not exist. Create a KEY=value file with "
        f"HF_WRITE_TOKEN=... (or HF_DOWNLOAD_TOKEN=...) and either save it as {default} or set {KEYS_ENV_VAR} to its path "
        f"(PowerShell: $env:{KEYS_ENV_VAR} = 'C:\\path\\to\\keys.env'; persistent: setx {KEYS_ENV_VAR} 'C:\\path\\to\\keys.env'; "
        f"bash: export {KEYS_ENV_VAR}=/path/to/keys.env)."
    )


def checkpoints_dir(environ=None, root: Path = ROOT) -> Path:
    """Folder that holds the pulled run folders: ``$FPCAMERA_CHECKPOINTS_DIR`` or ``<root>/artifacts/checkpoints/v2``."""
    environ = os.environ if environ is None else environ
    explicit = (environ.get(CHECKPOINTS_DIR_VAR) or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return Path(root) / DEFAULT_CHECKPOINTS_REL
