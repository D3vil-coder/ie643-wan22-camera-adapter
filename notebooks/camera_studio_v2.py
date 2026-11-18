"""Camera Studio v2: the self-sufficient, user-facing marimo notebook for the IE 643 v2 camera adapter.

Pipeline position: the last stage, after training, evaluation and calibration.  It turns ONE photograph into a
5 s, 1280 x 704, 24 fps video with a virtual camera move (pan / tilt by drawing a stroke, or zoom in / out) and one
strength slider whose label shows the real physical amount (strength s -> calibrated u -> degrees or zoom ratio,
see ``fpcamera.strength``).  Generation runs in-process through ``fpcamera.flow_infer.generate`` with the pipeline
loaded once; everything (install, download, load, generate) is a visible cell, top to bottom.

How to run::

    marimo run  notebooks/camera_studio_v2.py        # app view
    marimo edit notebooks/camera_studio_v2.py        # editable, code visible
    FPCAMERA_DRY_RUN=1 marimo edit notebooks/camera_studio_v2.py   # laptop UI test: no GPU, no downloads, placeholder mp4

On Molab (GPU sandbox) open the file and run all cells; the first run installs missing packages and downloads the
Wan2.2-TI2V-5B base model (about 32 GB) and the adapter bundle (HF repo D3vil-Reap3r/ie643-wan22-camera-adapter-v2).
Environment switches (all optional): FPCAMERA_DRY_RUN, FPCAMERA_WORK_DIR, FPCAMERA_USE_REPO_SRC, FPCAMERA_ADAPTER_REPO.

Cell map: SETUP (S0 configuration, S1 helpers, S2 GPU check, S3 packages, S4 base model, S5 adapter + code bundle,
S6 fpcamera modules + calibration, pipeline, engine ``run_generation``) and USER INTERFACE (photo, motion, drawing
canvas + direction detection, strength + preview, advanced settings, generate, result).  Details: docs/v2/INTERFACE_V2.md.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Camera Studio v2")


# =====================================================================================================
# Camera Studio v2 - the self-sufficient, user-facing notebook for the IE 643 v2 camera adapter.
#
#   * Run it top to bottom on a fresh GPU sandbox (Molab): it installs what is missing, downloads the
#     Wan2.2-TI2V-5B base model and the adapter bundle, loads the pipeline ONCE and generates videos
#     in-process with live progress. Every step is a visible cell below; nothing runs hidden.
#   * `FPCAMERA_DRY_RUN=1` skips installs, downloads and the model and makes a placeholder mp4 (a synthetic
#     pan/tilt/zoom of the input image) so the whole UI can be tried on a laptop without a GPU.
#   * Environment variables (all optional):
#       FPCAMERA_DRY_RUN=1          UI test mode, no GPU, no downloads
#       FPCAMERA_WORK_DIR=<dir>     where downloads, inputs and videos go (default ~/fpcamera_studio)
#       FPCAMERA_USE_REPO_SRC=1     import fpcamera from the repository's src/ instead of the code zip
#       FPCAMERA_ADAPTER_REPO=<id>  use another HF model repo for the adapter bundle
# =====================================================================================================


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        r"""
        # Camera Studio v2

        Turn **one photograph** into a 5-second video (1280 x 704, 24 fps) with a **virtual camera move**:
        pan, tilt or zoom, with a strength you control. A frozen *Wan2.2-TI2V-5B* image-to-video model is steered
        by a small trained camera adapter (IE 643, v2).

        **How to use it**

        1. **Photo**: upload one, or keep the built-in sample.
        2. **Motion**: for *Pan / Tilt*, **draw a stroke on the photo in the direction the camera should move**
           (the notebook detects left / right / up / down). *Zoom in* and *Zoom out* need no drawing.
        3. **Strength**: 0 is a static camera, 1 is the largest motion the adapter was trained on (8 degrees of pan or tilt,
           a 1.25x zoom). The label shows the real amount.
        4. **Generate**. Progress is live; the video, a download button and a run summary appear at the bottom.

        **Run the notebook from the top.** On a fresh GPU sandbox the *Setup* cells install missing packages and download
        the base model (about 32 GB) and the adapter, then load the model once: expect roughly 10 minutes the first time.
        Re-running skips every finished step. A 50-step video then takes about 6 minutes on an RTX PRO 6000 class GPU
        (about 43 GiB of GPU memory at peak); fewer steps (Advanced) are faster.
        """
    )
    return (mo,)


# =====================================================================================================
# SETUP
# =====================================================================================================


@app.cell
def _(mo):
    # ---- Setup 0/6: configuration -----------------------------------------------------------------
    # Only constants and imports of the standard library here (no third-party import yet, so the
    # install cell below can still change package versions before anything heavy is loaded).
    import hashlib
    import importlib.metadata as importlib_metadata
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    import threading
    import time
    import zipfile
    from pathlib import Path

    # Public Hugging Face *model* repo with the adapter and the code bundle (no token needed). Layout:
    #   adapter/adapter_000415.safetensors, adapter/ckpt_000415.json, adapter/config.json   (kept together)
    #   code/fpcamera_src.zip            (repository layout: src/fpcamera/...)
    #   configs/inference_v2.json, configs/strength_calibration_v2*.json
    ADAPTER_REPO = "D3vil-Reap3r/ie643-wan22-camera-adapter-v2"
    ADAPTER_REVISION = None  # None = latest main; put a commit hash here to pin the bundle
    ADAPTER_REPO = os.environ.get("FPCAMERA_ADAPTER_REPO") or ADAPTER_REPO

    # The frozen base model. generate() loads it with local_files_only=True, so it must be in the
    # default Hugging Face cache first (the download cell below puts it there).
    BASE_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    BASE_REVISION = "b8fff7315c768468a5333511427288870b2e9635"

    # Attention kernels offered under Advanced. "auto" = torch's own dispatch (the verified default);
    # the others are opt-in speed options that change numerics slightly (docs/v2/INFERENCE_V2.md).
    ATTN_BACKENDS = ("auto", "cudnn", "flash", "efficient")

    # Dry run: no installs, no downloads, no model; videos are a synthetic warp of the input image.
    DRY_RUN = os.environ.get("FPCAMERA_DRY_RUN", "").strip().lower() in ("1", "true", "yes")

    # Pinned runtime (requirements.txt + requirements-ui.txt of the repository).
    RUNTIME_PINS = {
        "torch": "2.11.0", "torchvision": "0.26.0", "diffusers": "0.36.0", "transformers": "4.57.6",
        "accelerate": "1.15.0", "safetensors": "0.8.0", "numpy": "2.4.6", "Pillow": "12.3.0",
        "huggingface-hub": "0.36.2", "imageio": "2.38.0", "imageio-ffmpeg": "0.6.0",
        "opencv-python": "5.0.0.93", "sentencepiece": "0.2.2", "protobuf": "6.33.6", "ftfy": "6.3.1",
        "tokenizers": "0.22.2", "anywidget": "0.11.0",
    }
    # The three packages fpcamera.wan.require_pinned_runtime() enforces. Everything else is only
    # installed when it is missing altogether (a newer compatible version is left alone).
    STRICT_PINS = ("torch", "diffusers", "transformers")

    # Where things live. On Molab nothing persists across a sandbox restart, so this is just a work folder.
    _home = Path(os.environ.get("FPCAMERA_WORK_DIR") or (Path.home() / "fpcamera_studio"))
    try:
        _home.mkdir(parents=True, exist_ok=True)
    except OSError:
        _home = Path(tempfile.gettempdir()) / "fpcamera_studio"
        _home.mkdir(parents=True, exist_ok=True)
    WORK_DIR = _home
    BUNDLE_DIR = WORK_DIR / "bundle"  # snapshot of ADAPTER_REPO
    RUNS_DIR = WORK_DIR / "runs"  # one folder per generated video
    INPUTS_DIR = WORK_DIR / "inputs"  # prepared input images
    STATE_DIR = WORK_DIR / "state"  # "finished" markers that make the setup idempotent
    LOG_DIR = WORK_DIR / "logs"
    for _d in (RUNS_DIR, INPUTS_DIR, STATE_DIR, LOG_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    # Quiet, notebook-friendly library output (set before huggingface_hub / transformers are imported).
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ.setdefault("USE_TF", "0")

    # The repository checkout this notebook sits in, if any (used by dry run and FPCAMERA_USE_REPO_SRC).
    _nb_dir = mo.notebook_dir()
    REPO_ROOT = _nb_dir.parent if (_nb_dir is not None and (_nb_dir.parent / "src" / "fpcamera").is_dir()) else None
    USE_REPO_SRC = DRY_RUN or os.environ.get("FPCAMERA_USE_REPO_SRC", "") == "1"

    mo.md(
        f"""
        ## Setup
        Work folder `{WORK_DIR}` · base model `{BASE_MODEL}` @ `{BASE_REVISION[:8]}` · adapter repo `{ADAPTER_REPO}`
        """
    )
    return (
        ADAPTER_REPO, ADAPTER_REVISION, ATTN_BACKENDS, BASE_MODEL, BASE_REVISION, BUNDLE_DIR, DRY_RUN, INPUTS_DIR,
        LOG_DIR, Path, REPO_ROOT, RUNS_DIR, RUNTIME_PINS, STATE_DIR, STRICT_PINS, USE_REPO_SRC, WORK_DIR, hashlib,
        importlib_metadata, json, os, shutil, subprocess, sys, threading, time, zipfile,
    )


@app.cell
def _(Path, mo, os, threading, time):
    # ---- Setup 1/6: small helpers shared by the setup cells ----------------------------------------
    def human_size(n_bytes):
        """12345678 -> '11.8 MB' (binary units)."""
        value = float(n_bytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if value < 1024 or unit == "TB":
                return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
            value /= 1024

    def dir_size(path):
        """Bytes under `path` (symlinks counted as links, so the HF cache's blobs are not counted twice)."""
        total = 0
        for root, _dirs, files in os.walk(path, followlinks=False):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    pass
        return total

    def fail(title, detail):
        """Stop this cell AND every cell that depends on it, showing a red message instead."""
        mo.stop(True, mo.callout(mo.md(f"**{title}**\n\n{detail}"), kind="danger"))

    def download_with_progress(*, title, total_bytes, watch_dir, task):
        """Run `task()` (a blocking download) in a thread and show a live progress bar.

        Progress is measured from the bytes that appear under `watch_dir` (works for any
        huggingface_hub version, with or without hf-xet). Exceptions of the task are re-raised here.
        """
        box = {}

        def _worker():
            try:
                box["result"] = task()
            except BaseException as error:  # noqa: BLE001 - re-raised in the calling thread below
                box["error"] = error

        thread = threading.Thread(target=_worker, daemon=True)
        total_mb = max(1, int(total_bytes / 1e6))
        start_bytes = dir_size(watch_dir)  # a resumed download starts part-way
        shown_mb = min(total_mb, int(start_bytes / 1e6))
        started = time.time()
        with mo.status.progress_bar(total=total_mb, title=title, subtitle="starting...", show_rate=False,
                                    show_eta=False, remove_on_exit=False) as bar:
            if shown_mb:
                bar.update(increment=shown_mb)
            thread.start()
            while thread.is_alive():
                thread.join(timeout=1.0)
                done = dir_size(watch_dir)
                mb = min(total_mb, int(done / 1e6))
                elapsed = max(1e-6, time.time() - started)
                rate = max(0.0, (done - start_bytes) / elapsed)
                left = (total_bytes - done) / rate if rate > 1e5 else None
                eta = f" · ~{left / 60:.0f} min left" if left is not None and left > 5 else ""
                bar.update(increment=max(0, mb - shown_mb),
                           subtitle=f"{human_size(done)} of {human_size(total_bytes)} · {human_size(rate)}/s{eta}")
                shown_mb = max(shown_mb, mb)
            if shown_mb < total_mb:
                bar.update(increment=total_mb - shown_mb, subtitle=f"{human_size(total_bytes)} done")
        if "error" in box:
            raise box["error"]
        return box.get("result")

    return dir_size, download_with_progress, fail, human_size


@app.cell
def _(DRY_RUN, mo, shutil, subprocess):
    # ---- Setup 2/6: GPU check ---------------------------------------------------------------------
    # nvidia-smi is queried (not torch) so that this works before torch is installed. No GPU: stop here
    # with a clear message; nothing below (installs, the 32 GB download, the model) will run.
    gpu_info = "dry run (no GPU needed)"
    gpu_memory_gib = None
    if not DRY_RUN:
        _smi = shutil.which("nvidia-smi")
        _line = ""
        if _smi:
            try:
                _line = subprocess.run([_smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
            except (OSError, subprocess.SubprocessError, IndexError):
                _line = ""
        mo.stop(
            not _line,
            mo.callout(
                mo.md(
                    "**No CUDA GPU found.** Wan2.2-TI2V-5B needs an NVIDIA GPU with about 48 GB or more of memory "
                    "(about 43 GiB peak at 1280 x 704, 121 frames).\n\n"
                    "On Molab choose a **GPU machine** for this notebook and run it again. To try only the user "
                    "interface without a GPU, start it with the environment variable `FPCAMERA_DRY_RUN=1` "
                    "(placeholder videos, no downloads)."
                ),
                kind="danger",
            ),
        )
        _name, _, _mem = _line.partition(",")
        gpu_memory_gib = float(_mem.strip() or 0) / 1024
        gpu_info = f"{_name.strip()} · {gpu_memory_gib:.0f} GiB"
    _small = gpu_memory_gib is not None and gpu_memory_gib < 48
    mo.callout(
        mo.md(f"**GPU**: {gpu_info}"
              + ("\n\nThis is below the ~48 GiB the model needs at 1280 x 704; generation will probably run out "
                 "of memory." if _small else "")),
        kind="warn" if _small else "success",
    )
    return (gpu_info,)


@app.cell
def _(DRY_RUN, RUNTIME_PINS, STRICT_PINS, STATE_DIR, gpu_info, importlib_metadata, mo, shutil, subprocess, sys, time):
    # ---- Setup 3/6: Python packages ---------------------------------------------------------------
    # Installs the pinned runtime only where something is missing: torch (any 2.11.x), diffusers 0.36.0 and
    # transformers 4.57.6 must match exactly (fpcamera.wan.require_pinned_runtime checks them); every other
    # pinned package is installed only if it is absent. Uses `uv pip` when available (Molab), else `pip`.
    _ = gpu_info  # ordering: runs after the GPU check

    def installed_version(dist):
        """Installed version of a distribution (OpenCV counts as installed under any of its wheel names), or None."""
        names = {"opencv-python": ("opencv-python", "opencv-python-headless", "opencv-contrib-python")}.get(dist, (dist,))
        for name in names:
            try:
                return importlib_metadata.version(name)
            except importlib_metadata.PackageNotFoundError:
                continue
        return None

    def satisfied(dist, pin, found):
        """True if the installed version meets the pin: torch any 2.11.x, diffusers / transformers exactly, anything else just present."""
        if found is None:
            return False
        if dist == "torch":
            return found.split("+")[0].startswith("2.11.")
        if dist in STRICT_PINS:
            return found == pin
        return True  # present: leave it alone

    _rows = []
    _todo = []
    for _dist, _pin in RUNTIME_PINS.items():
        _found = installed_version(_dist)
        _ok = satisfied(_dist, _pin, _found)
        _rows.append([_dist, _pin, _found or "missing", "ok" if _ok else "install"])
        if not _ok:
            _todo.append(f"{_dist}=={_pin}")
    if "torch==2.11.0" in _todo and "torchvision==0.26.0" not in _todo:
        _todo.append("torchvision==0.26.0")  # torch and torchvision must move together
    # A preinstalled torch (the GPU image's own 2.11.x build) must survive: pip/uv could otherwise pull a newer
    # torch while resolving accelerate. Constrain torch/torchvision to what is installed (as tools/molab_recover.py does).
    _constraints = []
    for _name in ("torch", "torchvision"):
        _have = installed_version(_name)
        if _have and f"{_name}=={RUNTIME_PINS[_name]}" not in _todo and satisfied(_name, RUNTIME_PINS[_name], _have):
            _constraints.append(f"{_name}=={_have.split('+')[0]}")

    deps_changed = []
    _log_tail = ""
    if DRY_RUN:
        _headline = mo.callout(mo.md("**Python packages**: dry run, install skipped."), kind="neutral")
    elif not _todo:
        _headline = mo.callout(mo.md("**Python packages**: every pinned dependency is present, nothing to install."),
                               kind="success")
    else:
        _constraint_args = []
        if _constraints:
            (STATE_DIR / "constraints.txt").write_text("\n".join(_constraints) + "\n")
            _constraint_args = ["-c", str(STATE_DIR / "constraints.txt")]
        if shutil.which("uv"):
            _cmd = [shutil.which("uv"), "pip", "install", "--python", sys.executable, *_constraint_args, *_todo]
        else:
            if subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode != 0:
                subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], capture_output=True)
            _cmd = [sys.executable, "-m", "pip", "install", *_constraint_args, *_todo]
        _tail = []
        _started = time.time()
        with mo.status.spinner(title=f"Installing {len(_todo)} package(s): {', '.join(_todo)}",
                               subtitle="this can take a few minutes (torch is large)") as _spin:
            _proc = subprocess.Popen(_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for _line in _proc.stdout:
                _tail.append(_line.rstrip())
                _spin.update(subtitle=f"{time.time() - _started:.0f}s · {_tail[-1][:110]}")
            _code = _proc.wait()
        _log_tail = "\n".join(_tail[-25:])
        _bad = []
        for _dist, _pin in RUNTIME_PINS.items():
            if not satisfied(_dist, _pin, installed_version(_dist)):
                _bad.append(_dist)
        if _code != 0 or _bad:
            mo.stop(True, mo.callout(mo.vstack([
                mo.md(f"**Installing the Python packages failed** (exit code {_code}; still wrong: {_bad or 'none'}).\n\n"
                      f"Command: `{' '.join(_cmd)}`"),
                mo.md(f"```\n{_log_tail}\n```"),
            ]), kind="danger"))
        # A package that was already imported keeps its old version in memory until the kernel restarts.
        _modules = {"Pillow": "PIL", "opencv-python": "cv2", "huggingface-hub": "huggingface_hub",
                    "imageio-ffmpeg": "imageio_ffmpeg"}
        deps_changed = [d.split("==")[0] for d in _todo
                        if _modules.get(d.split("==")[0], d.split("==")[0].replace("-", "_").lower()) in sys.modules]
        _headline = mo.callout(
            mo.md(f"**Python packages**: installed {', '.join(_todo)} in {time.time() - _started:.0f} s."
                  + (f"\n\n**Restart the kernel and run again**: {', '.join(deps_changed)} was already loaded in "
                     "this session with another version." if deps_changed else "")),
            kind="warn" if deps_changed else "success",
        )
    mo.stop(bool(deps_changed), _headline)
    deps_ready = True
    mo.vstack([
        _headline,
        mo.accordion({"Package details": mo.md("| package | pinned | found | action |\n|---|---|---|---|\n"
                                               + "\n".join("| " + " | ".join(r) + " |" for r in _rows)
                                               + (f"\n\n```\n{_log_tail}\n```" if _log_tail else ""))}),
    ])
    return (deps_ready,)


@app.cell
def _(DRY_RUN, deps_ready):
    # ---- Hugging Face client (imported only after the install cell, so a downgraded huggingface-hub is picked up) ----
    _ = deps_ready
    HfApi = None
    snapshot_download = None
    hf_cache_dir = None
    if not DRY_RUN:
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub import constants as _hf_constants

        hf_cache_dir = _hf_constants.HF_HUB_CACHE
    return HfApi, hf_cache_dir, snapshot_download


@app.cell
def _(BASE_MODEL, BASE_REVISION, DRY_RUN, HfApi, STATE_DIR, Path, deps_ready, download_with_progress, fail, hf_cache_dir,
      human_size, json, mo, shutil, snapshot_download, time):
    # ---- Setup 4/6: base model ---------------------------------------------------------------------
    # Wan-AI/Wan2.2-TI2V-5B-Diffusers at the pinned revision (about 32 GB) into the default Hugging Face
    # cache. Idempotent: a marker file lists every file with its size; if all of them are present the
    # cell finishes instantly without touching the network.
    _ = deps_ready
    base_model_dir = None
    if DRY_RUN:
        _view = mo.callout(mo.md("**Base model**: dry run, download skipped."), kind="neutral")
    else:
        _marker = STATE_DIR / "base_model.json"
        _repo_dir = Path(hf_cache_dir) / ("models--" + BASE_MODEL.replace("/", "--"))

        def _complete(files):
            """Snapshot folder if every expected file is present with the expected size, else None."""
            try:
                folder = Path(snapshot_download(BASE_MODEL, revision=BASE_REVISION, local_files_only=True))
            except Exception:  # noqa: BLE001 - not cached (yet)
                return None
            for name, size in files.items():
                target = folder / name
                if not target.exists() or target.stat().st_size != size:
                    return None
            return folder

        _files = None
        try:
            _record = json.loads(_marker.read_text())
            if _record.get("revision") == BASE_REVISION:
                _files = _record["files"]
        except (OSError, ValueError, KeyError):
            _files = None
        base_model_dir = _complete(_files) if _files else None
        _started = time.time()
        _note = "already present, skipped"
        if base_model_dir is None:
            try:
                _info = HfApi().model_info(BASE_MODEL, revision=BASE_REVISION, files_metadata=True)
            except Exception as _error:  # noqa: BLE001
                fail("Could not reach Hugging Face to list the base model files",
                     f"`{type(_error).__name__}: {_error}`\n\nCheck the network connection of the sandbox and run again.")
            _files = {s.rfilename: int(s.size or 0) for s in _info.siblings}
            _total = sum(_files.values())
            _repo_dir.mkdir(parents=True, exist_ok=True)
            _free = shutil.disk_usage(_repo_dir).free
            if _free < _total * 1.05 + 2e9:
                fail("Not enough free disk space for the base model",
                     f"Needs about {human_size(_total)}, only {human_size(_free)} free under `{_repo_dir.parent}`.")
            base_model_dir = _complete(_files)  # a complete cache from an earlier session, no marker yet
            if base_model_dir is None:
                download_with_progress(
                    title=f"Downloading {BASE_MODEL} ({human_size(_total)})", total_bytes=_total, watch_dir=_repo_dir,
                    task=lambda: snapshot_download(BASE_MODEL, revision=BASE_REVISION, max_workers=8))
                base_model_dir = _complete(_files)
                if base_model_dir is None:
                    fail("The base model download is incomplete", "Some files are missing or have the wrong size. Run the cell again to resume.")
                _note = f"downloaded in {(time.time() - _started) / 60:.1f} min"
            else:
                _note = "found in the Hugging Face cache, skipped"
            _marker.write_text(json.dumps({"revision": BASE_REVISION, "files": _files}))
        _view = mo.callout(mo.md(f"**Base model**: `{BASE_MODEL}` @ `{BASE_REVISION[:8]}` ({_note}). Folder `{base_model_dir}`"),
                           kind="success")
    _view
    return (base_model_dir,)


@app.cell
def _(ADAPTER_REPO, ADAPTER_REVISION, BUNDLE_DIR, DRY_RUN, HfApi, REPO_ROOT, USE_REPO_SRC, WORK_DIR, base_model_dir,
      download_with_progress, fail, hashlib, human_size, mo, shutil, snapshot_download, sys, zipfile):
    # ---- Setup 5/6: adapter, configs and code bundle -----------------------------------------------
    # snapshot_download of the public ADAPTER_REPO into BUNDLE_DIR (a re-run only fetches what changed),
    # then the code zip is unpacked once and put on sys.path. Needs the adapter files side by side
    # (adapter_*.safetensors, ckpt_*.json, config.json) because generate() reads the guide fill mode
    # and the checkpoint digest from them.
    _ = base_model_dir
    adapter_path = None
    if DRY_RUN:
        if REPO_ROOT is None:
            fail("Dry run needs the repository checkout", "Start the notebook from `notebooks/` inside the repository.")
        bundle_root = REPO_ROOT
        code_root = REPO_ROOT / "src"
        _bundle_note = f"dry run, configs and code come from the repository checkout `{REPO_ROOT}`"
    else:
        if "to be filled" in ADAPTER_REPO or "/" not in ADAPTER_REPO:
            fail("ADAPTER_REPO is not set", "Set `ADAPTER_REPO` in the first Setup cell to the public Hugging Face model repo of the adapter bundle.")
        BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
        _bundle_note = "downloaded"
        try:
            _info = HfApi().model_info(ADAPTER_REPO, revision=ADAPTER_REVISION, files_metadata=True)
            _total = sum(int(s.size or 0) for s in _info.siblings)
            download_with_progress(
                title=f"Downloading adapter bundle {ADAPTER_REPO} ({human_size(_total)})", total_bytes=max(_total, 1),
                watch_dir=BUNDLE_DIR,
                task=lambda: snapshot_download(ADAPTER_REPO, revision=ADAPTER_REVISION, local_dir=str(BUNDLE_DIR)))
        except Exception as _error:  # noqa: BLE001
            if not list((BUNDLE_DIR / "adapter").glob("adapter_*.safetensors")):
                fail(f"Could not download the adapter bundle `{ADAPTER_REPO}`",
                     f"`{type(_error).__name__}: {_error}`\n\nIs the repository public and spelled correctly?")
            _bundle_note = f"offline, using the copy from an earlier run ({type(_error).__name__})"
        bundle_root = BUNDLE_DIR

        _adapters = sorted((BUNDLE_DIR / "adapter").glob("adapter_*.safetensors"))
        if not _adapters:
            fail("The bundle has no adapter", f"Expected `adapter/adapter_NNNNNN.safetensors` in `{ADAPTER_REPO}`.")
        adapter_path = _adapters[-1]  # highest step number
        _step = adapter_path.stem.split("_")[-1]
        _missing = [n for n in (f"ckpt_{_step}.json", "config.json") if not (adapter_path.parent / n).is_file()]
        if _missing and not adapter_path.with_suffix(".json").is_file():
            fail("The adapter folder is incomplete",
                 f"`{adapter_path.parent}` lacks {_missing}. Keep `adapter_{_step}.safetensors`, `ckpt_{_step}.json` "
                 "and `config.json` together: without them the guide fill mode and the checkpoint digest are unknown.")

        if USE_REPO_SRC and REPO_ROOT is not None:
            code_root = REPO_ROOT / "src"
        else:
            _zip = BUNDLE_DIR / "code" / "fpcamera_src.zip"
            if not _zip.is_file():
                fail("The bundle has no code archive", f"Expected `code/fpcamera_src.zip` in `{ADAPTER_REPO}`.")
            _digest = hashlib.sha256(_zip.read_bytes()).hexdigest()[:12]
            _dest = WORK_DIR / "code" / _digest
            if not (_dest / ".extracted").is_file():
                shutil.rmtree(_dest, ignore_errors=True)
                _dest.mkdir(parents=True)
                with zipfile.ZipFile(_zip) as _archive:
                    _archive.extractall(_dest)
                (_dest / ".extracted").write_text(_digest)
            _packages = sorted(_dest.rglob("fpcamera/__init__.py"))
            if not _packages:
                fail("The code archive has no `fpcamera` package", f"Looked in `{_dest}`.")
            code_root = _packages[0].parent.parent  # <extract>/src
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    _adapter_note = "none (dry run)" if adapter_path is None else f"`{adapter_path.name}` ({human_size(adapter_path.stat().st_size)})"
    mo.callout(mo.md(f"**Adapter and code**: {_bundle_note}. Adapter: {_adapter_note}. Code on `sys.path`: `{code_root}`"),
               kind="success")
    return adapter_path, bundle_root, code_root


@app.cell
def _(DRY_RUN, Path, adapter_path, bundle_root, code_root, fail, json, mo):
    # ---- Setup 6/6: load the fpcamera modules, defaults and the strength calibration -----------------
    _ = code_root  # ordering: after sys.path is set
    import importlib.util as _importlib_util

    from fpcamera import border_metrics as bm
    from fpcamera import strength as st
    from fpcamera.flow_geometry import VideoContract, camera_path_v2

    if _importlib_util.find_spec("fpcamera.trajectory_direction") is None:
        fail("The code bundle is older than this notebook",
             "`fpcamera.trajectory_direction` (stroke -> pan/tilt) is missing from the code archive. "
             "Rebuild `code/fpcamera_src.zip` from the current repository and upload it to the adapter repo.")
    from fpcamera.trajectory_direction import detect_direction

    # Video contract of the v2 adapter and the trained maxima (configs/data_v2.json is not in the bundle,
    # so the maxima are given explicitly: 8 degrees pan/tilt, 1.25x focal zoom).
    CONTRACT = VideoContract(width=1280, height=704, frames=121, fps=24)
    max_amplitudes = {"yaw_deg": 8.0, "pitch_deg": 8.0, "zoom_ratio": 1.25}

    # Defaults of the Advanced panel (steps, seed, guidance, ...) from configs/inference_v2.json.
    inference_cfg = {"seed": 42, "steps": 50, "guidance": 5.0, "camera_guidance": 1.0, "camera_min_sigma": None,
                     "camera_residual_anchor": "auto", "prompt": None}
    _cfg_file = Path(bundle_root) / "configs" / "inference_v2.json"
    if _cfg_file.is_file():
        inference_cfg.update({k: v for k, v in json.loads(_cfg_file.read_text()).items() if not k.startswith("_")})
    TRAINING_CAPTION = "A photograph of a scene. One continuous camera shot."

    # Strength calibration: prefer a fitted file (fitted for this exact adapter first), else the identity
    # map (u = s), which the label then flags as "not calibrated".
    _adapter_sha = st.file_sha256(adapter_path) if (adapter_path is not None and not DRY_RUN) else None
    _candidates = sorted((Path(bundle_root) / "configs").glob("strength_calibration_v2*.json"))
    _referenced = (inference_cfg.get("calibration") or "").replace("\\", "/").split("/")[-1]

    def _rank(path):
        try:
            _record = json.loads(path.read_text())
            _is_identity = st.StrengthMap(_record).is_identity
            _sha = _record.get("checkpoint_sha256")
        except (OSError, ValueError):
            return (-1, 0, 0, path.name)
        return (0 if _is_identity else 1, int(bool(_sha and _sha == _adapter_sha)), int(path.name == _referenced), path.name)

    calibration_path = max(_candidates, key=_rank) if _candidates else None
    strength_map = st.load_calibration(calibration_path, root=bundle_root)
    if calibration_path is None:
        _cal_note = "none found: identity map, **not calibrated** (the label is the requested amount)"
    elif strength_map.is_identity:
        _cal_note = f"`{calibration_path.name}` is the identity map: **not calibrated** (the label is the requested amount)"
    else:
        _fitted_for = strength_map.record.get("checkpoint_sha256")
        _cal_note = (f"`{calibration_path.name}` (fitted"
                     + (" for this adapter" if _fitted_for and _fitted_for == _adapter_sha else
                        ", checkpoint digest not matching or not recorded" if _fitted_for else "") + ")")

    def describe_strength(motion, s):
        """Label for UI strength `s` of `motion`: calibrated request u and its physical amount."""
        u = strength_map.s_to_u(motion, s)
        if s <= 0:
            return {"u": 0.0, "amplitude": None, "text": f"{s:.2f} → static camera (no motion)"}
        amp = st.physical_amplitude(motion, u, max_amplitudes)
        name = st.MOTION_LABELS[motion].lower()
        amount = f"{amp['value']:.1f}° {name}" if amp["unit"] == "deg" else f"×{amp['value']:.2f} {name}"
        return {"u": u, "amplitude": amp, "text": f"{s:.2f} → {amount}"}

    mo.callout(mo.md(f"**Defaults and calibration**: {inference_cfg['steps']} steps, guidance {inference_cfg['guidance']}, "
                     f"seed {inference_cfg['seed']} · strength calibration {_cal_note}."), kind="success")
    return (
        CONTRACT, TRAINING_CAPTION, bm, calibration_path, camera_path_v2, describe_strength, detect_direction,
        inference_cfg, max_amplitudes, st, strength_map,
    )


@app.cell
def _(ATTN_BACKENDS, DRY_RUN, LOG_DIR, adapter_path, bundle_root, fail, inference_cfg, mo, sys, time):
    # ---- Load the pipeline ONCE (in-process) --------------------------------------------------------
    # The frozen Wan pipeline (T5, transformer, VAE) goes to the GPU here and stays there: every video
    # reuses it through generate(pipe=...), so there is no reload per generation and progress is live.
    # The loaded pipeline is also parked in a tiny module in sys.modules, so re-running this cell (or an
    # upstream cell) after an edit does not load a second copy into GPU memory.
    _ = (adapter_path, bundle_root, ATTN_BACKENDS, inference_cfg)  # ordering: only after setup 5/6 and 6/6 succeeded
    pipe = None
    generate_fn = None
    has_progress_hook = False
    if DRY_RUN:
        _view = mo.callout(mo.md("**Pipeline**: dry run, no model is loaded. Generate makes a synthetic pan/tilt/zoom "
                                 "of your photo so that the whole interface can be tested."), kind="neutral")
    else:
        import inspect as _inspect
        import types as _types

        import torch as _torch

        if not _torch.cuda.is_available():
            fail("PyTorch cannot see a CUDA GPU",
                 f"torch {_torch.__version__} reports `cuda.is_available() == False` although a GPU was found. "
                 "Check the Molab machine type and the installed torch wheel, then restart the kernel.")
        from fpcamera import flow_infer as _flow_infer

        generate_fn = _flow_infer.generate
        has_progress_hook = "progress_callback" in _inspect.signature(generate_fn).parameters
        _holder = sys.modules.setdefault("fpcamera_studio_cache", _types.ModuleType("fpcamera_studio_cache"))
        pipe = getattr(_holder, "pipe", None)
        _note = "already loaded in this session, reused"
        if pipe is None:
            _started = time.time()
            with mo.status.spinner(title="Loading Wan2.2-TI2V-5B onto the GPU",
                                   subtitle="text encoder, transformer and VAE; about 1-3 minutes the first time"):
                try:
                    pipe = _flow_infer.load_inference_pipeline(flow_shift=5.0, log_path=LOG_DIR / "library.log")
                except Exception as _error:  # noqa: BLE001
                    fail("Loading the Wan pipeline failed", f"`{type(_error).__name__}: {_error}`")
            _holder.pipe = pipe
            _note = f"loaded in {time.time() - _started:.0f} s"
        _view = mo.callout(
            mo.md(f"**Pipeline**: Wan2.2-TI2V-5B on `{_torch.cuda.get_device_name(0)}` ({_note}; "
                  f"{_torch.cuda.memory_allocated() / 2**30:.1f} GiB in use)."
                  + ("" if has_progress_hook else "\n\nThe code bundle's `generate()` has no `progress_callback`: "
                     "step-by-step progress is unavailable (the video itself is not affected).")),
            kind="success" if has_progress_hook else "warn",
        )
    _view
    return generate_fn, has_progress_hook, pipe


@app.cell
def _(deps_ready):
    # ---- Imports of the numeric stack (after the install cell) ----------------------------------------
    _ = deps_ready
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    return Image, ImageOps, cv2, np


# =====================================================================================================
# GENERATION ENGINE (real model or dry run)
# =====================================================================================================


@app.cell
def _(CONTRACT, DRY_RUN, Image, Path, adapter_path, bundle_root, calibration_path, camera_path_v2, cv2, describe_strength,
      generate_fn, has_progress_hook, inference_cfg, max_amplitudes, np, pipe, shutil, st, strength_map, subprocess, time):
    # ---- run_generation(): one function, two back ends ---------------------------------------------
    # Real: fpcamera.flow_infer.generate() with the cached pipeline, replicate guide fill passed explicitly
    #       (a fallback to zero fill once produced gray borders), backbone hash skipped (verified when the
    #       adapter was trained; hashing 5B weights per video would cost minutes).
    # Dry run: warps the input with the same camera homographies the real path would use and writes an mp4,
    #       so the UI flow (progress, video, download, summary) can be tested without a GPU.
    def write_mp4(frames, path, fps):
        """H.264 mp4 (browser playable): imageio-ffmpeg, else an ffmpeg binary, else OpenCV (mp4v)."""
        path = Path(path)
        try:
            import imageio.v2 as imageio

            imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8, ffmpeg_params=["-movflags", "+faststart"])
            return
        except Exception:  # noqa: BLE001 - fall through to the next writer
            pass
        exe = shutil.which("ffmpeg")
        if exe:
            height, width = frames[0].shape[:2]
            proc = subprocess.Popen(
                [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                 "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                stdin=subprocess.PIPE)
            for frame in frames:
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            proc.stdin.close()
            if proc.wait() == 0:
                return
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    def dry_generate(*, image_path, motion, strength, seed, steps, guidance, prompt, run_dir, progress_callback, **_unused):
        """Placeholder for generate(): same camera path, no model. Returns a run.json-like record."""
        started = time.perf_counter()
        run_dir.mkdir(parents=True, exist_ok=False)
        family, direction = st.split_motion(motion)
        control = describe_strength(motion, strength)
        homographies, _ = camera_path_v2(family, direction, control["u"], CONTRACT, max_yaw_deg=max_amplitudes["yaw_deg"],
                                         max_pitch_deg=max_amplitudes["pitch_deg"], max_zoom_ratio=max_amplitudes["zoom_ratio"])
        scale = 0.5  # half resolution keeps the placeholder fast
        size = (int(CONTRACT.width * scale), int(CONTRACT.height * scale))
        source = np.asarray(Image.open(image_path).convert("RGB").resize(size, Image.Resampling.LANCZOS))
        to_small = np.diag([scale, scale, 1.0])
        frames = []
        for homography in homographies:
            warp = to_small @ homography @ np.linalg.inv(to_small)
            frame = cv2.warpPerspective(source, warp, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            cv2.putText(frame, "DRY RUN - synthetic camera move, no model", (12, size[1] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2, cv2.LINE_AA)
            frames.append(frame)
        for step in range(1, int(steps) + 1):  # exercise the progress hook like generate() does
            time.sleep(0.04)
            progress_callback(step, int(steps))
        write_mp4(frames, run_dir / "video.mp4", CONTRACT.fps)
        return {
            "seconds": time.perf_counter() - started, "peak_gib": 0.0, "seed": seed, "steps": int(steps), "guidance": guidance,
            "guide_fill_mode": "replicate", "guide_fill_mode_source": "explicit", "completed_updates": None,
            "notes": ["dry run: placeholder video, no model was used"], "dry_run": True,
            "strength_control": {"label": control["text"], "strength_s": strength, "strength_u": control["u"],
                                 "calibration": {"identity": strength_map.is_identity}},
        }

    def run_generation(*, image_path, motion, strength, seed, steps, guidance, prompt, attn_backend, run_dir,
                       progress_callback):
        """Generate one video into the fresh folder `run_dir`; returns the run.json record (dict)."""
        if DRY_RUN:
            return dry_generate(image_path=image_path, motion=motion, strength=strength, seed=seed, steps=steps,
                                guidance=guidance, prompt=prompt, run_dir=run_dir, progress_callback=progress_callback)
        extra = {"progress_callback": progress_callback} if has_progress_hook else {}
        record = generate_fn(
            bundle_root, adapter_path, image_path, run_dir,
            motion=motion, strength=float(strength),
            calibration=str(calibration_path) if calibration_path else None,
            max_amplitudes=dict(max_amplitudes),
            seed=int(seed), steps=int(steps), guidance=float(guidance), prompt=prompt or inference_cfg.get("prompt"),
            camera_guidance=float(inference_cfg.get("camera_guidance", 1.0)),
            camera_min_sigma=inference_cfg.get("camera_min_sigma"),
            camera_residual_anchor=inference_cfg.get("camera_residual_anchor", "auto"),
            guide_fill_mode="replicate",  # always explicit; never rely on the checkpoint lookup
            pipe=pipe, verify_backbone=False, attn_backend=attn_backend, **extra,
        )
        if record.get("guide_fill_mode") != "replicate":  # belt and braces: gray borders otherwise
            raise RuntimeError(f"guide fill resolved to {record.get('guide_fill_mode')!r}, expected 'replicate'")
        return record

    return (run_generation,)


# =====================================================================================================
# USER INTERFACE
# =====================================================================================================


@app.cell
def _(Image, INPUTS_DIR, REPO_ROOT, bundle_root, np):
    # ---- Built-in sample image --------------------------------------------------------------------
    # The repository ships no photographs, so the built-in sample is generated here (a deterministic
    # landscape). Photos placed in `examples/inputs/` of the adapter bundle (or of the repo checkout in a
    # dry run) are offered as well.
    from PIL import ImageDraw, ImageFilter

    def make_sample_scene(width=1280, height=704, seed=7):
        """Deterministic procedural landscape: sky, sun, three ridges, meadow, pines, a cabin, film grain."""
        rng = np.random.default_rng(seed)
        ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
        horizon = 0.62
        top, low = np.array([54, 98, 176], np.float32), np.array([255, 208, 150], np.float32)
        img = (top + (low - top) * np.clip(ys / horizon, 0, 1) ** 1.5) * np.ones((1, width, 1), np.float32)
        glow = np.exp(-(((xs - 0.72) * width / height) ** 2 + (ys - 0.36) ** 2) / 0.006)
        img = img + glow * np.array([200, 170, 90], np.float32)
        img[(((xs - 0.72) * width / height) ** 2 + (ys - 0.36) ** 2)[..., 0] < 0.0016] = (255, 250, 230)
        x = xs[0, :, 0]
        for base, amp, color in ((0.50, 0.075, (150, 160, 190)), (0.56, 0.06, (96, 118, 150)), (0.60, 0.05, (58, 84, 96))):
            phase = rng.uniform(0, 6.28, 3)
            ridge = base + amp * (0.55 * np.sin(5 * x + phase[0]) + 0.3 * np.sin(11 * x + phase[1]) + 0.15 * np.sin(23 * x + phase[2]))
            img = np.where(ys > ridge[None, :, None], np.array(color, np.float32), img)
        meadow = np.array([70, 128, 62], np.float32) + np.array([30, 40, 10], np.float32) * np.clip((ys - horizon) / (1 - horizon), 0, 1)
        img = np.where(ys > horizon, meadow * (0.9 + 0.2 * rng.random((height, width, 1), dtype=np.float32)), img)
        pic = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(pic)
        for tx, ty, size in ((0.12, 0.80, 150), (0.22, 0.74, 110), (0.86, 0.78, 140), (0.93, 0.72, 100), (0.05, 0.70, 80)):
            cx, base_y = int(tx * width), int(ty * height)
            for tier in range(4):
                half = int(size * (0.5 - 0.1 * tier) * 0.8)
                draw.polygon([(cx - half, base_y - tier * size // 4), (cx + half, base_y - tier * size // 4),
                              (cx, base_y - (tier + 1) * size // 3)], fill=(24 + 6 * tier, 66 + 8 * tier, 40 + 4 * tier))
            draw.rectangle([cx - 5, base_y, cx + 5, base_y + 22], fill=(70, 46, 28))
        hx, hy = int(0.55 * width), int(0.76 * height)
        draw.rectangle([hx - 70, hy - 50, hx + 70, hy + 40], fill=(142, 92, 56))
        draw.polygon([(hx - 84, hy - 50), (hx + 84, hy - 50), (hx, hy - 112)], fill=(110, 48, 40))
        draw.rectangle([hx - 14, hy - 4, hx + 18, hy + 40], fill=(60, 36, 24))
        draw.rectangle([hx + 34, hy - 28, hx + 58, hy - 4], fill=(236, 214, 140))
        pic = pic.filter(ImageFilter.GaussianBlur(0.8))
        grain = rng.normal(0, 4.5, (height, width, 1)).astype(np.float32)
        return Image.fromarray(np.clip(np.asarray(pic, np.float32) + grain, 0, 255).astype(np.uint8))

    _builtin = INPUTS_DIR / "builtin_landscape.png"
    if not _builtin.is_file():
        make_sample_scene().save(_builtin)
    SAMPLES = {"Built-in landscape (generated)": str(_builtin)}
    for _folder in (bundle_root / "examples" / "inputs", REPO_ROOT / "examples" / "inputs" if REPO_ROOT else None):
        if _folder is not None and _folder.is_dir():
            for _p in sorted(_folder.glob("*")):
                if _p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                    SAMPLES.setdefault(f"Example: {_p.stem}", str(_p))
    return (SAMPLES,)


@app.cell
def _(SAMPLES, mo):
    # ---- 1. Photo: upload + sample picker ------------------------------------------------------------
    upload = mo.ui.file(filetypes=[".png", ".jpg", ".jpeg", ".webp"], multiple=False, kind="area",
                        max_size=30_000_000, label="Upload a photo (png, jpg or webp, up to 30 MB)")
    sample_choice = mo.ui.dropdown(options=SAMPLES, value=next(iter(SAMPLES)), label="...or use a sample")
    mo.vstack([
        mo.md("## 1 · Photo"),
        mo.hstack([upload, sample_choice], align="start", gap=1.5, wrap=True),
    ])
    return sample_choice, upload


@app.cell
def _(CONTRACT, INPUTS_DIR, Image, ImageOps, hashlib, mo, sample_choice, upload):
    # ---- The scene the model will see ---------------------------------------------------------------
    # An upload wins over the sample. The photo is EXIF-rotated, centre-cropped to the 16:9 frame of the
    # adapter (1280 x 704, as in training and in generate()) and saved as PNG; the JPEG data URL is only
    # the preview for the drawing canvas below.
    import base64 as _base64
    import io as _io

    try:
        if upload.value:
            _raw, _name = upload.value[0].contents, upload.value[0].name
            _source = f"uploaded `{_name}`"
        else:
            _raw = open(sample_choice.value, "rb").read()
            _source = "sample " + next(k for k, v in sample_choice.options.items() if v == sample_choice.value)
        _original = ImageOps.exif_transpose(Image.open(_io.BytesIO(_raw))).convert("RGB")
    except Exception as _error:  # noqa: BLE001
        mo.stop(True, mo.callout(mo.md(f"**This file could not be read as an image** (`{type(_error).__name__}: {_error}`). "
                                       "Upload a png, jpg or webp photo."), kind="warn"))
    _fitted = ImageOps.fit(_original, (CONTRACT.width, CONTRACT.height), method=Image.Resampling.LANCZOS)
    _png_path = INPUTS_DIR / f"{hashlib.sha256(_fitted.tobytes()).hexdigest()[:16]}.png"
    if not _png_path.is_file():
        _fitted.save(_png_path)
    _buffer = _io.BytesIO()
    _fitted.resize((960, 528), Image.Resampling.LANCZOS).save(_buffer, format="JPEG", quality=88)
    scene = {
        "path": _png_path, "image": _fitted, "source": _source, "original_size": _original.size,
        "preview_url": "data:image/jpeg;base64," + _base64.b64encode(_buffer.getvalue()).decode("ascii"),
    }
    _cropped = abs(_original.size[0] / _original.size[1] - CONTRACT.width / CONTRACT.height) > 0.01
    mo.md(f"Using {_source} · {_original.size[0]} x {_original.size[1]} px"
          + (" · centre-cropped to 16:9 (the preview below shows exactly what the model sees)" if _cropped
             else " · already 16:9"))
    return (scene,)


@app.cell
def _(mo):
    # ---- 2. Motion type -------------------------------------------------------------------------------
    motion_type = mo.ui.radio(
        options={"Pan / Tilt (draw a trajectory)": "pan_tilt", "Zoom in": "zoom_in", "Zoom out": "zoom_out"},
        value="Pan / Tilt (draw a trajectory)", label="Motion type")
    mo.vstack([
        mo.md("## 2 · Motion"),
        motion_type,
        mo.md("**Pan / Tilt:** press and drag on the photo **in the direction the camera should move** "
              "(drag right = pan right, the scene slides left; drag down = tilt down). "
              "The notebook measures the stroke and picks the closest of left / right / up / down. "
              "**Zoom:** nothing to draw."),
    ])
    return (motion_type,)


@app.cell
def _(deps_ready):
    # ---- Drawing widget (anywidget) -----------------------------------------------------------------------
    # A canvas over the photo. Mouse and touch (pointer events); the stroke is sent to Python when the
    # button/finger is released. JS only draws; the direction is decided in Python
    # (fpcamera.trajectory_direction.detect_direction, unit-tested). Python can set `disabled` (zoom),
    # `overlay_text/status` (the detected direction badge) and `veil_text`.
    _ = deps_ready
    try:
        import anywidget
        import traitlets

        class TrajectoryCanvas(anywidget.AnyWidget):
            """anywidget canvas over the photo: collects the user's stroke (mouse or touch) and sends the points to Python on
            release.  The direction itself is decided in Python (``trajectory_direction.detect_direction``); the widget only draws and
            exposes ``disabled`` (zoom), ``overlay_text`` / ``overlay_status`` (detected-direction badge) and ``veil_text``.
            """

            _esm = r"""
            function render({ model, el }) {
              const root = document.createElement("div");
              root.style.cssText = "width:100%;max-width:920px;font-family:inherit;";
              const stage = document.createElement("div");
              stage.style.cssText = "position:relative;width:100%;border-radius:10px;overflow:hidden;" +
                "border:1px solid rgba(127,127,127,.5);background:#1b1b1b;line-height:0;";
              const canvas = document.createElement("canvas");
              canvas.width = 960; canvas.height = 528;
              canvas.style.cssText = "display:block;width:100%;height:auto;touch-action:none;" +
                "user-select:none;-webkit-user-select:none;cursor:crosshair;";
              canvas.setAttribute("role", "img");
              canvas.setAttribute("aria-label", "Photo. Press and drag to draw the camera trajectory.");
              const veil = document.createElement("div");
              veil.style.cssText = "position:absolute;inset:0;display:none;align-items:center;justify-content:center;" +
                "text-align:center;padding:16px;background:rgba(10,10,10,.62);color:#fff;" +
                "font:600 15px system-ui,sans-serif;line-height:1.4;pointer-events:none;";
              stage.append(canvas, veil);
              const bar = document.createElement("div");
              bar.style.cssText = "display:flex;gap:12px;align-items:center;margin-top:8px;flex-wrap:wrap;" +
                "font:13px system-ui,sans-serif;";
              const clear = document.createElement("button");
              clear.type = "button"; clear.textContent = "Clear stroke";
              clear.style.cssText = "padding:6px 14px;border-radius:6px;border:1px solid rgba(127,127,127,.6);" +
                "background:transparent;color:inherit;cursor:pointer;font:inherit;";
              const hint = document.createElement("span");
              hint.style.opacity = "0.75";
              hint.textContent = "Press and drag on the photo, release to measure the direction.";
              bar.append(clear, hint);
              root.append(stage, bar);
              el.append(root);

              const ctx = canvas.getContext("2d");
              let img = null, current = null, dragging = false;

              function loadImage() {
                const src = model.get("image_data");
                if (!src) { img = null; draw(); return; }
                const next = new Image();
                next.onload = () => { img = next; canvas.width = next.naturalWidth; canvas.height = next.naturalHeight; draw(); };
                next.src = src;
              }
              function points() { return current || model.get("points") || []; }
              function arrowHead(x, y, angle, size) {
                ctx.beginPath();
                ctx.moveTo(x + Math.cos(angle) * size, y + Math.sin(angle) * size);
                ctx.lineTo(x + Math.cos(angle + 2.5) * size, y + Math.sin(angle + 2.5) * size);
                ctx.lineTo(x + Math.cos(angle - 2.5) * size, y + Math.sin(angle - 2.5) * size);
                ctx.closePath(); ctx.fill();
              }
              function draw() {
                const w = canvas.width, h = canvas.height, k = w / 960;
                const disabled = !!model.get("disabled");
                ctx.clearRect(0, 0, w, h);
                if (img) ctx.drawImage(img, 0, 0, w, h); else { ctx.fillStyle = "#222"; ctx.fillRect(0, 0, w, h); }
                const pts = points();
                if (!disabled && pts.length > 1) {
                  ctx.lineCap = "round"; ctx.lineJoin = "round";
                  const trace = (width, color) => {
                    ctx.lineWidth = width * k; ctx.strokeStyle = color; ctx.beginPath();
                    pts.forEach((p, i) => { const x = p[0] * w, y = p[1] * h; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
                    ctx.stroke();
                  };
                  trace(10, "rgba(255,255,255,.95)"); trace(5, "#ff7a00");
                  const a = pts[0], b = pts[pts.length - 1];
                  const angle = Math.atan2((b[1] - a[1]) * h, (b[0] - a[0]) * w);
                  ctx.fillStyle = "#ffffff"; arrowHead(b[0] * w, b[1] * h, angle, 22 * k);
                  ctx.fillStyle = "#ff7a00"; arrowHead(b[0] * w, b[1] * h, angle, 16 * k);
                  ctx.beginPath(); ctx.arc(a[0] * w, a[1] * h, 9 * k, 0, 6.2832);
                  ctx.fillStyle = "#1fbf6a"; ctx.fill(); ctx.lineWidth = 3 * k; ctx.strokeStyle = "#fff"; ctx.stroke();
                  const text = model.get("overlay_text");
                  if (text && !dragging) {
                    const colors = { ok: "#1f9d55", ambiguous: "#d97706", too_short: "#c0392b" };
                    ctx.font = `600 ${22 * k}px system-ui, sans-serif`;
                    const tw = ctx.measureText(text).width, pad = 12 * k;
                    ctx.fillStyle = colors[model.get("overlay_status")] || "#444";
                    ctx.beginPath(); ctx.roundRect(14 * k, 14 * k, tw + 2 * pad, 38 * k, 8 * k); ctx.fill();
                    ctx.fillStyle = "#fff"; ctx.textBaseline = "middle"; ctx.fillText(text, 14 * k + pad, 33 * k);
                  }
                }
                const veilText = model.get("veil_text");
                veil.textContent = veilText || "";
                veil.style.display = disabled ? "flex" : "none";
                canvas.style.cursor = disabled ? "not-allowed" : "crosshair";
                clear.disabled = disabled; clear.style.opacity = disabled ? "0.45" : "1";
              }
              function pointOf(ev) {
                const r = canvas.getBoundingClientRect();
                return [Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)),
                        Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height))];
              }
              function commit() {
                if (!dragging) return;
                dragging = false;
                const pts = current; current = null;
                model.set("points", pts || []); model.save_changes(); draw();
              }
              canvas.addEventListener("pointerdown", (ev) => {
                if (model.get("disabled") || (ev.pointerType === "mouse" && ev.button !== 0)) return;
                ev.preventDefault();
                try { canvas.setPointerCapture(ev.pointerId); } catch (error) { /* synthetic pointers cannot be captured */ }
                dragging = true; current = [pointOf(ev)]; draw();
              });
              canvas.addEventListener("pointermove", (ev) => {
                if (!dragging) return;
                ev.preventDefault();
                const p = pointOf(ev), last = current[current.length - 1];
                if (Math.hypot(p[0] - last[0], p[1] - last[1]) > 0.004) { current.push(p); draw(); }
              });
              canvas.addEventListener("pointerup", (ev) => { if (dragging) { current.push(pointOf(ev)); } commit(); });
              canvas.addEventListener("pointercancel", () => { dragging = false; current = null; draw(); });
              clear.addEventListener("click", () => { model.set("points", []); model.save_changes(); draw(); });
              for (const name of ["points", "disabled", "overlay_text", "overlay_status", "veil_text"])
                model.on("change:" + name, draw);
              model.on("change:image_data", loadImage);
              loadImage(); draw();
            }
            export default { render };
            """
            points = traitlets.List(traitlets.List(traitlets.Float()), default_value=[]).tag(sync=True)
            image_data = traitlets.Unicode("").tag(sync=True)
            disabled = traitlets.Bool(False).tag(sync=True)
            overlay_text = traitlets.Unicode("").tag(sync=True)
            overlay_status = traitlets.Unicode("").tag(sync=True)
            veil_text = traitlets.Unicode("").tag(sync=True)

    except ImportError:
        TrajectoryCanvas = None  # fallback: a plain direction picker is shown instead
    return (TrajectoryCanvas,)


@app.cell
def _(TrajectoryCanvas, mo, scene):
    # ---- The canvas over the photo (defined and shown here, so stroke changes never re-render it) --------
    # A new photo creates a new canvas (the old stroke is dropped). Without anywidget a radio group replaces
    # drawing.
    canvas = None
    fallback_direction = None
    if TrajectoryCanvas is None:
        fallback_direction = mo.ui.radio(
            options={"Pan left": "pan_left", "Pan right": "pan_right", "Tilt up": "tilt_up", "Tilt down": "tilt_down"},
            value="Pan right", inline=True, label="Direction")
        _view = mo.vstack([
            mo.callout(mo.md("The drawing widget (`anywidget`) is not available in this runtime. Pick the direction "
                             "instead."), kind="warn"),
            mo.image(scene["image"].resize((960, 528)), rounded=True),
            fallback_direction,
        ])
    else:
        canvas = mo.ui.anywidget(TrajectoryCanvas(image_data=scene["preview_url"]))
        _view = canvas
    _view
    return canvas, fallback_direction


@app.cell
def _(CONTRACT, canvas, detect_direction, fallback_direction, mo, motion_type):
    # ---- Direction detection and the camera choice -------------------------------------------------------
    # camera_choice["motion"] is what goes to generate(): one of pan_left / pan_right / tilt_up / tilt_down /
    # zoom_in / zoom_out, or None while there is no usable stroke. Zoom disables the canvas (veil on).
    _aspect = CONTRACT.width / CONTRACT.height
    camera_choice = {"motion": None, "source": None, "stroke": None}
    if motion_type.value in ("zoom_in", "zoom_out"):
        camera_choice = {"motion": motion_type.value, "source": "zoom", "stroke": None}
        if canvas is not None:
            canvas.overlay_text = ""
            canvas.veil_text = "Zoom is selected: there is nothing to draw. Choose Pan / Tilt to draw a trajectory."
            canvas.disabled = True
        _label = "Zoom in" if motion_type.value == "zoom_in" else "Zoom out"
        _readout = mo.md(f"**{_label}**: no trajectory needed (drawing is turned off).")
    elif canvas is None:
        camera_choice = {"motion": fallback_direction.value, "source": "picker", "stroke": None}
        _readout = mo.md(f"**{fallback_direction.selected_key}** (picked by hand).")
    else:
        canvas.disabled = False
        canvas.veil_text = ""
        _stroke = detect_direction(canvas.value.get("points"), aspect=_aspect)
        canvas.overlay_text = f"{_stroke.arrow} {_stroke.label} · {_stroke.confidence:.0%}" if _stroke.usable else ""
        canvas.overlay_status = _stroke.status
        camera_choice = {"motion": _stroke.motion, "source": "stroke", "stroke": _stroke}
        if _stroke.status == "empty" or _stroke.status == "too_short":
            _readout = mo.callout(mo.md(_stroke.message), kind="warn" if _stroke.status == "too_short" else "neutral")
        else:
            _vx, _vy = _stroke.screen_vector
            _rotation = {(1, 0): 0, (0, 1): 90, (-1, 0): 180, (0, -1): 270}[(_vx, _vy)]
            _color = "#1f9d55" if _stroke.status == "ok" else "#d97706"
            _arrow = (
                '<svg viewBox="-50 -50 100 100" width="76" height="76" aria-hidden="true">'
                '<circle r="46" fill="rgba(127,127,127,.14)" stroke="rgba(127,127,127,.5)"/>'
                f'<g transform="rotate({_rotation})" stroke="{_color}" stroke-width="9" stroke-linecap="round" '
                'stroke-linejoin="round" fill="none"><path d="M-26 0 H24 M8 -17 L26 0 L8 17"/></g></svg>'
            )
            _percent = round(_stroke.confidence * 100)
            _card = (
                '<div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap">'
                f"{_arrow}<div><div style=\"font-size:1.25rem;font-weight:650\">Detected: {_stroke.arrow} {_stroke.label}</div>"
                f'<div style="margin:6px 0 4px;font-size:.9rem">Confidence {_percent}% '
                f'<span style="opacity:.7">(share of the stroke along {"the horizontal" if _stroke.motion.startswith("pan") else "the vertical"} axis; '
                "50% = exact diagonal)</span></div>"
                f'<div style="width:240px;height:8px;border-radius:4px;background:rgba(127,127,127,.25)">'
                f'<div style="width:{(_percent - 50) * 2}%;height:100%;border-radius:4px;background:{_color}"></div></div>'
                "</div></div>"
            )
            _items = [mo.Html(_card)]
            for _w in _stroke.warnings:
                _items.append(mo.callout(mo.md(_w), kind="warn"))
            _readout = mo.vstack(_items)
    _readout
    return (camera_choice,)


@app.cell
def _(mo):
    # ---- 3. Strength slider --------------------------------------------------------------------------------
    strength_slider = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.5, show_value=True, full_width=True,
                                   label="Strength (0 = static camera, 1 = trained maximum)")
    mo.vstack([mo.md("## 3 · Strength"), strength_slider])
    return (strength_slider,)


@app.cell
def _(camera_choice, describe_strength, max_amplitudes, mo, strength_map, strength_slider):
    # ---- Live label: strength -> physical amount, through the calibration ------------------------------------
    _s = float(strength_slider.value)
    _motion = camera_choice["motion"]
    _calib = ("identity map, not calibrated" if strength_map.is_identity else "calibrated")
    if _motion is None:
        _body = (f"**{_s:.2f} → ?**  Draw a stroke to choose the direction; "
                 f"at strength 1 a pan or tilt reaches {max_amplitudes['yaw_deg']:.0f}°.")
        _detail = ""
    else:
        _info = describe_strength(_motion, _s)
        _body = f"**{_info['text']}**"
        _detail = (f"requests u = {_info['u']:.2f} of the trained maximum ({_calib})" if _info["amplitude"] else _calib)
    mo.callout(mo.md(f"### {_body}\n\n{_detail}"), kind="info")
    return


@app.cell
def _(CONTRACT, Image, bm, camera_choice, camera_path_v2, cv2, describe_strength, max_amplitudes, mo, np, scene, st,
      strength_slider):
    # ---- Preview: where the camera ends up, and what the model must invent -----------------------------------
    # Left: the first frame. Right: the photo warped by the homography of the LAST frame; the red region is
    # newly revealed (not in the photograph, so the model has to generate it, the adapter fills the guide
    # there by replicating the border). Cyan outlines: where the first view sits at 1/3, 2/3 and the end.
    # Geometry only, on the CPU; no model involved.
    _motion = camera_choice["motion"]
    if _motion is None or float(strength_slider.value) <= 0:
        _preview = mo.md("*The preview appears once a direction is chosen and the strength is above 0.*")
    else:
        _family, _direction = st.split_motion(_motion)
        _u = describe_strength(_motion, float(strength_slider.value))["u"]
        _h, _ = camera_path_v2(_family, _direction, _u, CONTRACT, max_yaw_deg=max_amplitudes["yaw_deg"],
                               max_pitch_deg=max_amplitudes["pitch_deg"], max_zoom_ratio=max_amplitudes["zoom_ratio"])
        _h = np.asarray(_h)
        _last = len(_h) - 1
        _first = np.asarray(scene["image"])
        _warped = cv2.warpPerspective(_first, _h[_last], (CONTRACT.width, CONTRACT.height), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
        _revealed = bm.revealed_mask_pixels(_h, CONTRACT.width, CONTRACT.height, frame_indices=[_last])[0]
        _view = _warped.astype(np.float32)
        _view[_revealed] = 0.4 * _view[_revealed] + 0.6 * np.array([235, 60, 60], np.float32)
        _view = _view.astype(np.uint8)
        _corners = np.array([[0, 0, 1], [CONTRACT.width - 1, 0, 1], [CONTRACT.width - 1, CONTRACT.height - 1, 1],
                             [0, CONTRACT.height - 1, 1]], dtype=np.float64)
        for _frame, _thick in ((_last // 3, 2), (2 * _last // 3, 2), (_last, 5)):
            _q = _corners @ _h[_frame].T
            _pts = (_q[:, :2] / _q[:, 2:3]).round().astype(np.int32)
            cv2.polylines(_view, [_pts], True, (40, 220, 255), _thick, cv2.LINE_AA)
        _size = (560, round(560 * CONTRACT.height / CONTRACT.width))
        _preview = mo.vstack([
            mo.hstack([
                mo.vstack([mo.md("**First frame**"), mo.image(scene["image"].resize(_size, Image.Resampling.LANCZOS), rounded=True)]),
                mo.vstack([mo.md("**Last frame (geometry only)**"),
                           mo.image(Image.fromarray(_view).resize(_size, Image.Resampling.LANCZOS), rounded=True)]),
            ], justify="start", gap=1, wrap=True),
            mo.md(f"Red = newly revealed, **{_revealed.mean():.0%}** of the last frame: the model has to invent it. "
                  "Cyan outlines = where the first view sits at 1/3, 2/3 and the end of the clip."),
        ])
    mo.vstack([mo.md("**Preview of the move**"), _preview])
    return


@app.cell
def _(ATTN_BACKENDS, TRAINING_CAPTION, inference_cfg, mo):
    # ---- 4. Advanced (collapsed) -----------------------------------------------------------------------------
    seed_input = mo.ui.number(start=0, stop=2**32 - 1, step=1, value=int(inference_cfg["seed"]), label="Seed")
    steps_input = mo.ui.number(start=1, stop=100, step=1, value=int(inference_cfg["steps"]),
                               label="Denoising steps (more = slower)")
    guidance_input = mo.ui.number(start=0.0, stop=20.0, step=0.5, value=float(inference_cfg["guidance"]),
                                  label="Text guidance")
    prompt_input = mo.ui.text_area(value="", rows=2, full_width=True,
                                   placeholder=f"empty = the training caption: {TRAINING_CAPTION}",
                                   label="Prompt override (optional)")
    attn_input = mo.ui.dropdown(options=list(ATTN_BACKENDS), value="auto",
                                label="Attention kernel (auto = verified default; others are speed options)")
    mo.vstack([
        mo.md("## 4 · Settings"),
        mo.accordion({"Advanced": mo.vstack([
            mo.hstack([seed_input, steps_input, guidance_input], justify="start", gap=1.5, wrap=True),
            prompt_input,
            attn_input,
            mo.md("The same seed, photo and settings reproduce the same video. The prompt only needs to be changed "
                  "to describe the scene differently; the camera move comes from the adapter, not from the text."),
        ])}),
    ])
    return attn_input, guidance_input, prompt_input, seed_input, steps_input


@app.cell
def _(mo):
    # ---- 5. Generate button and the result store ---------------------------------------------------------------
    # The finished run goes into `mo.state` so that it stays on screen when a slider is moved afterwards.
    get_result, set_result = mo.state(None)
    generate_button = mo.ui.run_button(label="Generate video", kind="success")
    mo.vstack([mo.md("## 5 · Generate"), generate_button])
    return generate_button, get_result, set_result


@app.cell
def _(RUNS_DIR, attn_input, camera_choice, generate_button, guidance_input, has_progress_hook, mo, prompt_input,
      run_generation, scene, seed_input, set_result, steps_input, strength_slider, sys, time):
    # ---- The generation itself ---------------------------------------------------------------------------------
    # Runs in this kernel (no subprocess): the cached pipeline does the work and the per-step callback of
    # generate() drives the progress bar. mo.stop() makes the cell do nothing unless the button was pressed.
    import traceback as _traceback
    import uuid as _uuid

    mo.stop(not generate_button.value)
    mo.stop(camera_choice["motion"] is None, mo.callout(mo.md("**Draw a stroke on the photo first** (or choose Zoom)."), kind="warn"))
    _steps = int(steps_input.value)
    _run_dir = RUNS_DIR / (time.strftime("%Y%m%d-%H%M%S") + "-" + _uuid.uuid4().hex[:6])  # fresh folder every time
    _started = time.perf_counter()
    _state = {"first_step_at": None, "done": 0}
    try:
        with mo.status.progress_bar(
            total=_steps, title="Generating video",
            subtitle="preparing: encoding the photo and the camera guide (a few seconds)",
            show_rate=False, show_eta=False, remove_on_exit=False,
        ) as _bar:

            def _on_step(step, total):
                """Called by generate() after every denoising step (step = 1..total)."""
                now = time.perf_counter()
                if _state["first_step_at"] is None:
                    _state["first_step_at"] = now
                elapsed = now - _started
                if step >= total:
                    time.sleep(0.25)  # the progress bar drops updates closer than 150 ms; make sure this last one shows
                    sub = f"{total}/{total} steps in {elapsed:.0f} s · decoding the video (VAE) and writing the mp4, about 30-60 s..."
                else:
                    per_step = (now - _state["first_step_at"]) / (step - 1) if step > 1 else None
                    left = per_step * (total - step) if per_step else None
                    eta = "" if left is None else f" · about {left:.0f} s left" if left < 90 else f" · about {left / 60:.1f} min left"
                    sub = f"step {step}/{total} · {elapsed:.0f} s elapsed{eta}"
                _bar.update(increment=step - _state["done"], subtitle=sub)
                _state["done"] = step

            _record = run_generation(
                image_path=scene["path"], motion=camera_choice["motion"], strength=float(strength_slider.value),
                seed=int(seed_input.value), steps=_steps, guidance=float(guidance_input.value),
                prompt=prompt_input.value.strip() or None, attn_backend=attn_input.value, run_dir=_run_dir,
                progress_callback=_on_step,
            )
    except Exception as _error:  # noqa: BLE001 - shown to the user, the kernel keeps its loaded pipeline
        mo.stop(True, mo.callout(mo.vstack([
            mo.md(f"**Generation failed**: `{type(_error).__name__}: {_error}`"),
            mo.accordion({"Traceback": mo.md(f"```\n{_traceback.format_exc()}\n```")}),
        ]), kind="danger"))
    _torch = sys.modules.get("torch")
    if _torch is not None and _torch.cuda.is_available():
        _torch.cuda.empty_cache()  # give the decode buffers back between videos
    set_result({
        "video_path": str(_run_dir / "video.mp4"), "run_dir": str(_run_dir), "record": _record,
        "motion": camera_choice["motion"], "strength": float(strength_slider.value),
        "stroke": camera_choice.get("stroke"), "wall_seconds": time.perf_counter() - _started,
        "has_progress_hook": has_progress_hook,
    })
    mo.callout(mo.md(f"Done in {time.perf_counter() - _started:.0f} s. The video is below."), kind="success")
    return


@app.cell
def _(DRY_RUN, Path, get_result, mo):
    # ---- Result: video, download, run summary ---------------------------------------------------------------------
    _result = get_result()
    if _result is None:
        _out = mo.md("## Result\n\n*Your video appears here after you press Generate.*")
    else:
        _video = Path(_result["video_path"])
        _record = _result["record"]
        _control = _record.get("strength_control") or {}
        _stroke = _result.get("stroke")
        _calibration = _control.get("calibration") or {}
        _rows = [
            ("Motion and amount", _control.get("label") or _result["motion"]),
            ("Strength s → requested u", f"{_control.get('strength_s')} → {_control.get('strength_u')}"
             + (" (identity calibration)" if _calibration.get("identity") else " (calibrated)")),
            ("Direction source", (f"stroke, confidence {_stroke.confidence:.0%}" if _stroke is not None and _stroke.usable
                                  else "zoom preset" if _result["motion"].startswith("zoom") else "picker")),
            ("Seed / steps / guidance", f"{_record.get('seed')} / {_record.get('steps')} / {_record.get('guidance')}"),
            ("Guide fill", f"{_record.get('guide_fill_mode')} ({_record.get('guide_fill_mode_source')})"),
            ("Generation time", f"{_record.get('seconds', 0):.0f} s (total with preparation {_result['wall_seconds']:.0f} s)"),
            ("Peak GPU memory", f"{_record.get('peak_gib', 0):.1f} GiB" if not _record.get("dry_run") else "n/a (dry run)"),
            ("Adapter updates", _record.get("completed_updates") or "n/a"),
            ("Notes", "; ".join(_record.get("notes") or []) or "none"),
        ]
        _table = "| Run summary | |\n|:---|:---|\n" +"\n".join(f"| {a} | {b} |" for a, b in _rows)
        _out = mo.vstack([
            mo.md("## Result"),
            mo.callout(mo.md("**Dry run**: this is a synthetic placeholder, not a model output."), kind="warn")
            if _record.get("dry_run") else mo.md(""),
            mo.video(src=_video, controls=True, loop=True, rounded=True),
            mo.download(data=lambda: _video.read_bytes(), filename=f"camera_{_result['motion']}_{Path(_result['run_dir']).name}.mp4",
                        mimetype="video/mp4", label="Download MP4"),
            mo.md(_table),
            mo.md(f"Files of this run: `{_result['run_dir']}`"),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    # ---- How it works (collapsed) ---------------------------------------------------------------------------------
    mo.accordion({
        "How it works": mo.md(
            r"""
            **The model.** Wan2.2-TI2V-5B turns a photograph into a video. It is kept completely frozen. A small trained
            adapter (a LoRA plus a guide projection, about 14 million parameters) is added to it.

            **Camera as geometry.** A camera move is a sequence of homographies: pan and tilt are small rotations
            (up to 8 degrees at the end of the clip), zoom is a change of focal length (up to x1.25), linearly eased over
            121 frames. The first frame is warped by these homographies to make a *guide video*. Where the warp reveals
            something that is not in the photograph, the guide is filled by **replicating** the border pixels (zero fill
            would show up as gray borders). The adapter reads the guide and the revealed-region mask, and the frozen model
            invents the new content.

            **One strength.** The slider value *s* (0 to 1) goes through a measured per-direction calibration to the strength
            *u* that is requested from the geometry and the adapter, so that the label ("0.50 → 4.0° pan right") matches the
            motion seen in the video as closely as the calibration allows. With the identity calibration the label is the
            requested amount, not a measured one.

            **Drawing a direction.** The adapter has one motion at a time, so the stroke is reduced to the closest of
            pan left / right and tilt up / down by its net displacement (compared on the 16:9 screen, not in normalised
            units). The confidence is the share of that displacement along the chosen axis: 100% for a straight horizontal or
            vertical drag, 50% for an exact diagonal (flagged as ambiguous). Dragging right means the camera pans right, so
            the scene slides left, as in the earlier trajectory interface.

            **Limits.** Motion is a 2D image motion; strong parallax, moving objects and textureless scenes are not
            guaranteed to follow the requested amount. The revealed region is invented and can contain artifacts. Each
            video should be checked by eye. Code: `src/fpcamera/` (`flow_infer.py`, `strength.py`, `trajectory_direction.py`);
            documentation in `docs/v2/`.
            """
        )
    })
    return


if __name__ == "__main__":
    app.run()
