"""Editable marimo interface for the checkpoint-300 Wan flow-camera adapter.

Preset availability uses a separate six-case empirical review profile. Pan and
tilt were not represented in training mode counts. Every new clip still needs
independent motion and preservation review.

Run `marimo edit notebooks/flow_camera_interface.py` from the
existing pinned Wan runtime. The inference child inherits that runtime; this
file deliberately has no isolated PEP 723 dependency environment.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Submission 1 · Camera motion")


@app.cell
def _():
    from pathlib import Path
    import sys as _app_sys
    import marimo as mo

    project_root = Path(__file__).resolve().parents[1]
    _app_sys.path.insert(0, str(project_root / "src"))
    return Path, mo, project_root


@app.cell
def _(Path, mo, project_root):
    import hashlib
    import json
    import os
    import signal
    import subprocess
    import sys as _sys
    import threading
    import time
    import uuid

    from PIL import Image, ImageOps
    from fpcamera.flow_geometry import VideoContract, camera_path
    from fpcamera.flow_preview import geometry_preview_svg
    from fpcamera.flow_view_path import camera_view_path

    PRESETS = {
        "Zoom in": "zoom_in",
        "Zoom out": "zoom_out",
        "Pan left": "pan_left",
        "Pan right": "pan_right",
        "Tilt up": "tilt_up",
        "Tilt down": "tilt_down",
    }
    EXPECTED_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    EXPECTED_REVISION = "b8fff7315c768468a5333511427288870b2e9635"
    EVALUATION_PROFILE_PATH = project_root / "configs" / "provenance" / "camera_control_evaluation_profile.json"
    RUN_TIMEOUT_SECONDS = 30 * 60
    OUTPUT_CONTRACT = {"width": 1280, "height": 704, "frames": 121, "fps": 24}

    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    def checkpoint_catalog():
        catalog = {}
        try:
            profile = json.loads(EVALUATION_PROFILE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            profile = {}
        for folder in (project_root / "artifacts" / "flow_poc_zoom",
                       project_root / "artifacts" / "flow_poc_all"):
            if not folder.is_dir():
                continue
            for candidate in sorted(folder.glob("adapter_*.safetensors")):
                sidecar = candidate.with_suffix(".json")
                record = {"path": str(candidate), "eligible": False,
                          "status": "missing_metadata", "evaluated": False}
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                    identity = meta.get("identity", {})
                    architecture = meta.get("architecture", {})
                    actual_hash = _sha256(candidate)
                    reasons = []
                    if actual_hash != meta.get("checkpoint_sha256"):
                        reasons.append("checkpoint hash mismatch")
                    if architecture.get("kind") != "wan_flow_control_lora":
                        reasons.append("wrong adapter architecture")
                    if identity.get("model") != EXPECTED_MODEL or identity.get("revision") != EXPECTED_REVISION:
                        reasons.append("base model/revision mismatch")
                    if meta.get("backbone_unchanged") is not True:
                        reasons.append("unchanged backbone audit missing or failed")
                    objective_name = str(identity.get("objective", meta.get("objective", ""))).lower()
                    objective = ("teacher_relative" if "teacher_relative" in objective_name else
                                 "uniform" if "uniform" in objective_name else "unknown")
                    record = {"path": str(candidate), "sidecar": str(sidecar),
                              "eligible": not reasons, "status": "eligible_for_inference" if not reasons else "rejected",
                              "reasons": reasons, "checkpoint_sha256": actual_hash,
                              "completed_updates": meta.get("completed_updates"),
                              "evaluated": meta.get("camera_evaluation_status") == "passed",
                              "objective": objective,
                              "mode_counts": meta.get("mode_counts", {}),
                              "training_status": meta.get("status"), "contract": identity.get("video_contract"),
                              "architecture": architecture}
                    if profile.get("checkpoint_sha256") == actual_hash:
                        record["evaluation_profile"] = profile
                except (OSError, ValueError, TypeError) as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                catalog[str(candidate)] = record
        return catalog

    def checkpoint_dropdown_options(checkpoints, project_root):
        """Build stable label-to-path options and choose the best eligible default."""
        placeholder = "No verified flow checkpoint available"
        eligible = [(path, item) for path, item in checkpoints.items() if item.get("eligible")]
        if not eligible:
            return {placeholder: ""}, placeholder

        def update_number(item):
            try:
                return int(item.get("completed_updates"))
            except (TypeError, ValueError):
                return -1

        def priority(pair):
            path, item = pair
            accepted_rank = 1 if item.get("evaluated") else 0
            objective_rank = 1 if item.get("objective") == "teacher_relative" else 0
            return accepted_rank, objective_rank, update_number(item), path

        options = {}
        for path, item in sorted(eligible, key=lambda pair: (*priority(pair)[:3], pair[0]), reverse=True):
            path_obj = Path(path)
            try:
                display_path = path_obj.resolve().relative_to((Path(project_root) / "artifacts").resolve()).as_posix()
            except ValueError:
                display_path = f"{path_obj.parent.name}/{path_obj.name}"
            label = display_path + (" · evaluated" if item.get("evaluated") else " · evaluation pending")
            options[label] = path

        selected_path, _ = max(eligible, key=priority)
        selected_label = next(label for label, path in options.items() if path == selected_path)
        return options, selected_label

    class _FlowJobController:
        """Run one bounded inference subprocess and preserve cancellation state."""
        def __init__(self):
            self.lock = threading.Lock()
            self.guard = threading.Lock()
            # Keep module handles on the controller itself. The monitor runs
            # in a background thread and must not depend on notebook globals
            # that can be replaced when cells re-run.
            self.clock = time
            self.process_api = subprocess
            self.signal_api = signal
            self.signal_worker = _signal_worker
            self.state_setter = set_job
            self.job = {"status": "idle", "message": "Upload an image and select a verified flow checkpoint."}
            self.process = None
            self.cancel_requested = False

        def update(self, **fields):
            with self.guard:
                self.job = {**self.job, **fields, "updated_at": self.clock.time()}
                self.state_setter(dict(self.job))

        def cancel(self):
            with self.guard:
                process = self.process
                if process is None or process.poll() is not None:
                    return False
                self.cancel_requested = True
                self.job = {**self.job, "status": "cancelling",
                            "message": "Cancellation requested. The worker will unwind the GPU lease."}
                self.state_setter(dict(self.job))
            self.signal_worker(process, self.signal_api.SIGINT)
            return True

        def submit(self, *, image_bytes, checkpoint, checkpoint_record, mode,
                   magnitude, prompt, seed, steps, camera_request=None):
            if not self.lock.acquire(blocking=False):
                raise RuntimeError("A flow-camera generation is already running in this UI.")
            run_dir = None
            try:
                gpu_lock = project_root / "artifacts" / "gpu.lock"
                if gpu_lock.exists():
                    raise RuntimeError("The shared GPU lease is occupied. Wait for the active GPU job to finish.")
                checkpoint = Path(checkpoint).expanduser().resolve()
                if not checkpoint.is_file() or not checkpoint_record.get("eligible"):
                    raise ValueError("Select a hash-verified flow checkpoint for this Wan revision.")
                image = ImageOps.exif_transpose(Image.open(__import__("io").BytesIO(image_bytes))).convert("RGB")
                image.load()
                run_dir = project_root / "artifacts" / "flow_ui_runs" / (
                    time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
                run_dir.mkdir(parents=True, exist_ok=False)
                source_path = run_dir / "uploaded_source.png"
                image.save(source_path)
                config = {
                    "project_root": str(project_root), "checkpoint": str(checkpoint),
                    "image": str(source_path), "output": str(run_dir), "mode": mode,
                    "magnitude": float(magnitude), "prompt": prompt.strip() or None,
                    "seed": int(seed), "steps": int(steps), "guidance": 5.0,
                    "flow_shift": 5.0, "frozen_baseline": False,
                    "camera_residual_anchor": "teacher", "camera_guidance": 1.0,
                    "camera_min_sigma": 0.6,
                }
                contract_fields = checkpoint_record.get("contract") or OUTPUT_CONTRACT
                contract = VideoContract(**contract_fields)
                if camera_request is None:
                    _homographies, camera_request = camera_path(mode, contract,
                                                                magnitude=float(magnitude))
                else:
                    expected = contract.record()
                    if camera_request.get("contract") != expected:
                        raise ValueError("Drawn camera path contract differs from the selected checkpoint.")
                request_path = run_dir / "ui_camera_request.json"
                request_path.write_text(json.dumps(camera_request, indent=2) + "\n",
                                        encoding="utf-8")
                config["camera_request_preview"] = str(request_path)
                config_path = run_dir / "worker_config.json"
                config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
                settings = {**config, "checkpoint_record": checkpoint_record,
                            "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
                            "output_contract": OUTPUT_CONTRACT,
                            "camera_request": camera_request,
                            "acceptance_status": "not_evaluated",
                            "interactive_path_drawing": "used" if mode == "drawn_camera_view_path" else "not_used"}
                settings_path = run_dir / "run_settings.json"
                settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

                # The CLI reaches the real flow_infer.generate() API. Its
                # function writes the actual preset request and acquires the
                # cross-process gpu.lock lease. Custom paths remain disabled.
                env = dict(os.environ)
                source_root = str(project_root / "src")
                env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
                env["FPCAMERA_ROOT"] = str(project_root)
                env["USE_TF"] = "0"
                env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
                log_path = run_dir / "worker.log"
                log = log_path.open("w", encoding="utf-8")
                command = [_sys.executable, "-m", "fpcamera.flow_infer",
                           "--project-root", str(project_root), "--checkpoint", str(checkpoint),
                           "--image", str(source_path), "--output", str(run_dir),
                           "--mode", mode, "--magnitude", str(float(magnitude)),
                           "--seed", str(int(seed)), "--steps", str(int(steps)),
                           "--camera-residual-anchor", "teacher",
                           "--camera-guidance", "1.0", "--camera-min-sigma", "0.6"]
                if mode == "drawn_camera_view_path":
                    command.extend(["--camera-request", str(request_path)])
                if prompt.strip():
                    command.extend(["--prompt", prompt.strip()])
                process = subprocess.Popen(
                    command, cwd=str(project_root),
                    env=env, stdout=log, stderr=subprocess.STDOUT, text=True,
                    start_new_session=(os.name != "nt"))
                with self.guard:
                    self.process = process
                    self.cancel_requested = False
                self.update(status="running", message="Starting native Wan image-to-video generation…",
                            run_dir=str(run_dir), settings_path=str(settings_path),
                            log_path=str(log_path), video_path=None, run_record=None,
                            started_at=time.time(), deadline=time.time() + RUN_TIMEOUT_SECONDS)

                def monitor(_clock=self.clock, _process_api=self.process_api,
                            _signal_api=self.signal_api, _send_signal=self.signal_worker,
                            _json=json, _timeout=RUN_TIMEOUT_SECONDS):
                    deadline = _clock.monotonic() + _timeout
                    cancellation_started = None
                    terminate_sent_at = None
                    kill_sent = False
                    try:
                        timed_out = False
                        while process.poll() is None:
                            progress_path = run_dir / "progress.json"
                            progress = None
                            try:
                                progress = _json.loads(progress_path.read_text(encoding="utf-8"))
                            except (OSError, ValueError):
                                pass
                            with self.guard:
                                cancelled = self.cancel_requested
                            if _clock.monotonic() >= deadline:
                                timed_out = True
                                if not cancelled:
                                    self.cancel()
                                    cancelled = True
                            if cancelled and cancellation_started is None:
                                cancellation_started = _clock.monotonic()
                            if cancellation_started is not None:
                                elapsed = _clock.monotonic() - cancellation_started
                                if elapsed >= 45 and terminate_sent_at is None and process.poll() is None:
                                    _send_signal(process, _signal_api.SIGTERM)
                                    terminate_sent_at = _clock.monotonic()
                                elif terminate_sent_at is not None and elapsed >= 55 and not kill_sent and process.poll() is None:
                                    _send_signal(process, _signal_api.SIGKILL)
                                    kill_sent = True
                            self.update(status="cancelling" if cancelled or timed_out else "running",
                                        message="Cancellation requested…" if cancelled else
                                        ("Time limit reached; stopping the worker…" if timed_out else
                                         "Generating with the frozen Wan backbone…"),
                                        progress=progress)
                            try:
                                process.wait(timeout=1)
                            except _process_api.TimeoutExpired:
                                continue
                        code = process.returncode
                        if code == 0 and (run_dir / "run.json").is_file() and (run_dir / "video.mp4").is_file():
                            record = _json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
                            self.update(status="complete_unverified", message="Video generated. Camera-control acceptance remains unverified.",
                                        video_path=str(run_dir / "video.mp4"), run_record=record,
                                        progress={"step": int(config["steps"]), "total": int(config["steps"])})
                        elif timed_out:
                            self.update(status="timed_out", message="Generation exceeded the 30 minute limit and was stopped.")
                        elif self.cancel_requested:
                            self.update(status="cancelled", message="Generation cancelled. Any partial run files are retained.")
                        else:
                            self.update(status="failed", message=f"Inference worker exited with code {code}. Inspect worker.log.")
                        if (project_root / "artifacts" / "gpu.lock").exists():
                            self.update(gpu_lock_present=True,
                                        message=self.job.get("message", "") + " Shared GPU lock is still present; inspect the worker before starting another job.")
                    except BaseException as error:
                        self.update(status="failed", message=f"{type(error).__name__}: {error}")
                    finally:
                        log.close()
                        with self.guard:
                            self.process = None
                        self.lock.release()

                mo.Thread(target=monitor, name="flow-camera-worker", daemon=True).start()
            except BaseException:
                if run_dir is not None:
                    # Preserve any settings/log files for inspection.
                    pass
                self.lock.release()
                raise

    def _signal_worker(process, signal_number, _os=os):
        try:
            if _os.name == "nt":
                process.send_signal(signal_number)
            else:
                _os.killpg(process.pid, signal_number)
        except ProcessLookupError:
            pass

    get_job, set_job = mo.state({"status": "idle", "message": "Upload an image and select a verified flow checkpoint."})
    controller = _FlowJobController()
    return Image, ImageOps, PRESETS, OUTPUT_CONTRACT, VideoContract, camera_path, camera_view_path, checkpoint_catalog, checkpoint_dropdown_options, controller, geometry_preview_svg, get_job


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.Html("<style>body{font-family:'Segoe UI',Arial,sans-serif}#root{max-width:1080px;margin:0 auto;padding:16px}h1{font-size:2rem;line-height:1.2;margin:.2rem 0 .65rem}h2,h3{font-size:1.2rem;line-height:1.35;margin:.9rem 0 .45rem}button,input,select,textarea{font-family:inherit}</style>"),
        mo.md("""
        # Camera motion

        **1. Choose a scene.  2. Set motion and strength.  3. Generate and review a five second clip.**

        Choose a pan or tilt path, or select a zoom preset.
        """),
        mo.accordion({
            "Model and evaluation": mo.md("""
            Wan2.2 TI2V-5B · 1280 × 704 · 121 frames · 24 fps. Zoom is represented in training examples. Pan and tilt have separate empirical inference reviews. Each generated clip needs its own motion and appearance review.
            """)
        })
    ])
    return


@app.cell
def _(mo):
    refresh = mo.ui.run_button(label="Refresh checkpoint readiness")
    refresh
    return (refresh,)


@app.cell
def _(Path, checkpoint_catalog, checkpoint_dropdown_options, mo, project_root, refresh):
    upload = mo.ui.file(filetypes=[".png", ".jpg", ".jpeg", ".webp"], multiple=False,
                        kind="area", max_size=30_000_000, label="Upload one scene image")
    strength = mo.ui.slider(start=0.0, stop=2.0, step=0.05, value=1.0,
                            label="Camera move strength", show_value=True)
    prompt = mo.ui.text_area(value="", placeholder="Describe visible scene details, such as lighting or setting",
                             label="Scene description (optional)", full_width=True)
    seed = mo.ui.number(start=0, stop=2**32 - 1, value=42, step=1, label="Seed")
    steps = mo.ui.number(start=1, stop=100, value=50, step=1, label="Denoising steps")
    _ = refresh.value
    checkpoints = checkpoint_catalog()
    options, option_label_key = checkpoint_dropdown_options(checkpoints, project_root)
    checkpoint = mo.ui.dropdown(options=options, value=option_label_key,
                                label="Flow adapter checkpoint", full_width=True)
    generate = mo.ui.run_button(label="Generate", kind="success", full_width=False)
    cancel = mo.ui.run_button(label="Cancel", kind="warn")
    refresh_status = mo.ui.run_button(label="Refresh")
    mo.vstack([mo.md("### 1. Choose a scene"), upload,
               mo.accordion({"Advanced settings": mo.vstack([
                   mo.md("Scene description (optional). Describe appearance or setting; camera motion is set below."), prompt,
                   mo.md("Defaults are seed 42 and 50 denoising steps."), mo.hstack([seed, steps]), checkpoint, refresh
               ])})])
    return cancel, checkpoint, checkpoints, generate, prompt, refresh_status, seed, steps, strength, upload


@app.cell
def _(Image, ImageOps, mo, project_root, upload):
    import hashlib as _hashlib
    import io as _upload_io

    _image_store = project_root / "artifacts" / "flow_ui_uploads"
    _image_store.mkdir(parents=True, exist_ok=True)
    if upload.value:
        _image = ImageOps.exif_transpose(Image.open(_upload_io.BytesIO(upload.value[0].contents))).convert("RGB")
        _buffer = _upload_io.BytesIO()
        _image.save(_buffer, format="PNG", optimize=True)
        _payload = _buffer.getvalue()
        _digest = _hashlib.sha256(_payload).hexdigest()
        _saved_path = _image_store / f"{_digest}.png"
        if not _saved_path.exists():
            _saved_path.write_bytes(_payload)
    _saved_paths = sorted(_image_store.glob("*.png"), key=lambda path: path.stat().st_mtime, reverse=True)
    _saved_options = {f"Saved image · {path.stem[:12]}": str(path) for path in _saved_paths}
    _saved_label = next(iter(_saved_options), "No saved image")
    saved_image = mo.ui.dropdown(options=_saved_options or {"No saved image": ""}, value=_saved_label,
                                 label="Selected scene image", full_width=True)
    mo.vstack([mo.md("### Confirm your scene"), saved_image])
    return (saved_image,)


@app.cell
def _(Path, saved_image):
    selected_image_path = Path(saved_image.value) if saved_image.value else None
    selected_image_bytes = selected_image_path.read_bytes() if selected_image_path and selected_image_path.is_file() else None
    return selected_image_bytes, selected_image_path


@app.cell
def _(Path, checkpoint, checkpoints, mo, refresh):
    _ = refresh.value
    selected = checkpoint.value or ""
    selected_record = checkpoints.get(selected, {})
    if not checkpoints:
        _checkpoint_view = mo.md("No verified checkpoint is available. Refresh checkpoint readiness, then select a compatible checkpoint.").callout(kind="warn")
    elif not selected:
        _checkpoint_view = mo.md("A compatible checkpoint is available. Select it above to continue.")
    elif not selected_record.get("eligible"):
        _checkpoint_view = mo.vstack([
            mo.md("This checkpoint did not pass the compatibility checks.").callout(kind="warn"),
            mo.accordion({"Checkpoint details": mo.json(selected_record)})
        ])
    else:
        _checkpoint_view = mo.vstack([
            mo.md(f"### Checkpoint ready for inference · {Path(selected).name}"),
            mo.accordion({"Checkpoint details": mo.json(selected_record)})
        ])
    mo.vstack([_checkpoint_view])
    return selected, selected_record


@app.cell
def _(PRESETS, camera_view_path, mo, selected_record, strength):
    _profile = selected_record.get("evaluation_profile", {})
    _profile_valid = bool(selected_record.get("eligible")
                          and _profile.get("status") == "empirically_reviewed_not_certified"
                          and _profile.get("camera_residual_anchor") == "teacher"
                          and _profile.get("checkpoint_sha256") == selected_record.get("checkpoint_sha256"))
    _reviewed = set(_profile.get("empirically_reviewed_modes", [])) if _profile_valid else set()
    _mode_options = {}
    if selected_record.get("eligible") and _profile.get("camera_residual_anchor") == "teacher":
        _mode_options["Pan or tilt path"] = "drawn"
    if "zoom_in" in _reviewed:
        _mode_options["Zoom in"] = "zoom_in"
    if "zoom_out" in _reviewed:
        _mode_options["Zoom out"] = "zoom_out"
    if not _mode_options:
        _mode_options = {"No reviewed motion available": ""}
    _mode_default = "Zoom in" if "Zoom in" in _mode_options else next(iter(_mode_options))
    input_mode = mo.ui.dropdown(options=_mode_options, value=_mode_default,
                                label="Motion type", full_width=True)
    def build_camera_request(control_mode, points, contract, strength):
        if control_mode in ("zoom_in", "zoom_out"):
            if control_mode not in _reviewed:
                raise ValueError("Choose an available zoom preset.")
            return control_mode, None
        if control_mode != "drawn":
            raise ValueError("Choose a reviewed motion type.")
        if points is None or len(points) < 2:
            raise ValueError("Draw at least two points on the image.")
        angle_limit = max(0.001, min(5.0, 5.0 * float(strength)))
        _, request = camera_view_path(
            points, contract, max_yaw_degrees=angle_limit,
            max_pitch_degrees=angle_limit)
        return "drawn_camera_view_path", request

    def classify_camera_path(points):
        if points is None or len(points) < 2:
            return "Custom camera path"
        start, end = points[0], points[-1]
        dx, dy = float(end[0]) - float(start[0]), float(end[1]) - float(start[1])
        length_squared = dx * dx + dy * dy
        if length_squared > 1e-12:
            for point in points[1:-1]:
                px, py = float(point[0]) - float(start[0]), float(point[1]) - float(start[1])
                t = max(0.0, min(1.0, (px * dx + py * dy) / length_squared))
                if ((px - t * dx) ** 2 + (py - t * dy) ** 2) ** 0.5 > 0.035:
                    return "Custom camera path"
        motions = []
        if abs(dx) >= 0.02:
            motions.append("pan right" if dx > 0 else "pan left")
        if abs(dy) >= 0.02:
            motions.append("tilt down" if dy > 0 else "tilt up")
        if len(motions) == 2:
            return "Combined " + " and ".join(motions)
        if motions:
            return motions[0].capitalize()
        return "Custom camera path"

    mo.vstack([mo.md("### 2. Choose a motion type"), input_mode, strength])
    return build_camera_request, input_mode


@app.cell
def _(input_mode, mo):
    preset = input_mode.value if input_mode.value in ("zoom_in", "zoom_out") else ""
    if input_mode.value == "drawn":
        _motion_message = "Draw a path on the scene to control horizontal and vertical camera view movement. Zoom is not combined with a drawn path."
    elif input_mode.value in ("zoom_in", "zoom_out"):
        _motion_message = "The camera follows the selected zoom preset. The drawn path below is inactive."
    else:
        _motion_message = "Select a verified checkpoint under Advanced settings to enable motion controls."
    mo.md(_motion_message)
    return (preset,)


@app.cell
def _(Image, ImageOps, PRESETS, input_mode, mo, selected_record, selected_image_bytes):
    import base64
    import io as _drawing_io

    _profile = selected_record.get("evaluation_profile", {})
    drawing_ready = bool(selected_record.get("eligible")) and _profile.get("camera_residual_anchor") == "teacher"
    use_drawing = input_mode.value == "drawn"
    drawing = None
    _drawing_notice = mo.md("Draw a path from the initial view toward the final view. The line controls horizontal and vertical camera motion.") if use_drawing else mo.md("Zoom is selected. The path is inactive and will not be sent with the request.")
    if selected_image_bytes and drawing_ready:
        try:
            import anywidget
            import traitlets
            if not hasattr(mo.ui, "anywidget"):
                raise RuntimeError("This marimo version does not expose mo.ui.anywidget.")

            class CameraPathCanvas(anywidget.AnyWidget):
                _esm = r"""
                function render({ model, el }) {
                  const canvas = document.createElement('canvas');
                  canvas.style.width = '100%'; canvas.style.maxWidth = '720px';
                  canvas.style.border = '1px solid #777'; canvas.style.borderRadius = '8px'; canvas.style.touchAction = 'none';
                  el.appendChild(canvas);
                  const ctx = canvas.getContext('2d');
                  let image = null;
                  function redraw() {
                    const disabled = model.get('disabled');
                    canvas.style.opacity = disabled ? '0.55' : '1';
                    canvas.style.filter = disabled ? 'grayscale(1)' : 'none';
                    canvas.style.pointerEvents = disabled ? 'none' : 'auto';
                    const src = model.get('image_data');
                    if (src && (!image || image.src !== src)) {
                      image = new Image(); image.onload = redraw; image.src = src;
                    }
                    const w = image && image.naturalWidth ? image.naturalWidth : 640;
                    const h = image && image.naturalHeight ? image.naturalHeight : 360;
                    const scale = Math.min(720 / w, 420 / h, 1);
                    canvas.width = Math.round(w * scale); canvas.height = Math.round(h * scale);
                    if (image && image.complete) ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
                    else { ctx.fillStyle = '#222'; ctx.fillRect(0,0,canvas.width,canvas.height); }
                    const pts = model.get('points') || [];
                    if (pts.length) {
                      ctx.strokeStyle = '#00e0ff'; ctx.fillStyle = '#ffcc00'; ctx.lineWidth = 3;
                      ctx.beginPath(); pts.forEach((p,i) => { const x=p[0]*canvas.width,y=p[1]*canvas.height; if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y); }); ctx.stroke();
                      pts.forEach((p,i) => { ctx.beginPath(); ctx.arc(p[0]*canvas.width,p[1]*canvas.height,5,0,Math.PI*2); ctx.fillStyle=i===0?'#00ff88':'#ffcc00'; ctx.fill(); });
                    }
                  }
                  function point(ev) { const r=canvas.getBoundingClientRect(); return [Math.max(0,Math.min(1,(ev.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(ev.clientY-r.top)/r.height))]; }
                  let drawing=false;
                  canvas.addEventListener('pointerdown', ev => { if(model.get('disabled'))return; drawing=true; canvas.setPointerCapture(ev.pointerId); model.set('points',[point(ev)]); model.save_changes(); redraw(); });
                  canvas.addEventListener('pointermove', ev => { if(!drawing || model.get('disabled'))return; const pts=(model.get('points')||[]).slice(); const p=point(ev); if(!pts.length || Math.hypot(p[0]-pts[pts.length-1][0],p[1]-pts[pts.length-1][1])>0.008) pts.push(p); model.set('points',pts); model.save_changes(); redraw(); });
                  canvas.addEventListener('pointerup', () => { drawing=false; model.save_changes(); });
                  canvas.addEventListener('pointercancel', () => { drawing=false; model.save_changes(); });
                  model.on('change:image_data', redraw); model.on('change:points', redraw); model.on('change:disabled', redraw); redraw();
                  return () => { model.off('change:image_data',redraw); model.off('change:points',redraw); model.off('change:disabled',redraw); el.removeChild(canvas); };
                }
                export default { render };
                """
                _view_name = traitlets.Unicode("AnyView").tag(sync=True)
                points = traitlets.List(traitlets.List(traitlets.Float()), default_value=[]).tag(sync=True)
                image_data = traitlets.Unicode("").tag(sync=True)
                disabled = traitlets.Bool(False).tag(sync=True)

            _drawing_image = ImageOps.exif_transpose(Image.open(_drawing_io.BytesIO(selected_image_bytes))).convert("RGB")
            _drawing_image.thumbnail((1280, 800))
            buffer = _drawing_io.BytesIO()
            _drawing_image.save(buffer, format="JPEG", quality=86)
            widget = CameraPathCanvas(image_data="data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
                                      disabled=not use_drawing)
            drawing = mo.ui.anywidget(widget)
            _drawing_notice = (mo.md("Draw a path from the initial view toward the final view. The line controls horizontal and vertical camera motion.")
                               if use_drawing else mo.md("Zoom is selected. The path is inactive and will not be sent with the request."))
        except (ImportError, RuntimeError, TypeError) as error:
            _drawing_notice = mo.md(f"Interactive drawing is unavailable in this runtime ({type(error).__name__}: {error}).").callout(kind="warn")
    elif not selected_image_bytes:
        _drawing_notice = mo.md("Select a scene image before drawing a path.")
    _drawing_items = [mo.md("### Camera path"), _drawing_notice]
    if drawing is not None:
        _drawing_items.append(drawing)
    mo.vstack(_drawing_items)
    return drawing, drawing_ready


@app.cell(hide_code=True)
def _(classify_camera_path, drawing, input_mode, mo):
    if input_mode.value == "drawn":
        _live_points = drawing.points if drawing is not None else []
        if len(_live_points) >= 2:
            _direction_message = f"Requested direction: **{classify_camera_path(_live_points)}**."
        else:
            _direction_message = "Draw a line to see the requested direction."
    else:
        _direction_message = "The path is inactive while zoom is selected."
    mo.md(_direction_message)
    return


@app.cell
def _(Image, ImageOps, OUTPUT_CONTRACT, VideoContract, camera_path, camera_view_path, classify_camera_path, drawing, drawing_ready, geometry_preview_svg, input_mode, mo, preset, selected_record, strength, selected_image_bytes, selected_image_path):
    _preview_items = []
    if selected_image_bytes:
        try:
            import io as _preview_io
            _preview_image = ImageOps.exif_transpose(Image.open(_preview_io.BytesIO(selected_image_bytes))).convert("RGB")
            _preview_contract = VideoContract(**(selected_record.get("contract") or OUTPUT_CONTRACT))
            if input_mode.value == "drawn":
                _points = drawing.points if drawing is not None else []
                if len(_points) >= 2:
                    _motion_label = classify_camera_path(_points)
                    _limit = max(0.001, min(5.0, 5.0 * float(strength.value)))
                    _homographies, _ = camera_view_path(
                        _points, _preview_contract, max_yaw_degrees=_limit,
                        max_pitch_degrees=_limit)
                    _, _svg = geometry_preview_svg(
                        _preview_image, _preview_contract, _homographies,
                        mode=_motion_label)
                    _preview_items = [mo.md(f"Preview of requested camera motion: **{_motion_label}**."), mo.Html(_svg)]
                else:
                    _preview_items = [mo.md("Draw at least two points to preview the path.")]
            else:
                _motion_label = input_mode.value.replace("_", " ")
                _homographies, _ = camera_path(
                    input_mode.value, _preview_contract, magnitude=float(strength.value))
                _, _svg = geometry_preview_svg(
                    _preview_image, _preview_contract, _homographies,
                    mode=_motion_label)
                _preview_items = [mo.md(f"Preview of requested camera motion: **{_motion_label}**."), mo.Html(_svg)]
        except Exception as error:
            _preview_items = [mo.md(f"Preview unavailable: {type(error).__name__}.").callout(kind="warn")]
    else:
        _preview_items = [mo.md("Choose a scene image to preview the requested motion.")]
    mo.accordion({"Preview of requested camera motion": mo.vstack(_preview_items)})
    return


@app.cell
def _(cancel, generate, mo, refresh_status):
    mo.vstack([mo.md("### 3. Generate and review"),
               mo.hstack([generate, cancel, refresh_status])])
    return


@app.cell
def _(OUTPUT_CONTRACT, VideoContract, build_camera_request, controller, drawing, drawing_ready, generate, input_mode, mo, prompt, seed, selected, selected_record, steps, strength, selected_image_bytes):
    if generate.value:
        try:
            if not selected_image_bytes:
                raise ValueError("Upload an image and select its saved copy before generating.")
            if not selected_record.get("eligible"):
                raise ValueError("No matching, hash-verified flow checkpoint is ready. Generation remains disabled until one exists.")
            if input_mode.value == "drawn" and (not drawing_ready or drawing is None):
                raise ValueError("A verified checkpoint and drawing canvas are required for a camera path.")
            points = drawing.points if input_mode.value == "drawn" else None
            contract = VideoContract(**(selected_record.get("contract") or OUTPUT_CONTRACT))
            _mode, camera_request = build_camera_request(
                input_mode.value, points, contract, strength.value)
            controller.submit(image_bytes=selected_image_bytes, checkpoint=selected,
                              checkpoint_record=selected_record, mode=_mode,
                              magnitude=float(strength.value), prompt=prompt.value,
                              seed=int(seed.value), steps=int(steps.value), camera_request=camera_request)
            mo.md("Generation started in a bounded child process. You can cancel or refresh status below.")
        except (ValueError, RuntimeError, OSError) as error:
            mo.md(str(error)).callout(kind="warn")
    return


@app.cell
def _(cancel, controller):
    if cancel.value:
        controller.cancel()
    return


@app.cell
def _(Path, get_job, mo, refresh_status):
    _ = refresh_status.value
    _job = get_job()
    _status_labels = {
        "complete_unverified": "Video ready",
        "running": "Generation running",
        "failed": "Generation failed",
        "cancelled": "Generation cancelled",
        "idle": "Ready",
    }
    _status_label = _status_labels.get(_job.get("status"), "Status unavailable")
    _status_message = _job.get("message", "")
    if _job.get("status") == "complete_unverified":
        _status_message = ""
    _job_items = [mo.md(f"### {_status_label}")]
    if _status_message:
        _job_items.append(mo.md(_status_message))
    _progress = _job.get("progress") or {}
    if isinstance(_progress.get("step"), (int, float)) and isinstance(_progress.get("total"), (int, float)) and _progress["total"] > 0:
        _fraction = min(1., max(0., _progress["step"] / _progress["total"]))
        _job_items.append(mo.Html(f'<progress value="{_fraction:.4f}" max="1" style="width:100%" aria-label="Denoising progress"></progress><span>{_progress["step"]} / {_progress["total"]} steps · {_fraction:.0%}</span>'))
    _video_path = _job.get("video_path")
    if _video_path and Path(_video_path).is_file():
        _video = Path(_video_path)
        _job_items.extend([mo.md("### Your generated video"), mo.md("Preview the finished MP4 here, or use Download original MP4 to save it to your device."), mo.video(str(_video)),
                      mo.download(lambda: _video.read_bytes(), filename="flow_camera_video.mp4",
                                  mimetype="video/mp4", label="Download original MP4")])
    for _key, _filename, _download_label in (("settings_path", "flow_camera_settings.json", "Download run settings"),
                                              ("log_path", "flow_camera_worker.log", "Download worker log")):
        _path = _job.get(_key)
        if _path and Path(_path).is_file():
            _candidate = Path(_path)
            if _key == "settings_path":
                _job_items.append(mo.download(_candidate.read_bytes(), filename=_filename,
                                              mimetype="application/json", label=_download_label))
    if _job.get("run_record"):
        _job_items.append(mo.accordion({"Run details": mo.json(_job["run_record"])}))
    if _job.get("run_dir") and Path(_job["run_dir"]).is_dir():
        _job_items.append(mo.md(f"Run artifacts: `{_job['run_dir']}`"))
    mo.vstack(_job_items)
    return


if __name__ == "__main__":
    app.run()
