import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Camera motion v2")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        r"""
        # Camera motion v2

        Turn **one photograph** into a five-second clip (Wan2.2 TI2V-5B, 1280 x 704, 121 frames, 24 fps) with a
        virtual camera move: pick **one motion** (pan, tilt or zoom) and **one strength** slider.

        * **Strength s in [0, 1]**: 0 is a static camera, 1 is the trained maximum (8 degrees of pan or tilt,
          or a 1.25x focal zoom). The slider is mapped through a measured per-direction *calibration*
          (`configs/strength_calibration_v2*.json`, see `docs/v2/CALIBRATION.md`) so that the amount you see
          is the amount on the label. Without a fitted file the identity map is used and the label says so.
        * **Preview** shows where the camera ends up at frame 120. The red region is *newly revealed*:
          the model has to invent it, because the photograph does not contain it.
        * **Generate** runs `python -m fpcamera.flow_infer` in a child process (so you can cancel it),
          shows the video, the run/request summary and the border-quality and motion measurements.

        **Launch** (from the repository root, in the pinned Wan runtime with a CUDA GPU):
        `marimo edit notebooks/camera_interface_v2.py` (editable) or
        `marimo run notebooks/camera_interface_v2.py` (app only). On Molab open the file and press run.

        **Needs**: `pip install -r requirements-ui.txt` (marimo, anywidget) on top of the Wan runtime
        (torch, diffusers, safetensors, opencv-python, imageio-ffmpeg); the pinned base model in the local
        Hugging Face cache (inference runs with `local_files_only`); at least one adapter checkpoint under
        `artifacts/` (`adapter_*.safetensors`, v1 sidecar `.json` or a v2 run directory with `checkpoints/`);
        a free GPU (`artifacts/gpu.lock` must not exist).
        """
    )
    return (mo,)


@app.cell
def _():
    from pathlib import Path
    import sys

    project_root = Path(__file__).resolve().parents[1]
    for _p in (project_root / "src", project_root / "scripts"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    return Path, project_root, sys


@app.cell
def _(Path, mo, project_root, sys):
    import hashlib
    import json
    import os
    import signal
    import subprocess
    import threading
    import time
    import uuid

    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    from fpcamera import border_metrics as bm
    from fpcamera import strength as st
    from fpcamera.flow_geometry import VideoContract

    TRAINING_CAPTION = "A photograph of a scene. One continuous camera shot."
    CONTRACT = VideoContract(width=1280, height=704, frames=121, fps=24)
    MOTIONS = {label: key for key, label in st.MOTION_LABELS.items() if key != "static"}
    RUN_TIMEOUT_SECONDS = 40 * 60
    RUNS_DIR = project_root / "artifacts" / "ui_v2_runs"
    UPLOADS_DIR = project_root / "artifacts" / "ui_v2_uploads"
    IDENTITY_NAME = "strength_calibration_v2.identity.json"

    # ---------------------------------------------------------------- catalogs
    def scan_checkpoints():
        """label -> path of every adapter checkpoint under artifacts/ (v2 first, newest step first)."""
        try:
            from fpcamera.flow_infer import resolve_checkpoint_meta
        except Exception:  # torch-less environment: list files without metadata
            resolve_checkpoint_meta = None
        found = []
        for path in sorted((project_root / "artifacts").glob("**/adapter_*.safetensors")):
            kind, steps = "?", None
            if resolve_checkpoint_meta is not None:
                try:
                    meta, source = resolve_checkpoint_meta(path)
                    kind = "v2" if source == "v2_manifest" else "v1"
                    steps = meta.get("completed_updates")
                    fill = meta.get("guide_fill_mode") or "zeros"
                except (OSError, ValueError, KeyError):
                    kind, fill = "no metadata", "?"
            else:
                fill = "?"
            rel = path.relative_to(project_root / "artifacts").as_posix()
            found.append((kind != "v2", -(steps or 0), rel, str(path), f"{rel} · {kind} · {steps} updates · fill {fill}"))
        found.sort()
        return {label: path for *_, path, label in found}

    def scan_calibrations():
        files = sorted((project_root / "configs").glob("strength_calibration_v2*.json"))
        files.sort(key=lambda p: (p.name == IDENTITY_NAME, p.name))  # fitted files first
        options = {}
        for p in files:
            tag = "identity (uncalibrated, u = s)" if p.name == IDENTITY_NAME else "fitted"
            options[f"{p.name} · {tag}"] = str(p)
        return options or {"identity (built in)": ""}

    def scan_examples():
        folder = project_root / "examples" / "inputs"
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")) if folder.is_dir() else []
        return {p.stem: str(p) for p in files}

    # ---------------------------------------------------------------- preview
    def build_preview(image_path, homographies, motion_label, width=560):
        """(first view, frame-120 view with the revealed region in red and first-view outlines)."""
        image = ImageOps.fit(
            ImageOps.exif_transpose(Image.open(image_path)).convert("RGB"),
            (CONTRACT.width, CONTRACT.height),
            method=Image.Resampling.LANCZOS,
        )
        first = np.asarray(image)
        last = len(homographies) - 1
        warped = cv2.warpPerspective(first, homographies[last], (CONTRACT.width, CONTRACT.height))
        revealed = bm.revealed_mask_pixels(homographies, CONTRACT.width, CONTRACT.height, frame_indices=[last])[0]
        view = warped.astype(np.float32)
        tint = np.array([235, 60, 60], dtype=np.float32)
        view[revealed] = 0.35 * view[revealed] + 0.65 * tint
        view = view.astype(np.uint8)
        corners = np.array(
            [[0, 0, 1], [CONTRACT.width - 1, 0, 1], [CONTRACT.width - 1, CONTRACT.height - 1, 1], [0, CONTRACT.height - 1, 1]],
            dtype=np.float64,
        )
        for frame, thickness in ((last // 3, 2), (2 * last // 3, 2), (last, 5)):
            q = corners @ homographies[frame].T
            pts = (q[:, :2] / q[:, 2:3]).round().astype(np.int32)
            cv2.polylines(view, [pts], True, (40, 220, 255), thickness, cv2.LINE_AA)
        height = round(width * CONTRACT.height / CONTRACT.width)
        left = image.resize((width, height), Image.Resampling.LANCZOS)
        right = Image.fromarray(view).resize((width, height), Image.Resampling.LANCZOS)
        return left, right, float(revealed.mean())

    # ---------------------------------------------------------------- job control
    def _signal_group(process, number):
        try:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, number)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    class JobController:
        """One bounded inference subprocess at a time; state is polled by the status cell."""

        def __init__(self):
            self.guard = threading.Lock()
            self.process = None
            self.job = {"status": "idle", "message": "Pick an image and a checkpoint, then press Generate."}

        def snapshot(self):
            with self.guard:
                return dict(self.job)

        def _update(self, **fields):
            with self.guard:
                self.job = {**self.job, **fields, "updated_at": time.time()}

        def busy(self):
            with self.guard:
                return self.process is not None and self.process.poll() is None

        def cancel(self):
            with self.guard:
                process = self.process
                if process is None or process.poll() is not None:
                    return False
                self.job = {**self.job, "status": "cancelling", "cancel_requested": True,
                            "message": "Cancelling: asking the worker to stop and release the GPU lease."}
            _signal_group(process, signal.SIGINT)
            return True

        def submit(self, command, run_dir, steps):
            if self.busy():
                raise RuntimeError("A generation is already running in this notebook.")
            gpu_lock = project_root / "artifacts" / "gpu.lock"
            if gpu_lock.exists():
                raise RuntimeError(f"The shared GPU lease {gpu_lock} exists: another job runs, or it is stale after a crash (delete it).")
            run_dir.mkdir(parents=True, exist_ok=False)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(project_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
            env.update(FPCAMERA_ROOT=str(project_root), USE_TF="0", HF_HUB_DISABLE_PROGRESS_BARS="1")
            log_path = run_dir / "worker.log"
            log = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command, cwd=str(project_root), env=env, stdout=log, stderr=subprocess.STDOUT, text=True,
                start_new_session=(os.name != "nt"),
            )
            with self.guard:
                self.process = process
            self._update(status="running", message="Loading Wan and generating...", run_dir=str(run_dir),
                         log_path=str(log_path), started_at=time.time(), steps=steps, cancel_requested=False,
                         progress=None, video_path=None)
            threading.Thread(target=self._monitor, args=(process, run_dir, log, steps), daemon=True).start()

        def _monitor(self, process, run_dir, log, steps):
            started = time.monotonic()
            sent_term = False
            try:
                while process.poll() is None:
                    progress = None
                    try:
                        progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        pass
                    snap = self.snapshot()
                    cancelling = snap.get("cancel_requested", False)
                    if time.monotonic() - started > RUN_TIMEOUT_SECONDS and not cancelling:
                        self.cancel()
                    if cancelling and not sent_term and time.time() - snap.get("updated_at", time.time()) > 45:
                        sent_term = True
                        _signal_group(process, signal.SIGTERM)
                    self._update(progress=progress)
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
                code = process.returncode
                snap = self.snapshot()
                if code == 0 and (run_dir / "run.json").is_file() and (run_dir / "video.mp4").is_file():
                    self._update(status="complete", message="Video ready.", video_path=str(run_dir / "video.mp4"),
                                 progress={"step": steps, "total": steps})
                elif snap.get("cancel_requested"):
                    self._update(status="cancelled", message="Cancelled. Partial files are kept in the run folder.")
                else:
                    self._update(status="failed", message=f"Worker exited with code {code}; see the log below.")
                lock = project_root / "artifacts" / "gpu.lock"
                try:  # a terminated worker cannot run its finally block: drop its own stale lease
                    if lock.exists() and lock.read_text().strip() == str(process.pid):
                        lock.unlink()
                except OSError:
                    pass
            except BaseException as error:  # noqa: BLE001 - the monitor must never die silently
                self._update(status="failed", message=f"{type(error).__name__}: {error}")
            finally:
                log.close()
                with self.guard:
                    self.process = None

    controller = JobController()
    return (
        CONTRACT, Image, ImageOps, MOTIONS, RUNS_DIR, TRAINING_CAPTION, UPLOADS_DIR, bm,
        build_preview, controller, hashlib, json, np, scan_calibrations, scan_checkpoints,
        scan_examples, st, time, uuid,
    )


@app.cell
def _(mo):
    refresh_catalogs = mo.ui.run_button(label="Rescan checkpoints / examples")
    refresh_catalogs
    return (refresh_catalogs,)


@app.cell
def _(MOTIONS, TRAINING_CAPTION, mo, refresh_catalogs, scan_calibrations, scan_checkpoints, scan_examples):
    _ = refresh_catalogs.value
    _ckpts = scan_checkpoints()
    _cals = scan_calibrations()
    _examples = scan_examples()

    upload = mo.ui.file(filetypes=[".png", ".jpg", ".jpeg", ".webp"], multiple=False, kind="area",
                        max_size=30_000_000, label="Upload a photograph")
    example = mo.ui.dropdown(options=_examples or {"No examples found in examples/inputs": ""},
                             value=next(iter(_examples), "No examples found in examples/inputs"),
                             label="...or pick an example", full_width=True)
    motion = mo.ui.dropdown(options=MOTIONS, value="Pan right", label="Motion", full_width=True)
    strength = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=1.0, label="Strength", show_value=True,
                            debounce=True, full_width=True)
    prompt = mo.ui.text_area(value=TRAINING_CAPTION, label="Scene description (optional; default = training caption)",
                             full_width=True, rows=2)
    seed = mo.ui.number(start=0, stop=2**32 - 1, value=42, step=1, label="Seed")
    steps = mo.ui.number(start=1, stop=100, value=50, step=1, label="Denoising steps")
    checkpoint = mo.ui.dropdown(options=_ckpts or {"No adapter checkpoint found under artifacts/": ""},
                                value=next(iter(_ckpts), "No adapter checkpoint found under artifacts/"),
                                label="Adapter checkpoint", full_width=True)
    calibration = mo.ui.dropdown(options=_cals, value=next(iter(_cals)), label="Strength calibration", full_width=True)
    min_sigma_auto = mo.ui.checkbox(value=True, label="camera_min_sigma: use the checkpoint's value")
    min_sigma = mo.ui.number(start=0.0, stop=1.0, step=0.05, value=0.6, label="camera_min_sigma (if not auto)")
    fill_mode = mo.ui.dropdown(options=["auto", "zeros", "replicate"], value="auto",
                               label="Guide fill mode (auto = what the checkpoint was trained with)")
    camera_gain = mo.ui.number(start=0.0, stop=2.0, step=0.1, value=1.0, label="Camera gain (1 = trained)")

    mo.vstack([
        mo.md("## 1. Scene"),
        mo.hstack([upload, example], widths=[1, 1], align="start"),
        mo.md("## 2. Motion and strength"),
        mo.hstack([motion, strength], widths=[1, 2], align="end"),
        mo.md("## 3. Model"),
        mo.hstack([checkpoint, calibration], widths=[2, 1], align="end"),
        mo.accordion({
            "Advanced": mo.vstack([
                prompt,
                mo.hstack([seed, steps, camera_gain], justify="start"),
                mo.hstack([min_sigma_auto, min_sigma], justify="start", align="end"),
                fill_mode,
                mo.md("`camera_min_sigma` is the noise level below which the adapter is switched off. "
                      "A different guide fill than the checkpoint was trained with is a train/inference mismatch "
                      "and is reported as a warning in the run summary."),
            ])
        }),
    ])
    return (calibration, camera_gain, checkpoint, example, fill_mode, min_sigma, min_sigma_auto, motion,
            prompt, seed, steps, strength, upload)


@app.cell
def _(Image, ImageOps, UPLOADS_DIR, example, hashlib, upload):
    import io as _io
    from pathlib import Path as _Path

    image_path = None
    image_source = None
    if upload.value:
        _img = ImageOps.exif_transpose(Image.open(_io.BytesIO(upload.value[0].contents))).convert("RGB")
        _buf = _io.BytesIO()
        _img.save(_buf, format="PNG")
        _payload = _buf.getvalue()
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        image_path = UPLOADS_DIR / f"{hashlib.sha256(_payload).hexdigest()[:16]}.png"
        if not image_path.exists():
            image_path.write_bytes(_payload)
        image_source = f"upload: {upload.value[0].name}"
    elif example.value:
        image_path = _Path(example.value)
        image_source = f"example: {image_path.stem}"
    return image_path, image_source


@app.cell
def _(calibration, motion, mo, project_root, st, strength):
    _cal_path = calibration.value or None
    strength_map = st.load_calibration(_cal_path, root=project_root)
    max_amplitudes = st.load_max_amplitudes(project_root)
    u_requested = strength_map.s_to_u(motion.value, strength.value)
    amount_label = st.amplitude_label(motion.value, u_requested, max_amplitudes)
    _note = ("identity map: u = s, **not calibrated**" if strength_map.is_identity
             else f"calibrated: s = {strength.value:.2f} requests u = {u_requested:.2f}")
    mo.callout(mo.md(f"### {amount_label}\n\nstrength s = {strength.value:.2f} · {_note}"), kind="info")
    return amount_label, max_amplitudes, strength_map, u_requested


@app.cell
def _(CONTRACT, build_preview, image_path, max_amplitudes, mo, motion, np, st, strength_map, u_requested):
    from fpcamera.flow_geometry import camera_path_v2

    if image_path is None:
        _view = mo.md("Upload a photograph or choose an example to see the preview.")
    else:
        try:
            _fam, _dir = st.split_motion(motion.value)
            _h, _ = camera_path_v2(_fam, _dir, u_requested, CONTRACT, max_yaw_deg=max_amplitudes["yaw_deg"],
                                   max_pitch_deg=max_amplitudes["pitch_deg"], max_zoom_ratio=max_amplitudes["zoom_ratio"])
            _left, _right, _frac = build_preview(image_path, np.asarray(_h), motion.value)
            _view = mo.vstack([
                mo.hstack([
                    mo.vstack([mo.md("**Input (frame 0)**"), mo.image(_left)]),
                    mo.vstack([mo.md("**Frame 120 (geometry only)**"), mo.image(_right)]),
                ], justify="start"),
                mo.md(f"Red = newly revealed, must be generated by the model ({_frac:.0%} of the last frame). "
                      "Cyan outlines = where the first view sits at frames 40, 80 and 120."),
            ])
        except Exception as _error:  # noqa: BLE001
            _view = mo.md(f"Preview unavailable: {type(_error).__name__}: {_error}").callout(kind="warn")
    mo.vstack([mo.md("## Preview"), _view])
    return (camera_path_v2,)


@app.cell
def _(mo):
    generate = mo.ui.run_button(label="Generate", kind="success")
    cancel = mo.ui.run_button(label="Cancel", kind="warn")
    refresh = mo.ui.refresh(options=["2s", "5s"], default_interval="2s", label="Status")
    mo.vstack([mo.md("## 4. Generate"), mo.hstack([generate, cancel, refresh], justify="start")])
    return cancel, generate, refresh


@app.cell
def _(RUNS_DIR, calibration, camera_gain, cancel, checkpoint, controller, fill_mode, generate, image_path, min_sigma,
      min_sigma_auto, mo, motion, project_root, prompt, seed, steps, strength, sys, time, uuid):
    _message = None
    if cancel.value:
        controller.cancel()
    if generate.value:
        try:
            if image_path is None:
                raise ValueError("Choose or upload an image first.")
            if not checkpoint.value:
                raise ValueError("No adapter checkpoint selected.")
            _run_dir = RUNS_DIR / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
            _cmd = [
                sys.executable, "-m", "fpcamera.flow_infer",
                "--project-root", str(project_root), "--checkpoint", checkpoint.value,
                "--image", str(image_path), "--output", str(_run_dir),
                "--motion", motion.value, "--strength", f"{strength.value:.4f}",
                "--steps", str(int(steps.value)), "--seed", str(int(seed.value)),
                "--camera-guidance", str(float(camera_gain.value)),
            ]
            if calibration.value:
                _cmd += ["--calibration", calibration.value]
            if not min_sigma_auto.value:
                _cmd += ["--camera-min-sigma", str(float(min_sigma.value))]
            if fill_mode.value != "auto":
                _cmd += ["--guide-fill-mode", fill_mode.value]
            if prompt.value.strip():
                _cmd += ["--prompt", prompt.value.strip()]
            controller.submit(_cmd, _run_dir, int(steps.value))
            _message = mo.md(f"Started `{_run_dir.name}`. This cell stays quiet; watch the status below.")
        except (ValueError, RuntimeError, OSError) as _error:
            _message = mo.md(str(_error)).callout(kind="warn")
    _message
    return


@app.cell
def _(controller, generate, cancel, mo, refresh):
    _ = (refresh.value, generate.value, cancel.value)
    job = controller.snapshot()
    _titles = {"idle": "Ready", "running": "Generating", "cancelling": "Cancelling", "complete": "Video ready",
               "cancelled": "Cancelled", "failed": "Failed"}
    _items = [mo.md(f"### {_titles.get(job['status'], job['status'])}"), mo.md(job.get("message", ""))]
    _p = job.get("progress") or {}
    if _p.get("total"):
        _f = min(1.0, max(0.0, _p["step"] / _p["total"]))
        _items.append(mo.Html(
            f'<progress value="{_f:.4f}" max="1" style="width:100%" aria-label="Denoising progress"></progress>'
            f"<div>{_p['step']} / {_p['total']} steps · {_f:.0%}"
            + (f" · {_p['seconds']:.0f}s" if _p.get("seconds") else "") + "</div>"))
    if job["status"] in ("failed", "cancelled") and job.get("log_path"):
        try:
            _tail = "\n".join(open(job["log_path"], encoding="utf-8", errors="replace").read().splitlines()[-25:])
            _items.append(mo.accordion({"Worker log (last lines)": mo.md(f"```\n{_tail}\n```")}))
        except OSError:
            pass
    mo.vstack(_items)
    return (job,)


@app.cell
def _(job, json, mo, np, project_root, st):
    from pathlib import Path as _P

    _out = mo.md("")
    if job.get("status") == "complete" and job.get("video_path"):
        _run = _P(job["run_dir"])
        _video = _P(job["video_path"])
        _items = [mo.md("## Result"), mo.video(src=str(_video), controls=True, loop=True, rounded=True),
                  mo.download(data=_video.read_bytes(), filename=f"camera_v2_{_run.name}.mp4", mimetype="video/mp4",
                              label="Download MP4")]
        _record = json.loads((_run / "run.json").read_text(encoding="utf-8"))
        _request = json.loads((_run / "request.json").read_text(encoding="utf-8"))
        _control = _record.get("strength_control") or {}
        _summary = {
            "requested": _control.get("label"),
            "strength s -> u": f"{_control.get('strength_s')} -> {_control.get('strength_u')}",
            "calibration": ("identity" if (_control.get("calibration") or {}).get("identity")
                            else (_control.get("calibration") or {}).get("path")),
            "calibration sha256": ((_control.get("calibration") or {}).get("sha256") or "")[:16],
            "checkpoint updates": _record.get("completed_updates"),
            "guide fill": f"{_record.get('guide_fill_mode')} ({_record.get('guide_fill_mode_source')})",
            "camera_min_sigma": _record.get("camera_min_sigma"),
            "residual anchor": _record.get("camera_residual_anchor"),
            "camera gain": _record.get("camera_guidance"),
            "seed / steps": f"{_record.get('seed')} / {_record.get('steps')}",
            "generation time (s)": round(_record.get("seconds", 0), 1),
            "peak GPU memory (GiB)": round(_record.get("peak_gib", 0), 1),
            "warnings": "; ".join(_record.get("notes") or []) or "none",
        }
        _items.append(mo.accordion({"Run summary": mo.ui.table([{"field": k, "value": str(v)} for k, v in _summary.items()],
                                                                selection=None, pagination=False)}, lazy=False))
        try:  # border metrics and tracker gain (CPU, a few seconds)
            import measure_border_quality as _mbq
            import calibrate_strength as _cs

            _bq = _mbq.clean(_mbq.measure_run(_run, mean_stride=4))
            _mean = _bq["mean_over_frames_with_R_gt_1pct"]
            _gain = _cs.measure_run_gain(_run)
            _rows = [
                {"metric": "gray-band score (R minus valid; lower is better)", "value": _mean.get("gray_band_score")},
                {"metric": "colorfulness R/valid (1 = same as valid)", "value": _mean.get("colorfulness_ratio")},
                {"metric": "high-frequency R/valid (1 = same texture)", "value": _mean.get("hf_ratio")},
                {"metric": "seam score (1 = no visible seam)", "value": _mean.get("seam_score")},
                {"metric": "revealed frames used", "value": _mean.get("n_frames")},
                {"metric": "tracker gain g (observed / requested displacement)", "value": _gain.get("gain")},
            ]
            if _gain.get("gain") is not None and _gain.get("strength_u") is not None:
                _rows.append({"metric": "observed amplitude a_obs = g * u (1 = trained max)",
                              "value": _gain["gain"] * _gain["strength_u"]})
            _rows = [{**r, "value": ("n/a" if r["value"] is None else round(float(r["value"]), 3))} for r in _rows]
            _items.append(mo.accordion({"Border quality and motion": mo.ui.table(_rows, selection=None, pagination=False)}, lazy=False))
        except Exception as _error:  # noqa: BLE001
            _items.append(mo.md(f"Measurements unavailable: {type(_error).__name__}: {_error}").callout(kind="warn"))
        _request_view = {k: v for k, v in _request.items() if k != "homographies_first_to_current"}
        _items.append(mo.accordion({"run.json": mo.json(_record), "request.json (without homographies)": mo.json(_request_view)}))
        _items.append(mo.md(f"Run folder: `{_run}`"))
        _out = mo.vstack(_items)
    _out
    return


if __name__ == "__main__":
    app.run()
