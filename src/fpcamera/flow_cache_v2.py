"""Resumable, sharded v2 synthetic camera cache (magnitude-diverse, six directions).

Differences from ``flow_cache`` (v1), which this module deliberately reuses for the
VAE path (``encode_rgb``: FP32, posterior mode, per-channel normalisation, bf16):

* one static clip + K moving clips per source, each with its own (family, direction,
  strength u) drawn by :class:`fpcamera.motion_sampling.MotionSampler`;
* per-source overscan chosen so the max-strength path in every direction stays on
  real source pixels; renders use BORDER_REFLECT_101 *and* a per-pixel coverage
  assertion, so no filled pixel can reach a target;
* one safetensors file per clip, written to a temp name and atomically renamed; the
  index row is appended only after the file is complete; restart skips verified
  files, adopts orphan files, and the index can always be rebuilt from the files;
* the module imports without torch; torch/diffusers are imported only when encoding.

Layout under the cache root::

    clips/<clip_id>.safetensors         latents [48,31,44,80] bf16, homographies [121,3,3] f32
    references/<source_id>.safetensors  image_latent [48,1,44,80], text_embeddings,
                                        negative_text_embeddings  (same keys as v1)
    references/<source_id>.png          rendered first view
    index.jsonl                         one row per clip (see ``make_row``)
    config_snapshot.json, progress.json, rejected.jsonl
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
import shutil
import struct
import time

import numpy as np
from PIL import Image

from .flow_geometry import (
    VideoContract,
    camera_path_v2,
    render_source,
    required_overscan,
    revealed_fraction,
)
from .motion_sampling import MotionSampler
from .runtime import file_sha256, write_json
from .session_deadline import should_stop

NORMALIZATION = "native_fp32_vae_mode_normalize_then_bf16"  # == flow_cache.NORMALIZATION
SCHEMA = "2"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "data_v2.json"
INDEX_FIELDS = (
    "id",
    "source_id",
    "split",
    "family",
    "direction",
    "magnitude_u",
    "yaw_deg",
    "pitch_deg",
    "zoom_ratio",
    "overscan",
    "latent_file",
    "reference_file",
    "sha256",
)


class SourceRejected(Exception):
    """A source image that cannot be used (too small for the required overscan); recorded in ``rejected.jsonl``."""

    pass


# --------------------------------------------------------------------- config
def load_config(path=None):
    """Read and check ``configs/data_v2.json`` (schema ``fpcamera_data_v2``)."""
    cfg = json.loads(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    if cfg.get("schema") != "fpcamera_data_v2":
        raise ValueError("Not a fpcamera_data_v2 config")
    return cfg


def contract_of(cfg) -> VideoContract:
    """The ``VideoContract`` described by a data config."""
    return VideoContract(**cfg["video_contract"])


def camera_kwargs(cfg):
    """Camera-path keyword arguments (max yaw/pitch/zoom, hfov, easing) from the config's ``camera`` section."""
    c = cfg["camera"]
    return {
        "max_yaw_deg": c["max_yaw_deg"],
        "max_pitch_deg": c["max_pitch_deg"],
        "max_zoom_ratio": c["max_zoom_ratio"],
        "hfov_degrees": c["hfov_degrees"],
        "easing": c["easing"],
    }


def load_sources(manifest_path):
    """Read the source manifest (``data/v2/sources_v2.json``) and check source ids are unique."""
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = data["records"]
    if len({r["source_id"] for r in records}) != len(records):
        raise ValueError("Duplicate source_id in manifest")
    return records


def population_sha(records):
    """Digest of the (source_id, split) population; stored in the config snapshot (``critical_config``) so a changed
    source set is detected when a build is resumed.
    """
    import hashlib

    payload = json.dumps(sorted((r["source_id"], r["split"]) for r in records))
    return hashlib.sha256(payload.encode()).hexdigest()


# ------------------------------------------------------------------- planning
@dataclass(frozen=True)
class ClipPlan:
    """Planned clip: id, source, split, kind (static | moving), family/direction and strength u."""

    id: str
    source_id: str
    split: str
    kind: str  # "static" | "moving"
    clip_index: int  # moving clips: 0..K-1; static: -1
    family: str
    direction: str | None
    u: float

    @property
    def mode(self):
        """v1-style mode string (``static`` or ``<family>_<direction>``)."""
        return "static" if self.kind == "static" else f"{self.family}_{self.direction}"


def plan_clips(records, cfg):
    """Deterministic clip plan for the whole manifest (independent of sharding)."""
    sampling = cfg["sampling"]
    sampler = MotionSampler(
        [(r["source_id"], r["split"]) for r in records],
        clips_per_source=sampling["moving_clips_per_source"],
        seed=cfg["seed"],
        u_min=sampling["u_min"],
        u_max=sampling["u_max"],
    )
    plans = []
    for r in records:
        sid, split = r["source_id"], r["split"]
        for s in range(sampling["static_clips_per_source"]):
            suffix = "static" if s == 0 else f"static{s}"
            plans.append(ClipPlan(f"{sid}__{suffix}", sid, split, "static", -1, "static", None, 0.0))
        for c in range(sampling["moving_clips_per_source"]):
            m = sampler.sample(sid, c)
            cid = f"{sid}__m{c}_{m.family}_{m.direction}_u{int(round(m.u * 1000)):04d}"
            plans.append(ClipPlan(cid, sid, split, "moving", c, m.family, m.direction, m.u))
    return plans, sampler


def source_overscan(image_size, contract, cfg):
    """Per-source overscan (constant across that source's clips)."""
    o = cfg["overscan"]
    if o["policy"] == "fixed":
        return float(o["fixed"])
    if o["policy"] != "per_source":
        raise ValueError("overscan.policy must be 'fixed' or 'per_source'")
    from .flow_geometry import V2_MOTIONS

    paths = [
        camera_path_v2(f, d, 1.0, contract, **camera_kwargs(cfg))[0]
        for f, d in V2_MOTIONS
    ]
    try:
        need = required_overscan(image_size, contract, paths, hi=o["cap"] + 0.5)
    except ValueError as error:
        raise SourceRejected(str(error))
    # overscan = how much extra real source margin is kept around the first view so that the strongest path in every
    # direction still renders from real pixels (never from filled borders).  It is chosen per source (floor, margin, cap
    # from the config); sources that would need more than the cap are rejected.
    value = max(o["floor"], need + o["margin"])
    value = math.ceil(value * 1000) / 1000
    if value > o["cap"]:
        raise SourceRejected(
            f"image {tuple(image_size)} needs overscan {value:.3f} > cap {o['cap']}"
        )
    return value


# ------------------------------------------------------------------ rendering
def render_clip(image, plan: ClipPlan, cfg, contract, overscan):
    """Return (rgb[F,H,W,3] uint8, homographies[F,3,3], camera_record, crop_record)."""
    import cv2

    h, record = camera_path_v2(
        plan.family, plan.direction, plan.u, contract, **camera_kwargs(cfg)
    )
    r = cfg["render"]
    rgb, crop = render_source(
        image,
        h,
        contract,
        overscan=overscan,
        border_mode=getattr(cv2, r["border_mode"]),
        assert_coverage=r["assert_coverage"],
        coverage_stride=r["coverage_stride"],
    )
    return rgb, h, record, crop


def make_row(plan: ClipPlan, cfg, contract, overscan, record, revealed_last, source):
    """Build the ``index.jsonl`` row of a finished clip: ids, split, motion (family, direction, u, physical amplitudes),
    overscan, file names, v1-compatible aliases (``mode``, ``cache_path``...), video contract, caption, source hash and
    the revealed fraction of the last frame.
    """
    return {
        "id": plan.id,
        "source_id": plan.source_id,
        "split": plan.split,
        "family": plan.family,
        "direction": plan.direction,
        "magnitude_u": plan.u,
        "yaw_deg": record["yaw_deg"],
        "pitch_deg": record["pitch_deg"],
        "zoom_ratio": record["zoom_ratio"],
        "overscan": overscan,
        "latent_file": f"clips/{plan.id}.safetensors",
        "reference_file": f"references/{plan.source_id}.safetensors",
        # --- extras (v1-trainer compatible names and bookkeeping) ---
        "mode": plan.mode,
        "record_id": plan.id,
        "cache_path": f"clips/{plan.id}.safetensors",
        "reference_path": f"references/{plan.source_id}.safetensors",
        "video_contract": asdict(contract),
        "normalization": NORMALIZATION,
        "caption": cfg["sources"]["training_caption"],
        "source_sha256": source["original_image_sha256"],
        "revealed_fraction_last": round(float(revealed_last), 5),
        "schema": SCHEMA,
    }


# ----------------------------------------------------------- file integrity
def _atomic_write_bytes_file(temp, final):
    try:
        with open(temp, "rb+") as f:
            os.fsync(f.fileno())
    except OSError:
        pass
    os.replace(temp, final)


def save_tensor_file(path, tensors, metadata):
    """Atomic: temp file in the same directory, fsync, rename."""
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    save_file(
        tensors,
        str(temp),
        metadata={"flow_cache_schema": SCHEMA, **{k: str(v) for k, v in metadata.items()}},
    )
    _atomic_write_bytes_file(temp, path)


def read_safetensors_metadata(path):
    """Header-only metadata read (no torch, works for bf16 files)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if n > 100 << 20:
            raise ValueError("Implausible safetensors header")
        header = json.loads(f.read(n))
    return header.get("__metadata__", {})


def append_jsonl(path, row):
    """Append one JSON line and fsync (the index row is only appended after the clip file is complete)."""
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path):
    """Read a JSON-lines file into a list (missing file -> empty list; blank lines skipped)."""
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # torn final line after a crash: ignore, the file is re-adopted below
            continue
    return rows


def clean_partials(cache_root):
    """Delete leftover ``*.partial`` files of an interrupted build."""
    for p in Path(cache_root).rglob("*.partial"):
        p.unlink(missing_ok=True)


def reconcile(cache_root, *, verify=True, log=print):
    """Return {id: row} of complete, verified clips; adopt orphan files.

    Sources of truth are the clip files themselves (row JSON is embedded in the
    safetensors metadata). Index rows pointing at missing/changed files are dropped.
    """
    root = Path(cache_root)
    clean_partials(root)
    indexed = {}
    for index in sorted(root.glob("index*.jsonl")):
        for row in read_jsonl(index):
            indexed[row["id"]] = row
    good, dropped, adopted = {}, 0, 0
    for row_id, row in indexed.items():
        f = root / row["latent_file"]
        ref = root / row["reference_file"]
        if not f.exists() or not ref.exists():
            dropped += 1
            continue
        if verify and file_sha256(f) != row["sha256"]:
            dropped += 1
            continue
        good[row_id] = row
    for f in sorted((root / "clips").glob("*.safetensors")) if (root / "clips").exists() else []:
        if f.stem in good:
            continue
        try:
            meta = read_safetensors_metadata(f)
            row = json.loads(meta["row_json"])
            ref = root / row["reference_file"]
            if not ref.exists():
                continue
            row["sha256"] = file_sha256(f)
            row["reference_sha256"] = file_sha256(ref)
            good[row["id"]] = row
            adopted += 1
        except Exception as error:  # corrupt/foreign file: leave for the builder to redo
            log(f"[reconcile] ignoring unreadable clip file {f.name}: {error!r}")
            f.rename(f.with_name(f.name + ".corrupt"))
    log(f"[reconcile] verified={len(good) - adopted} adopted={adopted} dropped={dropped}")
    return good


def rebuild_index(cache_root, *, verify=True, log=print):
    """Rewrite index.jsonl from the clip files (use after sharded runs)."""
    root = Path(cache_root)
    rows = reconcile(root, verify=verify, log=log)
    temp = root / "index.jsonl.partial"
    with open(temp, "w", encoding="utf-8") as stream:
        for row_id in sorted(rows):
            stream.write(json.dumps(rows[row_id], sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, root / "index.jsonl")
    return rows


def load_cached_v2(row, cache_root, *, verify=False):
    """Load one clip + its source reference (torch; same keys as v1 ``load_cached``)."""
    import torch
    from safetensors.torch import load_file

    root = Path(cache_root)
    path, reference = root / row["latent_file"], root / row["reference_file"]
    if verify and (
        file_sha256(path) != row["sha256"]
        or file_sha256(reference) != row["reference_sha256"]
    ):
        raise RuntimeError("Flow cache or source reference changed")
    data = load_file(str(path), device="cpu")
    refs = load_file(str(reference), device="cpu")
    contract = VideoContract(**row["video_contract"])
    if tuple(data["latents"].shape) != contract.latent_shape or tuple(
        refs["image_latent"].shape
    ) != (48, 1, contract.height // 16, contract.width // 16):
        raise ValueError("Cached tensors have the wrong contract")
    if tuple(data["homographies"].shape) != (contract.frames, 3, 3):
        raise ValueError("Camera path timing mismatch")
    if any(not torch.isfinite(x).all() for x in (*data.values(), *refs.values())):
        raise ValueError("Non-finite flow training cache")
    return {**data, **refs, "row": row, "contract": contract}


# ------------------------------------------------------------------- previews
def save_preview_strip(rgb, homographies, contract, path, *, frames=(0, 30, 60, 90, 120)):
    """Top row: frames; bottom row: same frames with newly revealed pixels tinted red."""
    import cv2
    from .flow_geometry import revealed_mask

    tw, th = contract.width // 4, contract.height // 4
    top, bottom = [], []
    for t in frames:
        t = min(t, len(rgb) - 1)
        small = cv2.resize(rgb[t], (tw, th), interpolation=cv2.INTER_AREA)
        mask = revealed_mask(homographies[t], contract, stride=4)[:th, :tw]
        tint = small.copy()
        tint[mask] = (0.55 * tint[mask] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
        top.append(small)
        bottom.append(tint)
    strip = np.concatenate([np.concatenate(top, 1), np.concatenate(bottom, 1)], 0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip).save(path)


# --------------------------------------------------------------------- build
def critical_config(cfg, records):
    """The config subset that must not change while a cache is being (re)built (contract, camera, sampling,
    overscan, render, seed, source-population hash); compared on resume unless ``force_config``.
    """
    return {
        "video_contract": cfg["video_contract"],
        "camera": cfg["camera"],
        "sampling": cfg["sampling"],
        "overscan": {k: cfg["overscan"][k] for k in ("policy", "fixed", "floor", "margin", "cap")},
        "render": cfg["render"],
        "seed": cfg["seed"],
        "normalization": cfg["cache"]["normalization"],
        "caption": cfg["sources"]["training_caption"],
        "population_sha256": population_sha(records),
    }


def check_disk(path, n_clips, cfg, *, force=False):
    """Estimate the cache size (clips x ``bytes_per_clip_estimate`` x safety factor + 2 GiB) and refuse to start
    if the free disk space is insufficient (unless ``force``).
    """
    per = cfg["cache"]["bytes_per_clip_estimate"]
    need = int(n_clips * per * cfg["cache"]["disk_safety_factor"]) + (2 << 30)
    Path(path).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    msg = (
        f"projected {n_clips} clips x {per / 1e6:.1f} MB = {n_clips * per / 1e9:.1f} GB "
        f"(+10% and 2 GiB margin => {need / 1e9:.1f} GB); free {free / 1e9:.1f} GB at {path}"
    )
    if free < need and not force:
        raise RuntimeError(
            "Insufficient disk: " + msg + ". Free space, lower moving_clips_per_source "
            "(K), build fewer sources (--limit/--shard), or pass --force-disk."
        )
    return msg, need, free


class Encoder:
    """Lazy VAE/T5 wrapper reusing the v1 native encode path."""

    def __init__(self, cache_root, caption, *, allow_download=False):
        import torch
        from . import flow_cache as v1
        from .wan import MODEL, REVISION, encode_text, load_encoders

        if v1.NORMALIZATION != NORMALIZATION:
            raise RuntimeError("v1/v2 normalisation labels diverged")
        self.torch, self.encode_rgb = torch, v1.encode_rgb
        self.model, self.revision = MODEL, REVISION
        log = Path(cache_root) / "encoder.log"
        try:
            self.pipe = load_encoders(log_path=log, local_files_only=not allow_download)
        except (OSError, ValueError) as error:
            print(f"[encoder] local-only load failed ({error!r}); retrying with download")
            self.pipe = load_encoders(log_path=log, local_files_only=False)
        self.text = encode_text(self.pipe, caption)
        self.negative = encode_text(self.pipe, "")
        self.pipe.text_encoder.to("cpu")
        self.pipe.text_encoder = None
        gc.collect()
        torch.cuda.empty_cache()

    def encode(self, rgb):
        """VAE-encode an RGB clip with the frozen native Wan VAE (``flow_cache.encode_rgb``)."""
        return self.encode_rgb(self.pipe.vae, rgb)

    def release(self, *objs):
        """Drop references and empty the CUDA cache between sources."""
        del objs
        gc.collect()
        self.torch.cuda.empty_cache()


def _json_line(**kw):
    print(json.dumps(kw, sort_keys=True, default=str), flush=True)


def build(
    project_root,
    cfg,
    *,
    cache_root=None,
    manifest=None,
    shard=(0, 1),
    limit=None,
    dry_run=False,
    preview_dir=None,
    preview_count=6,
    verify=True,
    force_disk=False,
    force_config=False,
    allow_download=False,
    use_lease=True,
    deadline=None,
):
    """Build (or resume) the v2 cache: plan all clips deterministically, then for this shard render each clip from real
    source pixels, VAE-encode, write ``clips/<id>.safetensors`` atomically and append the index row.

    Resumable: ``reconcile`` verifies existing files and adopts orphans; ``shard=(i, n)`` splits sources across
    processes; ``deadline`` stops cleanly between clips (``stopped_reason: deadline``); ``dry_run`` only plans and
    writes preview strips.  Takes the GPU lease unless ``use_lease`` is False.
    """
    project_root = Path(project_root)
    contract = contract_of(cfg)
    cache_root = Path(cache_root or project_root / cfg["cache"]["dir"])
    manifest = Path(manifest or project_root / cfg["sources"]["manifest"])
    records = load_sources(manifest)
    plans, sampler = plan_clips(records, cfg)
    by_source = {}
    for p in plans:
        by_source.setdefault(p.source_id, []).append(p)
    shard_i, shard_n = shard
    ordered = sorted(r["source_id"] for r in records)
    mine = [s for k, s in enumerate(ordered) if k % shard_n == shard_i]
    rec_by_id = {r["source_id"]: r for r in records}
    report = sampler.report()
    _json_line(event="plan", sources=len(records), clips=len(plans), shard=list(shard), report=report)

    if dry_run:
        return _dry_run(project_root, cfg, contract, mine, by_source, rec_by_id,
                        preview_dir, preview_count, limit)

    cache_root.mkdir(parents=True, exist_ok=True)
    snapshot = critical_config(cfg, records)
    snap_path = cache_root / "config_snapshot.json"
    if snap_path.exists() and json.loads(snap_path.read_text()) != snapshot and not force_config:
        raise RuntimeError(
            f"{snap_path} differs from the current config/population; the cache would mix "
            "incompatible clips. Use a new cache dir or --force-config."
        )
    if not snap_path.exists():
        write_json(snap_path, snapshot)

    done = reconcile(cache_root, verify=verify)
    index_path = cache_root / (
        "index.jsonl" if shard_n == 1 else f"index.shard{shard_i}of{shard_n}.jsonl"
    )
    # Rewrite this process's index to exactly the verified set it owns (drops stale rows).
    shard_rows = [done[k] for k in sorted(done) if k in {p.id for s in mine for p in by_source[s]}]
    tmp = index_path.with_name(index_path.name + ".partial")
    tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in shard_rows), encoding="utf-8")
    os.replace(tmp, index_path)

    todo_total = sum(1 for s in mine for p in by_source[s] if p.id not in done)
    todo_planned = min(todo_total, limit) if limit is not None else todo_total
    msg, _, _ = check_disk(cache_root, todo_planned, cfg, force=force_disk)
    _json_line(event="disk", message=msg, todo_clips=todo_total, will_build=todo_planned)
    if todo_planned == 0:
        _json_line(event="nothing_to_do", total_done=len(done))
        return {"built": 0, "total": len(done)}

    from contextlib import nullcontext

    from .runtime import gpu_lease

    lease = gpu_lease(project_root) if use_lease else nullcontext()
    built, started = 0, time.monotonic()
    times = []
    deadline_stop = False
    with lease:
        enc = Encoder(cache_root, cfg["sources"]["training_caption"], allow_download=allow_download)
        for sid in mine:
            todo = [p for p in by_source[sid] if p.id not in done]
            if not todo:
                continue
            if (limit is not None and built >= limit) or deadline_stop:
                break
            source = rec_by_id[sid]
            image_path = project_root / source["source_image"]
            if file_sha256(image_path) != source["original_image_sha256"]:
                raise ValueError(f"Source digest mismatch: {image_path}")
            image = Image.open(image_path).convert("RGB")
            try:
                overscan = source_overscan(image.size, contract, cfg)
            except SourceRejected as error:
                append_jsonl(cache_root / "rejected.jsonl", {"source_id": sid, "reason": str(error)})
                _json_line(event="source_rejected", source_id=sid, reason=str(error))
                continue
            ref_path = cache_root / "references" / f"{sid}.safetensors"
            first_latent = None
            for plan in todo:
                if limit is not None and built >= limit:
                    break
                if deadline is not None and should_stop(
                        time.time(), deadline, (sum(times) / len(times)) if times else 90.0):
                    deadline_stop = True
                    _json_line(event="deadline_stop", built=built)
                    print(f"DEADLINE_STOP built={built} total_done={len(done)}", flush=True)
                    break
                t0 = time.monotonic()
                try:
                    rgb, h, record, crop = render_clip(image, plan, cfg, contract, overscan)
                except ValueError as error:  # coverage assertion: reject this clip only
                    append_jsonl(cache_root / "rejected.jsonl", {"id": plan.id, "reason": str(error)})
                    _json_line(event="clip_rejected", id=plan.id, reason=str(error))
                    continue
                t_render = time.monotonic() - t0
                if first_latent is None:
                    if ref_path.exists():
                        from safetensors.torch import load_file

                        first_latent = load_file(str(ref_path), device="cpu")["image_latent"]
                    else:
                        first_latent = enc.encode(rgb[:1])
                        save_tensor_file(
                            ref_path,
                            {
                                "image_latent": first_latent,
                                "text_embeddings": enc.text,
                                "negative_text_embeddings": enc.negative,
                            },
                            {
                                "source_sha256": source["original_image_sha256"],
                                "normalization": NORMALIZATION,
                                "model": enc.model,
                                "revision": enc.revision,
                                "caption": cfg["sources"]["training_caption"],
                                "overscan": overscan,
                            },
                        )
                        png = ref_path.with_suffix(".png")
                        Image.fromarray(rgb[0]).save(png.with_name(png.name + ".partial"), format="PNG")
                        os.replace(png.with_name(png.name + ".partial"), png)
                t1 = time.monotonic()
                z = enc.encode(rgb)
                first_diff = float((z[:, :1].float() - first_latent.float()).abs().max())
                if first_diff > 0.0:
                    if first_diff > 1e-2:
                        raise RuntimeError(
                            f"Video first latent differs from image-only condition by {first_diff}"
                        )
                    _json_line(event="first_latent_small_diff", id=plan.id, max_abs=first_diff)
                row = make_row(
                    plan, cfg, contract, overscan, record,
                    revealed_fraction(h[-1], contract), source,
                )
                row["first_latent_max_abs_diff"] = first_diff
                import torch

                save_tensor_file(
                    cache_root / row["latent_file"],
                    {"latents": z, "homographies": torch.from_numpy(h).float().contiguous()},
                    {"row_json": json.dumps(row, sort_keys=True), "crop": json.dumps(crop)},
                )
                row["sha256"] = file_sha256(cache_root / row["latent_file"])
                row["reference_sha256"] = file_sha256(ref_path)
                append_jsonl(index_path, row)  # only after the clip file is complete
                done[plan.id] = row
                built += 1
                t_enc = time.monotonic() - t1
                times.append(t_render + t_enc)
                avg = sum(times) / len(times)
                remaining = todo_planned - built
                _json_line(
                    event="clip", id=plan.id, n=built, of=todo_planned,
                    s_render=round(t_render, 2), s_encode=round(t_enc, 2),
                    s_per_clip_avg=round(avg, 2),
                    eta_min=round(remaining * avg / 60, 1),
                    disk_free_gb=round(shutil.disk_usage(cache_root).free / 1e9, 1),
                )
                write_json(cache_root / "progress.json", {
                    "built_this_run": built, "total_done": len(done),
                    "s_per_clip_avg": avg, "last": plan.id,
                    "elapsed_s": time.monotonic() - started, "shard": list(shard),
                })
                del rgb, z
                enc.release()
    return {"built": built, "total": len(done), "seconds": time.monotonic() - started,
            "stopped_reason": "deadline" if deadline_stop else None}


def _dry_run(project_root, cfg, contract, mine, by_source, rec_by_id, preview_dir, preview_count, limit):
    """Render only (no VAE, no torch): coverage + revealed stats and preview strips."""
    stats, shown = [], 0
    t0 = time.monotonic()
    for sid in mine:
        source = rec_by_id[sid]
        image = Image.open(project_root / source["source_image"]).convert("RGB")
        try:
            overscan = source_overscan(image.size, contract, cfg)
        except SourceRejected as error:
            _json_line(event="source_rejected", source_id=sid, reason=str(error))
            continue
        for plan in by_source[sid]:
            if limit is not None and len(stats) >= limit:
                return {"rendered": len(stats), "stats": stats}
            t1 = time.monotonic()
            rgb, h, record, crop = render_clip(image, plan, cfg, contract, overscan)
            fr = revealed_fraction(h[-1], contract)
            stats.append({"id": plan.id, "mode": plan.mode, "u": plan.u, "overscan": overscan,
                          "revealed_last": round(fr, 4), "render_s": round(time.monotonic() - t1, 2)})
            _json_line(event="dry_clip", **stats[-1])
            if preview_dir and shown < preview_count:
                save_preview_strip(rgb, h, contract, Path(preview_dir) / f"{plan.id}.png")
                shown += 1
    return {"rendered": len(stats), "seconds": time.monotonic() - t0, "stats": stats}
