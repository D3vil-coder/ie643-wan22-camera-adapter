"""Per-frame drift and distortion analysis of generated videos (CPU): every frame, not three.

For every run folder (``video.mp4``, optional ``request.json`` / ``run.json``) it measures all frames:

* ``luma_ratio``     mean luma(t) / mean luma(0)                       (global exposure drift)
* ``da`` / ``db``    mean Lab a*, b* of frame t minus frame 0            (global chroma drift)
* ``valid_luma_ratio`` luma of the pixels that still show the first image (frame 0 warped by the request's homography)
                     over the luma of that same warped first image: drift measured on the content only, without the
                     content entering / leaving or the revealed band (needs a camera request; for a static
                     request or the frozen base model the whole frame is used)
* ``mc_resid``       motion-compensated frame-to-frame residual: mean |frame(t) - frame(t-1) warped to t| on luma
                     (Farneback flow t -> t-1 at half resolution, 8 px margin excluded; ``mc_resid_dm`` also removes the
                     global brightness change between the two frames, so a smooth dimming ramp does not look like
                     distortion). A frame whose content is
                     momentarily garbled (the user's "pixel distortion in a few frames") shows up as a spike.
* ``hf``             high-frequency energy: mean (Y - Gaussian_sigma2(Y))^2 of the luma
* spike detection    ``z_glob``: (x - median) / (1.4826 MAD) on the whole series (the plain rule);
                     ``z_run``: the same on the deviation from a same-phase running median (the VAE decodes 4 frames per
                     latent plane, so frames 1, 5, 9, ... have a different statistic in every video), robust to trends.
                     A frame is a spike when z_run > Z (default 4) and the deviation is above an absolute floor.

Outputs ``frame_stats.json`` next to the video, a curve plot ``frame_curves.png`` and a strip sheet ``frame_strip.png``
(every 10th frame, then the frames at detected spikes with their predecessor, labelled).

    python scripts/analyze_frames.py RUN_DIR [RUN_DIR ...] [--force] [--z 4] [--no-sheets]
    python scripts/analyze_frames.py --root artifacts/ff --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "fpcamera_frame_stats_v1"
MC_MARGIN = 8
RUN_WIN = 9
ABS_FLOOR = {"mc_resid": 0.4, "hf": 0.0}  # a spike must also exceed the running median by this much (gray levels)


def read_video(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames


def luma_of(rgb):
    rgb = rgb.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def lab_means(rgb):
    import cv2

    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    return float(lab[..., 1].mean()), float(lab[..., 2].mean())


def hf_energy(y, sigma=2.0):
    import cv2

    low = cv2.GaussianBlur(y, (0, 0), sigma)
    d = y - low
    return float(np.mean(d * d))


def mc_residual(prev_y, cur_y):
    """Mean |cur - warp(prev)| with Farneback flow cur -> prev, computed at half resolution."""
    import cv2

    h, w = cur_y.shape
    sy, sx = h // 2, w // 2
    a = cv2.resize(cur_y, (sx, sy), interpolation=cv2.INTER_AREA)
    b = cv2.resize(prev_y, (sx, sy), interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(a.astype(np.uint8), b.astype(np.uint8), None, 0.5, 4, 25, 3, 7, 1.5, 0)
    gx, gy = np.meshgrid(np.arange(sx, dtype=np.float32), np.arange(sy, dtype=np.float32))
    warped = cv2.remap(b, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    d = (a - warped)[MC_MARGIN:-MC_MARGIN, MC_MARGIN:-MC_MARGIN]
    # (raw mean |d|, mean |d - mean(d)|): the second removes a pure global brightness change between the two frames
    return float(np.abs(d).mean()), float(np.abs(d - d.mean()).mean())


def valid_luma_ratios(frames, y_all, homographies):
    """Per frame: mean luma over the still-valid pixels / mean luma of frame 0 warped there."""
    import cv2

    h, w = y_all[0].shape
    y0 = y_all[0].astype(np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    q = np.stack((xs, ys, np.ones_like(xs)), axis=-1)
    out = []
    for t, hom in enumerate(homographies):
        if t == 0:
            out.append(1.0)
            continue
        inv = np.linalg.inv(np.asarray(hom, dtype=np.float64))
        m = q @ inv.T
        den = np.where(np.abs(m[..., 2]) < 1e-9, 1e-9, m[..., 2])
        px, py = (m[..., 0] / den).astype(np.float32), (m[..., 1] / den).astype(np.float32)
        inside = (m[..., 2] > 1e-6) & (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)
        # erode: keep pixels 6 px inside the validity region to avoid boundary resampling effects
        inside = cv2.erode(inside.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
        if inside.sum() < 0.05 * h * w:
            out.append(float("nan"))
            continue
        ref = cv2.remap(y0, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out.append(float(y_all[t][inside].mean() / max(ref[inside].mean(), 1e-6)))
    return out


PERIOD = 4  # Wan's VAE decodes 4 video frames per latent plane: frames 1, 5, 9, ... have a different statistic


def running_median(x, win=RUN_WIN, period=PERIOD):
    """Phase-aware running median: for frame i the median of the frames i, i+-period, i+-2 period, ... (window ``win``
    frames wide on the same phase of the 4-frame latent period), robust against the periodic structure of the decoder."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    k = max(win // period // 2, 2)  # same-phase neighbours on each side
    out = np.empty(n)
    for i in range(n):
        idx = [j for j in range(i - k * period, i + k * period + 1, period) if 0 <= j < n]
        out[i] = np.median(x[idx])
    return out


def robust_z(dev):
    med = np.median(dev)
    mad = np.median(np.abs(dev - med))
    return (dev - med) / (1.4826 * mad + 1e-6)


def detect_spikes(series, z_thresh=4.0, abs_floor=0.0, skip_first=1, rel_floor=0.0):
    """Spike detection on a per-frame series. Returns (z_glob, z_run, spike_indices)."""
    x = np.asarray(series, dtype=np.float64)
    ok = np.isfinite(x)
    z_glob = np.full_like(x, np.nan)
    z_run = np.full_like(x, np.nan)
    if ok.sum() < 8:
        return z_glob, z_run, []
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x[ok] - med))
    z_glob[ok] = (x[ok] - med) / (1.4826 * mad + 1e-6)
    xf = np.where(ok, x, med)
    base = running_median(xf)
    dev = xf - base
    z_run = robust_z(dev)
    spikes = [int(i) for i in range(len(x)) if i >= skip_first and ok[i] and z_run[i] > z_thresh and dev[i] > abs_floor
              and dev[i] > rel_floor * abs(base[i])]
    return z_glob, z_run, spikes


def analyze_video(video_path, request_path=None, run_json=None, z_thresh=4.0):
    frames = read_video(video_path)
    n = len(frames)
    y_all = [luma_of(f) for f in frames]
    luma = np.array([float(y.mean()) for y in y_all])
    lab = np.array([lab_means(f) for f in frames])
    hf = np.array([hf_energy(y) for y in y_all])
    mc = np.full(n, np.nan)
    mc_dm = np.full(n, np.nan)
    for t in range(1, n):
        mc[t], mc_dm[t] = mc_residual(y_all[t - 1], y_all[t])
    valid = [float("nan")] * n
    homs, note = None, None
    adapter_on = True
    if run_json and Path(run_json).is_file():
        adapter_on = bool(json.loads(Path(run_json).read_text(encoding="utf8")).get("adapter_enabled", True))
    if request_path and Path(request_path).is_file() and adapter_on:
        homs = np.asarray(json.loads(Path(request_path).read_text(encoding="utf8"))["homographies_first_to_current"], dtype=np.float64)
        if len(homs) != n:
            homs, note = None, f"request has {len(homs)} homographies, video {n} frames"
    if homs is not None and not np.allclose(homs, np.eye(3)[None], atol=1e-9):
        valid = valid_luma_ratios(frames, y_all, homs)
    else:  # static request / frozen base model: the whole frame is content
        valid = list(luma / luma[0])
    d_luma = np.abs(np.diff(luma))
    zg_mc, zr_mc, sp_mc = detect_spikes(mc_dm, z_thresh, ABS_FLOOR["mc_resid"], rel_floor=0.15)  # spikes on the de-meaned residual
    zg_hf, zr_hf, sp_hf = detect_spikes(hf, z_thresh, ABS_FLOOR["hf"], rel_floor=0.05)
    # spikes in the frame-to-frame luma jump (flicker of the global mean)
    dl = np.concatenate([[0.0], d_luma])
    zg_dl, zr_dl, sp_dl = detect_spikes(dl, z_thresh, 0.5)

    def r(a, nd=4):
        return [None if not math.isfinite(float(v)) else round(float(v), nd) for v in a]

    valid_a = np.array(valid, dtype=np.float64)
    out = {
        "schema": SCHEMA,
        "video": str(video_path),
        "n_frames": n,
        "z_threshold": z_thresh,
        "frame0_luma": float(luma[0]),
        "luma_ratio": r(luma / luma[0]),
        "da": r(lab[:, 0] - lab[0, 0], 3),
        "db": r(lab[:, 1] - lab[0, 1], 3),
        "valid_luma_ratio": r(valid_a),
        "mc_resid": r(mc, 3),
        "mc_resid_dm": r(mc_dm, 3),
        "hf": r(hf, 3),
        "z_run_mc": r(zr_mc, 2),
        "z_glob_mc": r(zg_mc, 2),
        "z_run_hf": r(zr_hf, 2),
        "luma_jump": r(dl, 3),
        "spikes_mc": sp_mc,
        "spikes_hf": sp_hf,
        "spikes_luma_jump": sp_dl,
        "spikes_glob_mc": [int(i) for i in range(1, n) if np.isfinite(zg_mc[i]) and zg_mc[i] > z_thresh],
        "note": note,
    }
    ok = np.isfinite(valid_a)
    sel = lambda idx: float(luma[idx] / luma[0]) if idx < n else None  # noqa: E731
    out["summary"] = {
        "luma_ratio_20": sel(20), "luma_ratio_40": sel(40), "luma_ratio_80": sel(80), "luma_ratio_120": sel(n - 1),
        "luma_ratio_min": float((luma / luma[0]).min()), "luma_ratio_max": float((luma / luma[0]).max()),
        "valid_luma_ratio_min": float(np.nanmin(valid_a)) if ok.any() else None,
        "valid_luma_ratio_final": float(valid_a[-1]) if np.isfinite(valid_a[-1]) else None,
        "max_abs_da": float(np.abs(lab[:, 0] - lab[0, 0]).max()), "max_abs_db": float(np.abs(lab[:, 1] - lab[0, 1]).max()),
        "chroma_shift_final": float(np.hypot(lab[-1, 0] - lab[0, 0], lab[-1, 1] - lab[0, 1])),
        "luma_ratio_4": float(luma[4] / luma[0]) if n > 4 else None,
        "chroma_shift_4": float(np.hypot(lab[4, 0] - lab[0, 0], lab[4, 1] - lab[0, 1])) if n > 4 else None,
        "hf_frame1_over_median": float(hf[1] / max(np.median(hf[2:]), 1e-9)) if n > 2 else None,
        "max_luma_jump": float(d_luma.max()) if len(d_luma) else 0.0,
        "mc_resid_median": float(np.nanmedian(mc)), "mc_resid_max": float(np.nanmax(mc)),
        "mc_resid_dm_median": float(np.nanmedian(mc_dm)), "mc_resid_dm_max": float(np.nanmax(mc_dm)),
        "hf_first": float(hf[0]), "hf_median": float(np.median(hf)), "hf_ratio_last_first": float(hf[-1] / max(hf[0], 1e-9)),
        "n_spikes_mc": len(sp_mc), "n_spikes_hf": len(sp_hf), "n_spikes_luma_jump": len(sp_dl), "n_spikes_glob_mc": len(out["spikes_glob_mc"]),
        "spike_frames_mod4": {str(k): sum(1 for i in sp_mc if i % 4 == k) for k in range(4)},
    }
    return out, frames


def strip_sheet(frames, stats, out_path, every=10, tile_w=320, per_row=7):
    from PIL import Image, ImageDraw

    n = len(frames)
    h, w = frames[0].shape[:2]
    th = int(round(tile_w * h / w))
    reg = list(range(0, n, every))
    spk = sorted(set(stats["spikes_mc"]) | set(stats["spikes_hf"]) | set(stats["spikes_luma_jump"]))
    spk_tiles = []
    for i in spk[:14]:
        spk_tiles.extend([(i - 1, f"spike f{i - 1}"), (i, f"SPIKE f{i}")]) if i >= 1 else spk_tiles.append((i, f"SPIKE f{i}"))
    rows = [reg[k : k + per_row] for k in range(0, len(reg), per_row)]
    tiles = [[(i, f"f{i}  Y {stats['luma_ratio'][i]:.2f}") for i in row] for row in rows]
    for k in range(0, len(spk_tiles), per_row):
        tiles.append(spk_tiles[k : k + per_row])
    sheet = Image.new("RGB", (per_row * tile_w, len(tiles) * (th + 16)), (24, 24, 24))
    dr = ImageDraw.Draw(sheet)
    for ri, row in enumerate(tiles):
        for ci, (i, label) in enumerate(row):
            im = Image.fromarray(frames[i]).resize((tile_w, th), Image.LANCZOS)
            sheet.paste(im, (ci * tile_w, ri * (th + 16) + 16))
            dr.text((ci * tile_w + 4, ri * (th + 16) + 2), label, fill=(255, 230, 120) if "SPIKE" in label else (220, 220, 220))
    sheet.save(out_path)


def curves_plot(stats, out_path, title=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = stats["n_frames"]
    t = np.arange(n)
    f = lambda k: np.array([np.nan if v is None else v for v in stats[k]], dtype=float)  # noqa: E731
    fig, ax = plt.subplots(2, 3, figsize=(15, 6.5))
    ax[0, 0].plot(t, f("luma_ratio"), label="global"); ax[0, 0].plot(t, f("valid_luma_ratio"), label="valid-region")
    ax[0, 0].axhline(1, color="gray", lw=0.5); ax[0, 0].set_title("luma ratio vs frame 0"); ax[0, 0].legend()
    ax[0, 1].plot(t, f("da"), label="a*"); ax[0, 1].plot(t, f("db"), label="b*"); ax[0, 1].set_title("Lab chroma shift vs frame 0"); ax[0, 1].legend()
    ax[0, 2].plot(t, f("hf")); ax[0, 2].set_title("high-frequency energy (sigma 2)")
    ax[1, 0].plot(t, f("mc_resid"), label="raw"); ax[1, 0].plot(t, f("mc_resid_dm"), label="DC removed")
    ax[1, 0].set_title("motion-compensated residual"); ax[1, 0].legend()
    for i in stats["spikes_mc"]:
        ax[1, 0].axvline(i, color="r", alpha=0.4)
    ax[1, 1].plot(t, f("luma_jump")); ax[1, 1].set_title("|luma(t)-luma(t-1)|")
    ax[1, 2].plot(t, f("z_run_mc")); ax[1, 2].axhline(stats["z_threshold"], color="r", lw=0.6); ax[1, 2].set_title("spike z (running-median, mc)")
    for a in ax.ravel():
        a.set_xlabel("frame")
    fig.suptitle(title or Path(stats["video"]).parent.name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=70)
    plt.close(fig)


def process(run_dir, force=False, z=4.0, sheets=True):
    run_dir = Path(run_dir)
    cache = run_dir / "frame_stats.json"
    if cache.is_file() and not force:
        return json.loads(cache.read_text(encoding="utf8"))
    stats, frames = analyze_video(run_dir / "video.mp4", run_dir / "request.json", run_dir / "run.json", z)
    cache.write_text(json.dumps(stats) + "\n", encoding="utf8")
    if sheets:
        strip_sheet(frames, stats, run_dir / "frame_strip.png")
        curves_plot(stats, run_dir / "frame_curves.png", title="/".join(run_dir.parts[-2:]))
    return stats


def _worker(args):
    return process(*args)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--root", type=Path, default=None, help="every run folder (with video.mp4) below this directory")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--z", type=float, default=4.0)
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    runs = list(args.runs)
    if args.root:
        runs += sorted(p.parent for p in args.root.rglob("video.mp4"))
    if not runs:
        ap.error("no run folders")
    jobs = [(r, args.force, args.z, not args.no_sheets) for r in runs]
    if args.workers > 1 and len(jobs) > 1:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(_worker, jobs))
    else:
        res = [_worker(j) for j in jobs]
    for r, s in zip(runs, res):
        m = s["summary"]
        print(f"{'/'.join(r.parts[-2:]):40s} Y@40 {m['luma_ratio_40']:.3f} Y@120 {m['luma_ratio_120']:.3f} valid_min {m['valid_luma_ratio_min']} "
              f"chroma {m['chroma_shift_final']:.1f} maxdY {m['max_luma_jump']:.2f} spikes mc {m['n_spikes_mc']} hf {m['n_spikes_hf']} dY {m['n_spikes_luma_jump']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
