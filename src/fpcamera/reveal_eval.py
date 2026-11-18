"""Ground-truth reveal evaluation: generated revealed region R vs the real photograph.

Dev sources of the v2 cache are real photographs with margin (overscan). The cache builder renders the
clip ``render_clip(image, plan, cfg, contract, overscan)``; frame 0 of that clip is the first view the
model sees, and frames t > 0 show *real* content in R (it comes from the hidden source margin). We give
a model the GT first frame and the SAME homographies and compare its generated R with GT R.

Metric choices (all computed on R = ``border_metrics.revealed_mask_pixels``; V = ~R is a control):

* ``lpips``: LPIPS (AlexNet, spatial map) of the FULL generated frame vs the full GT frame, averaged
  over the pixels of the mask. Full-frame context avoids crop/mask artefacts at the R boundary; the
  spatial map (``lpips.LPIPS(net="alex", spatial=True)``) is upsampled to image size, so averaging it over
  R is a masked LPIPS. Receptive fields of AlexNet are large, so R values leak V context (documented).
* ``psnr`` / ``ssim``: MSE / Gaussian-window SSIM map averaged over the mask. Expected low in R:
  new content is not unique, so a good completion still differs pixel-wise.
* ``hist_bhatt``: Bhattacharyya distance between 16x16 Lab a*b* histograms of gen-R and GT-R
  (colour palette agreement; 0 identical, 1 disjoint).
* ``hf_dev`` = |ln clip(HF_gen_R / HF_GT_R, 0.2, 5)| (Laplacian variance, eroded 3 px): texture
  sharpness vs the real content.
* ``artifact_score_local`` (reference-free) of the generated frame, and of the GT frame for scale.

The module is pure numpy/cv2 (LPIPS is injected), so it is unit-testable on CPU.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import cv2
import numpy as np

from . import border_metrics as bm
from .flow_cache_v2 import ClipPlan, SourceRejected, plan_clips, render_clip, source_overscan

NAN = float("nan")
DEFAULT_FRAMES = (40, 80, 120)
MIN_PIXELS = 64
HIST_BINS = 16
U_BUCKETS = ((0.0, 0.5, "u<0.5"), (0.5, 0.75, "0.5<=u<0.75"), (0.75, 1.0001, "u>=0.75"))
REGION_KEYS = ("lpips", "psnr", "ssim", "hist_bhatt", "hf_dev")
SCORE_KEYS = ("artifact_score_local", "artifact_score_local_gt", "artifact_score_v2", "artifact_score_v2_gt")


# ------------------------------------------------------------------ selection
def _hash_key(seed, plan_id):
    return hashlib.sha256(f"{seed}|reveal_eval|{plan_id}".encode()).hexdigest()


def select_pairs(records, cfg, n=12, min_u=0.5, split="development"):
    """Deterministic balanced selection of (source, moving clip) dev pairs.

    Only clips with a revealed region are eligible (pan, tilt, zoom_out). Families are balanced
    (n/3 each; pan and tilt alternate between their two directions), clips prefer ``u >= min_u``,
    distinct sources are preferred, ties broken by a seeded hash of the clip id. Shortfalls are
    filled from lower-u clips of the same family. Returns a list of :class:`ClipPlan`.
    """
    plans, _ = plan_clips(records, cfg)
    dev = [p for p in plans if p.split == split and p.kind == "moving"
           and (p.family != "zoom" or p.direction == "out")]
    seed = cfg["seed"]
    per_family = {"pan": n // 3, "tilt": n // 3, "zoom": n - 2 * (n // 3)}
    chosen = []
    used_sources = set()
    for family, quota in per_family.items():
        fam = [p for p in dev if p.family == family]
        dirs = sorted({p.direction for p in fam})
        picked = []
        for relax in (False, True):  # first strict u >= min_u, then anything
            pool = [p for p in fam if p not in picked and (relax or p.u >= min_u)]
            pool.sort(key=lambda p: (-p.u if relax else 0.0, _hash_key(seed, p.id)))
            for pass_unique in (True, False):
                di = 0
                guard = 0
                while len(picked) < quota and guard < 4 * (len(pool) + 1):
                    guard += 1
                    d = dirs[di % len(dirs)] if dirs else None
                    di += 1
                    cand = [p for p in pool if p not in picked and p.direction == d
                            and (not pass_unique or p.source_id not in used_sources)]
                    if cand:
                        picked.append(cand[0])
                        used_sources.add(cand[0].source_id)
                if len(picked) >= quota:
                    break
            if len(picked) >= quota:
                break
        chosen.extend(picked[:quota])
    return chosen


def u_bucket(u):
    """Name of the strength bucket (``U_BUCKETS``) that contains ``u``."""
    for lo, hi, name in U_BUCKETS:
        if lo <= u < hi:
            return name
    return U_BUCKETS[-1][2]


# ------------------------------------------------------------------ GT render
def render_gt(image, plan, cfg, contract):
    """GT clip exactly as the cache builder renders it (same overscan policy, border mode, coverage check).

    Returns ``(rgb[F,H,W,3] uint8, homographies[F,3,3], overscan)``; ``rgb[0]`` is the first view.
    Raises :class:`SourceRejected` if the source is not usable (cache builder rejects it too).
    """
    overscan = source_overscan(image.size, contract, cfg)
    rgb, h, _record, _crop = render_clip(image, plan, cfg, contract, overscan)
    return rgb, h, overscan


def camera_request_for(homographies, contract):
    """``flow_infer.generate(camera_request=...)`` dict that replays exactly these homographies."""
    return {
        "homographies_first_to_current": np.asarray(homographies, dtype=np.float64).tolist(),
        "contract": {"width": contract.width, "height": contract.height,
                     "frames": contract.frames, "fps": contract.fps},
    }


# --------------------------------------------------------------------- metrics
def ssim_map(a, b):
    """Mean-over-channels SSIM map, 11x11 Gaussian window (sigma 1.5), uint8/float RGB in [0,255]."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    def blur(x):
        """Gaussian window (11x11, sigma 1.5) used by the SSIM map."""
        return cv2.GaussianBlur(x, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
    out = np.zeros(a.shape[:2])
    for c in range(a.shape[2]):
        x, y = a[..., c], b[..., c]
        mx, my = blur(x), blur(y)
        sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
        out += ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return out / a.shape[2]


def lab_ab_hist(rgb, mask, bins=HIST_BINS):
    """Normalised 2-D histogram of the Lab a*b* chroma of the masked pixels (palette descriptor)."""
    lab = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB)
    ab = lab[mask][:, 1:3].astype(np.float64)
    hist, _, _ = np.histogram2d(ab[:, 0], ab[:, 1], bins=bins, range=((0, 256), (0, 256)))
    return hist / max(hist.sum(), 1.0)


def bhattacharyya(p, q):
    """Bhattacharyya distance between two normalised histograms (0 identical, 1 disjoint)."""
    return float(math.sqrt(max(0.0, 1.0 - float(np.sqrt(p * q).sum()))))


def region_metrics(gen, gt, mask, lpips_map=None, ssim=None):
    """Metrics of generated vs GT frame over ``mask`` (bool [H,W])."""
    mask = np.asarray(mask, dtype=bool)
    nan = {k: NAN for k in REGION_KEYS}
    if mask.sum() < MIN_PIXELS:
        return nan
    g, t = np.asarray(gen, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    mse = float(((g[mask] - t[mask]) ** 2).mean())
    psnr = 99.0 if mse < 1e-12 else float(10 * math.log10(255.0**2 / mse))
    ssim = ssim_map(g, t) if ssim is None else ssim
    hist = bhattacharyya(lab_ab_hist(gen, mask), lab_ab_hist(gt, mask))
    h_gen, h_gt = bm._hf(bm._luma(g), mask), bm._hf(bm._luma(t), mask)
    ratio = bm._ratio(h_gen, h_gt)
    hf_dev = abs(math.log(min(max(ratio, 0.2), 5.0))) if math.isfinite(ratio) else NAN
    return {
        "lpips": float(lpips_map[mask].mean()) if lpips_map is not None else NAN,
        "psnr": psnr,
        "ssim": float(ssim[mask].mean()),
        "hist_bhatt": hist,
        "hf_dev": hf_dev,
    }


def frame_eval(gen, gt, revealed, lpips_map=None):
    """Per-frame record: R and V (control) region metrics plus reference-free scores."""
    r = np.asarray(revealed, dtype=bool)
    ssim = ssim_map(gen, gt)
    gen_m, gt_m = bm.frame_metrics(gen, r), bm.frame_metrics(gt, r)
    return {
        "R_fraction": float(r.mean()),
        "R": region_metrics(gen, gt, r, lpips_map, ssim),
        "V": region_metrics(gen, gt, ~r, lpips_map, ssim),
        "artifact_score_local": gen_m["artifact_score_local"],
        "artifact_score_local_gt": gt_m["artifact_score_local"],
        "artifact_score_v2": gen_m["artifact_score_v2"],
        "artifact_score_v2_gt": gt_m["artifact_score_v2"],
    }


def evaluate_video(gen_frames, gt_frames, homographies, frames, lpips_fn=None, min_fraction=0.01):
    """Evaluate ``frames`` (indices) of one generated video against GT.

    ``gen_frames``/``gt_frames``: dict or sequence indexable by frame index (uint8 RGB [H,W,3]).
    ``lpips_fn(gen, gt) -> map[H,W]`` (optional). Returns ``{"frames": {t: rec}, "mean": {...}}`` where
    ``mean`` averages (NaN-ignoring) the frames with R fraction > ``min_fraction``.
    """
    h = np.asarray(homographies, dtype=np.float64)
    per = {}
    for t in frames:
        height, width = gt_frames[t].shape[:2]
        mask = bm.revealed_mask_pixels(h, width, height, frame_indices=[t])[0]
        lmap = lpips_fn(gen_frames[t], gt_frames[t]) if lpips_fn is not None else None
        per[int(t)] = frame_eval(gen_frames[t], gt_frames[t], mask, lmap)
    return {"frames": per, "mean": mean_over_frames(per, min_fraction)}


def mean_over_frames(per, min_fraction=0.01):
    """Average per-frame metric dicts over frames whose revealed fraction exceeds ``min_fraction``
    (frames with almost no revealed region carry no signal).
    """
    rows = [r for r in per.values() if r["R_fraction"] > min_fraction]
    out = {"n_frames": len(rows), "R": {}, "V": {}}
    if not rows:
        return out
    def nm(vals):
        """NaN-safe mean of a list of per-frame values."""
        a = np.array(vals, dtype=np.float64)
        return float(np.nanmean(a)) if np.isfinite(a).any() else NAN
    for region in ("R", "V"):
        for k in REGION_KEYS:
            out[region][k] = nm([r[region][k] for r in rows])
    for k in SCORE_KEYS:
        out[k] = nm([r[k] for r in rows])
    out["R_fraction"] = nm([r["R_fraction"] for r in rows])
    return out


# ------------------------------------------------------------------- aggregate
def _flat(mean):
    flat = {f"R_{k}": v for k, v in mean.get("R", {}).items()}
    flat.update({f"V_{k}": v for k, v in mean.get("V", {}).items()})
    for k in SCORE_KEYS + ("R_fraction",):
        if k in mean:
            flat[k] = mean[k]
    return flat


def aggregate(records):
    """records: list of dict(model, pair, family, direction, u, mean). Mean over pairs, per model /
    family / magnitude bucket (NaN-ignoring). Returns nested dict with counts."""
    def reduce(rows):
        """Mean of every metric over the pairs of a group (NaN values ignored)."""
        keys = sorted({k for r in rows for k in _flat(r["mean"])})
        out = {"n_pairs": len(rows)}
        for k in keys:
            a = np.array([_flat(r["mean"]).get(k, NAN) for r in rows], dtype=np.float64)
            a = a[np.isfinite(a)]
            out[k] = float(a.mean()) if a.size else NAN
            out[k + "_std"] = float(a.std(ddof=1)) if a.size > 1 else NAN
        return out
    models = sorted({r["model"] for r in records})
    res = {}
    for m in models:
        rows = [r for r in records if r["model"] == m and r["mean"].get("n_frames", 0) > 0]
        entry = {"all": reduce(rows) if rows else {"n_pairs": 0}, "by_family": {}, "by_magnitude": {}}
        for fam in sorted({r["family"] for r in rows}):
            entry["by_family"][fam] = reduce([r for r in rows if r["family"] == fam])
        for _lo, _hi, name in U_BUCKETS:
            sub = [r for r in rows if u_bucket(r["u"]) == name]
            if sub:
                entry["by_magnitude"][name] = reduce(sub)
        res[m] = entry
    return res


TABLE_COLS = (
    ("R_lpips", "LPIPS_R"), ("R_psnr", "PSNR_R"), ("R_ssim", "SSIM_R"), ("R_hist_bhatt", "hist_R"),
    ("R_hf_dev", "HFdev_R"), ("artifact_score_local", "art_local"), ("artifact_score_local_gt", "art_local_GT"),
    ("V_lpips", "LPIPS_V"), ("V_psnr", "PSNR_V"),
)


def _f(v):
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.3f}"


def markdown(agg):
    """Markdown tables (all / per family / per magnitude bucket), one row per model."""
    heads = ["group", "model", "n"] + [c[1] for c in TABLE_COLS]
    lines = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for m, entry in agg.items():
        groups = [("all", entry["all"])]
        groups += [(f"family={k}", v) for k, v in entry["by_family"].items()]
        groups += [(f"magnitude={k}", v) for k, v in entry["by_magnitude"].items()]
        for name, row in groups:
            cells = [name, m, str(row.get("n_pairs", 0))] + [_f(row.get(k)) for k, _ in TABLE_COLS]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- contact sheet
def outline_mask(rgb, mask, color=(255, 0, 0), thickness=2):
    """Copy of ``rgb`` with the outline of ``mask`` drawn (for the qualitative reveal sheets)."""
    out = np.ascontiguousarray(rgb).copy()
    contours, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def contact_sheet(rows, homographies, frames, out_path, tile_w=360, label_w=110):
    """rows: list of (label, {t: rgb}); columns = frames; R outlined in red. Saves a PNG."""
    from PIL import Image, ImageDraw

    h0, w0 = next(iter(rows[0][1].values())).shape[:2]
    tile_h = int(round(tile_w * h0 / w0))
    masks = {t: bm.revealed_mask_pixels(homographies, w0, h0, frame_indices=[t])[0] for t in frames}
    sheet = Image.new("RGB", (label_w + tile_w * len(frames), (tile_h + 18) * len(rows) + 18), "white")
    draw = ImageDraw.Draw(sheet)
    for j, t in enumerate(frames):
        draw.text((label_w + j * tile_w + 4, 3), f"frame {t}", fill="black")
    for i, (label, fr) in enumerate(rows):
        y = 18 + i * (tile_h + 18)
        draw.text((4, y + tile_h // 2), label, fill="black")
        for j, t in enumerate(frames):
            tile = Image.fromarray(outline_mask(fr[t], masks[t], thickness=3)).resize((tile_w, tile_h), Image.BILINEAR)
            sheet.paste(tile, (label_w + j * tile_w, y))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


# -------------------------------------------------------------------- LPIPS
class LpipsScorer:
    """Spatial LPIPS (AlexNet) map for uint8 RGB frames; requires ``pip install lpips``."""

    def __init__(self, device="cpu"):
        import lpips
        import torch

        self.torch = torch
        self.device = device
        self.net = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(device).eval()

    def __call__(self, gen, gt):
        torch = self.torch

        def prep(x):
            """HWC uint8 -> NCHW float in [-1, 1] on the scorer's device."""
            return (torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float() / 127.5 - 1.0).to(self.device)

        with torch.no_grad():
            m = self.net(prep(gen), prep(gt))  # [1,1,h,w]
            if m.shape[-2:] != gen.shape[:2]:
                m = torch.nn.functional.interpolate(m, size=gen.shape[:2], mode="bilinear", align_corners=False)
        return m[0, 0].cpu().numpy().astype(np.float64)
