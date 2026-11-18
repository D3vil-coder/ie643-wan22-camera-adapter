"""Build the raster figures of the IE 643 Submission 2 report and poster.

Inputs (local, NOT in git): the E1 / G1 evaluation outputs under ``CKPT_ROOT`` (``$FPCAMERA_CHECKPOINTS_DIR``, else
``<repo root>/artifacts/checkpoints/v2``; the folders ``eval_E1`` and ``G1`` inside it hold videos, validation.json,
calibration rows).  Outputs: downscaled PNGs in ``docs/submission/figures/`` (these ARE committed, so the
report and the poster can be rebuilt without the evaluation folders).

    python docs/submission/build/make_figures.py                # everything
    python docs/submission/build/make_figures.py crops plots    # selected parts

Parts: a4_crops / a4_plot (figures of the A4 poster), crops (v1 | v1+replicate | v2 border crops), strength (slider demonstration), plots (validation loss and
calibration), overview (frame-120 overview of the E1 cases), thumbs (RGB analogue of the guide, used by the
method diagram).

Only COCO and gallery cases are used.  The personal-photo runs and the ball_* cases (brightness defect) are
deliberately not touched.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "docs" / "submission" / "figures"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

CKPT_ROOT = Path(os.environ.get("FPCAMERA_CHECKPOINTS_DIR") or REPO / "artifacts" / "checkpoints" / "v2")
E1 = CKPT_ROOT / "eval_E1"
G1 = CKPT_ROOT / "G1"

NAVY = "#0b1f6b"
GREEN = "#1a7a2e"
RED = "#b3382c"
AMBER = "#c9811a"
GRAY = "#6f6f6f"


def font(size, bold=False):
    names = ("arialbd.ttf", "segoeuib.ttf") if bold else ("arial.ttf", "segoeui.ttf")
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default()


def outline_inner(rgb, mask, color=(255, 40, 40), thickness=2):
    """Contour of R, but not along the image border (the mask is padded with ones before tracing)."""
    import cv2

    img = np.ascontiguousarray(rgb.copy())
    m = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=1)
    if mask.any():
        cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cs = [c - 1 for c in cs]
        cv2.drawContours(img, cs, -1, color, thickness)
    return img


def frame_and_mask(run_dir, frame):
    import render_reveal_sheets as r

    fr = r.read_frames(run_dir / "video.mp4", [frame])[frame]
    mask = r.revealed_masks(run_dir, [frame], fr.shape[1::-1])[frame]
    return fr, mask, outline_inner


def save_png(img, name, max_w=None, optimize=True):
    OUT.mkdir(parents=True, exist_ok=True)
    if max_w and img.width > max_w:
        img = img.resize((max_w, round(img.height * max_w / img.width)), Image.Resampling.LANCZOS)
    path = OUT / name
    img.save(path, optimize=optimize)
    print(f"wrote {path.name} {img.size} {path.stat().st_size / 1e3:.0f} kB")
    return path


# --------------------------------------------------------------------------------------
def build_crops(report=False, poster=False):
    """3 cases x 3 models (report variant: 2 cases, flatter tiles), native-resolution crops at frame 120, R outlined."""
    models = [
        ("v1 (zero-filled guide)", "v1_runs"),
        ("v1 + replicate fill (inference only)", "v1rep_runs"),
        ("v2 @ 415 updates", "showcase/matched/v2"),
    ]
    cases = [
        ("heldout_zoom_out", "zoom out x0.89", (0, 0, 520, 300)),
        ("computers_pan_left", "pan left 2.5 deg", (0, 60, 520, 360)),
        ("motorcycle_pan_right", "pan right 2.5 deg", (760, 170, 1280, 470)),
    ]
    if report:
        cases = cases[:2]
        cases = [(c, l, (x0, y0, x1, y0 + 185)) for c, l, (x0, y0, x1, y1) in cases]
    tw, th, pad, head, lab = 520, (185 if report else 300), 8, 44, (170 if report else 200)
    if poster:  # text-free tiles; labels are set in HTML
        head, lab = 0, 0
    W = lab + len(models) * (tw + pad) + pad
    H = head + len(cases) * (th + pad) + pad
    cv = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(cv)
    for j, (mname, _) in enumerate(models):
        if not poster:
            d.text((lab + pad + j * (tw + pad) + 4, 8), mname, fill=NAVY if j == 2 else "#222", font=font(21, bold=True))
    for i, (case, label, (x0, y0, x1, y1)) in enumerate(cases):
        y = head + i * (th + pad)
        if not poster:
            d.text((6, y + th // 2 - 26), label, fill="#222", font=font(21, bold=True))
            d.text((6, y + th // 2 + 4), "(" + case.split("_")[0] + ")", fill=GRAY, font=font(19))
        for j, (_, sub) in enumerate(models):
            fr, mask, outline = frame_and_mask(E1 / sub / case, 120)
            marked = outline(fr, mask, color=(255, 40, 40), thickness=2)
            tile = Image.fromarray(marked[y0:y1, x0:x1])
            cv.paste(tile, (lab + pad + j * (tw + pad), y))
    return save_png(cv, "border_crops_report.png" if report else ("border_crops_poster.png" if poster else "border_crops_f120.png"))


def build_strength(poster=False):
    """Slider demonstration: frame 120 for strength 0 (input), a matched mid amplitude and u = 1."""
    rows = [
        ("pan right", "motorcycle", ["showcase/matched/v2/motorcycle_pan_right", "showcase/sweep/motorcycle_pan_right/s1.00"],
         ["s = 0  (input)", "2.5 deg  (u = 0.31)", "8 deg  (u = 1)"]),
        ("zoom out", "food", ["showcase/matched/v2/heldout_zoom_out", "showcase/sweep/heldout_zoom_out/s1.00"],
         ["s = 0  (input)", "x0.89  (u = 0.51)", "x0.80  (u = 1)"]),
    ]
    tw, th, pad, head, lab = 480, 264, 8, 38, 130
    if poster:
        head, lab = 0, 0
    W = lab + 3 * (tw + pad) + pad
    H = head + len(rows) * (th + pad) + pad
    cv = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(cv)
    import render_reveal_sheets as r

    for j, tt in enumerate(rows[0][3]):
        if not poster:
            d.text((lab + pad + j * (tw + pad) + 4, 8), tt, fill="#222", font=font(20, bold=True))
    for i, (label, sub, runs, _) in enumerate(rows):
        y = head + i * (th + pad)
        if not poster:
            d.text((6, y + th // 2 - 24), label, fill="#222", font=font(22, bold=True))
            d.text((6, y + th // 2 + 6), f"({sub})", fill=GRAY, font=font(19))
        first = r.read_frames(E1 / runs[0] / "video.mp4", [0])[0]
        imgs = [first]
        for p in runs:
            fr, mask, _ = frame_and_mask(E1 / p, 120)
            imgs.append(outline_inner(fr, mask, color=(255, 40, 40), thickness=3))
        for j, im in enumerate(imgs):
            cv.paste(Image.fromarray(im).resize((tw, th), Image.Resampling.LANCZOS), (lab + pad + j * (tw + pad), y))
    return save_png(cv, "strength_poster.png" if poster else "strength_demo.png")


def build_overview():
    """Frame-120 overview (food, door, computers) for v1 | v1+rep | v2, downscaled."""
    cases = ["heldout_zoom_out", "motorcycle_pan_right", "computers_pan_left"]
    models = [("v1", "v1_runs"), ("v1 + replicate fill", "v1rep_runs"), ("v2 @ 415", "showcase/matched/v2")]
    tw, th, pad, head = 560, 308, 6, 36
    W = len(models) * (tw + pad) + pad
    H = head + len(cases) * (th + pad) + pad
    cv = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(cv)
    for j, (n, _) in enumerate(models):
        d.text((pad + j * (tw + pad) + 4, 6), n, fill=NAVY if j == 2 else "#222", font=font(22, bold=True))
    for i, case in enumerate(cases):
        for j, (_, sub) in enumerate(models):
            fr, mask, outline = frame_and_mask(E1 / sub / case, 120)
            marked = outline(fr, mask, color=(255, 40, 40), thickness=3)
            cv.paste(Image.fromarray(marked).resize((tw, th), Image.Resampling.LANCZOS),
                     (pad + j * (tw + pad), head + i * (th + pad)))
    return save_png(cv, "overview_f120.png")


# --------------------------------------------------------------------------------------
def _load_plot_data():
    v = json.load(open(E1 / "validation" / "validation.json"))["checkpoints"]
    order = [("v1_step0", 0), ("77", 77), ("201", 201), ("250", 250), ("415", 415)]
    steps = [s for _, s in order]

    def series(split, key):
        return [v[k]["summary"][split]["overall"][key] for k, _ in order]

    rows = json.load(open(G1 / "calibration_rows.json"))
    cal = json.load(open(REPO / "configs" / "strength_calibration_v2.json"))
    return steps, series("dev", "lb_R"), series("train_subset", "lb_R"), series("dev", "la"), series("train_subset", "la"), rows, cal


def _panel_val(ax, steps, dev, tr, fs, title, ylab=None, legend=True):
    ax.plot(steps, dev, "-o", color=NAVY, lw=1.8, ms=4.5, label="held-out (28 dev)")
    ax.plot(steps, tr, "--s", color=AMBER, lw=1.5, ms=4, label="train subset (28)")
    ax.set_title(title, fontsize=fs + 1, color="#222")
    ax.set_xlabel("update (0 = v1 adapter)", fontsize=fs)
    ax.set_xlim(-30, 500)
    ax.set_xticks([0, 77, 250, 415])
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=fs - 0.5)
    ax.annotate(f"{dev[-1]:.3f}", (415, dev[-1]), textcoords="offset points", xytext=(5, -3), fontsize=fs, color=NAVY)
    if legend:
        ax.legend(fontsize=fs - 0.5, frameon=False, loc="upper right")


def _panel_gain(ax, rows, fs, title):
    col = {"pan": NAVY, "tilt": GREEN, "zoom": AMBER}
    mk = {"pan": "o", "tilt": "s", "zoom": "^"}
    ax.axhline(1.0, color="#999", lw=0.9, ls="--")
    for fam in ("pan", "tilt", "zoom"):
        pts = [(r["u"], r["gain"]) for r in rows if r["family"] == fam]
        ax.scatter(*zip(*pts), s=22, color=col[fam], marker=mk[fam], alpha=0.8, label=f"{fam} (n={len(pts)})", linewidths=0)
    ax.set_xlabel(r"requested strength $u$", fontsize=fs)
    ax.set_ylabel(r"tracker gain $g$", fontsize=fs)
    ax.set_title(title, fontsize=fs + 1, color="#222")
    ax.set_xlim(0.1, 1.08)
    ax.legend(fontsize=fs - 1, frameon=False, loc="upper right", ncol=1, handletextpad=0.1, borderaxespad=0.2, labelspacing=0.2)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=fs - 0.5)


def _panel_map(ax, cal, fs, title):
    ax.plot([0, 1], [0, 1], color="#999", lw=1.0, ls="--", label="identity")
    for i, (dname, e) in enumerate(cal["directions"].items()):
        ax.plot(e["s"], e["u"], color=NAVY, lw=1.2, alpha=0.75, label="6 fitted maps" if i == 0 else None)
    ax.set_xlabel(r"slider $s$", fontsize=fs)
    ax.set_ylabel(r"requested $u(s)$", fontsize=fs)
    ax.set_title(title, fontsize=fs + 1, color="#222")
    ax.legend(fontsize=fs - 1, frameon=False, loc="lower right", handletextpad=0.4, labelspacing=0.2, borderaxespad=0.1)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=fs - 0.5)
    ax.set_aspect("equal", adjustable="box")


def build_plots():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7})
    steps, dev_b, tr_b, dev_a, tr_a, rows, cal = _load_plot_data()
    OUT.mkdir(parents=True, exist_ok=True)

    # report: three small panels in one row (17.8 cm wide at 100 %)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.62), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    _panel_val(axs[0], steps, dev_b, tr_b, 7, r"(a) revealed-region loss $L_B|_R$")
    _panel_gain(axs[1], rows, 7, "(b) delivered / requested motion")
    _panel_map(axs[2], cal, 7, r"(c) fitted maps $u(s)$")
    fig.tight_layout(pad=0.4, w_pad=0.8)
    fig.savefig(OUT / "results_panels.png", dpi=300)
    plt.close(fig)

    # poster: three panels in one row, large type
    fig, axs = plt.subplots(1, 3, figsize=(10.0, 3.7), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    _panel_val(axs[0], steps, dev_b, tr_b, 14, r"(a) $L_B|_R$, revealed region")
    _panel_gain(axs[1], rows, 14, "(b) delivered / requested")
    _panel_map(axs[2], cal, 14, r"(c) maps $u(s)$")
    fig.tight_layout(pad=0.5, w_pad=0.8)
    fig.savefig(OUT / "results_poster.png", dpi=200)
    plt.close(fig)
    # (older stand-alone variants, not used by the poster)
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    _panel_val(ax, steps, dev_b, tr_b, 14, r"revealed-region loss $L_B|_R$ (lower is better)")
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT / "validation_poster.png", dpi=200)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.1), gridspec_kw={"width_ratios": [1.25, 1]})
    _panel_gain(axs[0], rows, 14, "delivered / requested motion")
    _panel_map(axs[1], cal, 14, r"fitted PAV maps $u(s)$")
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT / "calibration_poster.png", dpi=200)
    plt.close(fig)
    print("wrote results_panels.png, validation_poster.png, calibration_poster.png")


def build_a4_crops():
    """A4 poster: the first two rows (food zoom-out, computers pan left) of the text-free poster crops (no local data needed)."""
    src = Image.open(OUT / "border_crops_poster.png").convert("RGB")
    th, pad = 300, 8
    crop = src.crop((0, 0, src.width, 2 * th + pad))
    return save_png(crop, "border_crops_a4.png")


def build_a4_plot():
    """A4 poster: held-out loss and tracker gain in two small panels, drawn at the size they are printed (3.62 in wide)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
    steps, dev_b, tr_b, _, _, rows, _ = _load_plot_data()
    fig, axs = plt.subplots(1, 2, figsize=(3.62, 1.5), gridspec_kw={"width_ratios": [1.0, 1.0]})
    _panel_val(axs[0], steps, dev_b, tr_b, 7, r"(a) held-out $L_B|_R$")
    axs[0].set_xlabel("update (0 = v1)", fontsize=7)
    _panel_gain(axs[1], rows, 7, "(b) delivered / requested")
    axs[1].set_ylabel(r"tracker gain $g$", fontsize=7, labelpad=1)
    fig.tight_layout(pad=0.3, w_pad=0.9)
    fig.savefig(OUT / "results_a4.png", dpi=300)
    plt.close(fig)
    print("wrote results_a4.png")


# --------------------------------------------------------------------------------------
def build_thumbs():
    """RGB analogue of the guide for the method diagram (the real guide lives in the 48-channel latent space)."""
    import cv2

    src = np.array(Image.open(E1 / "showcase" / "matched" / "v2" / "heldout_zoom_out" / "input.png").convert("RGB"))
    h, w = src.shape[:2]
    s = 0.80  # zoom-out x0.80 = u = 1
    c = np.array([(w - 1) / 2, (h - 1) / 2])
    M = np.array([[s, 0, c[0] * (1 - s)], [0, s, c[1] * (1 - s)]], dtype=np.float32)
    gray = tuple(int(v) for v in (128, 128, 128))
    zero = cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=gray)
    rep = cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpAffine(np.ones((h, w), np.uint8) * 255, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    tw = 480
    th = round(tw * h / w)
    outd = HERE.parent / "assets"
    outd.mkdir(exist_ok=True)
    for name, arr in (("guide_src", src), ("guide_zero", zero), ("guide_rep", rep), ("guide_valid", np.stack([valid] * 3, -1))):
        Image.fromarray(arr).resize((tw, th), Image.Resampling.LANCZOS).save(outd / f"{name}.png", optimize=True)
    # weight maps: 1 inside, alpha_R=0.1 (pass A) / 1+beta=5 (pass B) on R
    R = (valid == 0).astype(np.float32)
    wa = (1 - R) + 0.1 * R
    wb = 1 + 4 * R
    for name, arr, vmax in (("weight_A", wa, 1.0), ("weight_B", wb, 5.0)):
        g = (255 * (arr / vmax) ** 0.8).astype(np.uint8)
        rgb = np.stack([g, g, g], -1)
        Image.fromarray(rgb).resize((tw, th), Image.Resampling.NEAREST).save(outd / f"{name}.png", optimize=True)
    print("wrote guide / weight thumbnails to", outd)


PARTS = {"crops": build_crops, "crops_poster": lambda: build_crops(poster=True), "strength_poster": lambda: build_strength(poster=True), "crops_report": lambda: build_crops(report=True), "strength": build_strength, "overview": build_overview, "plots": build_plots, "thumbs": build_thumbs, "a4_crops": build_a4_crops, "a4_plot": build_a4_plot}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(PARTS)
    for p in todo:
        PARTS[p]()
