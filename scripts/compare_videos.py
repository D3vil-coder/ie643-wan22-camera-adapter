"""Per-frame comparison of two videos of equal size (CPU): PSNR, mean absolute difference, optional sheet.

Used to check that an opt-in speed option (``--attn-backend cudnn``) leaves the generated video visually
equivalent to the default path: same seed, same inputs, only the attention kernel differs, so differences
come from bf16 kernel noise amplified through the 50 sampling steps.

    python scripts/compare_videos.py A.mp4 B.mp4 --labels auto cudnn --json cmp.json --sheet cmp.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_all(path):
    """Decode every frame of a video as RGB uint8 arrays."""
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


def psnr(a, b):
    """PSNR in dB between two uint8 images (inf if identical)."""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def compare(path_a, path_b):
    """Per-frame PSNR and mean absolute difference of two equal-size videos (over their common length)."""
    fa, fb = read_all(path_a), read_all(path_b)
    n = min(len(fa), len(fb))
    ps = np.array([psnr(fa[i], fb[i]) for i in range(n)])
    mad = np.array([float(np.mean(np.abs(fa[i].astype(np.int16) - fb[i].astype(np.int16)))) for i in range(n)])
    finite = ps[np.isfinite(ps)]
    out = {
        "n_frames_a": len(fa), "n_frames_b": len(fb), "n_compared": n,
        "psnr_mean": float(finite.mean()) if len(finite) else float("inf"),
        "psnr_min": float(finite.min()) if len(finite) else float("inf"),
        "psnr_first": float(ps[0]), "psnr_last": float(ps[-1]),
        "psnr_by_frame_every_10": {str(i): float(ps[i]) for i in range(0, n, 10)},
        "mean_abs_diff_mean": float(mad.mean()), "mean_abs_diff_max": float(mad.max()),
    }
    return out, fa, fb


def sheet(fa, fb, labels, out_path, frames=(0, 40, 80, 120), tile_w=420, gain=4):
    """Write a contact sheet (A, B and an amplified |A-B| difference) for a few frames."""
    from PIL import Image, ImageDraw

    h, w = fa[0].shape[:2]
    th = round(tile_w * h / w)
    rows = [(labels[0], lambda i: fa[i]), (labels[1], lambda i: fb[i]),
            (f"|diff| x{gain}", lambda i: np.clip(np.abs(fa[i].astype(np.int16) - fb[i].astype(np.int16)) * gain, 0, 255).astype(np.uint8))]
    canvas = Image.new("RGB", (110 + len(frames) * (tile_w + 4), 22 + len(rows) * (th + 4)), (24, 24, 28))
    d = ImageDraw.Draw(canvas)
    for c, f in enumerate(frames):
        d.text((110 + c * (tile_w + 4) + 4, 4), f"frame {f}", fill=(220, 220, 225))
    for r, (name, fn) in enumerate(rows):
        d.text((6, 22 + r * (th + 4) + th // 2 - 6), name, fill=(240, 240, 245))
        for c, f in enumerate(frames):
            if f < min(len(fa), len(fb)):
                canvas.paste(Image.fromarray(fn(f)).resize((tile_w, th), Image.Resampling.LANCZOS),
                             (110 + c * (tile_w + 4), 22 + r * (th + 4)))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main(argv=None):
    """CLI entry; prints the comparison and optionally writes the sheet."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--sheet", type=Path, default=None)
    args = ap.parse_args(argv)
    res, fa, fb = compare(args.a, args.b)
    print(json.dumps(res, indent=2))
    if args.json:
        args.json.write_text(json.dumps(res, indent=2) + "\n", encoding="utf8")
    if args.sheet:
        sheet(fa, fb, args.labels, args.sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
