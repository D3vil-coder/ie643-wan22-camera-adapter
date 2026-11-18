"""Synthesize the flat test images of the F1 flat-image diagnosis (CPU, deterministic, no photographs).

* ``synthetic_landscape.png``: exactly the built-in landscape sample of ``notebooks/camera_studio_v2.py`` (same
  function, seed 7), so that the notebook's sample can be regenerated outside the notebook.
* ``table_object.png``: a smooth gradient wall above a plain table with one small mug (soft shadow, faint sensor
  grain). Very little texture to anchor to: the regime in which the flat-image drift is expected to be worst.

    python scripts/make_flat_inputs.py --out inputs/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def make_sample_scene(width=1280, height=704, seed=7):
    """Deterministic procedural landscape (verbatim copy of the notebook's ``make_sample_scene``)."""
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


def make_table_object(width=1280, height=704, seed=11):
    """Smooth wall gradient, plain wooden table (almost no grain), one small mug with a soft shadow."""
    rng = np.random.default_rng(seed)
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
    horizon = 0.52
    wall_top, wall_low = np.array([212, 206, 196], np.float32), np.array([188, 180, 168], np.float32)
    wall = wall_top + (wall_low - wall_top) * np.clip(ys / horizon, 0, 1)
    wall = wall + np.array([10, 8, 5], np.float32) * np.exp(-(((xs - 0.3) / 0.45) ** 2 + ((ys - 0.2) / 0.5) ** 2))
    tab_far, tab_near = np.array([150, 112, 78], np.float32), np.array([128, 92, 62], np.float32)
    table = tab_far + (tab_near - tab_far) * np.clip((ys - horizon) / (1 - horizon), 0, 1)
    table = table + np.array([9, 7, 5], np.float32) * np.sin(2 * np.pi * (xs * 1.3 + 0.2 * ys))
    img = np.where(ys > horizon, table, wall) * np.ones((1, width, 1), np.float32)
    pic = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGBA")
    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    cx, cy = int(0.38 * width), int(0.70 * height)
    sd.ellipse([cx - 70, cy + 22, cx + 130, cy + 62], fill=(30, 20, 12, 120))
    pic = Image.alpha_composite(pic, shadow.filter(ImageFilter.GaussianBlur(14)))
    d = ImageDraw.Draw(pic)
    d.rectangle([cx - 46, cy - 62, cx + 46, cy + 40], fill=(196, 64, 52, 255))
    d.ellipse([cx - 46, cy - 76, cx + 46, cy - 48], fill=(214, 92, 78, 255))
    d.ellipse([cx - 38, cy - 72, cx + 38, cy - 54], fill=(70, 38, 26, 255))
    d.ellipse([cx - 46, cy + 24, cx + 46, cy + 56], fill=(196, 64, 52, 255))
    d.arc([cx + 30, cy - 40, cx + 86, cy + 20], -80, 80, fill=(196, 64, 52, 255), width=10)
    pic = pic.filter(ImageFilter.GaussianBlur(0.9)).convert("RGB")
    grain = rng.normal(0, 2.0, (height, width, 1)).astype(np.float32)
    return Image.fromarray(np.clip(np.asarray(pic, np.float32) + grain, 0, 255).astype(np.uint8))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    make_sample_scene().save(args.out / "synthetic_landscape.png")
    make_table_object().save(args.out / "table_object.png")
    print("wrote", args.out / "synthetic_landscape.png", args.out / "table_object.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
