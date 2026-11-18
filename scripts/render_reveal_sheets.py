"""Contact sheets with the revealed region R outlined, and native-resolution border crops (CPU, local).

R is recomputed per frame from the run's recorded first->current homographies (``request.json``), exactly as
``scripts/measure_border_quality.py`` does (``border_metrics.revealed_mask_pixels``). Rows are run folders
(``video.mp4`` + ``request.json``), given as ``LABEL=RUN_DIR``.

    # frames 0/60/120, R outlined in red, one row per run
    python scripts/render_reveal_sheets.py sheet --out sheet.png --frames 0 60 120 \\
        "pan_left s=1=G1/userphoto/full/pan_left/s1.00" "pan_right s=1=G1/userphoto/attn_auto/pan_right/s1.00"

    # native-resolution crops of the revealed band at one frame, models side by side
    python scripts/render_reveal_sheets.py crops --out crops.png --frame 120 \\
        "v1=E1/v1_runs/computers_pan_left" "v2=E1/showcase/matched/v2/computers_pan_left"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import measure_border_quality as mbq  # noqa: E402
from fpcamera import border_metrics as bm  # noqa: E402

RED = (255, 40, 40)


def parse_rows(items):
    """Parse ``LABEL=RUN_DIR`` items into ``(label, path)`` rows."""
    rows = []
    for it in items:
        label, sep, path = it.rpartition("=")
        if not sep or not label:
            raise SystemExit(f"expected LABEL=RUN_DIR, got {it!r}")
        rows.append((label, Path(path)))
    return rows


def read_frames(video, indices):
    """Decode the listed frame indices of a video as RGB arrays."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    wanted, out, i, last = set(indices), {}, 0, max(indices)
    while i <= last:
        ok, bgr = cap.read()
        if not ok:
            break
        if i in wanted:
            out[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    missing = wanted - set(out)
    if missing:
        raise RuntimeError(f"{video}: frames {sorted(missing)} not decodable")
    return out


def revealed_masks(run_dir, frames, size):
    """Revealed-region masks R of a run at the given frames (from its recorded homographies)."""
    h, _, _, _ = mbq.load_homographies(run_dir)
    width, height = size
    return {t: bm.revealed_mask_pixels(h, width, height, frame_indices=[t])[0] for t in frames if t < len(h)}


def outline(rgb, mask, color=RED, thickness=2):
    """Draw the contour of ``mask`` onto a copy of ``rgb`` (no-op for an empty mask)."""
    import cv2

    img = np.ascontiguousarray(rgb.copy())
    m = mask.astype(np.uint8)
    if m.any():
        cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, color, thickness)
    return img


def _font(size):
    from PIL import ImageFont

    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def sheet(rows, frames, out_path, tile_w=420, label_w=190, outlined=True, thickness=2):
    """Contact sheet of several runs at the same frames, optionally with R outlined."""
    from PIL import Image, ImageDraw

    tile_h = round(tile_w * 704 / 1280)
    pad, head = 4, 26
    width = label_w + len(frames) * (tile_w + pad) + pad
    height = head + len(rows) * (tile_h + pad) + pad
    canvas = Image.new("RGB", (width, height), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    font, small = _font(15), _font(13)
    for c, f in enumerate(frames):
        draw.text((label_w + pad + c * (tile_w + pad) + 4, 5), f"frame {f}" + ("  (R outlined)" if outlined and f else ""),
                  fill=(220, 220, 225), font=small)
    for r, (label, run_dir) in enumerate(rows):
        y = head + pad + r * (tile_h + pad)
        fr = read_frames(run_dir / "video.mp4", frames)
        size = fr[frames[0]].shape[1::-1]
        masks = revealed_masks(run_dir, frames, size) if outlined else {}
        words, line = label.split(" "), ""
        lines = []
        for w in words:
            if len(line) + len(w) + 1 > 20 and line:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        lines.append(line)
        for k, ln in enumerate(lines[:4]):
            draw.text((8, y + 6 + 18 * k), ln, fill=(240, 240, 245), font=font if k == 0 else small)
        for c, f in enumerate(frames):
            img = outline(fr[f], masks[f], thickness=thickness + 1) if outlined and f in masks else fr[f]
            tile = Image.fromarray(img).resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            canvas.paste(tile, (label_w + pad + c * (tile_w + pad), y))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def crop_box(mask, margin=40, max_side=420, max_w=840):
    """Crop box (x0, y0, x1, y1) around R at native resolution.

    The revealed band is found from row/column coverage (rows or columns that are mostly R), which ignores the thin
    wedges a perspective warp adds along the other edges. A ring-like R (zoom out: both a row band and a column band)
    is cropped to the top-left corner region.
    """
    h, w = mask.shape
    rows = np.nonzero(mask.mean(1) > 0.6)[0]
    cols = np.nonzero(mask.mean(0) > 0.6)[0]
    if rows.size >= 40 and cols.size >= 40:  # ring (a few mostly-R columns of a tilt wedge do not count)
        return 0, 0, min(w, 2 * max_side), min(h, max_side + 100)
    if rows.size >= max(cols.size, 1) and rows.size:  # horizontal band (tilt): the band, left part of the frame
        return 0, int(max(0, rows.min() - margin)), min(w, max_w), int(min(h, rows.max() + 1 + margin))
    if cols.size:  # vertical band (pan)
        return int(max(0, cols.min() - margin)), 0, int(min(w, cols.max() + 1 + margin)), h
    return 0, 0, min(w, max_side), min(h, max_side)  # nothing revealed (zoom in): top-left corner


def crops(rows, frame, out_path, outlined=True, scale=1):
    """Native-resolution crop of the revealed band at ``frame``; models side by side (same box for all)."""
    from PIL import Image, ImageDraw

    tiles = []
    box = None
    for label, run_dir in rows:
        fr = read_frames(run_dir / "video.mp4", [frame])[frame]
        mask = revealed_masks(run_dir, [frame], fr.shape[1::-1])[frame]
        if box is None:
            box = crop_box(mask)
        x0, y0, x1, y1 = box
        raw = Image.fromarray(fr[y0:y1, x0:x1])
        marked = Image.fromarray(outline(fr, mask, thickness=1)[y0:y1, x0:x1])
        tiles.append((label, raw, marked))
    cw, ch = tiles[0][1].size
    pad, head = 6, 22
    font = _font(14)
    if cw >= 600 and outlined:  # wide band: one row per model, [plain | outlined]
        canvas = Image.new("RGB", (pad + 2 * (cw * scale + pad), len(tiles) * (head + ch * scale + pad)), (24, 24, 28))
        draw = ImageDraw.Draw(canvas)
        for i, (label, raw, marked) in enumerate(tiles):
            y = i * (head + ch * scale + pad)
            for j, (name, img) in enumerate((("", raw), (" (R outlined)", marked))):
                x = pad + j * (cw * scale + pad)
                draw.text((x + 2, y + 3), label + name, fill=(230, 230, 235), font=font)
                canvas.paste(img.resize((cw * scale, ch * scale), Image.Resampling.NEAREST) if scale != 1 else img, (x, y + head))
    else:
        mode_cols = 2 if outlined else 1
        width = pad + len(tiles) * mode_cols * (cw * scale + pad)
        canvas = Image.new("RGB", (width, head + ch * scale + pad), (24, 24, 28))
        draw = ImageDraw.Draw(canvas)
        x = pad
        for label, raw, marked in tiles:
            for name, img in (("", raw), (" (R outlined)", marked))[:mode_cols]:
                draw.text((x + 2, 3), label + name, fill=(230, 230, 235), font=font)
                canvas.paste(img.resize((cw * scale, ch * scale), Image.Resampling.NEAREST) if scale != 1 else img, (x, head))
                x += cw * scale + pad
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path, box


def main(argv=None):
    """CLI entry with sub-commands ``sheet`` (full frames) and ``crops`` (native-resolution border crops)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("sheet")
    a.add_argument("rows", nargs="+")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--frames", type=int, nargs="+", default=[0, 60, 120])
    a.add_argument("--tile-w", type=int, default=420)
    a.add_argument("--no-outline", action="store_true")
    b = sub.add_parser("crops")
    b.add_argument("rows", nargs="+")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--frame", type=int, default=120)
    b.add_argument("--scale", type=int, default=1)
    b.add_argument("--no-outline", action="store_true", help="plain crops only (default: plain + R outlined)")
    args = ap.parse_args(argv)
    rows = parse_rows(args.rows)
    if args.cmd == "sheet":
        print(sheet(rows, args.frames, args.out, tile_w=args.tile_w, outlined=not args.no_outline))
    else:
        path, box = crops(rows, args.frame, args.out, outlined=not args.no_outline, scale=args.scale)
        print(path, "box", box)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
