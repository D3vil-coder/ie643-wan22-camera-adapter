"""Draw the method / workflow diagrams as SVG and rasterise them with headless Chrome.

    python docs/submission/build/make_diagrams.py

Needs ``docs/submission/assets/guide_*.png`` and ``weight_*.png`` (written by ``make_figures.py thumbs``).
Outputs (docs/submission/figures/):
    method_report.svg/.png   wide two-panel method figure for the 2-page report
    method_poster.svg        taller, large-type version for the poster (embedded inline by build_poster.py)
    workflow_poster.svg      project workflow for the poster
The thumbnails are an RGB analogue of the 48-channel latent guide (the real guide lives in latent space).
"""
from __future__ import annotations

import base64
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
FIG = HERE.parent / "figures"

NAVY, GREEN, RED, AMBER, GRAY = "#0b1f6b", "#1a7a2e", "#b3382c", "#c9811a", "#555555"
FONT = "'Segoe UI', Arial, Helvetica, sans-serif"


def b64(name):
    return "data:image/png;base64," + base64.b64encode((ASSETS / name).read_bytes()).decode()


def sub(text):
    """x_A -> x with subscript A; supports _{...}, _c, and ^{...}; escapes & and <."""
    text = text.replace("&", "&amp;").replace("<", "&lt;")
    text = re.sub(r"_\{([^}]*)\}|_(\w)", lambda m: f'<tspan baseline-shift="sub" font-size="72%">{m.group(1) or m.group(2)}</tspan>', text)
    text = re.sub(r"\^\{([^}]*)\}|\^(\w)", lambda m: f'<tspan baseline-shift="super" font-size="72%">{m.group(1) or m.group(2)}</tspan>', text)
    return text


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill="none", stroke="none", sw=2, rx=0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=26, fill="#222", bold=False, anchor="start", italic=False, weight=None):
        wt = ' font-weight="700"' if bold else (f' font-weight="{weight}"' if weight else "")
        it = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"{wt}{it}>{sub(s)}</text>')

    def image(self, x, y, w, h, name, stroke=None, sw=4):
        self.add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{b64(name)}" preserveAspectRatio="none"/>')
        if stroke:
            self.rect(x, y, w, h, stroke=stroke, sw=sw)

    def arrow(self, x1, y1, x2, y2, color="#444", sw=4, head=14):
        import math

        ang = math.atan2(y2 - y1, x2 - x1)
        bx, by = x2 - head * math.cos(ang), y2 - head * math.sin(ang)
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" stroke-width="{sw}"/>')
        px, py = -math.sin(ang), math.cos(ang)
        pts = f"{x2},{y2} {bx + head * 0.55 * px:.1f},{by + head * 0.55 * py:.1f} {bx - head * 0.55 * px:.1f},{by - head * 0.55 * py:.1f}"
        self.add(f'<polygon points="{pts}" fill="{color}"/>')

    def save(self, path):
        body = "\n".join(self.parts)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
               f'font-family="{FONT}"><rect width="{self.w}" height="{self.h}" fill="white"/>\n{body}\n</svg>\n')
        Path(path).write_text(svg, encoding="utf-8")
        return svg


# ----------------------------------------------------------------------------------------------------------------
def method_report():
    s = Svg(1800, 500)
    # panel frames
    s.rect(8, 8, 884, 484, fill="#fafbff", stroke="#cdd3ea", rx=14)
    s.rect(908, 8, 884, 484, fill="#fafbff", stroke="#cdd3ea", rx=14)
    s.text(30, 52, "(a) Guide construction", 32, NAVY, bold=True)
    s.text(930, 52, "(b) Hybrid objective (one update)", 32, NAVY, bold=True)

    # (a) first-frame latent -> warp -> two fills
    s.image(30, 190, 200, 110, "guide_src.png", stroke="#888", sw=2)
    s.text(130, 330, "first-frame latent z_1", 24, "#222", anchor="middle")
    s.text(130, 358, "(RGB analogue)", 22, GRAY, anchor="middle")
    s.text(300, 244, "warp", 23, "#222", anchor="middle")
    s.text(300, 276, "H_t^{-1}", 23, "#222", anchor="middle", italic=True)

    s.image(350, 92, 200, 110, "guide_zero.png", stroke=RED)
    s.text(450, 232, "v1: zero fill", 26, RED, bold=True, anchor="middle")
    s.text(450, 260, "latent 0 is roughly gray", 22, "#222", anchor="middle")
    s.image(350, 316, 200, 110, "guide_rep.png", stroke=GREEN)
    s.text(450, 456, "v2: replicate fill", 26, GREEN, bold=True, anchor="middle")
    s.text(450, 482, "nearest-edge features", 22, "#222", anchor="middle")
    s.arrow(236, 232, 345, 150, "#444", sw=3, head=12)
    s.arrow(236, 260, 345, 370, "#444", sw=3, head=12)

    s.image(640, 92, 200, 110, "guide_valid.png", stroke="#555", sw=2)
    s.text(740, 232, "validity V (kept)", 24, "#222", anchor="middle")
    s.text(740, 260, "0 on revealed region R", 22, GRAY, anchor="middle")
    s.rect(620, 316, 244, 136, fill="white", stroke=NAVY, rx=10)
    s.text(742, 352, "guide C (51 ch)", 26, NAVY, bold=True, anchor="middle")
    s.text(742, 384, "48 + 1 + 2 channels", 22, "#222", anchor="middle")
    s.text(742, 412, "zero-init Conv3d,", 22, "#222", anchor="middle")
    s.text(742, 438, "added at patch embed", 22, "#222", anchor="middle")
    s.arrow(560, 372, 612, 372, "#444", sw=3, head=12)
    s.arrow(740, 270, 740, 308, "#444", sw=3, head=12)

    # (b) two passes
    for x0, title, col, wimg, wtxt, l1, l2, role in (
        (930, "Pass A: teacher-relative", NAVY, "weight_A.png", "w_A: 1 valid, 0.1 on R",
         "x_A = (1-σ_A) z_s + σ_A ε", "tgt = teacher(x_A) + (z_s - z_m)/σ_A", "keeps the camera motion"),
        (1366, "Pass B: flow matching", GREEN, "weight_B.png", "w_B: 1 valid, 5 on R",
         "x_B = (1-σ_B) z_m + σ_B ε", "tgt = ε - z_m", "generates the revealed content"),
    ):
        s.rect(x0, 80, 418, 392, fill="white", stroke=col, rx=10)
        s.text(x0 + 209, 118, title, 27, col, bold=True, anchor="middle")
        s.image(x0 + 109, 140, 200, 110, wimg, stroke="#555", sw=2)
        s.text(x0 + 209, 282, wtxt, 23, "#222", anchor="middle")
        s.text(x0 + 209, 330, l1, 24, "#222", anchor="middle", italic=True)
        s.text(x0 + 209, 362, l2, 24 if "teacher" not in l2 else 22, "#222", anchor="middle", italic=True)
        s.text(x0 + 209, 432, role, 25, col, bold=True, anchor="middle")
    return s


def method_poster():
    s = Svg(1000, 806)
    # --- A: guide construction -----------------------------------------------------------
    s.text(0, 34, "A   Guide construction (per latent frame t)", 33, NAVY, bold=True)
    tw, th, y = 220, 121, 54
    xs = [0, 260, 520, 780]
    specs = [("guide_src.png", "#888", ["first-frame", "latent z_1"], "#222"),
             ("guide_zero.png", RED, ["v1: zero fill", "latent 0 ≈ gray"], RED),
             ("guide_rep.png", GREEN, ["v2: replicate", "edge features"], GREEN),
             ("guide_valid.png", "#555", ["validity V", "0 on revealed R"], "#222")]
    for x, (img, col, (l1, l2), tc) in zip(xs, specs):
        s.image(x, y, tw, th, img, stroke=col, sw=5)
        s.text(x + tw / 2, y + th + 34, l1, 26, tc, bold=True, anchor="middle")
        s.text(x + tw / 2, y + th + 64, l2, 24, "#333", anchor="middle")
    s.arrow(226, y + th / 2, 254, y + th / 2, "#444", sw=4, head=12)
    s.text(500, 268, "z_1 is warped by H_t^{-1} (bilinear); thumbnails are an RGB analogue", 22, GRAY, anchor="middle", italic=True)
    s.rect(0, 284, 1000, 84, fill="#eef1fb", stroke=NAVY, rx=12, sw=2)
    s.text(500, 319, "guide C = [ 48 guide | 1 validity | 2 displacement ]  (51 ch)", 27, NAVY, bold=True, anchor="middle")
    s.text(500, 353, "zero-init Conv3d, summed into Wan's frozen patch-embedding output", 24, "#222", anchor="middle")

    # --- B: hybrid objective ---------------------------------------------------------------
    s.text(0, 412, "B   Hybrid objective (two passes per update)", 33, NAVY, bold=True)
    for x0, title, col, wimg, wtxt, l1, l2, role in (
        (0, "Pass A: teacher-relative", NAVY, "weight_A.png", "w_A: 1 valid, 0.1 on R",
         "x_A = (1-σ_A) z_s + σ_A ε", "tgt = teacher(x_A) + (z_s - z_m)/σ_A", "keeps the camera motion"),
        (510, "Pass B: flow matching", GREEN, "weight_B.png", "w_B: 1 valid, 5 on R",
         "x_B = (1-σ_B) z_m + σ_B ε", "tgt = ε - z_m", "generates the revealed content"),
    ):
        s.rect(x0, 428, 490, 322, fill="white", stroke=col, rx=12, sw=3)
        s.text(x0 + 245, 466, title, 29, col, bold=True, anchor="middle")
        s.image(x0 + 145, 480, 200, 110, wimg, stroke="#555", sw=2)
        s.text(x0 + 245, 622, wtxt, 24, "#222", anchor="middle")
        s.text(x0 + 245, 656, l1, 25, "#222", anchor="middle", italic=True)
        s.text(x0 + 245, 692, l2, 23 if "teacher" in l2 else 25, "#222", anchor="middle", italic=True)
        s.text(x0 + 245, 735, role, 25, col, bold=True, anchor="middle")
    s.text(500, 790, "L = L_A + L_B, weighted MSE normalised by Σw; 10 % static requests use pass A only", 24, GRAY, anchor="middle")
    return s


def workflow_poster():
    s = Svg(1000, 432)
    cards = [
        ("1  Submission 1 (v1)", "Adapter, 300 updates, zero-filled guide", "Review: gray borders, no slider", RED),
        ("2  Diagnose", "H1 to H4; replicate fill removes gray", "but leaves stretch streaks", AMBER),
        ("3  Data v2", "636 COCO photos, 1,908 clips, six", "directions, strength u, overscan", NAVY),
        ("4  Train v2", "Replicate guide + passes A and B,", "415 updates, 37 to 41 s each", NAVY),
        ("5  Evaluate, calibrate", "Held-out loss, GT reveal, tracker gain,", "PAV strength map (28 videos)", GREEN),
        ("6  Camera Studio v2", "Self-installing marimo notebook,", "tested on a fresh Molab GPU", GREEN),
    ]
    w, h, gx, gy = 470, 128, 60, 24
    for i, (t, l1, l2, col) in enumerate(cards):
        r, c = divmod(i, 2)
        x, y = c * (w + gx), r * (h + gy)
        s.rect(x, y, w, h, fill="white", stroke=col, rx=14, sw=4)
        s.rect(x, y, 16, h, fill=col, rx=0)
        s.text(x + 34, y + 38, t, 28, col, bold=True)
        s.text(x + 34, y + 75, l1, 23, "#222")
        s.text(x + 34, y + 107, l2, 23, GRAY)
        if c == 0:
            s.arrow(x + w + 4, y + h / 2, x + w + gx - 4, y + h / 2, "#555", sw=4, head=14)
    return s


def rasterise(svg_path, png_path, scale=2):
    chrome = next((p for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                               r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe") if Path(p).exists()), None)
    if chrome is None:
        raise SystemExit("Chrome/Edge not found")
    text = Path(svg_path).read_text(encoding="utf-8")
    w, h = (int(v) for v in re.search(r'viewBox="0 0 (\d+) (\d+)"', text).groups())
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--force-device-scale-factor={scale}",
                    f"--window-size={w},{h}", f"--screenshot={png_path}", Path(svg_path).resolve().as_uri()],
                   check=True, capture_output=True, timeout=120)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for name, fn in (("method_report", method_report), ("method_poster", method_poster), ("workflow_poster", workflow_poster)):
        p = FIG / f"{name}.svg"
        fn().save(p)
        print("wrote", p.name)
    rasterise(FIG / "method_report.svg", FIG / "method_report.png", 2)
    print("rasterised method_report.png")
    if "--preview" in sys.argv:
        for n in ("method_poster", "workflow_poster"):
            rasterise(FIG / f"{n}.svg", HERE.parent / "assets" / f"_preview_{n}.png", 1)
