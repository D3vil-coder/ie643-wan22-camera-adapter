"""Build the IE 643 poster (A4 portrait, 595.28 x 841.89 pt) from ``poster_template.html`` with headless Chrome.

    python docs/submission/build/build_poster.py            # --check prints how full the two columns are
    python docs/submission/build/build_poster.py --png out.png

* LaTeX tokens {{m:...}} / {{M:...}} in the template become MathML (latex2mathml); Chrome renders MathML natively.
* The GitHub and demo-video QR boxes are empty placeholders in the template (the codes are added later by hand); this script does not draw QR codes.
* Output: NeuralReaper_Submission2_Poster.html (self-contained next to figures/ and assets/) and ..._Poster.pdf (one A4 page).
* ``--check`` prints how full the two columns are (overflow is clipped by the frame, so look at the numbers).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import latex2mathml.converter

HERE = Path(__file__).resolve().parent
SUB = HERE.parent
TEMPLATE = SUB / "poster_template.html"
HTML_OUT = SUB / "NeuralReaper_Submission2_Poster.html"
PDF_OUT = SUB / "NeuralReaper_Submission2_Poster.pdf"
PAGE_W, PAGE_H = f"{595.28 / 72:.4f}in", f"{841.89 / 72:.4f}in"


def mathml(latex: str, display: bool) -> str:
    m = latex2mathml.converter.convert(latex, display="block" if display else "inline")
    return m


def render_html() -> str:
    html = TEMPLATE.read_text(encoding="utf-8")
    html = re.sub(r"\{\{M:(.+?)\}\}\s*(?=\n|<)", lambda m: mathml(m.group(1), True), html, flags=re.S)
    html = re.sub(r"\{\{m:(.+?)\}\}", lambda m: mathml(m.group(1), False), html, flags=re.S)
    return html


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print column fill information")
    ap.add_argument("--png", type=Path, help="also write a PNG preview of the whole page at this path (scaled)")
    args = ap.parse_args(argv)

    HTML_OUT.write_text(render_html(), encoding="utf-8")
    print("wrote", HTML_OUT.name)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome")
        except Exception:
            browser = p.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": 794, "height": 1123})  # 595.28 x 841.89 pt in CSS px
        page.goto(HTML_OUT.resolve().as_uri())
        page.wait_for_load_state("networkidle")
        page.evaluate("document.fonts.ready")
        if args.check:
            info = page.evaluate("""() => {
              const out = {};
              for (const id of ['left', 'right']) {
                const c = document.getElementById(id);
                const kids = [...c.children];
                const last = kids[kids.length - 1].getBoundingClientRect();
                const fr = document.querySelector('.frame').getBoundingClientRect();
                const px2pt = 0.75;
                out[id] = {top: c.getBoundingClientRect().top * px2pt, lastBottom: last.bottom * px2pt,
                           frameInnerBottom: (fr.bottom - parseFloat(getComputedStyle(document.querySelector('.frame')).borderBottomWidth)) * px2pt};
              }
              return out; }""")
            for k, v in info.items():
                print(f"{k:5s} content ends at {v['lastBottom']:.0f} pt, frame inner bottom {v['frameInnerBottom']:.0f} pt, "
                      f"free {v['frameInnerBottom'] - v['lastBottom']:.0f} pt")
        if args.png:
            page.screenshot(path=str(args.png), full_page=False, scale="css")
        page.pdf(path=str(PDF_OUT), width=PAGE_W, height=PAGE_H, print_background=True, margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                 prefer_css_page_size=True)
        browser.close()
    print("wrote", PDF_OUT.name)


if __name__ == "__main__":
    sys.exit(main())
