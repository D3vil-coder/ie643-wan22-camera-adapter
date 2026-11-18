"""Render every page of a PDF to PNG for visual checks:  python render_pdf.py in.pdf out_prefix [dpi]"""
import sys
import fitz

pdf, prefix = sys.argv[1], sys.argv[2]
dpi = int(float(sys.argv[3])) if len(sys.argv) > 3 else 110
d = fitz.open(pdf)
print("pages", len(d), d[0].rect)
for i, p in enumerate(d):
    pix = p.get_pixmap(dpi=dpi)
    pix.save(f"{prefix}_{i + 1}.png")
    print(f"{prefix}_{i + 1}.png", pix.width, pix.height)
