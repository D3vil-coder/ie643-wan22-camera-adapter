"""Build the 2-page IE 643 Submission 2 report as an editable Word document.

    python docs/submission/build/build_report.py            # writes NeuralReaper_Submission2_Report.docx
    powershell -File docs/submission/build/docx_to_pdf.ps1   # (Windows + Word) docx -> PDF

All text lives in this file.  Inline markup in the strings: **bold**, //italic//, $latex$ (a native Word
equation), ^^sup^^, ~~sub~~, [[placeholder]] (yellow highlight, to be replaced by the planner).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import latex2mathml.converter
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from lxml import etree

HERE = Path(__file__).resolve().parent
SUB = HERE.parent
FIG = SUB / "figures"
OUT = SUB / "NeuralReaper_Submission2_Report.docx"
XSL_PATH = Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")
NAVY = RGBColor(0x0B, 0x1F, 0x6B)
BODY_PT = 9.0
TEXT_W = 18.2  # cm, text width
FONT = "Times New Roman"

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_xsl = etree.XSLT(etree.parse(str(XSL_PATH)))


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def omml(latex: str):
    mml = latex2mathml.converter.convert(latex)
    root = _xsl(etree.fromstring(mml.encode("utf-8"))).getroot()
    om = root if root.tag == f"{{{M_NS}}}oMath" else root.find(f".//{{{M_NS}}}oMath")
    return parse_xml(etree.tostring(om))


TOKEN = re.compile(r"(\*\*.+?\*\*|//.+?//|\$.+?\$|\^\^.+?\^\^|~~.+?~~|\[\[.+?\]\])")


def run(p, text, size=None, bold=None, italic=None, color=None, sup=False, sub=False, highlight=False, font=None):
    r = p.add_run(text)
    r.font.name = font or FONT
    r._element.rPr.rFonts.set(qn("w:eastAsia"), font or FONT)
    if size:
        r.font.size = Pt(size)
    if bold is not None:
        r.bold = bold
    if italic is not None:
        r.italic = italic
    if color is not None:
        r.font.color.rgb = color
    if sup:
        r.font.superscript = True
    if sub:
        r.font.subscript = True
    if highlight:
        rPr = r._element.get_or_add_rPr()
        h = OxmlElement("w:highlight")
        h.set(qn("w:val"), "yellow")
        rPr.append(h)
    return r


def rich(p, text, size=BODY_PT, bold=False, italic=False, color=None):
    for tok in TOKEN.split(text):
        if not tok:
            continue
        if tok.startswith("**"):
            run(p, tok[2:-2], size, True, italic, color)
        elif tok.startswith("//"):
            run(p, tok[2:-2], size, bold, True, color)
        elif tok.startswith("$"):
            p._p.append(omml(tok[1:-1]))
        elif tok.startswith("^^"):
            run(p, tok[2:-2], size, bold, italic, color, sup=True)
        elif tok.startswith("~~"):
            run(p, tok[2:-2], size, bold, italic, color, sub=True)
        elif tok.startswith("[["):
            run(p, "[" + tok[2:-2] + "]", size, True, False, RGBColor(0x9C, 0x1C, 0x1C), highlight=True)
        else:
            run(p, tok, size, bold, italic, color)
    return p


def para(doc, text="", size=BODY_PT, align="justify", before=0, after=1.5, keep_next=False, bold=False, italic=False, color=None, left=None, first=None):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before, pf.space_after, pf.line_spacing = Pt(before), Pt(after), 1.0
    pf.keep_with_next = keep_next
    pf.widow_control = True
    if left is not None:
        pf.left_indent = Cm(left)
    if first is not None:
        pf.first_line_indent = Cm(first)
    p.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY, "left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER,
                   "right": WD_ALIGN_PARAGRAPH.RIGHT}[align]
    if text:
        rich(p, text, size, bold, italic, color)
    return p


def heading(doc, text, level=1):
    size = 10 if level == 1 else BODY_PT
    p = para(doc, "", before=4 if level == 1 else 1.5, after=1, keep_next=True, align="left")
    run(p, text, size, True, level == 2, NAVY if level == 1 else None)
    return p


def equation(doc, latex, number, after=1.5, size=BODY_PT):
    """Centred inline-style equation with a right-aligned number (tab stops)."""
    p = para(doc, "", after=after, align="left", size=size)
    ts = p.paragraph_format.tab_stops
    ts.add_tab_stop(Cm(TEXT_W / 2), WD_TAB_ALIGNMENT.CENTER)
    ts.add_tab_stop(Cm(TEXT_W), WD_TAB_ALIGNMENT.RIGHT)
    run(p, "\t", size)
    p._p.append(omml(latex))
    run(p, f"\t({number})", size)
    return p


def set_cell_margins(table, top=15, bottom=15, left=50, right=50):
    tblPr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for k, v in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        e = OxmlElement(f"w:{k}")
        e.set(qn("w:w"), str(v))
        e.set(qn("w:type"), "dxa")
        mar.append(e)
    tblPr.append(mar)


def set_borders(table, color="8A8FA6"):
    tblPr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge, sz in (("top", 8), ("bottom", 8), ("insideH", 3)):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), str(sz))
        e.set(qn("w:color"), color)
        b.append(e)
    tblPr.append(b)


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), fill)
    tcPr.append(sh)


def table(doc, rows, widths_cm, size=8.2, header_fill="E8ECF8", align=None):
    t = doc.add_table(rows=len(rows), cols=len(widths_cm))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    set_cell_margins(t)
    set_borders(t)
    for i, row in enumerate(rows):
        for j, txt in enumerate(row):
            c = t.cell(i, j)
            c.width = Cm(widths_cm[j])
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            if align and align[j] == "c":
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            rich(p, txt, size, bold=(i == 0))
            if i == 0:
                shade(c, header_fill)
    return t


def picture(doc, name, width_cm, after=1, align="center"):
    p = para(doc, "", after=after, align=align, before=1)
    p.add_run().add_picture(str(FIG / name), width=Cm(width_cm))
    p.paragraph_format.keep_with_next = True
    return p


def caption(doc, label, text, after=3):
    p = para(doc, "", size=8.3, align="justify", after=after)
    run(p, label + " ", 8.3, True)
    rich(p, text, 8.3)
    return p


# ---------------------------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------------------------
def build():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin, sec.bottom_margin = Cm(1.0), Cm(0.95)
    sec.left_margin = sec.right_margin = Cm(1.4)
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    st.font.size = Pt(BODY_PT)

    # ---- header block ----------------------------------------------------------------------
    p = para(doc, "", after=1, align="left", size=8.5)
    p.paragraph_format.tab_stops.add_tab_stop(Cm(TEXT_W), WD_TAB_ALIGNMENT.RIGHT)
    run(p, "IE 643: Deep Learning Theory and Practice  |  Project Report: Submission 2 (Novelty Assessment)", 8.5, True)
    run(p, "\tJuly to December 2026", 8.5)
    p = para(doc, "", after=1, align="left")
    run(p, "Trajectory Conditioned Motion Control Adapter for Image to Video Diffusion, Version 2: Learned Border Completion and Calibrated Strength", 12.5, True, color=NAVY)
    p = para(doc, "", after=1, align="left", before=0)
    run(p, "Team Name: Neural Reaper Team", 9.5, True)
    run(p, "        Members: Ansh Savla (24b0746)", 9.5, True)

    # ---- abstract --------------------------------------------------------------------------
    heading(doc, "Abstract")
    para(doc, (
        "Submission 1 gave a 14.26 M-parameter camera-motion adapter on a frozen Wan2.2 TI2V 5B. Review raised two gaps: gray or distorted content in newly revealed "
        "borders, and no controllable motion strength. The first traces to a zero-filled guide and a regression-only loss; a replicate-padded guide and a hybrid loss that "
        "adds flow matching on the true clip, weighted towards the revealed region, fix it. Continuous-strength training in six directions and an isotonic calibration of "
        "one slider fix the second. Held-out revealed-region loss falls from 0.230 to 0.140 without an overfitting gap, tracker gain stays near one and no gray bands "
        "appear in inspected cases; flat images remain a limitation."))

    # ---- 1 introduction --------------------------------------------------------------------
    heading(doc, "1  What changed relative to Submission 1")
    para(doc, (
        "Submission 1 [7] trained a 14,261,248-parameter adapter (51-channel guide, rank-16 LoRA [2]) on the frozen Wan2.2 TI2V 5B [1], trained on zoom clips of "
        "one magnitude (×1.12) with a teacher-relative loss; pan and tilt were never trained. Review raised two gaps. **(G1)** Zoom-out and pan expose "
        "pixels present in no source frame, and v1 filled them with gray, low-detail bands and rectangular edges. **(G2)** Strength was not controllable: one training "
        "magnitude, a 0 to 2 slider that extrapolated, tracker gains of 0.915 to 1.13. Version 2 closes both with the base model and "
        "adapter architecture unchanged (Table 1, Fig. 1). Of the plan in Section 8 of Submission 1, items 8.1 (strength calibration), 8.2 (revealed-region objective) and 8.3 "
        "(six-direction training) are done; 8.4 (component ablations) and 8.5 (bends, speed changes) are not."))
    table(doc, [
        ["", "v1 (Submission 1)", "v2 (this report)"],
        ["Guide fill outside the image", "zero padding × validity (latent 0)", "replicate padding; validity and displacement channels kept"],
        ["Objective", "teacher-relative MSE, σ in [0.6, 1]", "pass A (R down-weighted) + pass B flow matching on the true clip, σ in [0.2, 1]"],
        ["Training data", "36 photos, 108 clips, zoom ×1.12 only", "636 photos, 1,908 clips, six directions, strength u in [0.15, 1]"],
        ["Strength control", "one magnitude; slider extrapolated", "continuous u; slider s mapped to u(s) by a PAV fit"],
        ["Training", "300 updates, 20.7 s each", "415 further updates, 37 to 41 s each (RTX PRO 6000)"],
    ], [3.6, 5.4, 9.2], size=7.8)
    caption(doc, "Table 1.", "v1 and v2 side by side.", after=1)
    picture(doc, "method_report.png", 16.0)
    caption(doc, "Figure 1.", "(a) Guide construction: the revealed region is filled with zeros (v1) or replicated edge features (v2); thumbnails are an RGB "
            "analogue of the latent guide. (b) The two passes of one update and their per-location weights (R is black in (a)).", after=2)

    # ---- 2 diagnosis -----------------------------------------------------------------------
    heading(doc, "2  Diagnosis of the gray border")
    para(doc, (
        "Four hypotheses were tested on the v1 adapter (50 steps, seed 42). **H1:** a zero-padded guide multiplied by the validity is latent 0 on the revealed region R, "
        "which the adapter can copy. **H2:** in R the v1 target is independent of the noise sample, so MSE converges to a conditional mean. **H3:** the hand-off to the "
        "frozen model at σ = 0.6 keeps a gray layout. **H4:** too little data. Replicate fill at inference alone removes the bands (held-out zoom-out: colourfulness "
        "R/valid 0.668 → 0.974, high-frequency energy R/valid 0.386 → 0.715), supporting H1; a hand-off at σ = 0.3 or 0 did not help (H3 rejected). H2 is consistent but "
        "not isolated: replicate fill alone leaves stretched edge streaks, i.e. copied rather than generated content. This training-free “v1 + replicate” baseline keeps "
        "the motion (path-gain ratio 0.98 to 1.19, cosine ≥ 0.986). The final v2 adapter with a zero-filled guide again gives a flat gray frame: the fill belongs to the method."))

    # ---- 3 method --------------------------------------------------------------------------
    heading(doc, "3  Method")
    heading(doc, "3.1  Guide fill (fixes H1).", 2)
    para(doc, (
        "For a latent position $q$ at time $t$ let $p=H_t^{-1}q$ be its first-view position, $z_1$ the first-image latent, $S(\\cdot,p)$ bilinear sampling, $V_t(q)$ the "
        "validity (1 inside the first image) and $R=1-V$ the revealed mask. v1 used zero padding and multiplied by $V$; v2 uses border padding and does not:"), after=1)
    equation(doc, r"G_t^{v1}(q)=V_t(q)\,S(z_1,p),\qquad G_t^{v2}(q)=S\big(z_1,\mathrm{clip}(p)\big)", 1)
    para(doc, (
        "The 51-channel guide $C_t=[G_t,V_t,d_t]$ keeps validity and displacement $d_t=(q-p)/(W-1,H-1)$ unchanged (Fig. 1a)."))
    heading(doc, "3.2  Hybrid objective (fixes H2).", 2)
    para(doc, (
        "With $z_s,z_m$ the static and requested clip latents, $v_0$ the frozen teacher, $v_\\theta$ the adapted network and $x_\\sigma(z)=(1-\\sigma)z+\\sigma\\varepsilon$, "
        "each update adds two passes (the clean first latent plane is an anchor and excluded):"), after=1)
    equation(doc, r"L_A=\sum w_A\,\lVert v_\theta(x_A)-v_0(x_A)-(z_s-z_m)/\sigma_A\rVert^2\Big/\sum w_A,\quad w_A=(1-R)+\alpha R,\ x_A=x_{\sigma_A}(z_s)", 2, after=1)
    equation(doc, r"L_B=\sum w_B\,\lVert v_\theta(x_B)-(\varepsilon-z_m)\rVert^2\Big/\sum w_B,\quad w_B=1+\beta R,\ x_B=x_{\sigma_B}(z_m)", 3, after=2)
    para(doc, (
        "with $\\alpha=0.1$, $\\beta=4$, $\\sigma_A\\sim\\mathrm{TN}(0.8,0.075,[0.6,1])$, $\\sigma_B\\sim\\mathrm{TN}(0.7,0.15,[0.2,1])$, $L=L_A+L_B$ and 10 % static requests (pass A "
        "only). Pass A is the v1 objective, down-weighted on R so that its mean-regression target barely acts on holes. Pass B is conditional flow matching [4] on the real "
        "clip; its target contains $\\varepsilon$ and real overscan content $z_m$, so the network must commit to concrete content in R instead of its mean. Training "
        "warm-starts from v1 update 300 (AdamW, learning rate 10^^−4^^, cosine over 1,000 updates, batch 2) and stopped at 415."))
    heading(doc, "3.3  Data (fixes H4, enables strength).", 2)
    para(doc, (
        "636 COCO 2017 sources [3] (600 new, chosen round-robin over categories, plus the 36 v1 photos; 608 train, 28 development), each with one static and two moving "
        "clips: 1,908 clips, about 20 GB of latents. A moving clip has a balanced random direction and a stratified strength $u\\sim U[0.15,1]$, i.e. an end-of-clip yaw or "
        "pitch of 8°·u or a focal ratio exp(±u ln 1.25), with linear easing over 121 frames. The overscan is set per source as max(1.35, required + 0.02) (about 1.38 for "
        "4:3, 1.53 for 16:9) and coverage is asserted, so every revealed pixel of a training target is real source content (revealed fraction at $u=1$: 0.16 to 0.36)."))
    heading(doc, "3.4  Strength control (fixes G2).", 2)
    para(doc, (
        "For tracked displacements $a_i$ and displacements $e_i$ expected from the requested homographies (RANSAC inliers), the signed gain is $g=\\sum a_i\\!\\cdot\\! e_i/"
        "\\sum e_i\\!\\cdot\\! e_i$ and the observed amplitude, in units of the trained maximum (8° or ×1.25), is $\\bar a=g\\,u$. The slider $s\\in[0,1]$ should show "
        "$\\bar a(u(s))=s$. Per direction we fit an increasing amplitude curve through the origin by pool-adjacent-violators isotonic regression [5] and invert it:"), after=1)
    equation(doc, r"A=\mathrm{argmin}_f\sum_j w_j(\bar a_j-f_j)^2\ \ \mathrm{s.t.}\ 0=f_0\le f_1\le\dots\le f_J,\qquad u(s)=\mathrm{min}\{u:\ A(u)\ge s\}", 4)
    para(doc, (
        "(linear interpolation, capped at $u=1$). It uses 28 existing v2 videos (8 images, 50 steps, seed 42), not a dedicated sweep; camera gain stays 1."))
    # ---- 4 results -------------------------------------------------------------------------
    heading(doc, "4  Experiments and results")
    para(doc, (
        "**Held-out loss.** On the 28 development sources (140 rows, fixed noise), $L_B|_R$ falls from 0.230 (the v1 adapter under the v2 loss, i.e. the warm start) to "
        "0.174, 0.150, 0.146 and 0.140 at updates 77, 201, 250 and 415 (Fig. 3a); the paired difference 415 − 250 is −0.0058 (SE 0.0006), so the loss still falls, more "
        "slowly. The held-out minus train-subset gap stays near −0.009 from update 77: no sign of overfitting. Training stopped at 415 of 1,000 planned "
        "updates, once the crops (Fig. 2) showed no gray band."))
    picture(doc, "border_crops_report.png", 13.2)
    caption(doc, "Figure 2.", "Native-resolution crops at frame 120 (red: boundary of R) at the amplitudes of Submission 1. v1 leaves a grainy gray band; replicate fill alone "
            "extrudes fragments (orange blob, teal smear); v2 continues the bowl rim and the wall.")
    para(doc, (
        "**Border content.** The reference-free local-ring score (lower is better) puts v2 first on 2 of 4 matched cases (food 0.199 against 0.554 for v1 and 0.246 for "
        "v1 + replicate; computers 1.757 against 2.135 and 2.069) and below v1 on 3 of 4. It disagrees with visual inspection on the others (flowers: v1 scores best "
        "despite speckle noise; motorcycle: v1 + replicate scores best despite a doubled edge), so it is a failure-mode detector only. The v2 tracker gain is 0.95 to 1.17 "
        "(v1: 0.92 to 1.25), cosine ≥ 0.999."))
    para(doc, (
        "**Ground-truth reveal evaluation.** Development sources have real hidden margin. On 7 of 8 selected pairs v2 reaches LPIPS [6] 0.276 (PSNR 19.4 dB) in R against "
        "0.068 (29.0 dB) in the valid region as a control for codec, VAE and drift; per pair 0.111 to 0.568, worst at $u\\approx0.95$. No v1 number exists yet, so this is "
        "a reference level, not a comparison."))
    para(doc, (
        "**Strength.** The pooled gain is 0.96 to 1.02 at every tested level (Fig. 3b) and 0.989 (zoom-out ×0.80) and 0.993 (pan 8°) at $u=1$ on two "
        "images. The fitted maps differ from the identity by at most 0.047 (tilt down at s = 0.25, one video), otherwise by at most 0.021 (Fig. 3c); the in-sample "
        "residual is 0.007 and there is no held-out verification. Single videos reach 0.76 to 1.17 at small amplitudes: the slider is linear within tracker noise."))
    picture(doc, "results_panels.png", 15.8)
    caption(doc, "Figure 3.", "(a) Held-out and train-subset revealed-region loss. (b) Tracker gain of the 28 calibration videos. (c) The six fitted PAV maps.")

    # ---- 5 interface -----------------------------------------------------------------------
    heading(doc, "5  Interface: Camera Studio v2")
    para(doc, (
        "One marimo notebook installs the pinned runtime, downloads the base model and the public adapter bundle, loads the pipeline once and generates. The user uploads a "
        "photo and either draws a stroke (its net displacement gives pan or tilt and the direction) or picks zoom in/out, which disables drawing; the slider (0 to 1, step 0.05) is labelled with the physical amount after calibration, e.g. “0.50 → 4.0° pan right”. A stroke now only "
        "selects the direction, so arbitrary bends are no longer supported. Deployed as 27 visible cells on a fresh Molab GPU sandbox, it installed, downloaded the 34 GB model and "
        "generated three videos (361 to 364 s per 50-step video) without gray borders."))

    # ---- 6 limitations ---------------------------------------------------------------------
    heading(doc, "6  Limitations")
    para(doc, (
        "(i) **Flat or low-texture images** show global colour or brightness drift and occasional per-frame distortion that three-frame contact sheets missed (the four videos "
        "of one bright test image dim to about 0.65× of the input luma with a brown shift; every other test video stays within ±5 %). "
        "A diagnosis on a separate branch attributes the drift to the adapter rather than the base model (base Wan keeps luma at 0.99; v2 gives 1.08, 0.90 and 0.63 at updates 77, 250 and 415) "
        "and to a spatially near-uniform part of the adapter residual. An opt-in inference-time filter that removes the low-frequency part of the residual (--residual-lowfreq blur:8) "
        "restores luma to 0.999 on the bright image and removes the per-frame spikes on a synthetic flat landscape. Its effect on detailed images is not verified, and the sparse motion tracker "
        "disagrees with three other measures of the motion (gain 0.14 on the bright image), so it is not enabled by default. "
        "(ii) Reference-free border scores can disagree with visual inspection. (iii) Large reveals are weakest: the ×1.25 zoom-out fill is soft and LPIPS in R (0.276) is far "
        "above its control (0.068). (iv) The evidence is thin: one seed, four matched cases, seven ground-truth pairs, in-sample calibration, 415 of 1,000 planned updates; "
        "v1 and v1 + replicate lack a ground-truth evaluation and no loss ablation was run (training clips are 2D homographies without parallax, as in v1)."))

    heading(doc, "7  Conclusion")
    para(doc, (
        "The gray borders came from a zero-filled guide the adapter could copy and a loss that rewarded the conditional mean in revealed regions. A replicate-padded guide plus "
        "a flow-matching term weighted towards those regions removes the bands without weakening the motion; continuous-strength training with an isotonic map makes the "
        "slider near-linear. Code: https://github.com/D3vil-coder/ie643-wan22-camera-adapter (tag v2-final-415); weights: Hugging Face D3vil-Reap3r/ie643-wan22-camera-adapter-v2."))

    heading(doc, "References")
    refs = (
        "[1] Team Wan. Wan: open and advanced large-scale video generative models. arXiv:2503.20314, 2025. "
        "[2] E. J. Hu et al. LoRA: low-rank adaptation of large language models. ICLR 2022. "
        "[3] T.-Y. Lin et al. Microsoft COCO: common objects in context. ECCV 2014. "
        "[4] Y. Lipman et al. Flow matching for generative modeling. ICLR 2023. "
        "[5] R. E. Barlow et al. Statistical Inference under Order Restrictions. Wiley, 1972. "
        "[6] R. Zhang et al. The unreasonable effectiveness of deep features as a perceptual metric. CVPR 2018. "
        "[7] Neural Reaper Team. Submission 1 report: trajectory conditioned motion control adapter for image to video diffusion, 2026.")
    para(doc, refs, size=7.4, after=0)

    doc.core_properties.title = "IE 643 Submission 2 report: camera-motion adapter v2"
    doc.core_properties.author = "Neural Reaper Team"
    doc.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
