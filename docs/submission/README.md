# IE 643 final submission: report and poster (camera-motion adapter v2)

Team Neural Reaper (Ansh Savla, 24b0746). Subject: how v2 closes the two gaps raised on Submission 1, namely gray or distorted
borders (replicate-padded guide, hybrid loss with flow matching on the revealed region, per-source overscan data) and
variable motion strength (continuous-strength training, PAV isotonic calibration, one slider). All numbers come from
`docs/ANALYSIS_AND_PLAN.md`, `docs/v2/*.md` and `docs/experiments/*.md` (tag `v2-final-415`).

| File | What it is |
|---|---|
| `NeuralReaper_Submission2_Report.docx` / `.pdf` | the report, 2 pages (A4), editable Word source and PDF |
| `NeuralReaper_Submission2_Poster.pdf` | the poster, one A4 page (595 x 842 pt; header, frame and section order as `IE643_Poster_Template.pdf`, which is A0 and has the same aspect ratio) |
| `poster_template.html` | editable poster source (plain HTML/CSS) |
| `NeuralReaper_Submission2_Poster.html` | the same with the math tokens expanded (generated; open it in Chrome to preview) |
| `figures/` | figures used (PNG and SVG, about 6 MB) |
| `assets/` | IIT Bombay and IE 643 logos (cut from the template) and RGB thumbnails for the method diagram |
| `build/` | scripts that rebuild everything |

## Still open (left for the author)

1. The two QR boxes on the poster ("GitHub repository", "Demo video") are empty on purpose; paste the QR images in later
   (GitHub: https://github.com/D3vil-coder/ie643-wan22-camera-adapter; the demo video is not made yet).
2. Report references are still in the short form of Submission 1; the poster uses the "authors. title, venue, year. URL" format of the instructions.

Done: the flat-image outcome is in report Section 6 (limitation i) and, as one line under Future Work, on the poster (a diagnosis and an
opt-in inference filter on branch `exp/flat-image-fix`, not enabled by default); the public repository URL is in the report (Section 7)
and the README.

## Rebuild

Windows with Microsoft Word, Chrome and Python 3.12 (`python-docx`, `latex2mathml`, `lxml`, `playwright`, `pymupdf`,
`matplotlib`, `pillow`, `opencv-python`). Run from the repository root.

```powershell
# report: edit the text in build/build_report.py (inline markup: **bold**, //italic//, $latex$, [[placeholder]]),
python docs/submission/build/build_report.py                      # writes the .docx (native Word equations)
powershell -NoProfile -ExecutionPolicy Bypass -File docs/submission/build/docx_to_pdf.ps1   # Word -> PDF, prints the page count
# the .docx can also be edited in Word directly; File > Export > PDF gives the same PDF. Keep it at 2 pages.

# poster (A4, one page): edit poster_template.html, then
python docs/submission/build/build_poster.py --check              # --check prints how full the two columns are (free pt per column)
python docs/submission/build/render_pdf.py docs/submission/NeuralReaper_Submission2_Poster.pdf out_prefix 40    # PNG preview
```

The figures are committed, so the documents rebuild without the evaluation outputs. To regenerate them (needs the local,
untracked evaluation folders `<checkpoints_dir>\eval_E1`, `G1` and the repository config):

```powershell
python docs/submission/build/make_figures.py thumbs crops crops_report crops_poster strength_poster plots   # crops, plots, thumbnails
python docs/submission/build/make_diagrams.py                                                                # method and workflow diagrams (Chrome rasterises the SVG)
```

## Notes on what is shown

* Only COCO and gallery cases appear (food, computers, motorcycle, flowers in the tables). The personal-photo runs and the `ball_*`
  cases (brightness defect) are not used as examples.
* Figures with thumbnails of the guide are an RGB analogue of the 48-channel latent guide (stated in the captions).
* Not yet available and said so in the documents: a v1 baseline on the ground-truth reveal pairs, a held-out check of the strength
  calibration (it is fitted on 28 videos and evaluated in sample), seeds other than 42, and loss ablations (A only, B only).
* The poster frame is extended down to the page margin (the template leaves the lower quarter of the A0 page empty); width,
  colours, header and section order follow the template.
