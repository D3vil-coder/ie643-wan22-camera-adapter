"""Before/after examples and strength sweeps for v2 (Molab GPU; metrics/sheets are CPU).

(a) matched before/after (default, ``--match-v1``): for each of the 11 cases the v1 run's mode and
    magnitude (Submission 1 showcase: pan 0.5 = 2.5 deg at v1's 5 deg max, tilt 0.35 = 1.75 deg,
    zoom 1.0 = x1.12; override per case from ``--v1-runs-root/<case>/run.json``) are turned into the
    identical *physical* amplitude and v2 is run at exactly that (``amplitude_deg`` / ``zoom_ratio`` of
    ``flow_infer``: u = deg/8 or ln r/ln 1.25, no calibration, so the geometry equals v1's). Optional
    rows: ``--include-v1`` (regenerate v1: original recipe, zeros fill, teacher anchor, camera_min_sigma
    0.6), ``--v1-videos-root`` (reuse existing v1 run dirs ``<root>/<case>/{video.mp4,run.json,...}``
    instead), ``--include-v1-replicate`` (v1 checkpoint + replicate guide fill, same v1 magnitudes).
(b) extended range: 3 cases x calibrated strength s in {0.25, 0.5, 0.75, 1.0}; s = 1 is 8 deg / x1.25,
    beyond v1's amplitude, so it is shown separately from the matched comparison.

Every output gets the border metrics of ``scripts/measure_border_quality.py`` and the
tracker gain of ``scripts/calibrate_strength.py``. Then it writes contact sheets
(rows = v1 / v1+replicate / v2 matched, or strengths; columns = input, frames 40/80/120), a markdown summary and an
offline gallery (``gallery/index.html`` + ``gallery_data.json``/``.js``; videos are
referenced relatively under ``gallery/media``). Resumable: finished runs
(``run.json``) are never regenerated, cached metrics are reused.

    python scripts/generate_examples_v2.py --project-root . \\
        --checkpoint artifacts/v2_full/checkpoints/adapter_001000.safetensors \\
        --calibration configs/strength_calibration_v2.json --include-v1 --include-v1-replicate

    python scripts/generate_examples_v2.py --project-root . --no-generate    # rebuild sheets/gallery
    python scripts/generate_examples_v2.py --write-gallery-template          # regenerate index.html only
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
import re
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibrate_strength as cs  # noqa: E402
import measure_border_quality as mbq  # noqa: E402
from fpcamera import strength as st  # noqa: E402
from fpcamera.session_deadline import add_deadline_args, deadline_from_args  # noqa: E402

CASES = (
    "ball_pan_left",
    "ball_pan_right",
    "ball_tilt_up",
    "ball_tilt_down",
    "heldout_zoom_in",
    "heldout_zoom_out",
    "computers_pan_left",
    "flowers_tilt_up",
    "kitchen_zoom_out",
    "living_room_zoom_in",
    "motorcycle_pan_right",
)
SWEEP_CASES = ("ball_pan_right", "flowers_tilt_up", "kitchen_zoom_out")
SWEEP_STRENGTHS = (0.25, 0.5, 0.75, 1.0)
# Submission 1 (v1) magnitudes per family and v1's physical maxima (camera_path defaults).
V1_MAGNITUDES = {"pan": 0.5, "tilt": 0.35, "zoom": 1.0}
V1_MAX = {"pan_deg": 5.0, "tilt_deg": 5.0, "zoom_ratio": 1.12}
MODELS = ("v1", "v1_replicate", "v2")
MODEL_LABELS = {"v1": "v1 (Submission 1)", "v1_replicate": "v1 + replicate fill", "v2": "v2 (matched amplitude)"}
SHEET_LABELS = {"v1": "v1", "v1_replicate": "v1+replicate", "v2": "v2 (matched)"}
SHEET_FRAMES = (40, 80, 120)
V1_CHECKPOINT = "artifacts/flow_poc_all/adapter_000300.safetensors"
MOTION_RE = re.compile(r"(pan_left|pan_right|tilt_up|tilt_down|zoom_in|zoom_out)$")
BORDER_KEYS = ("gray_band_score", "colorfulness_ratio", "hf_ratio", "seam_score")
SCHEMA = "fpcamera_gallery_v2"


# --------------------------------------------------------------------------- cases
def case_motion(name):
    """Motion name (e.g. ``pan_left``) contained in a case name."""
    m = MOTION_RE.search(name)
    if not m:
        raise ValueError(f"cannot derive a motion from case name {name!r}")
    return m.group(1)


def find_input(inputs_dir, name):
    """Input image of a case in the inputs folder (png/jpg/jpeg/webp)."""
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = inputs_dir / f"{name}{ext}"
        if p.is_file():
            return p
    return None


def discover(inputs_dir, wanted):
    """Resolve the wanted case names to input files; returns ``(found, missing)``."""
    found, missing = {}, []
    for name in wanted:
        p = find_input(inputs_dir, name)
        (found.__setitem__(name, p) if p else missing.append(name))
    return found, missing


def v1_case_params(case, v1_runs_root=None):
    """``(motion, v1 magnitude)`` of a case: family table, overridden by ``<root>/<case>/run.json``."""
    motion = case_motion(case)
    magnitude = V1_MAGNITUDES[motion.split("_")[0]]
    if v1_runs_root is not None:
        rj = Path(v1_runs_root) / case / "run.json"
        if rj.is_file():
            rec = json.loads(rj.read_text(encoding="utf8"))
            if rec.get("magnitude") is not None:
                magnitude = float(rec["magnitude"])
            if rec.get("mode") not in (None, motion):
                print(f"warning: {rj} mode {rec['mode']!r} != case motion {motion!r}; keeping {motion!r}")
    return motion, magnitude


def v2_matched_kwargs(motion, magnitude):
    """``flow_infer.generate`` kwargs giving v2 the physical amplitude of v1 ``camera_path(mode, magnitude)``."""
    family = motion.split("_")[0]
    if family == "zoom":
        return dict(motion=motion, zoom_ratio=V1_MAX["zoom_ratio"] ** magnitude)
    return dict(motion=motion, amplitude_deg=V1_MAX[f"{family}_deg"] * magnitude)


def v1_kwargs(motion, magnitude, fill):
    """``generate()`` arguments for the v1 comparison model (v1 mode + magnitude, teacher anchor, sigma 0.6)."""
    return dict(mode=motion, magnitude=magnitude, guide_fill_mode=fill,
                camera_residual_anchor="teacher", camera_min_sigma=0.6)


# --------------------------------------------------------------------------- metrics
def run_metrics(run_dir, gain_key=cs.GAIN_KEY):
    """Border metrics (mean over frames with R > 1%, plus frames 40/80/120) and tracker gain."""
    run_dir = Path(run_dir)
    cached = run_dir / "border_quality.json"
    if cached.is_file():
        bq = json.loads(cached.read_text(encoding="utf8"))
    else:
        bq = mbq.clean(mbq.measure_run(run_dir, mean_stride=4))
    gain = cs.measure_run_gain(run_dir, gain_key)
    mean = bq.get("mean_over_frames_with_R_gt_1pct", {})
    run = json.loads((run_dir / "run.json").read_text(encoding="utf8"))
    u = gain.get("strength_u")
    g = gain.get("gain")
    return {
        "border": {k: mean.get(k) for k in BORDER_KEYS},
        "n_revealed_frames": mean.get("n_frames", 0),
        "border_at_frame": {
            f: {k: (bq.get("at_frame", {}).get(str(f), {}) or {}).get(k) for k in (*BORDER_KEYS, "revealed_fraction")}
            for f in SHEET_FRAMES
        },
        "gain": g,
        "cosine": gain.get("cosine_pathwide"),
        "strength_u": u,
        "strength_s": gain.get("strength_s"),
        "a_obs": (g * u) if (g is not None and u is not None) else None,
        "guide_fill_mode": run.get("guide_fill_mode"),
        "seconds": run.get("seconds"),
        "completed_updates": run.get("completed_updates"),
        "label": (run.get("strength_control") or {}).get("label"),
    }


def fmt(v, nd=3):
    """Format a metric for the markdown tables (``n/a`` for missing / non-finite)."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


# --------------------------------------------------------------------------- sheets
def read_frames(video, indices):
    """Decode the listed frame indices of a video as RGB arrays."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    wanted, out, i = set(indices), {}, 0
    last = max(indices)
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


def _font(size):
    from PIL import ImageFont

    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def contact_sheet(rows, input_image, out_path, tile_w=360, label_w=170, frames_at=SHEET_FRAMES):
    """rows: [(label, video_path)] -> PNG with columns input, frame 40, 80, 120."""
    from PIL import Image, ImageDraw

    inp = Image.open(input_image).convert("RGB")
    tile_h = round(tile_w * 704 / 1280)
    inp = inp.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
    head_h, pad = 28, 4
    width = label_w + 4 * (tile_w + pad) + pad
    height = head_h + len(rows) * (tile_h + pad) + pad
    sheet = Image.new("RGB", (width, height), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    font, small = _font(15), _font(13)
    for c, title in enumerate(("input", *(f"frame {f}" for f in frames_at))):
        draw.text((label_w + pad + c * (tile_w + pad) + 4, 6), title, fill=(220, 220, 225), font=small)
    for r, (label, video) in enumerate(rows):
        y = head_h + pad + r * (tile_h + pad)
        frames = read_frames(video, frames_at)
        draw.text((8, y + tile_h // 2 - 18), label[:22], fill=(240, 240, 245), font=font)
        if len(label) > 22:
            draw.text((8, y + tile_h // 2 + 2), label[22:44], fill=(240, 240, 245), font=small)
        tiles = [inp] + [
            Image.fromarray(frames[f]).resize((tile_w, tile_h), Image.Resampling.LANCZOS) for f in frames_at
        ]
        for c, tile in enumerate(tiles):
            sheet.paste(tile, (label_w + pad + c * (tile_w + pad), y))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


# --------------------------------------------------------------------------- summary
def summary_markdown(entries, sweeps, meta):
    """Render the matched before/after table and the strength sweeps as markdown."""
    lines = ["# v2 examples: matched before/after and extended-range strength sweeps", ""]
    lines += [f"- {k}: {v}" for k, v in meta.items()] + [""]
    lines += [
        "Border metrics are means over frames whose revealed fraction is above 1 % "
        "(`n/a` = nothing revealed, e.g. zoom in). gray_band_score: gray fraction in R minus in valid "
        "(lower is better); colorfulness_ratio and hf_ratio: R / valid (closer to 1 is better); "
        "seam_score: gradient at the R/valid boundary / inside (closer to 1 is better). gain: signed "
        "path-wide tracker gain (1 = exactly the requested motion).",
        "",
        "## Matched before / after (v2 at the same physical amplitude as v1)",
        "",
        "| case | motion | v1 magnitude | physical amplitude | model | gray-band | colorfulness R/V | HF R/V | seam | gain |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        for model in MODELS:
            m = e.get(model)
            if not m:
                continue
            b = m["metrics"]["border"]
            lines.append(
                f"| {e['name']} | {e['motion']} | {fmt(e.get('v1_magnitude'), 2)} | {e.get('amplitude') or 'n/a'} | "
                f"{MODEL_LABELS[model]} | {fmt(b['gray_band_score'])} | "
                f"{fmt(b['colorfulness_ratio'])} | {fmt(b['hf_ratio'])} | {fmt(b['seam_score'])} | "
                f"{fmt(m['metrics']['gain'])} |"
            )
    if sweeps:
        lines += ["", "## Extended range (v2 only): calibrated strength sweep, up to 8 deg / x1.25", "",
                  "| case | s | u (calibrated) | label | observed a_obs | gain | gray-band | colorfulness R/V | HF R/V | seam |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for sw in sweeps:
            for item in sw["entries"]:
                mt = item["metrics"]
                b = mt["border"]
                lines.append(
                    f"| {sw['name']} | {item['s']:.2f} | {fmt(mt['strength_u'])} | {item.get('label') or ''} | "
                    f"{fmt(mt['a_obs'])} | {fmt(mt['gain'])} | {fmt(b['gray_band_score'])} | "
                    f"{fmt(b['colorfulness_ratio'])} | {fmt(b['hf_ratio'])} | {fmt(b['seam_score'])} |"
                )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- gallery
def _copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size != Path(src).stat().st_size:
        shutil.copy2(src, dst)
    return dst


def gallery_entry_media(gallery, rel, run_dir):
    """Copy video + input of a run into gallery/media/<rel>/ and return relative hrefs."""
    media = gallery / "media" / rel
    out = {"video": _copy(Path(run_dir) / "video.mp4", media / "video.mp4").relative_to(gallery).as_posix()}
    if (Path(run_dir) / "input.png").is_file():
        out["input"] = _copy(Path(run_dir) / "input.png", media / "input.png").relative_to(gallery).as_posix()
    return out


def write_gallery(gallery, entries, sweeps, meta):
    """Write ``gallery_data.json`` (+ data js) for the offline gallery next to ``index.html``."""
    gallery = Path(gallery)
    gallery.mkdir(parents=True, exist_ok=True)
    data = {"schema": SCHEMA, "generated": meta["generated"], "meta": meta, "cases": [], "sweeps": []}
    data["sections"] = {"cases": "matched", "sweeps": "extended"}
    for e in entries:
        item = {"name": e["name"], "motion": e["motion"], "section": "matched",
                "amplitude": e.get("amplitude"), "v1_magnitude": e.get("v1_magnitude"),
                "contact_sheet": e.get("sheet_rel")}
        for model in MODELS:
            m = e.get(model)
            if m:
                item[model] = {**gallery_entry_media(gallery, f"{e['name']}/{model}", m["run_dir"]),
                               "metrics": m["metrics"], "request": m.get("label")}
        data["cases"].append(item)
    for sw in sweeps:
        item = {"name": sw["name"], "motion": sw["motion"], "section": "extended", "contact_sheet": sw.get("sheet_rel"), "entries": []}
        for it in sw["entries"]:
            item["entries"].append({
                "s": it["s"], "label": it.get("label"), "metrics": it["metrics"],
                **gallery_entry_media(gallery, f"{sw['name']}/sweep_s{it['s']:.2f}", it["run_dir"]),
            })
        data["sweeps"].append(item)
    text = json.dumps(mbq.clean(data), indent=2)
    (gallery / "gallery_data.json").write_text(text + "\n", encoding="utf8")
    (gallery / "gallery_data.js").write_text("window.GALLERY_DATA = " + text + ";\n", encoding="utf8")
    return data


def write_template(gallery):
    """Write the gallery's ``index.html`` template and an empty data file."""
    gallery = Path(gallery)
    gallery.mkdir(parents=True, exist_ok=True)
    (gallery / "index.html").write_text(GALLERY_HTML, encoding="utf8")
    empty = {"schema": SCHEMA, "generated": None, "meta": {}, "cases": [], "sweeps": []}
    path = gallery / "gallery_data.json"
    if not path.exists():
        path.write_text(json.dumps(empty, indent=2) + "\n", encoding="utf8")


GALLERY_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Camera motion v2 gallery</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#17181c;--muted:#5d6370;--line:#d9dce3;--accent:#1f6f8b;--good:#1a7f4b}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--card:#1d2026;--fg:#e8eaee;--muted:#a1a7b3;--line:#303540;--accent:#5bb5d3;--good:#5fd39a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:16px}
h1{font-size:1.6rem;margin:.4rem 0}h2{font-size:1.25rem;margin:2rem 0 .6rem}h3{font-size:1.05rem;margin:0 0 .5rem}
.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:0 0 16px}
.grid{display:grid;gap:12px}.g2{grid-template-columns:repeat(2,minmax(0,1fr))}.g3{grid-template-columns:repeat(3,minmax(0,1fr))}.g4{grid-template-columns:repeat(4,minmax(0,1fr))}
@media (max-width:760px){.g2,.g3,.g4{grid-template-columns:minmax(0,1fr)}}
video,img{width:100%;height:auto;border-radius:6px;background:#000;display:block}
.tag{display:inline-block;font-size:.75rem;padding:1px 8px;border:1px solid var(--line);border-radius:99px;color:var(--muted);margin-left:6px}
table{border-collapse:collapse;width:100%;font-size:.85rem;margin:.5rem 0}th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}button{font:inherit;padding:4px 12px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;cursor:pointer}
button:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.empty{padding:24px;border:1px dashed var(--line);border-radius:10px;color:var(--muted)}
details{margin-top:8px}summary{cursor:pointer;color:var(--accent)}
</style>
</head>
<body>
<main>
<h1>Camera motion v2 gallery</h1>
<p class="muted" id="meta">Loading...</p>
<div id="app"></div>
</main>
<script src="gallery_data.js" onerror="void 0"></script>
<script>
(function(){
const el=(tag,attrs,...kids)=>{const n=document.createElement(tag);for(const[k,v]of Object.entries(attrs||{})){if(k==='text')n.textContent=v;else if(v!==undefined&&v!==null)n.setAttribute(k,v)}for(const kid of kids){if(kid)n.append(kid)}return n};
const f=(v,d=3)=>(v===null||v===undefined||Number.isNaN(v))?'n/a':Number(v).toFixed(d);
const B=[['gray_band_score','gray-band'],['colorfulness_ratio','colorfulness R/V'],['hf_ratio','HF R/V'],['seam_score','seam']];
function video(src,label){const v=el('video',{controls:'',loop:'',muted:'',preload:'metadata',playsinline:'','aria-label':label});v.muted=true;const s=el('source',{src:src,type:'video/mp4'});v.append(s);return v}
function metricsTable(rows){const t=el('table');const h=el('tr');['run',...B.map(b=>b[1]),'gain'].forEach(x=>h.append(el('th',{text:x})));t.append(h);
 rows.forEach(r=>{const tr=el('tr');tr.append(el('td',{text:r.name}));B.forEach(b=>tr.append(el('td',{text:f(r.m.border[b[0]])})));tr.append(el('td',{text:f(r.m.gain)}));t.append(tr)});return t}
function render(d){
 const app=document.getElementById('app');app.replaceChildren();
 const meta=document.getElementById('meta');
 if(!d||!(d.cases&&d.cases.length)&&!(d.sweeps&&d.sweeps.length)){meta.textContent='';app.append(el('div',{class:'empty',text:'No examples yet. Run scripts/generate_examples_v2.py, which fills gallery_data.json/js and gallery/media.'}));return}
 const m=d.meta||{};meta.textContent=['generated '+(d.generated||'?'),m.checkpoint_v2&&('v2 checkpoint '+m.checkpoint_v2),m.steps&&(m.steps+' steps'),m.seed!==undefined&&('seed '+m.seed),m.calibration&&('calibration '+m.calibration)].filter(Boolean).join(' · ');
 if(d.cases.length){app.append(el('h2',{text:'Matched before / after: v2 at the same physical amplitude as v1'}));
  d.cases.forEach(c=>{const card=el('section',{class:'card'});card.append(el('h3',{text:c.name},el('span',{class:'tag',text:c.motion}),el('span',{class:'tag',text:'matched'+(c.amplitude?' \u00b7 '+c.amplitude:'')})));
   const KS=['v1','v1_replicate','v2'].filter(k=>c[k]);const NM={v1:'v1 (submission 1)',v1_replicate:'v1 + replicate fill',v2:'v2 (matched amplitude)'};
   const g=el('div',{class:'grid '+(KS.length>2?'g3':'g2')});const vids=[];
   KS.forEach(k=>{const box=el('div');box.append(el('div',{class:'muted',text:NM[k]}));const v=video(c[k].video,c.name+' '+k);vids.push(v);box.append(v);g.append(box)});
   card.append(g);
   if(vids.length>1){const b=el('button',{type:'button',text:'Play both'});b.onclick=()=>vids.forEach(v=>{v.currentTime=0;v.play()});card.append(el('p',{},b))}
   card.append(metricsTable(KS.map(k=>({name:NM[k],m:c[k].metrics}))));
   if(c.contact_sheet){const dt=el('details');dt.append(el('summary',{text:'Contact sheet'}),el('img',{src:c.contact_sheet,alt:'Contact sheet for '+c.name,loading:'lazy'}));card.append(dt)}
   app.append(card)})}
 if(d.sweeps.length){app.append(el('h2',{text:'Extended range (v2 only): strength sweep up to 8 deg / x1.25, beyond v1'}));
  d.sweeps.forEach(s=>{const card=el('section',{class:'card'});card.append(el('h3',{text:s.name},el('span',{class:'tag',text:s.motion}),el('span',{class:'tag',text:'extended'})));
   const g=el('div',{class:'grid g4'});const vids=[];
   s.entries.forEach(e=>{const box=el('div');box.append(el('div',{class:'muted',text:'s = '+f(e.s,2)+(e.label?' · '+e.label:'')}));const v=video(e.video,s.name+' s='+e.s);vids.push(v);box.append(v);g.append(box)});
   card.append(g);const b=el('button',{type:'button',text:'Play all'});b.onclick=()=>vids.forEach(v=>{v.currentTime=0;v.play()});card.append(el('p',{},b));
   const t=el('table');const h=el('tr');['s','u','observed a_obs','gain',...B.map(x=>x[1])].forEach(x=>h.append(el('th',{text:x})));t.append(h);
   s.entries.forEach(e=>{const tr=el('tr');[f(e.s,2),f(e.metrics.strength_u),f(e.metrics.a_obs),f(e.metrics.gain),...B.map(x=>f(e.metrics.border[x[0]]))].forEach(x=>tr.append(el('td',{text:x})));t.append(tr)});card.append(t);
   if(s.contact_sheet){const dt=el('details');dt.append(el('summary',{text:'Contact sheet'}),el('img',{src:s.contact_sheet,alt:'Strength sweep contact sheet for '+s.name,loading:'lazy'}));card.append(dt)}
   app.append(card)})}
}
if(window.GALLERY_DATA){render(window.GALLERY_DATA)}
else{fetch('gallery_data.json').then(r=>r.json()).then(render).catch(()=>render(null))}
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- main
def main(argv=None):
    """CLI entry: for each case generate the v1 (before) and v2 (after) videos and the extended strength sweeps,
    measure border quality and tracker gain, and build the gallery; resumable and deadline-aware.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--checkpoint", type=Path, default=None, help="v2 adapter (adapter_NNNNNN.safetensors)")
    ap.add_argument("--v1-checkpoint", type=Path, default=Path(V1_CHECKPOINT))
    ap.add_argument("--match-v1", action=argparse.BooleanOptionalAction, default=True,
                    help="matched before/after: v2 at the same physical amplitude as each v1 case (default on; "
                    "--no-match-v1 skips the matched section and only builds the strength sweep)")
    ap.add_argument("--v1-runs-root", type=Path, default=None,
                    help="Submission-1 style runs <root>/<case>/run.json; mode/magnitude there override the built-in table")
    ap.add_argument("--v1-videos-root", type=Path, default=None,
                    help="existing v1 run dirs <root>/<case>/{video.mp4,run.json,request.json,input.png} used as the v1 row "
                    "(no regeneration)")
    ap.add_argument("--include-v1", action="store_true",
                    help="regenerate the v1 baseline (same seed/steps) unless --v1-videos-root supplies it")
    ap.add_argument("--include-v1-replicate", action="store_true",
                    help="also generate v1 checkpoint + replicate guide fill at the v1 magnitudes")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="strength calibration JSON (default: configs/strength_calibration_v2.json if present, else identity)")
    ap.add_argument("--inputs-dir", type=Path, default=Path("examples/inputs"))
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--sweep-cases", nargs="*", default=list(SWEEP_CASES))
    ap.add_argument("--sweep-strengths", type=float, nargs="+", default=list(SWEEP_STRENGTHS))
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/v2_examples"))
    ap.add_argument("--gallery-dir", type=Path, default=Path("gallery"))
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--guide-fill-mode", choices=("zeros", "replicate"), default=None,
                    help="v2 fill mode; default: recorded by the checkpoint")
    ap.add_argument("--no-lease", action="store_true",
                    help="do not take the GPU lease (share the GPU with a concurrent training run)")
    ap.add_argument("--no-generate", action="store_true", help="CPU only: metrics, sheets, gallery from existing runs")
    ap.add_argument("--write-gallery-template", action="store_true", help="only (re)write gallery/index.html and exit")
    ap.add_argument("--attn-backend", choices=("auto", "cudnn", "flash", "efficient"), default="auto",
                    help="opt-in SDPA kernel priority for the transformer (speed only, not bit-identical); auto = unchanged")
    ap.add_argument("--compile-blocks", action="store_true",
                    help="opt-in regional torch.compile of the transformer blocks, once per generated video "
                         "(minutes of compile each; speed only, not bit-identical)")
    add_deadline_args(ap)
    args = ap.parse_args(argv)
    deadline = deadline_from_args(args)

    root = args.project_root.resolve()
    gallery = args.gallery_dir if args.gallery_dir.is_absolute() else root / args.gallery_dir
    if args.write_gallery_template:
        write_template(gallery)
        print(f"wrote {gallery / 'index.html'}")
        return 0
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    inputs_dir = args.inputs_dir if args.inputs_dir.is_absolute() else root / args.inputs_dir
    cases, missing = discover(inputs_dir, args.cases)
    if missing:
        print(f"warning: no input image for {missing} under {inputs_dir}")
    if not cases:
        sys.exit(f"No example inputs found in {inputs_dir}")
    for name in args.sweep_cases:
        if name not in cases:
            print(f"warning: sweep case {name} has no input; skipped")
    sweep_cases = [n for n in args.sweep_cases if n in cases]
    ckpt = None
    if args.checkpoint:
        ckpt = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    calibration = args.calibration
    if calibration is None:
        default = root / "configs" / "strength_calibration_v2.json"
        calibration = default if default.is_file() else root / "configs" / "strength_calibration_v2.identity.json"
        print(f"calibration: {calibration.name}" + ("  (IDENTITY: no fitted calibration yet)" if "identity" in calibration.name else ""))
    calibration = calibration if calibration.is_absolute() else root / calibration
    v1_ckpt = args.v1_checkpoint if args.v1_checkpoint.is_absolute() else root / args.v1_checkpoint

    # ---- plan
    def run_dir_of(model, case):
        """Run folder of (model, case); an external v1 videos root is used if given and complete."""
        if model == "v1" and args.v1_videos_root is not None:
            ext = (args.v1_videos_root if args.v1_videos_root.is_absolute() else root / args.v1_videos_root) / case
            if (ext / "video.mp4").is_file():
                return ext
        return out / "matched" / model / case

    def sweep_run(case, s):
        """Run folder of one strength-sweep point ``<out>/sweep/<case>/s<s>``."""
        return out / "sweep" / case / f"s{s:.2f}"

    case_info = {}
    jobs = []
    for case, path in cases.items():
        motion, mag = v1_case_params(case, args.v1_runs_root)
        v2kw = v2_matched_kwargs(motion, mag)
        amp = st.amplitude_label(motion, st.physical_to_u(
            motion, v2kw.get("amplitude_deg"), v2kw.get("zoom_ratio")))
        case_info[case] = dict(motion=motion, magnitude=mag, amplitude=amp.split(" \u00b7 ", 1)[-1])
        if not args.match_v1:
            continue
        jobs.append(dict(image=path, run_dir=run_dir_of("v2", case), generate_kwargs=v2kw))
        if args.include_v1 and run_dir_of("v1", case) == out / "matched" / "v1" / case:
            jobs.append(dict(image=path, run_dir=run_dir_of("v1", case), checkpoint=v1_ckpt,
                             generate_kwargs=v1_kwargs(motion, mag, "zeros")))
        if args.include_v1_replicate:
            jobs.append(dict(image=path, run_dir=run_dir_of("v1_replicate", case), checkpoint=v1_ckpt,
                             generate_kwargs=v1_kwargs(motion, mag, "replicate")))
    for case in sweep_cases:
        for s in args.sweep_strengths:
            jobs.append(dict(image=cases[case], run_dir=sweep_run(case, s), motion=case_motion(case),
                             strength=float(s), calibration=str(calibration)))
    todo = cs.pending(jobs)
    print(f"{len(jobs)} runs planned, {len(todo)} pending (steps={args.steps}, seed={args.seed})")
    errors = {}
    if todo and not args.no_generate:
        if ckpt is None or not ckpt.is_file():
            sys.exit("--checkpoint (v2 adapter) is required to generate")
        if (args.include_v1 or args.include_v1_replicate) and not v1_ckpt.is_file():
            sys.exit(f"v1 checkpoint not found: {v1_ckpt}")
        gen = cs.Generator(root, ckpt, args.steps, args.seed, args.guide_fill_mode, log_name="examples_v2_library.log",
                           use_lease=not args.no_lease, attn_backend=args.attn_backend,
                           compile_blocks=args.compile_blocks)
        gen.deadline = deadline
        try:
            errors = gen.run(todo)
        finally:
            gen.close()
        if gen.deadline_hit:
            return 0  # clean stop; re-run to continue, the gallery is built once all runs exist

    # ---- metrics, sheets, gallery
    entries, sweeps, report_errors = [], [], {}

    def metrics_of(run_dir):
        """Border + gain metrics of a run folder; errors are recorded and generation continues."""
        try:
            return run_metrics(run_dir)
        except Exception as exc:  # keep going; report at the end
            report_errors[str(run_dir)] = f"{type(exc).__name__}: {exc}"
            return None

    sheets = gallery / "contact_sheets"
    for case, path in cases.items():
        if not args.match_v1:
            break
        info = case_info[case]
        entry = {"name": case, "motion": info["motion"], "v1_magnitude": info["magnitude"],
                 "amplitude": info["amplitude"]}
        rows = []
        for model in MODELS:
            d = run_dir_of(model, case)
            if (d / "run.json").is_file() and (d / "video.mp4").is_file():
                m = metrics_of(d)
                if m is not None:
                    entry[model] = {"run_dir": d, "metrics": m, "label": m.get("label")}
                    rows.append((SHEET_LABELS[model], d / "video.mp4"))
        if "v2" not in entry and "v1" not in entry:
            continue
        if rows:
            first = next((entry[k]["run_dir"] for k in ("v2", "v1_replicate", "v1")
                          if k in entry and (entry[k]["run_dir"] / "input.png").is_file()), None)
            if first is not None:
                sheet = contact_sheet(rows, first / "input.png", sheets / f"{case}_before_after.png")
                entry["sheet_rel"] = sheet.relative_to(gallery).as_posix()
        entries.append(entry)
    for case in sweep_cases:
        sweep = {"name": case, "motion": case_motion(case), "entries": []}
        rows = []
        for s in args.sweep_strengths:
            d = sweep_run(case, s)
            if not ((d / "run.json").is_file() and (d / "video.mp4").is_file()):
                continue
            m = metrics_of(d)
            if m is None:
                continue
            label = m.get("label") or f"s={s:.2f}"
            sweep["entries"].append({"s": float(s), "run_dir": d, "metrics": m, "label": label})
            rows.append((f"s={s:.2f}  {label}", d / "video.mp4"))
        if rows:
            sheet = contact_sheet(rows, Path(sweep["entries"][0]["run_dir"]) / "input.png", sheets / f"{case}_strength.png")
            sweep["sheet_rel"] = sheet.relative_to(gallery).as_posix()
            sweeps.append(sweep)

    ck_sha = st.file_sha256(ckpt) if ckpt is not None and ckpt.is_file() else None
    cal_sha = st.file_sha256(calibration) if calibration.is_file() else None
    meta = {
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "checkpoint_v2": ck_sha[:12] if ck_sha else None,
        "checkpoint_v2_path": str(ckpt) if ckpt else None,
        "calibration": f"{calibration.name} ({cal_sha[:12] if cal_sha else 'n/a'})",
        "steps": args.steps,
        "seed": args.seed,
        "match_v1": bool(args.match_v1),
        "include_v1": bool(args.include_v1),
        "include_v1_replicate": bool(args.include_v1_replicate),
        "v1_videos_root": str(args.v1_videos_root) if args.v1_videos_root else None,
        "v1_runs_root": str(args.v1_runs_root) if args.v1_runs_root else None,
        "sections": "matched = v2 at v1's physical amplitude; extended = v2 calibrated strength sweep up to 8 deg / x1.25",
    }
    out.mkdir(parents=True, exist_ok=True)
    md = summary_markdown(entries, sweeps, {k: v for k, v in meta.items() if k not in ("generated",)})
    (out / "summary.md").write_text(md, encoding="utf8")
    write_template(gallery)
    write_gallery(gallery, entries, sweeps, meta)
    print(md)
    print(f"gallery: {gallery / 'index.html'}")
    problems = {**errors, **report_errors}
    if problems:
        (out / "errors.json").write_text(json.dumps(problems, indent=2), encoding="utf8")
        print(f"{len(problems)} problems; see {out / 'errors.json'}; re-run to retry")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
