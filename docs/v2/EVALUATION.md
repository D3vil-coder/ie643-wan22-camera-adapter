# Evaluation: metrics, caveats and protocol

Goal: judge how well a model fills the newly revealed region R of a camera move, and whether the camera
actually moved by the requested amount. R is recomputed from the homographies
(`border_metrics.revealed_mask_pixels`; frame-t pixel `q` is revealed iff `H[t]^-1 q` falls outside the
first image or behind the camera). V = everything else.

## 1. Reference-free border metrics (`src/fpcamera/border_metrics.py`, `scripts/measure_border_quality.py`)

No ground truth needed; they compare R with the valid region of the *same* generated frame.

| metric | definition | caveat |
|---|---|---|
| `colorfulness_ratio`, `hf_ratio` | Hasler-Susstrunk colorfulness / Laplacian variance of R divided by the same on the whole V (HF regions eroded 3 px) | conflates content with artifact: flowers generated next to a white wall are "over-saturated" and "over-sharp" although plausible |
| `glow_term` | abs(mean luma R - mean luma of V within 24 px of R) / 255 | genuine brightness gradients (sky) count |
| `artifact_score_v1` | `max(0,1-c)+max(0,1-h)+glow` | one-sided: misses over-sharp / over-saturated borders |
| `artifact_score_v2` | `abs(ln c)+abs(ln h)+glow`, ratios clipped to [0.2, 5] | symmetric, but still global reference |
| `gray_band_score`, `seam_score` | gray-fill fraction difference; gradient at the R/V seam vs inside V | gray band is a failure-mode detector, not a quality score |
| **`artifact_score_local`** | like v2 but R is compared with a **local ring**: valid pixels 8-48 px from R in the same frame (`local_ring_mask`); keys `colorfulness_ring`, `hf_ring`, `colorfulness_ratio_local`, `hf_ratio_local`, `glow_term_local` | removes most "different content far away" bias; ratios use `(a+1)/(b+1)` so a flat/gray ring (wall) stays defined |

Local-ring caveats: if the model continues a texture correctly the score is about 0 (tested with a stripe
pattern; gray fill gives > 2.5, over-saturated streaks > 1). It still cannot tell "wrong but similar-looking"
content from right content, large R (zoom-out, 36 % of the frame) can contain a different object class than the ring
(legitimately), and it is blind to semantic errors. Use it as a failure-mode detector and always report it together with
the GT-referenced numbers below. Frame statistics are averaged over frames with R > 1 %.

## 2. Ground-truth reveal evaluation (`scripts/eval_dev_reveal.py`, `src/fpcamera/reveal_eval.py`)

Dev sources are real photographs with hidden margin, so the real content of R exists. For each selected dev
(source, motion) pair the GT clip is rendered **exactly like the cache builder** (`flow_cache_v2.render_clip`: same
per-source overscan, reflect border, coverage assertion; tests assert bit-equality with an independent
`camera_path_v2` + `render_source` call). The model gets GT frame 0 (the first view) and the *same*
homographies (`flow_infer.generate(camera_request=...)`), 50 steps, seed 42, neutral caption.

Selection (deterministic from the data seed): 12 pairs = 4 pan (2 left, 2 right), 4 tilt (2 up, 2 down), 4 zoom_out,
dev split only, `u >= 0.5` preferred, distinct sources preferred (zoom_in has no R). The list is written to
`<out>/selection.json`; sources the cache builder rejected are skipped.

Metrics on R per frame (40/80/120 by default, or `--frame-stride 4`), averaged over frames with R > 1 %, and
the same on V as a control (a model that copies the first frame should be near-perfect on V only where V is static;
V metrics show generic drift / codec / VAE loss, so subtract mentally when reading R):

| metric | how | reading |
|---|---|---|
| LPIPS | AlexNet spatial map of the **full** generated vs full GT frame (`lpips` package), averaged over the mask. Chosen over a bounding-box crop with GT-pasted pixels because that creates an artificial seam; the cost is that AlexNet receptive fields leak V context into R values | lower = perceptually closer; the main GT number |
| PSNR / SSIM | MSE and 11x11 Gaussian SSIM map over the mask | expected low in R: plausible completions differ pixel-wise from the truth. Report, do not optimise |
| `hist_bhatt` | Bhattacharyya distance of 16x16 Lab a*b* histograms of gen-R vs GT-R | colour palette agreement, independent of layout; 0 identical, 1 disjoint |
| `hf_dev` | abs(ln clip(HF_gen_R / HF_GT_R, 0.2, 5)) | texture sharpness vs the real thing (blur and over-sharpening both penalized) |
| `artifact_score_local` (+`_gt`) | reference-free score of the generated frame; the GT frame's own score shows the floor of natural content | gen minus GT is the excess artifact |

Aggregation: per pair (mean over frames) then mean (and std) over pairs, per model, per family and per
magnitude bucket (`u<0.5`, `0.5<=u<0.75`, `u>=0.75`). Outputs under `--output-dir` (default `artifacts/reveal_eval`):
`models/<name>/<pair>/{video.mp4, run.json, reveal_metrics.json}`, `reveal_eval.json` (protocol + per-pair +
aggregate), `reveal_eval.md` (tables), `sheets/<pair>.png` (rows GT / each model, columns frames 40/80/120, R outlined red).
Generated videos are H.264 compressed while GT is rendered losslessly; this adds a small constant error to every model
and to V, hence the V control.

### Usage (Molab)

```bash
pip install lpips                       # AlexNet LPIPS; first use downloads torchvision AlexNet weights
export PYTHONPATH=src
# v1 and v1+rep (fixed checkpoint, run once; default models if no --model is given)
python scripts/eval_dev_reveal.py --project-root . --session-hours 3 --no-lease
# one v2 checkpoint per milestone
python scripts/eval_dev_reveal.py --project-root . --models v2 \
    --model v2=artifacts/v2_full/checkpoints/adapter_001000.safetensors --output-dir artifacts/reveal_eval
# metrics, tables and sheets only (CPU, after generation)
python scripts/eval_dev_reveal.py --project-root . --no-generate
python scripts/eval_dev_reveal.py --project-root . --dry-run          # selection + time estimate
```

`--model name=ckpt[:fill[:recipe]]` is repeatable (fill `zeros|replicate|auto`, recipe `v1|default`; recipe `v1` =
teacher anchor + camera_min_sigma 0.6 as in Submission 1). The defaults are `v1 = v1 ckpt, zeros, recipe v1` and
`v1+rep = same ckpt, replicate`. Resumable (finished `run.json` and metrics are reused; `--models` restricts the
set), deadline flags `--session-hours/--deadline-margin-minutes/--max-minutes`, `--no-lease` to share the GPU.
Generation is always real (no frames-only shortcut). Use one output dir for all models so the tables are joint.

Time: about 8 min/video (50 steps at 1280x704x121) -> **about 1.6 h GPU per model for 12 pairs** (about 3.2 h for
v1 + v1+rep, about 4.8 h for all three). Metrics add about 1 min/video on CPU
(LPIPS on 1280x704 frames, CPU). Run v1 and v1+rep once (their numbers do not change) and only v2 per milestone.

## 3. Tracker motion metrics (camera adherence)

`scripts/measure_flow_video.py`, `scripts/calibrate_strength.py` (tracker gain, requested vs delivered
amplitude) and `scripts/measure_flow_preservation.py` / `measure_temporal_consistency.py` check that the
camera moves by the requested amount and that content is preserved. They judge the *motion*, not the border
content: a model can score a perfect gain and fill R with gray. Caveats: feature tracking needs texture (low
confidence on flat scenes), the gain is measured on V content only, and delivered amplitude is a property of the
tracker, not of human perception. Report gain together with the border metrics, never alone.

## 4. Protocol for the paper

1. **Models**: `v1` (Submission 1 checkpoint, zeros), `v1+rep` (same checkpoint, replicate fill: isolates the fill-mode effect),
   and `v2` (final checkpoint; earlier milestones only in the appendix / training curve).
2. **GT-referenced (main table)**: `eval_dev_reveal.py`, 12 dev pairs, 50 steps, seed 42, frames 40/80/120
   (optionally also `--frame-stride 4` for the supplementary table). Report per model: LPIPS_R, hist_bhatt_R, hf_dev_R,
   artifact_score_local (and its GT floor), PSNR/SSIM_R in a footnote with the caveat, LPIPS_V as the control; per-family
   and per-magnitude breakdown in the appendix with n and std (12 pairs, so differences below the std are not claims).
3. **Reference-free (qualitative cases)**: `measure_border_quality.py` on the 11 showcase cases
   (`artifact_score_local` and `artifact_score_v2`; the global v2 only for continuity with S3), seeds 42.
4. **Motion**: tracker gain / delivered-vs-requested on the matched cases and the strength sweep.
5. **Visuals**: `sheets/*.png` for all 12 pairs (rows GT / v1 / v1+rep / v2, R outlined), a figure with 3-4 hand-picked pairs, and
   a statement which failure modes (gray band, blur, over-saturation, seams) remain.
6. **Seeds**: one fixed seed (42) for every model, so differences are not seed noise; if time permits repeat the v2 run
   with seeds 43/44 (another 1.6 h per seed) and report the seed std of the aggregate. State exactly which metrics are
   reference-free and which use ground truth.

## 5. Evaluation session runbook (Molab, resumable across sessions)

1. Recover with the eval folders registered (restores what exists on HF and starts a backup daemon per folder):
   ```bash
   python tools/molab_recover.py --resume-training v2_full --checkpoints 77,250 \
       --resume-eval validate_v2full:artifacts/v2_full/validation --resume-eval reveal_v2:artifacts/reveal_eval
   ```
   (a training-only recovery is unchanged; omit `--resume-training` and give `--checkpoints-run v2_full` to avoid touching `latest.json`/training.)
2. Overfitting validation: `python scripts/validate_v2.py --run-dir artifacts/v2_full --cache data/v2/cache_1280x704_f121_v2/index.jsonl
   --steps 77,250,<latest> --checkpoint artifacts/flow_poc_all/adapter_000300.safetensors=v1_step0` (v1 as the step-0 reference;
   `validation.json` is saved after each checkpoint, so a re-run continues).
3. GT reveal evaluation of v2@latest: `python scripts/eval_dev_reveal.py --project-root . --models v2 --model v2=artifacts/v2_full/checkpoints/adapter_<latest>.safetensors --output-dir artifacts/reveal_eval`. Re-run the identical command after a `DEADLINE_STOP`; finished pair videos are kept.
4. v1 and v1+rep in separate sessions as time allows: `python scripts/eval_dev_reveal.py --project-root . --models <v1 names>` (default models, section 2), restoring
   the same `reveal_v2` folder first so `reveal_eval.json` accumulates all models. Use `--no-generate` for CPU-only metrics/tables.
5. Before the sandbox expires check `logs/backup_eval_*.log`, or run `python tools/hf_backup.py once-dir <dir> --path-in-repo eval/<name>` to flush.
