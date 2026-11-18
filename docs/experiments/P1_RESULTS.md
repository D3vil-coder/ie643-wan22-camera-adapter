# P1 results: v1 gray-border diagnostics (Molab, 2026-11-16)

Full automatic output: `docs/experiments/P1_diagnosis_summary.md` (verbatim `diagnosis_summary.md`). Setup: v1 adapter
`adapter_000300`, 50 steps, seed 42, 5 variants per image. Metrics are means over frames with revealed fraction R > 1%
(R@120 = revealed fraction at frame 120). Columns: gray score (gray-band score), color ratio (colorfulness R/valid),
hf ratio (HF energy R/valid), seam, latent std R/V and mean|z| R/V (final latent), x0hat std and HF R/V at sigma 0.6.
Images are in the pulled scratch folder, not in git.

## heldout, zoom_out

| variant | fill | min_sigma | R@120 | gray score | gray@120 | color ratio | hf ratio | seam | lat std R/V | lat mean-abs R/V | x0hat std R/V | x0hat hf R/V |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_baseline | zeros | 0.600 | 0.204 | 0.000 | 0.000 | 0.668 | 0.386 | 0.999 | 1.057 | 1.027 | 1.046 | 0.139 |
| fill_replicate | replicate | 0.600 | 0.204 | -0.004 | -0.004 | 0.974 | 0.715 | 0.943 | 1.090 | 1.109 | 1.088 | 0.676 |
| min_sigma_0.3 | zeros | 0.300 | 0.204 | 0.002 | 0.004 | 0.632 | 0.296 | 1.005 | 1.091 | 1.055 | 1.101 | 0.121 |
| min_sigma_0.0 | zeros | 0.000 | 0.204 | 0.003 | 0.005 | 0.619 | 0.283 | 1.006 | 1.107 | 1.072 | 1.101 | 0.121 |
| frozen_base | zeros | 0.000 | 0.204 | -0.005 | -0.005 | 0.960 | 0.874 | 0.997 | 1.052 | 1.066 | 1.055 | 0.781 |

## motorcycle, pan_right

| variant | fill | min_sigma | R@120 | gray score | gray@120 | color ratio | hf ratio | seam | lat std R/V | lat mean-abs R/V | x0hat std R/V | x0hat hf R/V |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_baseline | zeros | 0.600 | 0.107 | 0.017 | 0.032 | 0.817 | 0.416 | 0.943 | 0.940 | 0.945 | 0.931 | 0.371 |
| fill_replicate | replicate | 0.600 | 0.107 | -0.042 | -0.015 | 0.934 | 0.420 | 0.971 | 0.980 | 0.997 | 0.968 | 0.391 |
| min_sigma_0.3 | zeros | 0.300 | 0.107 | 0.027 | 0.034 | 0.803 | 0.399 | 0.940 | 0.961 | 0.968 | 0.966 | 0.345 |
| min_sigma_0.0 | zeros | 0.000 | 0.107 | 0.021 | 0.022 | 0.791 | 0.393 | 0.939 | 0.970 | 0.978 | 0.966 | 0.345 |
| frozen_base | zeros | 0.000 | 0.107 | -0.039 | -0.039 | 1.233 | 0.419 | 0.941 | 0.991 | 1.001 | 0.992 | 0.428 |

## Automatic verdicts

- H1 (zero-guide shortcut): heldout__zoom_out = NOT APPLICABLE, motorcycle__pan_right = NOT APPLICABLE
- H2 (deterministic regression target): NOT APPLICABLE on both
- H3 (hand-off at sigma 0.6): NOT APPLICABLE on both
- Reason on both: "baseline shows no gray-band problem" (heldout gray_band_score=0.000, motorcycle 0.017, threshold 0.15).
- The `heldout__pan_right` and `motorcycle__zoom_out` cases in the summary are NO DATA (see anomalies).

## Anomalies and interpretation

1. The gray-band score is miscalibrated. In the contact sheets v1_baseline clearly has artifacts in R (dark green/gray vertical
   bands and a white glow at the border, frames 80 and 120, heldout zoom-out), but gray_band_score is 0.000 (heldout) / 0.017
   (motorcycle), so every verdict says NOT APPLICABLE. The score appears to look only for mid-gray pixels; the artifact is
   dark or glowing. The other metrics do see it: colorfulness ratio 0.668 and HF ratio 0.386 (heldout), 0.817 / 0.416 (motorcycle)
   for v1_baseline, versus 0.960 / 0.874 for the frozen base on heldout. The seam score is about 1.0 for baseline and `zeros` variants.
2. fill_replicate is the effective fix visually and numerically: colorfulness ratio 0.974, HF ratio 0.715 (heldout), x0hat HF R/V
   0.676 vs 0.139; the contact sheet shows pink pattern extended instead of bands. On motorcycle the gain is smaller (color 0.934,
   HF ratio 0.420 vs 0.416).
3. Lowering camera_min_sigma (0.3 / 0.0) with zero fill does not help (color ratio 0.632 / 0.619, bands still visible), so H3 is not supported by the
   metrics, though the verdict machinery could not conclude. Latent R/V std and mean|z| are about 1.0 (no latent collapse, so no
   support for the "latent near 0" version of H1/H2); x0hat HF R/V is very low (0.12-0.14) for zero-fill variants, i.e. R is smooth/blurry (H2-like), and
   returns to 0.676 with replicate fill, which is the zero-guide shortcut (H1) acting through the guide, not through collapsed latents.
4. The summary enumerates the full image x mode cross product, so two bogus cases (heldout__pan_right, motorcycle__zoom_out) are reported as NO DATA
   when `--summarize-only` is given both images and both modes. Harmless; fix by pairing images with modes.
5. frozen_base is only a control (R = area that would be revealed).

## Recalibration addendum (2026-11-16): border artifact score

The gray-band gate above is replaced by `artifact_score` (`src/fpcamera/border_metrics.py`):
`max(0, 1 - colorfulness_ratio) + max(0, 1 - hf_ratio) + glow`, both ratios clipped to [0, 1.5], with
`glow = |mean luma in R - mean luma of valid pixels within 24 px of the boundary| / 255`, computed per frame then averaged.
`gray_band_score` stays as a secondary column. Verdicts (`scripts/diagnose_v1_borders.py`): H1 supported if
`fill_replicate` lowers the score by >= 25% relative to `v1_baseline`; H3 supported if a `min_sigma_*` variant does;
H2 is reported as an indicator only (x0hat HF R/valid < 0.3), not separable from H1 at inference. Cases are now paired
(`--images` zipped with `--modes`, or explicit `--cases name=path:mode`), so no NO-DATA rows are produced.

Threshold rationale (honest): "problem present" is `artifact_score > 0.40`. It was chosen by hand on two cases only.
From the table above, ratios-only scores are v1_baseline 0.95 (heldout) and 0.77 (motorcycle) versus frozen_base 0.17 (heldout);
0.40 sits between them. The motorcycle frozen_base control has hf ratio 0.419 (low natural texture in that region), so its
ratios-only score is about 0.58 and WOULD be flagged: the HF term is content dependent, so the gate is a screening
heuristic, not a validated detector.

Check on the saved frames only (frames 40/80/120, regenerated locally from the pulled PNGs with the final glow term; the full
per-frame means over the 30 R-frames could not be recomputed because the Molab sandbox ended, see below):

| case | variant | artifact (mean of 3 frames) | glow |
|---|---|---|---|
| heldout zoom_out | v1_baseline | 0.721 | 0.084 |
| heldout zoom_out | fill_replicate | 0.264 (-63%) | 0.034 |
| heldout zoom_out | min_sigma_0.3 | 0.736 | 0.092 |
| heldout zoom_out | min_sigma_0.0 | 0.748 | 0.095 |
| heldout zoom_out | frozen_base | 0.263 | 0.025 |
| motorcycle pan_right | v1_baseline | 0.728 | 0.037 |
| motorcycle pan_right | fill_replicate | 0.716 (-2%) | 0.081 |
| motorcycle pan_right | min_sigma_0.3 | 0.750 | 0.037 |
| motorcycle pan_right | min_sigma_0.0 | 0.758 | 0.039 |
| motorcycle pan_right | frozen_base | 0.564 | 0.034 |

Reading (3 frames per run, indicative): H1 supported on heldout zoom_out (-63%), NOT supported on motorcycle pan_right (-2%);
H3 not supported on either (min_sigma variants are equal or slightly worse). The earlier "all NOT APPLICABLE" verdicts were an
artifact of the gray-band gate.

Not completed: the Molab sandbox was terminated (HTTP 410 "sandbox terminated") after the tracker run, before results were
pulled, so the motion table (signed gain, cosine, ADE/FDE, coverage for the 10 runs) and the Molab re-run of `--summarize-only`
are still to do after the P1 outputs are regenerated. `P1_diagnosis_summary.md` is the pre-recalibration output (gray-band gate,
cross-product NO-DATA rows) and is kept only as the raw record.
