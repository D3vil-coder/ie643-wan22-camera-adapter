# E1: evaluation of v2_full at step 415 (Molab, 2026-11-17/18)

Evaluation only, no training. Sandbox start 19:01:52 UTC, hard kill about 22:02 UTC; every job used `--session-hours 3` with a
15 min margin (20 min for the reveal eval). Recovery was running 12 min after the sandbox start (`molab_recover`: code, deps, token, model,
restore from 21 cache shards in 107 s, extras, checkpoints 250 and 415, three eval daemons). Checkpoint 201 is **not on HF** (pruned), 77 and 201
were pushed from the laptop (sha256 verified against `ckpt_*.json`). All numbers below are single runs, seed 42 for generation, fixed noise seed
20261117 for validation.

Checkpoints (adapter sha256 prefix): v1 adapter (step 0) `9e2963f2`, 77 `dc71ac8a`, 201 `2bd2f1d9`, 250 `89a379b5`, 415 `83df1e3f`.
Commits: HEAD at the start `2cb54ae`; E1 fixes `495b84a` and `7387f6a` (section 6).

Local outputs (not in git): `<checkpoints_dir>\eval_E1\`
(`validation\`, `showcase\`, `v1_runs\`, `v1rep_runs\`, `sheets\`, `reveal_eval\`). Backed up on HF dataset repo `ie643-camera-v2-cache`
under `eval/validate_v2full`, `eval/e1_showcase`, `eval/reveal_v2`.

## 1. Held-out validation loss (`scripts/validate_v2.py`, 140 rows per checkpoint, 28 dev sources, 1 fixed draw per row)

Same fixed (sigma, eps) draws for every checkpoint, replicate-fill guide, losses are the training losses. `L_B|R` = pass-B loss restricted to
the revealed region (46 dev rows with R > 0; zoom-in rows have no R). Step 0 is the v1 adapter evaluated with the v2 loss (it was trained with a
zero-fill guide, so its L_A is inflated by the fill mismatch; read it as "where the warm start began", not as a v1 quality score).

| step | dev L_A | dev L_A\|valid | dev L_B | **dev L_B\|R** | train-subset L_A | train L_B | train L_B\|R | dev - train L_B\|R |
|---|---|---|---|---|---|---|---|---|
| 0 (v1 adapter) | 0.2908 | 0.2899 | 0.1309 | **0.2301** | 0.2232 | 0.1514 | 0.2506 | -0.0205 |
| 77 | 0.1311 | 0.1291 | 0.0967 | **0.1736** | 0.0988 | 0.1089 | 0.1829 | -0.0093 |
| 201 | 0.0916 | 0.0895 | 0.0816 | **0.1500** | 0.0784 | 0.0917 | 0.1588 | -0.0088 |
| 250 | 0.0900 | 0.0878 | 0.0782 | **0.1461** | 0.0660 | 0.0878 | 0.1557 | -0.0096 |
| 415 | 0.0632 | 0.0609 | 0.0749 | **0.1403** | 0.0506 | 0.0840 | 0.1495 | -0.0091 |

By family (dev, L_B|R): pan 0.2129 / 0.1621 / 0.1400 / 0.1358 / 0.1301 and tilt 0.2319 / 0.1741 / 0.1513 / 0.1480 / 0.1424 and zoom 0.2630 / 0.1969 /
0.1684 / 0.1643 / 0.1579 (columns: step 0, 77, 201, 250, 415; zoom n = 9 rows with R, pan 19, tilt 18). Full tables: `validation\validation.md`, rows in `rows_*.json`.

Paired differences on dev (same rows and same noise at both checkpoints, SE clustered by source, 28 sources):

| comparison | L_B\|R diff | SE | L_A diff | SE |
|---|---|---|---|---|
| 415 - step 0 | -0.0898 | 0.0068 | -0.2276 | 0.0193 |
| 415 - 77 | -0.0333 | 0.0020 | -0.0678 | 0.0050 |
| 415 - 201 | -0.0096 | 0.0008 | -0.0284 | 0.0033 |
| 415 - 250 | -0.0058 | 0.0006 | -0.0268 | 0.0028 |
| 250 - 201 | -0.0038 | 0.0005 | -0.0016 | 0.0028 |

Reading (facts only): dev L_B|R is lower at 415 than at every earlier checkpoint and 39 % below step 0; the per-update rate of improvement is falling
(77 to 201: -0.019 per 100 updates, 201 to 250: -0.008, 250 to 415: -0.0035) but is still significantly negative on the paired test. The dev-minus-train gap of
L_B|R is constant at about -0.009 from step 77 to 415 (dev is *lower* than the train subset; different sources, so this is not a like-for-like gap), i.e. no sign
of overfitting in this monitor. The training-log L_B plateau (about 0.079 since step 250) is not reproduced on held-out data: dev L_B 0.0782 to 0.0749.
This is a loss, not a quality metric, and a single noise draw per row; 28 dev sources.

## 2. Showcase videos at step 415 (`scripts/generate_examples_v2.py`, 50 steps, seed 42, replicate fill read from the checkpoint's run config)

(a) `--match-v1` matched physical amplitude (v1 magnitudes: zoom 1.0 = x0.893, pan 0.5 = 2.5 deg, tilt 0.35 = 1.75 deg), (b) extended range at u = 1 (calibration is the
identity file, so s = u): heldout zoom-out x0.80 (= 1/1.25) and motorcycle pan right 8.0 deg. 6 videos, 365-386 s each, about 43 GiB peak.

Videos (`showcase\`): `matched\v2\<case>\video.mp4` (4 cases), `sweep\heldout_zoom_out\s1.00\video.mp4`, `sweep\motorcycle_pan_right\s1.00\video.mp4`
(each folder also has `run.json`, `request.json`, `border_quality.json`, `measure.json`, `path_adherence.json`). v1 videos: `v1_runs\<case>\video.mp4` (Submission 1 originals).
v1+rep videos: `v1rep_runs\<case>\video.mp4` (from P1b, **see caveat 3 in section 6**).

### 2.1 Border metrics (R vs valid; mean over every 4th frame with R > 1 %, then frame 120; all computed locally with the same code and frame source)

`local` = `artifact_score_local` (R vs a 8-48 px ring of valid pixels, the better-posed score), `v2` = `artifact_score_v2` (R vs the whole valid area, symmetric
|ln ratio|); lower is better for both. gray-band: gray fraction in R minus in valid (about 0 = no gray fill). colorf and HF are R/V ratios (1 = same as valid).
Both artifact scores are failure-mode detectors: they cannot tell plausible from correct content (flowers: R legitimately differs from the frame average).

| case / model | local mean | v2 mean | local @120 | v2 @120 | gray-band | colorf R/V | HF R/V | seam | R frac @120 |
|---|---|---|---|---|---|---|---|---|---|
| heldout_zoom_out / v1 | 0.554 | 0.753 | 0.673 | 0.905 | -0.003 | 0.725 | 0.738 | 0.897 | 0.204 |
| heldout_zoom_out / v1+rep | 0.246 | **0.419** | 0.323 | 0.529 | -0.004 | 0.974 | 0.715 | 0.943 | 0.204 |
| heldout_zoom_out / **v2@415** | **0.199** | 0.428 | **0.186** | **0.479** | -0.004 | 0.953 | 0.785 | 0.936 | 0.204 |
| motorcycle_pan_right / v1 | 1.515 | 1.403 | 1.378 | 1.192 | 0.040 | 1.040 | 0.262 | 0.895 | 0.055 |
| motorcycle_pan_right / v1+rep | **1.028** | **0.977** | **1.158** | **1.144** | -0.052 | 0.902 | 0.458 | 0.933 | 0.055 |
| motorcycle_pan_right / v2@415 | 1.146 | 1.188 | 1.239 | 1.245 | -0.064 | 1.116 | 0.348 | 0.971 | 0.055 |
| computers_pan_left / v1 | 2.135 | 1.992 | 2.057 | 2.017 | 0.065 | 0.695 | 0.068 | 0.597 | 0.055 |
| computers_pan_left / v1+rep | 2.069 | 1.213 | 2.198 | 1.323 | -0.071 | 1.504 | 0.458 | 1.284 | 0.055 |
| computers_pan_left / **v2@415** | **1.757** | **0.739** | **1.554** | **1.135** | -0.063 | 1.151 | 0.563 | 1.153 | 0.055 |
| flowers_tilt_up / v1 | **0.434** | **0.742** | **0.407** | **0.737** | -0.006 | 1.119 | 1.567 | 0.966 | 0.055 |
| flowers_tilt_up / v1+rep | 0.867 | 1.332 | 0.773 | 1.214 | -0.013 | 1.268 | 2.646 | 1.055 | 0.055 |
| flowers_tilt_up / v2@415 | 0.491 | 0.845 | 0.458 | 0.799 | -0.009 | 1.254 | 1.558 | 0.969 | 0.055 |
| heldout_zoom_out / v2@415 EXT x0.80 (u=1) | 0.173 | 0.402 | 0.179 | 0.418 | -0.004 | 0.971 | 0.729 | 0.980 | 0.361 |
| motorcycle_pan_right / v2@415 EXT 8 deg (u=1) | 1.125 | 1.293 | 1.118 | 1.298 | -0.061 | 1.304 | 0.366 | 0.965 | 0.164 |

Bold = best of the three models for the matched case on that metric. Counts (lower is better, matched cases only, mean over frames):

- `local`: v2@415 is best on 2 of 4 cases (heldout zoom-out 0.199, computers 1.757) and second on the other 2 (motorcycle 1.146 vs v1+rep 1.028; flowers 0.491 vs v1 0.434).
  v2 is lower than v1 on 3 of 4 (not flowers) and lower than v1+rep on 3 of 4 (not motorcycle).
- `v2` score: v2@415 is best on 1 of 4 (computers 0.739); v1+rep is best on heldout (0.419 vs 0.428) and motorcycle (0.977 vs 1.188); v1 is best on flowers (0.742 vs 0.845).
- For continuity with S3: the `v2` score of v2@50/77 was heldout 0.44, motorcycle 1.42, computers 1.69, flowers 1.54; at 415 it is 0.428, 1.188, 0.739, 0.845 (lower on all four).
- Gray-band is about 0 or negative for v2 on every case and for v1+rep; v1 has +0.040 / +0.065 on the two pan cases.

### 2.2 Motion (tracker, `signed_motion_gain_pathwide`, 1 = requested motion realised)

| case | v1 | v1+rep | v2@415 | v2@415 cosine | v2@415 EXT (u=1) |
|---|---|---|---|---|---|
| heldout_zoom_out | 1.245 | 1.336 | 1.032 | 0.999 | 0.989 |
| motorcycle_pan_right | 1.106 | 1.147 | 0.983 | 0.999 | 0.993 |
| computers_pan_left | 0.915 | 0.962 | 0.948 | 0.999 | n/a |
| flowers_tilt_up | 0.925 | 1.019 | 1.165 | 1.000 | n/a |

At u = 1 the delivered motion is 0.989 and 0.993 of the request (x0.80 zoom-out, 8.0 deg pan), so the strength control reaches the extended range with unit gain on these two cases.

### 2.3 Contact sheets and crops (`sheets\`)

- `OVERVIEW_frame120_all_cases.png`: frame 120, rows = 4 cases, columns v1 / v1+rep / v2@415 / v2@415 EXT, R outlined in red.
- `<case>_matched.png` and `<case>_matched_Routlined.png`: rows v1 / v1+rep / v2@415 at frames 0, 60, 120 (plain and with R outlined).
- `<case>_matched_plus_extended[_Routlined].png` (heldout_zoom_out, motorcycle_pan_right): the same plus the u = 1 row.
- `<case>_border_crop_f120.png`: native-resolution crop of the revealed band at frame 120 (R outlined), models side by side.
- Invalid run kept for the record: `showcase\zerofill_invalid\matched\v2\heldout_zoom_out` (section 6, item 1).

## 3. GT reveal evaluation, v2@415 (`scripts/eval_dev_reveal.py`, dev split, 50 steps, seed 42, frames 40/80/120)

Selection (`reveal_eval/selection.json`, deterministic from the data seed): 8 dev pairs = 2 pan, 2 tilt, 4 zoom-out; every source was checked to be in the project's `development` split.
Model gets the GT first frame and the same homographies as the GT clip (exactly as the cache builder renders it); 50 steps, seed 42, neutral caption; metrics on frames 40/80/120, mean over frames with R > 1 %.
**7 of 8 pairs were generated** (the job stopped itself on the deadline; ~365 s per pair): not run = `coco2017val_000000554328_ann633522__m0_zoom_out_u0896`.
`reveal_eval.json` lists all 8 selected pairs under `protocol.pairs`; the aggregate and the per-pair table cover only the 7 finished pairs (n is small: no claim is made from means or stds here).
**There is no v1 / v1+rep row yet (E2), so none of these numbers say whether v2 is better than anything.**

Aggregate (LPIPS_R: lower is better, AlexNet spatial map of the full frame averaged over R; PSNR/SSIM on R; hist_R = Bhattacharyya distance of Lab a*b* histograms; HFdev_R = |ln(HF gen / HF GT)|;
art_local = reference-free local-ring score of the generated frame and `art_local_GT` the same score of the GT frame (the floor of natural content); `_V` = same metric on the valid region as control):

| group | model | n | LPIPS_R | PSNR_R | SSIM_R | hist_R | HFdev_R | art_local | art_local_GT | LPIPS_V | PSNR_V |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | v2 | 7 | 0.276 | 19.445 | 0.655 | 0.173 | 0.605 | 0.754 | 0.690 | 0.068 | 28.994 |
| family=pan | v2 | 2 | 0.377 | 16.959 | 0.641 | 0.220 | 0.643 | 0.733 | 0.767 | 0.066 | 29.959 |
| family=tilt | v2 | 2 | 0.233 | 25.531 | 0.792 | 0.177 | 0.837 | 0.847 | 0.459 | 0.078 | 30.155 |
| family=zoom | v2 | 3 | 0.238 | 17.044 | 0.574 | 0.139 | 0.426 | 0.707 | 0.792 | 0.064 | 27.576 |
| magnitude=0.5<=u<0.75 | v2 | 4 | 0.224 | 22.801 | 0.769 | 0.151 | 0.634 | 0.621 | 0.739 | 0.064 | 30.655 |
| magnitude=u>=0.75 | v2 | 3 | 0.347 | 14.970 | 0.504 | 0.202 | 0.567 | 0.932 | 0.623 | 0.074 | 26.778 |

Per pair (same columns; `R frac` = revealed fraction averaged over the evaluated frames):

| pair | family / dir | u | R frac | LPIPS_R | LPIPS_V | PSNR_R | SSIM_R | hist_R | HFdev_R | art_local | art_local GT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tr_000000259553__m0_pan_left_u0593 | pan / left | 0.59 | 0.069 | 0.186 | 0.050 | 23.2 | 0.825 | 0.131 | 0.401 | 0.456 | 1.154 |
| tr_000000148240__m0_pan_right_u0950 | pan / right | 0.95 | 0.107 | 0.568 | 0.082 | 10.7 | 0.456 | 0.308 | 0.886 | 1.011 | 0.379 |
| tr_000000334432__m0_tilt_down_u0518 | tilt / down | 0.52 | 0.086 | 0.356 | 0.064 | 12.3 | 0.640 | 0.136 | 0.601 | 1.521 | 0.868 |
| val_000000229747~159951__m0_tilt_up_u0666 | tilt / up | 0.67 | 0.110 | 0.111 | 0.091 | 38.7 | 0.943 | 0.219 | 1.073 | 0.173 | 0.049 |
| tr_000000147675__m1_zoom_out_u0961 | zoom / out | 0.96 | 0.245 | 0.358 | 0.087 | 12.5 | 0.218 | 0.215 | 0.228 | 1.213 | 1.315 |
| tr_000000386009__m0_zoom_out_u0681 | zoom / out | 0.68 | 0.182 | 0.241 | 0.052 | 16.9 | 0.666 | 0.120 | 0.463 | 0.334 | 0.885 |
| tr_000000357383__m1_zoom_out_u0810 | zoom / out | 0.81 | 0.212 | 0.116 | 0.052 | 21.7 | 0.837 | 0.084 | 0.587 | 0.573 | 0.176 |

Observations (n = 7, descriptive only):
- The valid-region control is not zero: mean LPIPS_V 0.068 and PSNR_V 29.0 dB (H.264 + VAE loss, and any drift of the generated camera from the GT path), so an LPIPS_R of about 0.1 to 0.2 is close to that floor and values above about 0.35 are far from the real content.
- LPIPS_R per pair ranges from 0.111 to 0.568; the three worst pairs are pan right u 0.95 (0.568), zoom out u 0.96 (0.358), tilt down u 0.52 (0.356). There is no clean relation with u at this n: the u >= 0.75 pairs average 0.347 versus 0.224 for the 4 pairs with u < 0.75, but a u = 0.81 pair is among the best. Family and magnitude are confounded.
- PSNR/SSIM in R are low where the real new content is unpredictable; the EVALUATION.md caveat applies. `art_local` of the generated frame is above the GT floor on 4 of 7 pairs and below it on the others (excess = gen minus GT).
- Contact sheets (GT row vs v2 row, R outlined red): `reveal_eval/sheets/<pair>.png`; videos `reveal_eval/models/v2/<pair>/video.mp4`; GT first frames `reveal_eval/pairs/<pair>/gt_first.png`.

## 4. What was not done

- v1 and v1+replicate on the reveal eval (E2): no GT-referenced comparison exists yet, so the reveal numbers above have no baseline row.
- Steps 77, 201, 250 were not rendered (validation loss only); no seeds other than 42; no tracker-gain sweep over strengths.
- The v1+rep videos are not regenerated with the v1 recipe (teacher anchor); see caveat 3.

## 5. Timeline (UTC)

19:01:52 sandbox start; 19:14 recovery done; 19:15-20:09 validation (5 checkpoints, 9.6 to 15 min each); 20:09-20:19 first showcase batch (invalid, zero fill);
20:19:30-20:57:33 showcase regenerated (6 videos, about 6.2 min each); 20:58-21:34 reveal eval (6 pairs, stopped itself on the 20 min margin at 21:34:24);
21:37-about 21:44 one more pair generated under the 15 min margin rule; metrics by a side loop after each pair; final pull about 21:47 (clean stop target 21:46:52).

## 6. Anomalies, fixes, caveats

1. **Silent zero-fill (my error, fixed).** The recovery restored checkpoints with `--checkpoints` (adapter + manifest only), so `artifacts/v2_full/config.json` was missing on the sandbox.
   `flow_infer.resolve_checkpoint_meta` then found no `fill_mode` and `resolve_guide_fill_mode` silently defaulted to `zeros` for a replicate-trained adapter
   (`run.json`: `guide_fill_mode: zeros`, source `default`; S3 had `replicate`, source `checkpoint`). The first video (heldout zoom-out, 6 min) showed a flat gray-brown frame around the shrunk image
   (local artifact score 2.89, gray-band 0.57), which is how it was caught. Killed the job, pushed `config.json`, re-ran all six videos (about 10 min of GPU lost); the invalid run is kept under `zerofill_invalid`.
   The validation run was **not** affected (no config.json there either, but `normalize_config({})` defaults to `fill_mode: replicate`, recorded in `validation.json` identity; the objective block matches the training config).
   Fix `7387f6a`: `restore_from_hf.restore_checkpoints` also fetches `config.json` and the progress files (`latest.json` untouched), and `flow_infer.generate` raises when a v2 checkpoint has no run `config.json`
   unless `guide_fill_mode` is passed explicitly (test added; 187 tests pass). Useful data point: v2@415 with a zero-fill guide produces a gray frame, i.e. the replicate fill is essential at inference.
2. `restore_from_hf.py --eval` crashed with `EntryNotFoundError` when `eval/<name>` did not exist on HF yet (fresh start); `tree()` now treats a missing folder as empty (`495b84a`, test added).
   `molab_recover --checkpoints` also aborts the whole call when one requested step is missing on HF (201 was pruned); I re-ran it with 250,415 and pushed 77/201 by hand.
3. **v1+rep videos are the P1b ones** (`--include-v1-replicate` was not run): same checkpoint, seed 42, 50 steps, replicate fill, `camera_min_sigma` 0.6, but `camera_residual_anchor: frozen`, whereas the Submission 1 v1 videos
   and `generate_examples_v2.py --include-v1-replicate` use `teacher`. The v1+rep rows are therefore not strictly like-for-like with v1; the effect of the anchor is unknown.
   Their border metrics reproduce the numbers in `P1b_REPLICATE_BASELINE.md` / S3 (heldout 0.419, motorcycle 0.977, computers 1.213, flowers 1.332 for `artifact_score_v2`).
4. The reveal eval's own run does not compute metrics when it stops at the deadline, so metrics were computed by a side loop that re-ran `eval_dev_reveal.py --no-generate` after every finished video (resumable, same code path).
5. `validate_v2` orders step 0 after the numeric steps and `validation.png` omits it; the table in section 1 is the full set.
