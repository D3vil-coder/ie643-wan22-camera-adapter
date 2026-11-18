# G1: generation and strength calibration of v2_full step 415 (Molab, 2026-11-17/18)

Generation and measurement only, no training. Model = **v2_full step 415** (adapter sha256 `83df1e3f...`, replicate guide fill read from the run `config.json`;
`run.json` of every v2 video says `guide_fill_mode: replicate`, source `checkpoint`). 50 steps, seed 42, neutral caption, 1280x704x121 frames, same recipe as E1.
All numbers are single runs; the metrics are the ones of `docs/v2/EVALUATION.md` (computed locally from the pulled videos with the repository code, `scripts/measure_runs.py`).

Sandbox: NVIDIA RTX PRO 6000 Blackwell (97 887 MiB), start 22:19:57 UTC on 2026-11-17, recovery finished at 22:26 UTC (3 min: code 3 s, deps 66 s, token 1 s, model 97 s,
extras 3 s, checkpoint 415 + `config.json`; **no training-cache restore**), GPU queue started 22:27:29 UTC. 26 videos in one queue, pipeline loaded once (20 s). The last backup
commit on the HF dataset repo is 01:06:35 UTC, i.e. the sandbox lived at least 2 h 46 min and the whole queue finished; at 10:02 UTC the sandbox answered HTTP 410 (gone).
The queue runner log (outside the backed-up folder) was lost with the sandbox; times come from `run.json`.

**Seconds per video: 355 s steady state** (min 354.7, max 355.7, 24 runs without the backbone hash; the first run 367.0 s includes the hash check), about 6 min including pipeline overhead, 43.0 GiB peak.

Local outputs (not in git): `<checkpoints_dir>\G1\` (videos, `run.json`, `request.json`, `border_quality.json`, `measure.json`, `queue\`, `sheets\`, `all_runs_metrics.{json,md}`,
`calibration_fit.md`, `calibration_rows.json`, `calibration_plots\`, `global_*.json`). Backed up on the private HF dataset repo `ie643-camera-v2-cache`, folder `eval/g1`
(a copy restored from there is in `<checkpoints_dir>\G1_hf`, the videos and images that had already been pulled directly were identical by sha256). **The user-supplied test photo (1280x705) is personal and is not distributed: it is only in these private local locations, in no commit and on no public repo. `userphoto` below stands for it in table labels and in the local run-folder and sheet names.**
Baselines: E1 (`<checkpoints_dir>\eval_E1\`), Submission 1 videos (`<course_folder>\submission 1\examples\`, copied to `G1\v1_runs\` for the 7 new cases).

New code (commit `9522d30` and the follow-up): `scripts/generate_batch.py` (queue runner: one loaded pipeline, job files dropped into a folder, resumable), `scripts/measure_runs.py`
(border metrics + tracker, parallel), `scripts/render_reveal_sheets.py` (frames with R outlined, native-resolution border crops), `scripts/compare_videos.py` (per-frame PSNR),
`scripts/fit_calibration_from_runs.py` (calibration from existing videos).

## 1. Attention-kernel check (photo, pan right, s = 1, same seed)

| | `--attn-backend auto` (official video) | `--attn-backend cudnn` |
|---|---|---|
| denoising loop (50 steps, `progress.json`) | 328.2 s | 317.8 s (**3.2 % faster**) |
| `run.json` total seconds | 367.0 s (incl. the post-run backbone hash) | 344.9 s (no hash) |
| steady-state `auto` total of the 24 later runs (no hash) | 354.8 to 355.7 s | 344.9 s (2.8 % faster) |
| peak memory | 42.97 GiB | 42.97 GiB |
| per-frame PSNR auto vs cudnn | mean **45.37 dB**, min 44.21, first frame 50.95, last 44.21 | |
| local / v2 artifact score (mean) | 2.204 / 3.242 | 2.215 / 3.242 |
| gray-band, colorf R/V, HF R/V, seam | 0.626, 0.122, 0.122, 0.972 | 0.626, 0.122, 0.121, 0.970 |
| tracker gain / cosine | 0.987 / 0.999 | 0.988 / 1.000 |

Contact sheet with |difference| x4: `sheets\attn_auto_vs_cudnn.png` (the difference is a faint edge pattern, no visible change). **Decision: stay on `auto` for everything.** The videos are equivalent (45 dB, identical border
and tracker metrics), but cuDNN is only 3 % faster on this Blackwell GPU, below the 10 % adoption threshold. (The first-run warm-up is in the 328 s; the steady-state comparison above shows the same 3 %.) The cudnn video is kept
(`userphoto\attn_cudnn\...`) and excluded from the calibration so that the same input/direction/u is not counted twice.

## 2. User photo at full strength (v2@415, s = 1, identity calibration, so u = s)

Videos: `userphoto\full\<direction>\s1.00\video.mp4` for pan_left, tilt_up, tilt_down, zoom_in, zoom_out and `userphoto\attn_auto\pan_right\s1.00\video.mp4` (the official pan right). Sheet: `sheets\userphoto_s1_all_directions.png`
(rows = directions, frames 0 / 60 / 120, R outlined in red), border crops `sheets\userphoto_s1_<direction>_border_crop_f120.png`.

| photo, v2@415, s = 1 | local mean | v2 mean | local @120 | v2 @120 | gray-band | colorf R/V | HF R/V | seam | R frac @120 | gain | cosine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pan_left (8.0 deg) | 1.605 | 0.855 | 1.827 | 0.864 | -0.032 | 0.566 | 0.814 | 1.146 | 0.164 | 1.005 | 1.000 |
| pan_right (8.0 deg) | 2.204 | 3.242 | 2.009 | 3.230 | 0.626 | 0.122 | 0.122 | 0.972 | 0.164 | 0.987 | 0.999 |
| tilt_up (8.0 deg) | 1.092 | 0.542 | 1.316 | 0.345 | 0.220 | 1.031 | 0.630 | 1.015 | 0.241 | 1.018 | 1.000 |
| tilt_down (8.0 deg) | 0.338 | 0.794 | 0.290 | 0.872 | -0.104 | 0.707 | 0.659 | 1.062 | 0.241 | 0.973 | 1.000 |
| zoom_in (x1.25) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.000 | 0.974 | 0.998 |
| zoom_out (x0.80) | 0.316 | 1.496 | 0.355 | 1.536 | 0.185 | 0.728 | 0.314 | 1.032 | 0.361 | 1.031 | 0.998 |

Border metrics: R vs valid / local ring, lower is better for the two artifact scores; gray-band about 0 = no gray fill; colorf and HF are R/valid ratios (1 = same). gain = signed path-wide tracker gain (1 = requested motion realised).
zoom in reveals nothing (no R), so only the gain applies.

**Direction check.** Every video has a positive signed gain (0.97 to 1.03, cosine >= 0.998) and the frames show the requested motion: pan_right the content (bottle) moves left, pan_left right, tilt_up down, tilt_down up,
zoom_in the bottle grows, zoom_out it shrinks. Delivered amplitude is within 3 % of the request in all six (gain 0.973 to 1.031), so s = 1 really is 8 deg / x1.25.

**The reference-free scores of this photo must not be read as artifacts.** The photo has a white marble wall on the right and a brown table plus teal bottle on the left, so any strip of wall is "gray" and "less colourful" than the valid area.
Applying the *same R mask* to the input photo itself (its own right-edge pixels, i.e. natural content) gives the floor below; the generated pan_right (gray-band 0.628) is at or below it:

| photo, direction | metric | same mask on the input photo | generated frame 120 |
|---|---|---|---|
| pan_right s=1 | gray-band | 0.696 | 0.628 |
| pan_right s=1 | colorf R/V | 0.174 | 0.112 |
| pan_right s=1 | HF R/V | 0.132 | 0.125 |
| pan_right s=1 | local score | 2.693 | 2.009 |
| pan_left s=1 | gray-band | -0.072 | -0.022 |
| pan_left s=1 | colorf R/V | 0.601 | 0.505 |
| pan_left s=1 | HF R/V | 0.323 | 0.906 |
| pan_left s=1 | local score | 0.720 | 1.827 |
| tilt_up s=1 | gray-band | 0.265 | 0.143 |
| tilt_up s=1 | colorf R/V | 0.827 | 1.052 |
| tilt_up s=1 | HF R/V | 0.552 | 0.746 |
| tilt_up s=1 | local score | 0.642 | 1.316 |
| tilt_down s=1 | gray-band | -0.124 | -0.057 |
| tilt_down s=1 | colorf R/V | 0.681 | 0.701 |
| tilt_down s=1 | HF R/V | 0.783 | 0.610 |
| tilt_down s=1 | local score | 0.436 | 0.290 |
| zoom_out s=1 | gray-band | 0.246 | 0.172 |
| zoom_out s=1 | colorf R/V | 0.653 | 0.720 |
| zoom_out s=1 | HF R/V | 0.266 | 0.306 |
| zoom_out s=1 | local score | 1.170 | 0.355 |

So pan_right's gray-band 0.63, colorf 0.12 and HF 0.12 are what natural wall pixels score at that mask; the local ring score of 2.0 is also below the 2.7 of the input's own pixels. The other directions are not saturated by scene content:
pan_left local 1.83 is above its floor 0.72 and tilt_up local 1.32 above 0.64, so those two have a ring mismatch beyond natural content (see observations).

## 3. User photo at half strength (slider demonstration)

Videos `userphoto\half\<direction>\s0.50\video.mp4`; sheet `sheets\userphoto_s05_vs_s1.png` (s = 1 row followed by the s = 0.5 row for each direction).

| photo, v2@415, s = 0.5 | local mean | v2 mean | local @120 | v2 @120 | gray-band | colorf R/V | HF R/V | seam | R frac @120 | gain | cosine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pan_right (4.0 deg) | 2.468 | 3.250 | 2.294 | 3.243 | 0.633 | 0.126 | 0.108 | 0.979 | 0.087 | 1.009 | 0.999 |
| tilt_up (4.0 deg) | 0.631 | 1.126 | 0.907 | 0.887 | 0.274 | 1.060 | 0.348 | 0.997 | 0.124 | 1.017 | 1.000 |
| zoom_out (x0.89) | 0.272 | 1.481 | 0.298 | 1.530 | 0.190 | 0.729 | 0.317 | 1.027 | 0.203 | 1.002 | 0.998 |

The delivered amplitude is half: a_obs = g u = 0.504 (pan right, 4 deg), 0.509 (tilt up), 0.501 (zoom out) against 0.987, 1.018 and 1.031 at s = 1, and R shrinks accordingly (R frac @120 0.087 / 0.124 / 0.203 versus 0.164 / 0.241 / 0.361). Visually the bottle travels about half as far.

## 4. Gallery completion: v2@415 at v1-matched amplitude, 7 new cases (11 cases in total with E1)

v2 at the physical amplitude of the Submission 1 run of each case (`--match-v1` logic: pan 2.5 deg, tilt 1.75 deg, zoom x1.12, i.e. u = 0.3125 / 0.2188 / 0.5079, no calibration): ball_pan_left, ball_pan_right, ball_tilt_up, ball_tilt_down,
heldout_zoom_in, kitchen_zoom_out, living_room_zoom_in. v1 videos = the 7 Submission 1 originals (all present; magnitudes in their `run.json` are 0.5 / 0.35 / 1.0, 50 steps, seed 42, teacher anchor, zeros fill; the 4 E1 cases were identical copies).
Nothing had to be regenerated for v1. The input of each v2 run was checked against the v1 `input_sha256`.
Videos `gallery\matched\v2\<case>\video.mp4`, v1 `v1_runs\<case>\video.mp4`. Sheets `sheets\gallery_<case>_v1_vs_v2.png` (rows v1 / v2, frames 0 / 60 / 120, R outlined) and native-resolution crops of the revealed band at frame 120 `sheets\gallery_<case>_border_crop_f120.png`.

| case / model | local mean | v2 mean | local @120 | v2 @120 | gray-band | colorf R/V | HF R/V | seam | R frac @120 | gain | cosine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ball_pan_left / v1 | 0.450 | 0.843 | 0.559 | 0.895 | -0.000 | 0.512 | 1.000 | 0.919 | 0.055 | 0.859 | 0.993 |
| ball_pan_left / **v2@415** | 0.215 | 1.221 | 0.336 | 1.343 | -0.000 | 0.557 | 1.798 | 0.995 | 0.055 | 0.996 | 0.997 |
| ball_pan_right / v1 | 0.083 | 0.802 | 0.069 | 0.805 | -0.000 | 0.535 | 0.881 | 0.830 | 0.055 | 1.007 | 0.999 |
| ball_pan_right / **v2@415** | 0.077 | 1.022 | 0.075 | 1.054 | -0.000 | 0.532 | 0.688 | 0.905 | 0.055 | 1.089 | 0.999 |
| ball_tilt_up / v1 | 0.146 | 0.665 | 0.215 | 0.729 | -0.000 | 0.596 | 1.122 | 0.980 | 0.055 | 0.568 | 0.892 |
| ball_tilt_up / **v2@415** | 0.364 | 1.091 | 0.413 | 1.128 | -0.000 | 0.568 | 1.649 | 1.124 | 0.055 | 0.911 | 0.994 |
| ball_tilt_down / v1 | 0.752 | 1.260 | 1.042 | 1.554 | -0.000 | 0.551 | 0.580 | 0.866 | 0.055 | 0.633 | 0.947 |
| ball_tilt_down / **v2@415** | 0.123 | 0.664 | 0.053 | 0.656 | -0.000 | 0.560 | 1.019 | 0.933 | 0.055 | 0.762 | 0.995 |
| heldout_zoom_in / v1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.000 | 1.146 | 0.999 |
| heldout_zoom_in / **v2@415** | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.000 | 0.982 | 0.998 |
| kitchen_zoom_out / v1 | 1.498 | 1.410 | 1.831 | 1.717 | 0.008 | 0.708 | 0.355 | 0.703 | 0.204 | 1.130 | 0.998 |
| kitchen_zoom_out / **v2@415** | 0.487 | 0.744 | 0.207 | 0.489 | -0.034 | 0.768 | 1.356 | 0.928 | 0.204 | 0.971 | 0.998 |
| living_room_zoom_in / v1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.000 | 1.107 | 0.998 |
| living_room_zoom_in / **v2@415** | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.000 | 0.988 | 0.995 |

Reading (facts only): `local` score (lower is better), v2 lower than v1 on 4 of the 5 cases that have a revealed region (ball_pan_left, ball_tilt_down, kitchen_zoom_out, ball_pan_right by a hair; ball_tilt_up is better in v1);
`v2` score, v2 lower on 2 of 5 (ball_tilt_down, kitchen_zoom_out), v1 lower on 3 (ball pan left and right, tilt up). Gray-band is 0.000 for all ball cases and -0.034 / +0.008 on kitchen (v2 / v1). Tracker gain closer to 1 in v2 on 6 of 7 cases
(v1 0.57 to 1.15, v2 0.76 to 1.09; the exception is ball_pan_right, 1.007 in v1 and 1.089 in v2); the worst v2 gain is ball_tilt_down at the small 1.75 deg amplitude (0.762, v1 0.633) and ball_tilt_up 0.911 (v1 0.568).
**See section 7: the four ball videos of v2@415 darken by 30 to 40 %.**

## 5. v1 against v2 on the user photo at matched amplitude (pan 2.5 deg, tilt 1.75 deg, zoom x0.89)

v1 = Submission 1 recipe on the v1 adapter (zeros fill, teacher anchor, camera_min_sigma 0.6) run by `generate_batch.py` with the photo; v2@415 at the same physical amplitude. Videos `userphoto_match\v1\<direction>\` and `userphoto_match\v2\<direction>\`.
Sheet `sheets\userphoto_v1_vs_v2_matched.png`, crops `sheets\userphoto_match_<direction>_border_crop_f120.png`.

| photo, matched amplitude | local mean | v2 mean | local @120 | v2 @120 | gray-band | colorf R/V | HF R/V | seam | R frac @120 | gain | cosine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pan_right (2.5 deg) v1 | 1.363 | 2.854 | 1.044 | 2.589 | 0.596 | 0.161 | 0.295 | 0.958 | 0.055 | 1.024 | 0.998 |
| pan_right (2.5 deg) **v2@415** | 2.534 | 3.253 | 2.529 | 3.249 | 0.628 | 0.136 | 0.107 | 0.952 | 0.055 | 1.030 | 0.999 |
| tilt_up (1.75 deg) v1 | 0.316 | 0.888 | 0.359 | 0.696 | 0.392 | 1.003 | 0.429 | 0.989 | 0.055 | 0.892 | 0.988 |
| tilt_up (1.75 deg) **v2@415** | 0.418 | 1.369 | 0.339 | 1.464 | 0.269 | 1.062 | 0.273 | 1.023 | 0.055 | 1.018 | 0.999 |
| zoom_out (x0.89) v1 | 0.350 | 1.129 | 0.496 | 0.757 | 0.294 | 0.671 | 0.511 | 0.992 | 0.204 | 1.339 | 0.991 |
| zoom_out (x0.89) **v2@415** | 0.300 | 1.484 | 0.320 | 1.552 | 0.191 | 0.729 | 0.311 | 1.029 | 0.204 | 0.994 | 0.997 |

Reading (facts only): the reference-free scores do **not** favour v2 on this photo: `local` v1 lower for pan_right (1.363 vs 2.534) and tilt_up (0.316 vs 0.418), v2 lower for zoom_out (0.300 vs 0.350); `v2` score v1 lower on all three. Gray-band is
0.596 (v1) vs 0.628 (v2) for pan right, both at the natural-content floor of this scene (section 2). Tracker gain: v1 1.024 / 0.892 / 1.339, v2 1.030 / 1.018 / 0.994 (v2 closer to the request on tilt and zoom).
The crops show what the scores cannot (section 8).

## 6. Strength calibration (`configs/strength_calibration_v2.json`)

Fitted with `scripts/fit_calibration_from_runs.py` from **28 existing v2@415 videos**: 22 from G1 (photo s = 1 x 6, s = 0.5 x 3, extra levels pan_left 0.5, tilt_down 0.5, zoom_in 0.75, photo matched x 3, gallery matched x 7) and 6 from E1 (4 matched, 2 extended u = 1),
8 different images, 12 skipped (the duplicate cudnn run, the 3 v1 photo runs, 7 v1 videos, the invalid zero-fill E1 run). No dedicated 72-video sweep was run (no GPU time was spent on it); the fit uses the same tracker gain, isotonic (PAV) fit and inversion as `calibrate_strength.py`
(`docs/v2/CALIBRATION.md`). Full data and the fit are in `G1\calibration_fit.md`, `G1\calibration_rows.json`, plot `G1\calibration_plots\calibration_fit.png`.

Extra calibration levels I added to the queue (3 videos, 18 GPU minutes):

| direction | u | gain g | a_obs | note |
|---|---|---|---|---|
| pan_left s=0.5 | 0.500 | 0.984 | 0.492 | extra calibration level (not in the task list) |
| tilt_down s=0.5 | 0.500 | 0.993 | 0.497 | extra calibration level (not in the task list) |
| zoom_in s=0.75 | 0.750 | 1.018 | 0.763 | extra calibration level (not in the task list) |

**Gain per family versus requested strength u** (mean signed path-wide gain g over the videos at that level, n videos; g = 1 means the request is realised exactly):

| family | u level | n | mean g |
|---|---|---|---|
| pan (left+right) | 0.3125 | 5 | 1.009 |
| pan | 0.5000 | 2 | 0.996 |
| pan | 1.0000 | 3 | 0.995 |
| tilt (up+down) | 0.2188 | 4 | 0.964 |
| tilt | 0.5000 | 2 | 1.005 |
| tilt | 1.0000 | 2 | 0.996 |
| zoom (in+out) | 0.5000 / 0.5079 | 6 | 0.994 |
| zoom | 0.7500 | 1 | 1.018 |
| zoom | 1.0000 | 3 | 0.998 |

Per direction: pan_left 0.972 / 0.984 / 1.005 at u = 0.3125 / 0.5 / 1; pan_right 1.034 / 1.009 / 0.990; tilt_up 1.031 / 1.017 / 1.018 at u = 0.2188 / 0.5 / 1; tilt_down 0.762 / 0.993 / 0.973 (**one video per level**); zoom_in 0.985 / 1.018 / 0.974 at u = 0.5079 / 0.75 / 1 (n = 2 / 1 / 1);
zoom_out 0.999 / 1.010 at u = 0.5 to 0.508 / 1 (n = 4 / 2). The adapter delivers essentially the requested amplitude over the whole range (0.2 to 1) in every direction: the gain is within about +-3 % of 1 except two small-amplitude, single-image
cases (ball_tilt_down g = 0.762, ball_tilt_up 0.911; flowers_tilt_up 1.165 from E1).

Fit (rule: a direction is fitted only with >= 3 videos on >= 2 distinct u levels (levels closer than 0.02 merged) and a level >= 0.9; **all six directions met it, so no direction fell back to the identity map**):

| direction | videos / levels | u(s = 0.25) | u(s = 0.5) | u(s = 0.75) | u(s = 1) | max abs(u - s) | saturates_at |
|---|---|---|---|---|---|---|---|
| pan_left | 4 / 3 | 0.257 | 0.508 | 0.752 | 0.995 | 0.009 | - |
| pan_right | 6 / 3 | 0.242 | 0.495 | 0.753 | 1.000 | 0.010 | - |
| tilt_up | 5 / 3 | 0.243 | 0.491 | 0.737 | 0.982 | 0.018 | - |
| tilt_down | 3 / 3 | 0.290 | 0.504 | 0.766 | 1.000 | 0.047 | 0.973 |
| zoom_in | 4 / 3 | 0.254 | 0.508 | 0.738 | 1.000 | 0.021 | 0.974 |
| zoom_out | 6 / 2 | 0.249 | 0.499 | 0.745 | 0.990 | 0.010 | - |

**The fitted map is the identity within the tracker noise** (the tracker has a few percent noise, `docs/v2/CALIBRATION.md` section 7): the largest deviation is 0.047 (tilt_down at s = 0.25, set by one 1.75 deg video with g = 0.762), the others stay below 0.021.
Mean |a_obs - u| of the 28 videos without calibration is 0.014 (max 0.052); the in-sample residual around the fitted curve is 0.007 (max 0.029). That in-sample number is not a held-out verification: **no `--verify` pass (new videos at s = 0.33 / 0.66 / 1 through the fitted file) was run**, and
each knot is the mean of 1 to 3 videos. tilt_down and zoom_in show `saturates_at` 0.973 / 0.974 because their only u = 1 video has g = 0.973 / 0.974, one image each, which is within noise of 1. Recommendation for the planner: the slider is already linear to within about 0.03 to 0.05;
the committed file is the honest fit but could equally be replaced by the identity file without a visible change. If kept, it is tied to checkpoint 415 by `checkpoint_sha256`.

## 7. Defect found: brightness drop on the ball image (v2@415)

All four videos of the "ball" image (blue ball on bright oak) darken strongly in the first second, in every direction, while v1 does not and no other v2 case does. Global luma of the frame (mean over all pixels; frame 0 identical to the input):

| video | frame-0 luma | luma @40 / @0 | luma @120 / @0 | max abs frame-to-frame luma change |
|---|---|---|---|---|
| gallery/matched/v2/ball_pan_left | 152.9 | 0.672 | 0.631 | 4.66 |
| gallery/matched/v2/ball_pan_right | 152.9 | 0.749 | 0.640 | 2.42 |
| gallery/matched/v2/ball_tilt_up | 152.9 | 0.700 | 0.660 | 4.63 |
| gallery/matched/v2/ball_tilt_down | 152.9 | 0.773 | 0.700 | 3.23 |
| v1_runs/ball_pan_left | 152.9 | 1.048 | 1.051 | 1.30 |
| v1_runs/ball_pan_right | 152.9 | 1.044 | 1.032 | 0.94 |
| v1_runs/ball_tilt_up | 152.9 | 1.054 | 1.054 | 1.34 |
| v1_runs/ball_tilt_down | 152.9 | 1.059 | 1.065 | 1.26 |
| gallery/matched/v2/heldout_zoom_in | 117.2 | 0.985 | 0.981 | 0.45 |
| gallery/matched/v2/kitchen_zoom_out | 118.1 | 1.008 | 1.022 | 0.34 |
| gallery/matched/v2/living_room_zoom_in | 127.8 | 0.990 | 0.984 | 0.47 |
| E1/v2/flowers_tilt_up | 192.0 | 0.961 | 0.939 | 1.33 |
| userphoto/full/tilt_up/s1.00 | 118.1 | 1.028 | 1.091 | 0.47 |
| userphoto/full/tilt_down/s1.00 | 118.1 | 0.976 | 0.928 | 0.41 |

For ball_pan_left the luma goes 151 (frame 0), 148 (5), 136 (10), 115 (20), 100 (40) and then stays near 94 to 98: a dimming ramp over frames 5 to 40, with colours shifting to a darker brown (RGB mean 186,146,103 to 127,87,65) and a visible vignette towards the corners
(`sheets\gallery_ball_tilt_down_v1_vs_v2.png`, `sheets\gallery_ball_pan_left_v1_vs_v2.png`). The valid-region luma ratio of every other v2 video is 0.95 to 1.05 (the photo's +-5 to 9 % comes from the content that enters and leaves the frame). Max frame-to-frame luma change of all non-ball v2 videos is
<= 0.5 gray levels (no flicker in the global mean); during the ball dimming ramp it is 2.4 to 4.7. This is not a border problem (R is a thin strip) but a global exposure/colour drift of the sampled video, with the same seed and the same input in four different motions, so it is reproducible for this image.
Not investigated (no GPU): another seed, another checkpoint (250), the zoom case with this image, whether `camera_min_sigma` or the replicate fill interacts. This must be known before the ball cases are shown as v2 results.

## 8. Visual observations (I looked at every sheet and the crops listed; I did not play the videos, flicker is judged from the global luma curve and the frames 0 / 60 / 120 only)

User photo, v2@415, s = 1:
- **pan_right**: R is the right 16 % wedge. At native resolution the marble crack pattern continues across the old image edge with no seam, no gray fill, no streaks; its contrast is slightly lower than in the valid area (softer cracks). The high gray-band score is the white wall (section 2).
- **pan_left**: R at the left: the brown table surface with its plastic wrinkles continues; slightly softer than the valid region. No seam, no gray.
- **tilt_up**: R is the top band (24 % of the frame at frame 120). Right side: the red bag and the wall continue plausibly. **Top-left: a hazy light gray area with indistinct smeared shapes (a stack of objects with white highlights, blurred together) looks mushy**; no flat fill, but it is the weakest border in the set.
- **tilt_down**: R is the bottom band: dark brown table and baseboard continue with wrinkle texture, a little smoother than the valid table. No seam.
- **zoom_in**: the bottle grows smoothly, no distortion, edges stay sharp; nothing is revealed.
- **zoom_out**: the ring around the shrunk image is filled with the table surface at the left and the wall at the top (crop of the top-left corner at native resolution; I did not crop the bottom and right sides, the sheet shows the same kind of continuation); no seam, texture slightly softer than the valid area. No warping of the valid content.
- Half strength: the same motions at about half the travel; border content as above, smaller.

v1 vs v2 on the photo at matched amplitude (crops): **pan_right** - v2's 64 px wedge continues the marble cracks; v1's wedge is a duller, darker, nearly textureless gray-olive strip (the Submission 1 "gray band" look), although the scores rate v1 and v2 about equal (section 5). tilt_up (1.75 deg, 40 px band): v1 smoother
gray, v2 keeps faint wall cracks on the right; both continue the bag. zoom_out: v2's ring has sharper table/wall texture, v1 is smoother; both without seam.

New gallery cases (v1 vs v2 at matched amplitude): **ball_pan_left**: v1's left strip is a darker vertical band with an abrupt edge about 50 px wide (visible seam); v2's strip is seamless wood grain but the whole frame is dark (section 7).
**ball_pan_right / tilt_up / tilt_down**: both models continue the wood grain without a seam (v1's tilt_up band is softer, v2's grain is sharper); v2 is much darker and shows corner vignetting. **kitchen_zoom_out**: v1's ring at the top and left is a grainy, blotchy dark texture (visible noise); v2's ring continues shelf items and wall with more natural content; the blue
glare patch in the top-left corner is in the input. **heldout_zoom_in, living_room_zoom_in**: v1 and v2 both fine, no distortion, no artifact I could see. Direction of all seven matches the request (content moves in the same direction as v1; tracker gain positive 0.76 to 1.09).

No flicker was visible in the sampled frames; the only temporal artifact found is the ball dimming. I saw no warping of the scene in any video other than the genuine zoom/pan/tilt geometry.

## 9. What was not done

- **6b** GT reveal evaluation of the v1 adapter on the 8 E1 dev pairs (plus the 8th v2 pair `coco2017val_000000554328_ann633522__m0_zoom_out_u0896`): not run, the queue filled the 2 h 46 min. The dev sources and E1's `reveal_v2` folder were restored on the sandbox (CPU) for it. Command for the next session:
  `python scripts/eval_dev_reveal.py --project-root . --pairs 8 --model v1=artifacts/flow_poc_all/adapter_000300.safetensors:zeros:v1 --model v2=artifacts/v2_full/checkpoints/adapter_000415.safetensors --output-dir artifacts/reveal_eval` (restored E1 folder, ~9 videos = 55 min).
- **6c** extended-range (s = 1) videos of the 11 gallery cases in their gallery direction: not run (66 GPU minutes).
- **6a** v1 baselines: not needed, all 11 Submission 1 videos exist.
- No calibration `--verify` pass (section 6); one seed (42) only; no seed or checkpoint study of the ball dimming.

## 10. Index of outputs (all under `<checkpoints_dir>\G1\`)

- Sheets: `sheets\userphoto_s1_all_directions.png`, `sheets\userphoto_s05_vs_s1.png`, `sheets\userphoto_v1_vs_v2_matched.png`, `sheets\attn_auto_vs_cudnn.png`, `sheets\userphoto_s1_<dir>_border_crop_f120.png`, `sheets\userphoto_match_<dir>_border_crop_f120.png`,
  `sheets\gallery_<case>_v1_vs_v2.png`, `sheets\gallery_<case>_border_crop_f120.png`.
- Videos: `userphoto\attn_auto\pan_right\s1.00`, `userphoto\attn_cudnn\pan_right\s1.00`, `userphoto\full\{pan_left,tilt_up,tilt_down,zoom_in,zoom_out}\s1.00`, `userphoto\half\{pan_right,tilt_up,zoom_out}\s0.50`, `userphoto\calib_extra\...`, `userphoto_match\{v1,v2}\{pan_right,tilt_up,zoom_out}`, `gallery\matched\v2\<case>`, `v1_runs\<case>` (each folder `video.mp4`, `run.json`, `request.json`, `input.png`, metrics json).
- Tables: `all_runs_metrics.md/.json`, `calibration_fit.md`, `calibration_rows.json`, `calibration_plots\calibration_fit.png`, `global_drift_luma.json`, `global_luma_curves_summary.json`, `userphoto_metric_floor_input_image.json`, `userphoto\attn_compare_auto_vs_cudnn.json`.
- Job files that produced the videos: `queue\*.json`.
