# P1b: v1 + replicate guide fill at inference (training-free baseline), Molab 2026-11-16

Question: does swapping the zero guide fill of the v1 adapter for a replicate (border-padding) fill at inference
weaken the camera motion relative to plain v1? It is also the paper's training-free baseline row.

Setup: `adapter_000300`, `guide_fill_mode=replicate`, `camera_min_sigma=0.6`, 50 steps, seed 42, default prompt,
Molab RTX PRO 6000 shared with a running cache build (`--no-lease`, ~17 min/video instead of the ~6 min estimate).
Magnitudes were taken from the original v1 runs (their `run.json`) so the comparison is like for like: pan 0.5, tilt 0.35,
zoom 1.0 (not 1.0 everywhere). Driver: `scripts/diagnose_v1_borders.py --variants fill_replicate --cases name=path:mode[@magnitude] --no-lease`
(both options added in this phase). Tracker: `measure_flow_video.py` -> `summarize_flow_adherence.py` (no reference tracks, as for v1).
v1 numbers are from the Submission 1 `path_adherence.json` files. Border metrics for v1 were recomputed locally from the
Submission 1 `video.mp4` (mean over every 4th frame with R > 1%, decoded mp4 frames), for v1+replicate on the Molab
output video the same way, so both sides use identical code and frame source.

Coverage: 8 of 11 cases completed (deadline: no new video after 17:55 UTC). Not run: `ball_pan_right`, `ball_tilt_down`,
`living_room_zoom_in`. Every family (pan, tilt, zoom_in, zoom_out) has at least one case. Single seed, so differences of a
few percent are within seed noise (not measured).

## Motion (path-wide signed gain, 1.0 = requested motion fully realised)

| case | magnitude | v1 gain | v1+rep gain | ratio | v1 cos | rep cos | v1 ADE/FDE px | rep ADE/FDE px | v1 cov | rep cov |
|---|---|---|---|---|---|---|---|---|---|---|
| heldout_zoom_out | 1.00 | 1.245 | 1.336 | 1.07 | 0.996 | 0.986 | 4.3 / 7.3 | 6.1 / 11.5 | 1.00 | 1.00 |
| motorcycle_pan_right | 0.50 | 1.106 | 1.147 | 1.04 | 0.999 | 0.997 | 3.1 / 5.6 | 4.6 / 7.4 | 1.00 | 1.00 |
| kitchen_zoom_out | 1.00 | 1.130 | 1.103 | 0.98 | 0.998 | 0.997 | 2.8 / 5.3 | 2.6 / 4.8 | 1.00 | 1.00 |
| computers_pan_left | 0.50 | 0.915 | 0.962 | 1.05 | 0.997 | 0.999 | 2.7 / 5.3 | 23.1 / 25.8 | 1.00 | 0.99 |
| flowers_tilt_up | 0.35 | 0.925 | 1.019 | 1.10 | 0.997 | 0.997 | 1.8 / 4.3 | 1.5 / 2.4 | 1.00 | 1.00 |
| ball_pan_left | 0.50 | 0.859 | 0.877 | 1.02 | 0.993 | 0.997 | 4.3 / 8.9 | 3.4 / 7.5 | 1.00 | 1.00 |
| ball_tilt_up | 0.35 | 0.568 | 0.563 | 0.99 | 0.892 | 0.987 | 9.3 / 18.2 | 18.6 / 26.5 | 1.00 | 0.99 |
| heldout_zoom_in | 1.00 | 1.146 | 1.359 | 1.19 | 0.999 | 0.992 | 2.9 / 4.8 | 7.1 / 12.8 | 1.00 | 1.00 |

ADE/FDE are the missing-penalty versions in px (a track the tracker loses counts as one image diagonal). Coverage = mean fraction of
expected-visible tracks found. Note `computers_pan_left` v1+rep ADE 23.1 px is dominated by a few lost tracks (visible-only ADE 1.6 px,
coverage 0.99), not by a motion error; `ball_tilt_up` ADE/FDE rise similarly (visible-only 7.8 vs v1 9.3).

### By motion family (mean path-wide gain, n cases)

| family | n | v1 | v1+replicate | ratio |
|---|---|---|---|---|
| zoom (out x2, in x1) | 3 | 1.174 | 1.266 | 1.08 |
| pan | 3 | 0.960 | 0.995 | 1.04 |
| tilt | 2 | 0.747 | 0.791 | 1.06 |

## Border quality (revealed region R, mean over frames with R > 1%)

| case | v1 artifact | rep artifact | v1 color ratio | rep color ratio | v1 HF ratio | rep HF ratio |
|---|---|---|---|---|---|---|
| heldout_zoom_out | 0.603 | 0.359 | 0.725 | 0.974 | 0.738 | 0.715 |
| motorcycle_pan_right | 0.760 | 0.680 | 1.040 | 0.902 | 0.262 | 0.458 |
| kitchen_zoom_out | 0.945 | 0.333 | 0.708 | 0.766 | 0.355 | 1.101 |
| computers_pan_left | 1.255 | 0.567 | 0.695 | 1.504 | 0.068 | 0.458 |
| flowers_tilt_up | 0.181 | 0.126 | 1.119 | 1.268 | 1.567 | 2.646 |
| ball_pan_left | 0.619 | 0.521 | 0.512 | 0.523 | 1.000 | 1.486 |
| ball_tilt_up | 0.438 | 0.420 | 0.596 | 0.592 | 1.122 | 1.532 |
| heldout_zoom_in | n/a | n/a | n/a | n/a | n/a | n/a |

`heldout_zoom_in` has R = 0 (zoom-in reveals nothing), so border metrics do not apply. artifact_score = lower is better (>0.40 flagged, see P1_RESULTS.md); colorfulness and HF ratios are R/valid, 1.0 = same as valid area.

## Answer

No: replicate fill does not weaken the motion. Path-wide gain changes by a factor of 0.98 to 1.19 per case (mean ratio about
1.06), i.e. equal or slightly stronger motion in every family: zoom 1.17 -> 1.27, pan 0.96 -> 1.00, tilt 0.75 -> 0.79. Cosine is unchanged
(0.99+) except `ball_tilt_up`, where it improves (0.89 -> 0.99). The largest increase is `heldout_zoom_in` (1.15 -> 1.36), a
case with R = 0, where the only difference is the guide in already-valid regions' neighbourhood; the guide is not multiplied by validity
in replicate mode so the conditioning is slightly stronger, which plausibly explains gains above 1. No case lost more than 2%. With n = 1 seed per case
we cannot call the 4-10% gains significant, only that there is no sign of a reduction.

Border side: the artifact score falls in 6 of 7 applicable cases and is flat (0.44 -> 0.42) in `ball_tilt_up`; largest drops are
`computers_pan_left` 1.26 -> 0.57, `kitchen_zoom_out` 0.95 -> 0.33, `heldout_zoom_out` 0.60 -> 0.36. Effects are not uniform:
`motorcycle_pan_right` only 0.76 -> 0.68 (its colorfulness ratio actually drops 1.04 -> 0.90), and several rep colorfulness ratios exceed 1 (computers 1.50),
meaning R is more saturated than the valid area, so the score is not a complete judge; see the contact sheets.
Consequently v1 + replicate fill is a cheap, training-free improvement of borders without a motion penalty and is the right baseline for any trained
fix (v2) to beat.

## Artifacts (local scratch, not in git)

`scratchpad/p1b/<case>/{video.mp4, path_adherence.json, border_quality.json, run.json, request.json, small_{040,080,120}.jpg, contact_sheet.jpg}`
and `scratchpad/p1b/contact_sheet_all.jpg` (rows: v1 from Submission 1, v1+replicate; columns frames 40/80/120).
