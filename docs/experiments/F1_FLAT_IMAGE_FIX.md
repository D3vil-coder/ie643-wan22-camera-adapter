# F1: flat-image colour/brightness drift and frame distortion of v2@415 (Molab, 2026-11-18)

Branch `exp/flat-image-fix` (from tag `v2-final-415`). **Not merged into `master`.** The public HF repo is untouched; the model is the frozen v2_full step 415
(sha256 `83df1e3f...`, replicate fill, teacher anchor, `camera_min_sigma` 0.6, `camera_guidance` 1.0). All videos 1280x704x121, 50 steps, seed 42 unless the table says otherwise,
neutral caption, physical amplitude (calibration bypassed). Every number is one run. **Status: partial.** The sandbox gave 12 videos; the planned second half (more flat
images, the regression set with a fix) did not run (section 8). The recommendation in section 7 is therefore provisional.

## 1. Summary

* **The drift is caused by the adapter, not by Wan.** Base Wan on the ball image stays at 0.994 luma ratio and 0.7 chroma shift; v2@415 goes to 0.631 luma / 9.6 chroma (pan_left 2.5 deg).
  A *static* request (s = 0) already gives 0.901, so the adapter has a bias without any camera motion, and the motion request adds the rest (0.631).
* **It builds up with training and is produced at high noise levels.** Step 77: 1.076 (slightly brighter), step 250: 0.896, step 415: 0.631. A second seed: 0.641. Switching the adapter on only for
  sigma >= 0.8 (`camera_min_sigma 0.8`, 28 active steps instead of 39): 0.631, identical to the default, so the 11 steps with sigma in [0.6, 0.8) are not needed to produce the drift; it comes from sigma >= 0.8 (the first 28 of 50 steps).
* **The adapter's residual carries a large spatially-uniform part.** In the recorded residual of the adapter (adapter-on minus teacher, latent space), 25 to 36 % of its energy on the ball (15 to 24 % on the
  synthetic landscape) is the per-frame, per-channel spatial mean, i.e. global colour/brightness, not motion (section 4). This is consistent with the working hypothesis (Pass-B absolute-velocity training pulls global colour toward
  the data mean when the image does not anchor it); it is **not a proof** of that training mechanism, I did not run an ablation of the training loss.
* **Inference-time fix tested: `--residual-lowfreq blur:8` (F2) and `mean` (F1).** Remove the low spatial frequencies of the adapter's residual at each adapter-active step.
  Ball: both remove the dimming (luma 0.997 / 0.999 at frame 120, chroma 2.1 / 2.2). Synthetic landscape (zoom_out x1.25): **F2 removes the sky over-saturation (chroma shift 18.6 -> 2.5) and the first-frames burst; F1 only partly does**
  (7.0, yellow-green meadow, a luma jump of 8.3 gray levels in the first second).
* **Motion: not resolved.** The repository's sparse tracker reports a gain of 0.235 (F1) and 0.135 (F2) on the ball against 0.996 for the default. Three independent measurements (ball centroid, patch-wise phase correlation, visual position
  of the wood knot) say the macro motion is kept (about -8 to -10 % versus the default, v1 is -12 %), so I believe the tracker is failing on the changed fine texture, but I have not proven it. The acceptance rule (gain within +-5 % of the default) is **not met**:
  not on the ball (tracker 0.14 / 0.24; independent measures -8 / -10 %), and not on the synthetic zoom (F2 gain 1.245 vs 1.352 = -8 %).
* **Regression set (test photo, motorcycle, heldout, COCO person): not run with a fix.** Only default-adapter baselines exist (all within +-6 % luma, no spikes). So nothing can be said yet about "unchanged within noise" for a fix.
* **Per-frame distortion:** reproduced on the synthetic landscape (frames 1 to 4: grain-like speckle at frame 1, sudden jump to a saturated blue, motion-compensated residual spikes). Not seen in any ball video, in the E1/G1 baselines
  of the regression images, or in base Wan. Details in section 3.

## 2. What ran

Sandbox: RTX PRO 6000 Blackwell, started 12:00:49 UTC (17:30 IST) 2026-11-18, recovery finished 12:11 UTC (G1 recipe: `molab_recover --only probe code deps token model extras`, no training cache; run `config.json` and
checkpoint 415 / 250 restored from the private HF repo, checkpoint 77 pushed from the laptop because the remote retention had pruned it), queue runner started 12:14:53 UTC. `scripts/generate_batch.py`, one loaded pipeline,
job files made by `scripts/make_ff_queue.py`, `hf_backup.py watch-dir artifacts/ff -> eval/flatfix` of the private dataset repo ran as a daemon. 12 jobs, all finished (adapter runs about 355 s, base Wan about 235 s).
The queue was empty after that and the GPU idled: **The second half (extras and regression) had not been queued when the session was interrupted; the sandbox answered HTTP 410 when work resumed (21:00 IST).**
I restored the finished runs from the private backup (`eval/flatfix`, 108 files, all sha256 of the videos equal to the ones already pulled).

Incident worth knowing (about 2 minutes of GPU time): I restarted the runner with a SIGTERM to its exact PID to pick up new code; the killed process left `artifacts/gpu.lock`, which `gpu_lease` does **not** clear for a dead pid
(only the train lock logic does), so the new runner exited with "already holds artifacts/gpu.lock". Remove the lock by hand after confirming the PID is dead (`/proc/<pid>` was a zombie, GPU memory 3 MiB).

Local outputs (not in git): `<checkpoints_dir>\flatfix\` : `runs\<image>\<variant>\` (video.mp4, run.json, request.json, border_quality.json, measure.json, **frame_stats.json, frame_strip.png, frame_curves.png**),
`sheets\`, `table_ball.md`, `table_synth.md`, `measure_phase1.json`, `trace_summary.json`, `hf_eval\` (raw restore), `inputs\`, `queue\`. Private backup: dataset repo `D3vil-Reap3r/ie643-camera-v2-cache`, folder `eval/flatfix`.
Inputs (`inputs\`): `ball.png` (G1's), `synthetic_landscape.png` (the notebook sample, `scripts/make_flat_inputs.py`; its first frame matches D1's at 33 dB after video coding, the D1 file hash differs because the D1 input is not stored),
`table_object.png` (synthesized, not run), `airplane_sky.png` and `coach_person.png` (COCO val pairs of E1, not run), `motorcycle.png`, `heldout.png`, `user_photo.jpeg` (a user-supplied test photo: personal, not distributed, only on the local machine and on the sandbox, never in git).

New code on the branch (CPU-tested: the whole suite, 310 tests of which 22 are new, passes; defaults unchanged):

| commit (original history) | content |
|---|---|
| `32c5bbc` | `--residual-lowfreq mean|blur:K` in `flow_anchor.frozen_camera_difference` / `flow_infer.generate`, `scripts/make_flat_inputs.py` |
| `74e53cf` | `scripts/analyze_frames.py` (per-frame analysis, strips, curves), `scripts/summarize_ff.py`, `scripts/make_ff_queue.py` |
| `229ffc7` | `trace_residual` diagnostic (per-step residual statistics in run.json) |
| `233ae48` | `--residual-lowfreq-mask valid` (validity-weighted estimate; **implemented and unit-tested, never run on GPU**) |

## 3. Per-frame analysis (all 121 frames of every video)

`scripts/analyze_frames.py` computes per frame: global luma and Lab a*/b* relative to frame 0, a valid-region luma ratio (frame 0 warped by the request's homography, eroded), the motion-compensated residual
(Farneback flow t -> t-1 at half resolution; `mc_resid_dm` also removes the global brightness change so a dimming ramp is not counted as distortion), high-frequency energy (luma minus Gaussian sigma 2), and spike detection
(robust z of the deviation from a same-phase running median > 4, plus an absolute and a relative floor). Strips: every 10th frame, then the frames at the detected spikes with their predecessor.

Findings that matter for reading any per-frame number:

* **Period-4 structure in every video** (also base Wan and v1): the Wan VAE decodes four frames per latent plane, so frames 1, 5, 9, ... have a different statistic (mc residual 2x the neighbours, HF jitter). The detector uses a
  same-phase baseline; a plain "z > 4 against the video median" flags these frames in every video (4 to 20 frames), so it cannot be used.
* **Ball (all 8 adapter variants, base Wan, v1): no per-frame spike** in the mc residual or in the HF energy. The ball defect is a smooth drift: the luma ramp of the default is 1.00 -> 0.90 (frame 10) -> 0.77 (20) -> 0.67 (40) -> 0.63 (120),
  a* +4, b* -9 (browner), max luma step 4.6 gray levels per frame during the ramp, no flicker.
* **Synthetic landscape, default v2@415 (zoom_out x1.25): a real frame distortion in the first four frames.** Frame 1 (the first generated frame after the clean anchor) shows a fine mosaic/speckle texture in the sky and a violet-blue cast
  (HF energy 3.4x the video median, base Wan 1.6x); frames 2 to 4 jump to a saturated blue (luma 1.00 -> 0.93, chroma 14 within four frames; mc spikes at frames 1, 2, 3, 4). It is invisible in a 0/40/80/120 sheet. Sheet: `sheets\synth_first_frames.png`
  (rows: base, default, F1, F2; columns frames 0, 1, 2, 4, 8, 30). After frame 4 the sky stays electric blue and the meadow drifts to lime green (`sheets\synth_variants_f0_20_40_80_120.png`).
* I did not see this in the other videos I could analyze (ball, test photo, motorcycle, heldout: no spike). The airplane (grey sky) and the table image, the most likely candidates for more of the same, were not run.

## 4. Phase 1: diagnosis (ball image, pan_left 2.5 deg = u 0.3125, G1's setting; synthetic landscape zoom_out x1.25)

Ball (Y = mean luma ratio to frame 0, valid-region luma ratio in `frame_stats.json` agrees to 0.01; chroma = hypot of the change of the frame-mean Lab a*, b*; ball dx = blue-ball centroid shift, texture dx = median patch-wise phase-correlation shift of the wood, both px at frame 120):

| run | Y@4 | Y@20 | Y@40 | Y@80 | Y@120 | chroma@120 | max step | mc spikes | tracker gain / cos | ball dx | texture dx |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base Wan (no adapter) | 0.997 | 0.988 | 0.994 | 0.995 | **0.994** | 0.7 | 0.52 | 0 | n/a | 0.0 | 0.0 |
| v2@415, s = 0 (static request) | 0.991 | 0.937 | 0.921 | 0.907 | **0.901** | 3.0 | 1.87 | 0 | n/a | -2.9 | ~0 |
| v2@77 | 1.030 | 1.057 | 1.059 | 1.066 | **1.076** | 3.0 | 1.58 | 0 | 1.033 / 0.999 | 50.1 | 52.6 |
| v2@250 | 0.995 | 0.967 | 0.936 | 0.914 | **0.896** | 5.1 | 1.52 | 0 | 0.884 / 0.997 | 40.2 | |
| **v2@415 default (G1 video)** | 1.001 | 0.768 | 0.671 | 0.650 | **0.631** | 9.6 | 4.64 | 0 | 0.996 / 0.997 | 32.9 | 51.5 |
| v2@415, seed 43 | 0.996 | 0.788 | 0.687 | 0.664 | **0.641** | 11.0 | 3.48 | 0 | 1.035 / 0.998 | 35.2 | |
| v2@415, `camera_min_sigma` 0.8 | 0.997 | 0.778 | 0.662 | 0.641 | **0.631** | 9.5 | 4.36 | 0 | 0.983 / 0.996 | 32.4 | |
| v1 (Submission 1) | 1.019 | 1.051 | 1.048 | 1.050 | 1.051 | 1.9 | 1.28 | 0 | 0.859 / 0.993 | 45.6 | 45.5 |

Reading:

1. Base Wan does not drift on this image. The defect is introduced by the adapter (v1 does not have it).
2. The adapter dims by 10 % even with a **static** request, and by 37 % with the pan request: there is an adapter-wide bias plus a motion-dependent part. A static request should be the identity.
3. The defect is not monotone in training: step 77 is slightly *brighter* (+7.6 %), 250 is -10 %, 415 is -37 %. It grew over the training that followed the Pass-B objective.
4. Another seed gives the same (0.641): it is not a lucky seed.
5. A shorter adapter window (sigma >= 0.8: 28 active steps instead of 39) gives exactly the same dimming and the same chroma shift (0.631 / 9.5): the drift is produced at sigma >= 0.8, where the adapter's
   residual is also largest (section below). Shortening the window (variant F3) to 0.8 cannot help, a window that is much shorter would also cut the motion (sigma >= 0.9 is 18 steps; not tried), and lowering `camera_guidance` was not tried (the motion would weaken with it).
6. The motion itself is realised in all of them (tracker gain 0.88 to 1.04); in the dimming videos the ball moves 33 px while the wood texture moves 51 px (the ball also drifts 10 px upward): the drift also disturbs the geometry a little.

Synthetic landscape, zoom_out x1.25 (u = 1):

| run | Y@4 | Y@20 | Y@40 | Y@80 | Y@120 | chroma@4 | chroma@120 | max step | mc spikes | HF frame1 / median | tracker gain |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base Wan | 0.989 | 0.975 | 0.968 | 0.948 | 0.937 | 0.9 | 2.7 | 0.71 | none | 1.6 | n/a |
| v2@415 default | 0.932 | 0.937 | 0.934 | 0.932 | 0.930 | **14.2** | **18.6** | 5.51 | frames 1, 2, 3, 4 | **3.4** | 1.352 |

Here base Wan itself loses 6 % luma over the clip (it hallucinates a camera and leaves entering from the top-left in this static request, so part of that is scene content). The adapter's problem is the chroma: a saturation jump within four frames (a* +10, b* -10), the
sky ending electric blue and the meadow lime green. The D1 notebook video (same image, calibrated u 0.990) had the same numbers (Y 0.929, chroma 18.7, spikes at frames 1 to 4), so this is reproducible and not a seed effect of the notebook.

### Residual statistics (mechanism evidence)

`trace_residual` records, at every adapter-active step, the rms of the raw residual (adapter(requested) - teacher, latent frames 1.., 48 channels) and the rms of its spatial mean per frame and channel, and of its Gaussian low-pass (sigma 8 cells = 128 px).
Energy shares (rms ratio squared) averaged over sigma buckets, default trajectory:

| run | sigma 0.9-1 : rms, share(mean), share(low-pass 8) | 0.8-0.9 | 0.7-0.8 | 0.6-0.7 |
|---|---|---|---|---|
| ball v2@250 | 0.207 / 0.26 / 0.28 | 0.102 / 0.23 / 0.27 | 0.096 / 0.26 / 0.30 | 0.100 / 0.25 / 0.29 |
| ball v2@415 seed 43 | 0.213 / 0.32 / 0.37 | 0.091 / 0.27 / 0.32 | 0.091 / 0.26 / 0.30 | 0.096 / 0.25 / 0.28 |
| synthetic v2@415 | 0.288 / 0.24 / 0.49 | 0.129 / 0.19 / 0.31 | 0.119 / 0.18 / 0.26 | 0.118 / 0.15 / 0.23 |

A quarter to a third of the residual energy is a per-frame, per-channel constant and about a third is below 128 px, for a signal that should be mostly local displacement. The residual is twice as large at sigma > 0.9 as later. Caveat: in the fix runs the
same statistic of the raw (unfiltered) residual is 2 to 3x larger (ball F1: 0.41 / 0.31 / 0.27 / 0.25), because the removed component is requested again at every step; the adapter keeps pushing toward the colour state that the filter denies it.
It also means the filtered residual is a different, smaller signal than the one the adapter was trained to give.

## 5. Phase 2: inference-time fix

Flag: `--residual-lowfreq mean` (F1) or `--residual-lowfreq blur:K` (F2, K in latent cells, 1 cell = 16 px; here K = 8 = 128 px). At each adapter-active step the residual (adapter-on minus teacher, FP32, per latent frame and channel) is
high-passed before it is added (`flow_anchor.lowfreq_highpass`). Default off; with the flag absent the code path and the video are unchanged (unit-tested bit-for-bit on the mock anchor). No retraining, no extra forward pass.

### Ball (same job as the default)

| run | Y@20 | Y@40 | Y@120 | valid Y min | chroma@120 | max step | mc spikes | tracker gain / cos | ball dx | texture dx (patches) | gray-band | local / v2 score | HF R/V | seam |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 0.768 | 0.671 | 0.631 | 0.618 | 9.6 | 4.64 | 0 | 0.996 / 0.997 | 32.9 | 51.5 | 0.000 | 0.215 / 1.221 | 1.80 | 0.995 |
| F1 `mean` | 0.996 | 0.994 | **0.997** | 0.983 | 2.1 | 0.39 | 0 | **0.235** / 0.899 | 44.2 | 46.1 | 0.000 | 0.236 / 0.835 | 1.17 | 0.927 |
| F2 `blur:8` | 0.996 | 0.995 | **0.999** | 0.986 | 2.2 | 0.38 | 0 | **0.135** / 0.865 | 45.4 | 47.6 | 0.000 | 0.238 / 0.816 | 1.16 | 0.940 |

* Drift fixed (luma within 0.3 % of frame 0, chroma 2.1 to 2.2, a bit above base Wan's 0.7; max luma step 0.4). No spikes. Border: no gray (0.000), the local ring score is within noise (0.236 vs 0.215) and the reference-free v2 score and HF ratio are lower (better) than the default.
* **Motion: the tracker gain collapses (0.235 and 0.135, cosine 0.90 / 0.87) while the other three measurements keep the motion.** The ball moves 44 to 45 px (v1: 45.6; the default's 32.9 is the one that is off), the wood texture moves 46 to 48 px versus 51.5 for the default
  (-10 % and -8 %; v1 -12 %), and the wood knot sits at the same place in the crops of default, F1, F2 and v1 at frame 120 (`sheets\ball_knot_crop.png`). The fine grain correlates less with frame 0 (phase-correlation response at frame 40: default 0.74, F1 0.46, F2 0.18),
  so fine texture is partly re-synthesised and the sparse tracker, which follows that fine texture, loses it. I cannot tell from these numbers whether a viewer would see the fine grain "swim"; at strip size (`runs\ball\v415_f2blur8\frame_strip.png`) the video looks clean, the colour is stable
  and the ball and wood move consistently. **The pre-registered acceptance (tracker gain within +-5 % of the default) is not met on the ball.** Playing the video is the missing check.

### Synthetic landscape (zoom_out x1.25)

| run | Y@4 | Y@40 | Y@120 | chroma@4 | chroma@120 | max step | mc spikes | tracker gain / cos | gray-band | local / v2 score | colorf R/V | HF R/V | seam |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base Wan | 0.989 | 0.968 | 0.937 | 0.9 | 2.7 | 0.71 | none | n/a | n/a | n/a | n/a | n/a | n/a |
| default | 0.932 | 0.934 | 0.930 | 14.2 | 18.6 | 5.51 | 1, 2, 3, 4 | 1.352 / 0.997 | -0.006 | 0.080 / 0.442 | 1.39 | 1.09 | 1.001 |
| F1 `mean` | 0.975 | 0.922 | 0.927 | 3.7 | 7.0 | **8.32** | frame 4 (+ 14 HF-spike frames) | 1.334 / 0.997 | -0.013 | 0.199 / 0.824 | 1.52 | 1.49 | 0.997 |
| F2 `blur:8` | 0.985 | 0.952 | 0.929 | 0.9 | **2.5** | 0.75 | none | 1.245 / 0.994 | -0.014 | 0.321 / 0.534 | 1.48 | 1.09 | 1.018 |

* **F2 fixes the image**: chroma at frame 4 is 0.9 (base 0.9, default 14.2), no mc spike, max luma step 0.75 (base 0.71), the sky keeps its colour and the meadow stays green (`sheets\synth_variants_f0_20_40_80_120.png`, `sheets\synth_full.png`). The frame-1 speckle is reduced (HF 2.5x median vs 3.4x default, 1.6x base) but not gone.
* **F1 is not enough on this image**: the frame-1 colour jump is smaller but the sky still saturates at the top, the meadow goes yellow-green at the bottom (a spatially varying low-frequency drift that a per-frame mean cannot remove) and there is a new luma jump of 8.3 gray levels in the first second (frames 1, 3, 4, 5, 10, 25 to 29). F1 is worse than F2 here.
* Border: F2 keeps the revealed ring continuous but the top band of the ring is a more saturated blue than the (now correct) sky below it (`sheets\synth_top_f120.png`: default electric blue everywhere, F2 a soft, mildly stronger blue band, F1 in between). The local ring score is worse than the default (0.321 vs 0.080), which is the same mismatch;
  the default scores well only because its whole sky is saturated. This is what the unrun `--residual-lowfreq-mask valid` is meant to address (estimate the low-frequency part from valid cells and subtract it everywhere, so the band cannot keep the adapter's cast).
* Tracker gain on the zoom: default 1.352, F1 1.334, F2 1.245 (F2 is -8 % of the default; neither is near the +-5 % rule, and the default itself is 35 % over the requested zoom on this textureless image, so I do not trust the tracker on it either; G1's other zooms are within +-3 %).

## 6. Visual observations (I looked at these; I did not play the videos)

* Ball default: smooth darkening of the whole frame to a brown (G1 also noted darker corners; I did not re-check that); strip `runs\ball\v415_default\frame_strip.png`. Ball F1/F2: the wood stays light oak for the whole clip, the ball keeps its blue; the knot, grain lines and the ball shadow move consistently.
* Ball s = 0: the same oak, 10 % darker at the end, no motion (ball -2.9 px).
* Synthetic default: frame 1 shows a speckled sky texture with a violet cast; from frame 2 the sky is saturated blue; at frame 120 the top ring is flat electric blue and the meadow lime green. F2: no speckle burst beyond the input grain, sky colour as the input, a mild blue band in the top ring.
* Sheets (all under `<checkpoints_dir>\flatfix\sheets\`): `ball_variants_f0_20_40_80_120.png`, `synth_variants_f0_20_40_80_120.png`, `synth_full.png`, `synth_first_frames.png`, `synth_top_f120.png`, `ball_knot_crop.png`.
  Per run: `runs\<image>\<variant>\frame_strip.png` (every 10th frame + spike frames) and `frame_curves.png` (the six per-frame curves). Videos: `runs\<image>\<variant>\video.mp4`.

## 7. Recommendation (provisional)

1. **Do not change any default.** v2@415 stays as published; nothing in this branch is on by default.
2. **Candidate fix: `--residual-lowfreq blur:8`**, in `generate()` `residual_lowfreq="blur:8"`; in a queue job file `"generate_kwargs": {"residual_lowfreq": "blur:8", ...}`. On the two flat images tested it removes the drift and the early-frame burst; it is cheap (no extra forward, no retraining).
   `mean` is not recommended (insufficient on the synthetic image, introduced a luma jump). Open problems that block adopting it: (a) the repository tracker's motion gain on the ball (0.135) disagrees with three other measurements (-8 %), needs the video to be played and a better tracker check;
   (b) the regression set has not been run with it; (c) the revealed ring on zoom_out is slightly bluer than the sky; (d) only two flat images, one seed each. The acceptance rule (flat drift and spikes gone **and** regression unchanged within noise) is therefore **not yet established**.
3. Next GPU session (needs a new sandbox, section 9), in this order: regression set and the other flats with `blur:8` (9 runs, ~55 min), then the untested variants (`blur:8` + `--residual-lowfreq-mask valid`, `blur:4`, `blur:16`, and the frozen-anchor reference `--camera-residual-anchor frozen`, which cancels the static bias but not the motion-dependent part) on ball and synthetic (about 50 min).
4. If the regression set shows a loss or the tracker issue is real, the fix must be in training (section 10).

## 8. What did not finish

* Flat set: airplane (grey sky) and the synthesized table image: **not run** (neither default nor fix). So "flat images" = 2 images (ball, synthetic).
* Regression set with a fix (test photo, motorcycle_pan_right, heldout_zoom_out, COCO person `coach`): **not run**. Baselines exist for photo (pan_right 8 deg, G1), motorcycle (pan_right 2.5 deg, E1) and heldout (zoom_out x1.12, E1); all within +-6 % luma, no spikes, chroma shift 0.7 to 2.4;
  the coach has no baseline either. Photo/motorcycle/heldout baselines come from earlier sessions (same checkpoint, seed, steps, replicate fill; older code version, same math).
* `--residual-lowfreq-mask valid`, `blur:K` other than 8, `camera_guidance` below 1, frozen anchor: implemented where code was needed, **none run on GPU**.
* Phase 3 (training): not started (plan below).
* Not measured: LPIPS or GT reveal metrics of the fix; the video was not played.

## 9. Next GPU session: job files and time estimate

Job files are produced by `scripts/make_ff_queue.py` (`--defaults`, `--fix '{"residual_lowfreq":"blur:8"}' --tag f2blur8 --prefix 25`, `--only airplane table` etc.); images are pushed to `artifacts/ff_inputs`, which is where the jobs read them (the images are in
`<checkpoints_dir>\flatfix\inputs\`). The queue runner and daemon recipe is the one of section 2. About 6.0 min per adapter run (355 s plus load), 7.5 min with the frozen anchor, 4 min for base Wan; recovery about 11 min (G1 recipe, checkpoint 77 is not needed).

| block | runs | GPU minutes |
|---|---|---|
| A1 other flats: airplane default + F2, table default + F2 | 4 | 24 |
| A2 regression: coach default + F2, photo / motorcycle / heldout F2 (baselines exist) | 5 | 30 |
| B1 ball and synthetic: F2 + valid mask, blur:4, blur:16 | 6 | 36 |
| B2 frozen anchor (`camera_residual_anchor: "frozen"`) on ball and synthetic | 2 | 15 |
| **total** | 17 | **about 105 min + 11 min recovery** (one sandbox) |

Analysis afterwards is CPU only (`analyze_frames.py`, `measure_runs.py`, `summarize_ff.py`, about 1 min per video).

## 10. Phase 3 plan (not started): short fine-tune from step 415 in a new run `v2_flatfix`

Only if the inference fix fails the regression set or the motion check. Back up to the private repos only (`runs/v2_flatfix`, `eval/flatfix`), session discipline of `SESSION_CHUNKING.md`.

* **Loss term (low-frequency consistency).** In Pass A the adapter output is teacher-relative with target `(z_s - z_m)/sigma` added to the teacher; add `L_lf = || LP(pred - teacher) - LP(target - teacher) ||^2` with LP = per-frame spatial mean plus the Gaussian low-pass of 8 cells,
  weight about 1, so that the part of the adapter's residual below 128 px has to equal what the data says (near zero for a pan or tilt, the known geometry for a zoom), and for the static request (p_static = 0.1, extend to 0.3) to be exactly zero. This directly targets the static bias (0.901 at s = 0) and the motion-dependent drift.
* **Pass B.** Keep it but add the same low-frequency match to the teacher at sigma > 0.7 (where the colour is decided), or lower its weight there; Pass B was introduced for the revealed region and the gray-border fix, whose gains must not be lost (check with `validate_v2` L_B|R and the border metrics).
* **Data.** Add flat and synthetic sources (smooth gradient walls, plain tables, flat skies, a few synthetic landscapes), encode with the existing builder (about 6.8 s per clip); a few hundred clips are enough for a short run.
* **Cost.** 100 to 150 updates at about 41 s = 70 to 100 min, plus about 10 min to restore the cache shards and the evaluation (the same 17-run queue, ~105 min): one 3 h sandbox for the fine-tune and a second for the evaluation, or the fine-tune plus a reduced evaluation.
* **Acceptance.** Ball, synthetic and the extra flats within +-5 % luma and chroma shift below 3; no per-frame spike; regression set unchanged within noise; tracker gain +-5 %.

## 11. Reproduction

```
python scripts/make_flat_inputs.py --out inputs/                           # synthetic landscape, table image
python scripts/make_ff_queue.py --queue-dir artifacts/ff_queue --groups 00_p1_ball 02_f_ball ...   # see --list
python scripts/generate_batch.py --project-root . --queue-dir artifacts/ff_queue \
    --checkpoint artifacts/v2_full/checkpoints/adapter_000415.safetensors --steps 50 --seed 42
python scripts/analyze_frames.py --root runs --workers 6                   # frame_stats.json, frame_strip.png, frame_curves.png
python scripts/measure_runs.py runs/ball/v415_f2blur8 --workers 4          # border metrics + tracker gain
python scripts/summarize_ff.py runs --runs ball/v415_default ball/v415_f2blur8 --md t.md --sheet s.png
python -m fpcamera.flow_infer ... --motion zoom_out --zoom-ratio 1.25 --residual-lowfreq blur:8      # single video
```
