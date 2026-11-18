# Strength calibration (v2)

The interface has one strength slider `s` in [0, 1] per motion. `s = 1` is the largest motion the adapter was trained on
(8 degrees of pan or tilt, a 1.25x focal zoom; `configs/data_v2.json`, `camera`). The adapter does not realise a requested
amount exactly (v1 measured gains of 0.915 to 1.13, and tilt was weaker than pan), so the slider is passed through a measured,
monotone, per-direction map `s -> u`, where `u` is the strength that is actually requested from the geometry and the
adapter. Code: `src/fpcamera/strength.py` (map, isotonic fit, inversion), `scripts/calibrate_strength.py` (sweep, fit, plots,
verify). Tests: `tests/test_strength.py`.

Calibrated vs physical strength: the slider `s` goes through this calibration (`s -> u`, `amplitude_source: "calibrated"`), while `--amplitude-deg` / `--zoom-ratio`
in `flow_infer` set the requested physical amplitude directly (`u = deg / 8`, `u = ln r / ln 1.25`, `amplitude_source: "physical"`) and bypass it. Use physical amplitudes for
like-for-like comparison with v1 (`generate_examples_v2.py --match-v1`); use `s` for the user-facing slider.

## 1. Definitions

For direction `d` in {pan_left, pan_right, tilt_up, tilt_down, zoom_in, zoom_out}:

* Requested strength `u`: the end-of-clip amplitude as a fraction of the trained maximum, linear easing over 121 frames:
  yaw or pitch `= +-u * 8 deg`, focal zoom ratio `= exp(+-u * ln 1.25)` (`flow_geometry.camera_path_v2`).
* Signed motion gain `g(d, u, image)`: `signed_motion_gain_pathwide` of `scripts/summarize_flow_adherence.py`, computed after
  `scripts/measure_flow_video.py` tracked the decoded video. With tracked displacements `a_t,i` and the displacements `e_t,i`
  expected from the requested homographies (RANSAC inliers, expected-visible, non-zero targets):

  `g = sum(a . e) / sum(e . e)`

  `g = 1` means the video moves exactly as requested; `g < 1` under-delivers; the sign handles direction.
* Observed amplitude (in units of the trained maximum): `a_obs = g * u`. The requested path already contains the factor `u`, so
  the amount actually seen is the gain times the request.

The goal of the calibration is `a_obs(u(s)) = s` for every `s`, so that `s = 0.5` shows half of the trained maximum
amplitude. The label of the UI is then physical: `physical_amplitude(direction, a)` (5.6 degrees, x0.86, ...). The label shows the
amplitude of the **requested** path after calibration, i.e. what the calibration claims the video will show.

## 2. Sweep

`scripts/calibrate_strength.py` generates `u in {0.25, 0.5, 0.75, 1.0}` x 6 directions x N development images (default 3,
evenly spread over the `development` split of `data/v2/sources_v2.json`; `--images name=path` overrides), the same seed
for every run, with the identity map (so `u = s` during the sweep). That is 72 videos for N = 3.

Steps default to 30 (`--steps`). The sweep measures where the content goes, not final texture, and 72 runs at 50 steps cost
about 40 % more GPU time. The final examples and the interface use 50 steps; the verify pass uses the same `--steps` as the sweep, so
the reported errors are for 30-step runs. If you want the verification at 50 steps, run the script again with `--force --steps 50`
into another `--output-dir` (the cost is another 72 + 54 runs).

The pipeline is loaded once (`flow_infer.load_inference_pipeline`, one GPU lease), the checkpoint is verified on the first call
only. It is resumable: a run folder with `run.json` is never regenerated, one with `measure.json` is never re-tracked,
an existing calibration file is kept (use `--force` to refit). `--no-generate` measures and fits whatever exists, on CPU.

## 3. Fit: isotonic regression through the origin

Per direction collect the samples `(u_k, a_k)` with `a_k = g_k * u_k`.

1. Merge samples at equal `u` into their weighted mean `abar(u_j)`, `j = 1..J` (weights = number of images).
2. Prepend the fixed knot `(0, 0)` (static request, no motion; carries a very large weight).
3. Solve the monotone least-squares problem

   `min_{f_0 <= f_1 <= ... <= f_J}  sum_j w_j (abar_j - f_j)^2,   f_0 = 0`

   with pool-adjacent-violators (`isotonic_increasing`): whenever a block mean exceeds the next block mean the two are merged
   into their weighted mean. The result `A(u)` is the piecewise-linear interpolation through `(u_j, f_j)`. The isotonic
   constraint absorbs tracker noise and non-monotone sampling ripples without assuming a parametric form (the gain need not be constant).

## 4. Inversion

For a target `s` in the grid `s = 0, 0.05, ..., 1` (the slider step), find the smallest `u` with `A(u) = s`:

`u(s) = min { u : A(u) >= s },  linear interpolation inside the segment`

The smallest solution is used because the isotonic fit can have flat stretches (many `u` give the same `a`); the least extrapolating one is chosen.

* If `A` never reaches `s` (`max A = A(1) < s`) the request is capped at the largest tested `u`
  (`u(s) = 1`), and `saturates_at = A(1)` is recorded when `A(1) < 1 - 0.02`. The slider then keeps moving the label but not the video
  above `s = saturates_at`; report that as a limit of the adapter. (The 0.02 tolerance is tracker noise.)
* If the adapter overshoots (`A(1) > 1`), `u(1) < 1` and the whole map is compressed.
* `u(0) = 0` always; the map is non-decreasing and clamped to [0, 1].

The file stores `s` and `u` lists (knots of the inverse map). At run time `u = interp(s; s_knots, u_knots)`
(`StrengthMap.s_to_u`).

## 5. File format

```json
{
  "version": 1,
  "created": "2026-11-17T08:00:00+00:00",
  "checkpoint_sha256": "<sha256 of the adapter this was fitted for, or null>",
  "max_amplitudes": {"yaw_deg": 8.0, "pitch_deg": 8.0, "zoom_ratio": 1.25},
  "directions": {"pan_left": {"s": [0, 0.05, ..., 1], "u": [0, 0.06, ..., 1]}, "...": "6 keys"},
  "fit": {"method": "...", "per_direction": {"pan_left": {"u_knots": [], "a_obs_isotonic": [],
          "a_obs_max": 0.93, "saturates_at": 0.93, "mean_gain": 0.92, "n_points": 12}}, "steps": 30, "seed": 42}
}
```

Validation (`StrengthMap`): version 1; `s` strictly increasing in [0, 1]; `u` non-decreasing in [0, 1]; the origin maps to 0; unknown
directions are an error; a missing direction falls back to identity with a warning. `configs/strength_calibration_v2.identity.json`
is the no-op file (`u = s`), the default until a calibration has been fitted. `configs/strength_calibration_v2.json` now exists (section 9) and is the default of `generate_examples_v2.py` and of the publish tool. At inference `flow_infer` warns when a calibration was
fitted for another checkpoint (`checkpoint_sha256`) or with other maximum amplitudes, and records the file path, its sha256, `s`, `u` and
the physical amplitude in `request.json` and `run.json` (`strength_control`).

## 6. Outputs

In `--output-dir` (default `artifacts/calibration_v2`):

* `plots/<direction>.png`: left panel, requested `u` vs observed `a_obs` (per-image points, the isotonic fit, the ideal diagonal);
  right panel, UI strength `s` vs observed amplitude (the predicted `A(u(s))` and, with `--verify`, the measured verification points).
* `calibration_report.md`: sweep table (`a_obs` per `u`, mean gain, `saturates_at`), the fitted `u(s)` at 0.25/0.5/0.75/1 and the verification table.
* `sweep_rows.json` (all per-video measurements), `verify_report.json`, `errors.json` if runs failed.
* `configs/strength_calibration_v2.json` (`--calibration-out`).

## 7. Verification (`--verify`)

Re-run `s in {0.33, 0.66, 1.0}` x 6 directions x the same images using the fitted file (so `u = u(s)`; each run records the file's
sha256). For every video compute `a_obs = g * u` from the requested path of that run and report
`|a_obs - s|` per direction and level, plus the overall mean and maximum. Directions with `saturates_at < 1` are expected to
miss at `s` above their saturation; read them together with `saturates_at`. A sensible acceptance bar is a mean absolute error of at most
about 0.05 (the tracker alone has a few percent of noise, see the synthetic-video test in `tests/test_strength.py`, which recovers a
known gain of 0.6 within 0.06).

## 8. Caveats

* The tracker measures apparent 2D image motion on sparse corners. Parallax, object motion, blur and textureless images bias `g`;
  that is why three images and the isotonic pooling are used, and why `cosine` is stored next to the gain.
* Pan/tilt are small-angle approximations of image shifts and zoom of an image scale; `a_obs` is relative to the *requested* homographies,
  not to physical camera angles.
* The map is specific to a checkpoint (and to `camera_min_sigma`, camera gain, the sampler and 30 vs 50 steps in principle). Refit after
  every training change; the checkpoint hash in the file makes a mismatch visible.
* Only `u in [0.25, 1]` is sampled; below 0.25 the map is the linear interpolation to the origin.

## 9. First fitted file: `configs/strength_calibration_v2.json` (G1, 2026-11-18, checkpoint 415)

Fitted by `scripts/fit_calibration_from_runs.py` (not by the 72-video sweep of section 2) from **28 existing v2@415 videos** (50 steps, seed 42, replicate fill): 22 from session G1 and 6 from E1, 8 different images
(photo, ball, computers, motorcycle, flowers, heldout, kitchen, living room). The same tracker gain, isotonic fit through the origin and inversion as above; the requested level of every video is its recorded `strength_u`, whether it was requested with the slider or as a physical amplitude.

| direction | videos | u levels | mean g by level |
|---|---|---|---|
| pan_left | 4 | 0.3125, 0.5, 1 | 0.972, 0.984, 1.005 |
| pan_right | 6 | 0.3125, 0.5, 1 | 1.034, 1.009, 0.990 |
| tilt_up | 5 | 0.2188, 0.5, 1 | 1.031, 1.017, 1.018 |
| tilt_down | 3 | 0.2188, 0.5, 1 | 0.762, 0.993, 0.973 (one video per level) |
| zoom_in | 4 | 0.5079, 0.75, 1 | 0.985, 1.018, 0.974 |
| zoom_out | 6 | 0.5 to 0.5079, 1 | 0.999, 1.010 |

Rule for the identity fallback (`--min-videos 3 --min-levels 2 --top-level 0.9`, levels closer than 0.02 merged): a direction is fitted only when it has that much data; otherwise it stays `u = s` and the report says so. All six directions passed, so none fell back.
Family-pooled gains (pan 1.009 / 0.996 / 0.995 at u = 0.3125 / 0.5 / 1, tilt 0.964 / 1.005 / 0.996 at 0.2188 / 0.5 / 1, zoom 0.994 / 1.018 / 0.998 at 0.5 / 0.75 / 1) show that the adapter realises the requested amplitude to within about +-3 %,
so the fitted map is the identity within tracker noise (largest |u - s| 0.047, tilt_down at s = 0.25 from a single 1.75 deg video; the others <= 0.021). Mean |a_obs - u| without calibration 0.014, in-sample residual after the fit 0.007.
Limits: knots are means of 1 to 3 videos; no held-out `--verify` run was made; `saturates_at` 0.973 / 0.974 for tilt_down / zoom_in rests on one image each. Data, per-video gains and plots: `docs/experiments/G1_GENERATION_CALIBRATION.md` section 6,
and (local, not in git) `G1\calibration_fit.md`, `calibration_rows.json`, `calibration_plots\calibration_fit.png`. The file stores the adapter hash (`checkpoint_sha256` `83df1e3f...`), the list of videos used (`fit.runs`) and the rule (`fit.rules`).
