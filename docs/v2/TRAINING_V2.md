# v2 training: hybrid objective, warm start, resumable runs

Code: `src/fpcamera/flow_train_v2.py` (trainer), `src/fpcamera/checkpointing.py` (fault tolerance),
`scripts/train_v2.py` (CLI), config `configs/train_v2.json` (1000 updates). The pilot is the first 150 updates of this same run (`--stop-after-updates 150`); the separate pilot config was removed.
Tests: `tests/test_train_v2_logic.py` (CPU, tiny mock transformer and the real `FlowControlLoRA`).

## 1. Objective

Notation. `z_s` is the static clip latent and `z_m` the requested (moving) clip latent of the same source, both
`[48,31,44,80]`. `first` is the clean first-image latent plane. `R = 1 - valid` is the revealed mask at latent
resolution, taken from channel 48 of `build_image_condition(..., fill_mode)` built from the **requested**
homographies, first plane excluded, shape `[1,1,30,44,80]`, broadcast over the 48 channels.

Wan is rectified flow: `x_s = (1-s) z + s ε`, the network predicts the velocity `v = ε - z`.
v1's teacher-relative target is consistent with that: for an ideal teacher `(x_s - z_s)/s = ε - z_s`.

**Pass A, teacher-relative (v1, unchanged sign convention).** `s_A ~ TruncNormal(0.8, 0.075, [0.6, 1])`

```
x_A      = (1 - s_A) z_s + s_A ε_A                 plane 0 := first, timestep of plane 0 := 0
target_A = teacher(x_A) + (z_s - z_m) / s_A        teacher = adapter disabled, no grad
w_A      = (1 - R) + α_R R                         α_R = 0.1
L_A      = Σ w_A (pred_A - target_A)²  /  Σ w_A    (w broadcast over 48 channels)
```

**Pass B, ordinary conditional flow matching on the moving clip.** `s_B ~ TruncNormal(0.7, 0.15, [0.2, 1])`

```
x_B      = (1 - s_B) z_m + s_B ε_B                 plane 0 := first, timestep of plane 0 := 0
target_B = ε_B - z_m
w_B      = λ_B (1 + β R)                           λ_B = 1, β = 4
L_B      = Σ w_B (pred_B - target_B)² / Σ w_B
```

Both passes use the adapter guide built from the requested motion, the same text conditioning (10 % dropout to the
negative embedding, per microbatch) and exclude plane 0 from the loss (anchor, as v1).
With probability `p_static = 0.1` the request is the source's static clip: pass A only (`z_m = z_s`, `R = 0`, an
identity regulariser); set `objective.static_pass_b` to run B too.
One update = `effective_batch` (2) microbatches of one source each; `L = mean over microbatches of (L_A + L_B)`.
Backward is called separately for A and B, so only one adapted forward graph is alive at a time.
The reported `loss_b` is the mean over the microbatches that ran pass B.

## 2. Why this fixes the gray borders (and what v1 did)

In newly revealed regions the true content is not a function of the model inputs (guide, text, noisy static latent).
v1's target there is `teacher + (z_s - z_m)/σ`, deterministic given those inputs and independent of the noise sample.
The MSE-optimal answer for a target that cannot be inferred is its conditional mean, which for normalized latents is
near 0, i.e. mid-gray, with a sharp edge at the validity boundary. Nothing in v1 ever rewarded generating content.

Pass B conditions on the noisy sample of the real moving clip. The target `ε_B - z_m` contains ε, so for a given
`x_B` the network can only reduce the loss by committing to a concrete clean latent that is consistent with the
visible noisy content; the expected velocity field is the standard flow-matching one, whose ODE samples new
content instead of a mean. Revealed positions get 5× weight in B (`1 + β`) and 0.1× in A, so the regression target barely
acts on holes while pass A still supervises motion on valid content. The guide fill changes too
(`fill_mode: "replicate"`, edge features instead of latent 0), removing the shortcut "output the guide" for holes.
The validity channel still says which features are extrapolated.

## 3. Hyper-parameters

| Item | Value |
|---|---|
| optimizer | AdamW, lr 1e-4, weight decay 0.01, fused, FP32 adapter, BF16 autocast backbone |
| schedule | 25 warm-up updates, cosine to 0.1x over `updates` (1000; the pilot is the first 150 updates of the same run) |
| grad clip | 1.0 (global norm of the adapter) |
| batch | 2 microbatches of 1 source per update |
| α_R, λ_B, β | 0.1, 1, 4 |
| σ_A / σ_B | TruncNormal(0.8, 0.075, [0.6, 1]) / TruncNormal(0.7, 0.15, [0.2, 1]) |
| p_static, text dropout | 0.1, 0.1 |
| activation checkpointing | `full` (as v1) |
| guide fill | `replicate` |
| warm start | v1 step-300 adapter (`--init-from`), optimizer state is fresh |
| memory guard | stops (after a checkpoint) if peak allocated exceeds `max_peak_gib` (80) |

All of it lives in the config JSON (see `DEFAULT_CONFIG` in `flow_train_v2.py`).

## 4. Data

`--cache INDEX[:WEIGHT]` is repeatable. Each cache gets its own seeded permutation (every source once per epoch) and a
cache is chosen per microbatch with probability proportional to its weight. Both schemas are accepted and can be mixed:

* v2 rows `{id, source_id, split, family, direction, magnitude_u, latent_file, reference_file, sha256, ...}`, paths
  relative to the index directory; each source needs exactly one `static` row and at least one moving row
  (`families` in the config can restrict the moving families).
* v1 rows `{record_id, mode in static/zoom_in/zoom_out, cache_path, reference_path, ...}` (paths relative to the
  project root; found by walking up from the index). They map to family `static`/`zoom`, direction `in`/`out`,
  `magnitude_u = null`. `build_image_condition` is called with the configured fill mode for these too.

Reference files provide `image_latent`, `text_embeddings`, `negative_text_embeddings`, as in v1. SHA-256 values are
verified the first time each file is read in a process.

## 5. Fault tolerance and resume semantics

Checkpoints go to `<run-dir>/checkpoints/` every 25 updates **or** every 600 s (whichever first), and on SIGTERM/SIGINT.
Each file is written as `*.tmp`, fsynced and renamed (`os.replace`); `latest.json` (step, file names, sha256) is written
last. Retention: newest 3 plus every step divisible by 250. A checkpoint is `adapter_NNNNNN.safetensors`
(FP32 adapter, loadable by the inference code), `resume_NNNNNN.pt` (optimizer, update counter, LR state, Python/NumPy/
torch/CUDA RNG, sampler state incl. permutations and cursor) and `ckpt_NNNNNN.json` (manifest).

* **Resume is automatic.** Re-running the same command resumes if `<run-dir>/latest.json` exists. `--no-resume` moves the
  old state to `<run-dir>/old_<timestamp>/` (never deletes) and starts over.
* **Integrity.** On load the sha256 of both files is checked; if the newest checkpoint (or `latest.json`) is corrupt or
  incomplete the previous one is used, and the skipped names are logged.
* **Determinism.** The noise and σ of update `k`, microbatch `m` come from generators seeded by `(seed, k, m)`; data order
  comes from the saved sampler state. A resumed run reproduces the same losses and weights as an uninterrupted one
  (tested bit-for-bit on CPU). On GPU, non-deterministic kernels can cause tiny float differences; the sequence of
  data, noise levels and timesteps is still identical.
* **Signals.** The first SIGTERM/SIGINT finishes the current update (~20 s), writes an emergency checkpoint and exits with
  code 75; a second one raises `KeyboardInterrupt` immediately (a mid-update state cannot be checkpointed; resume then
  uses the last periodic checkpoint). A hard kill loses at most 25 updates / 10 minutes.
* **Session deadline (Molab chunking).** `--session-hours H` (sandbox lifetime, measured from the sandbox start via
  `/proc/1`, else `MOLAB_SESSION_START`, else process start), `--max-minutes M` (from now), `--deadline-epoch T`
  (absolute) and `--deadline-margin-minutes` (default 15, subtracted from the earliest constraint). Before every update
  the trainer stops if `now + est_update + est_ckpt + 90 s upload grace > deadline` (`est_update` = EMA of s/update,
  initial 45 s; `est_ckpt` = last measured checkpoint write, initial 60 s). It then writes a checkpoint (reason
  `deadline`) if the current step is not already saved, logs `DEADLINE_STOP step=N`, writes `run_status.json` with
  `status: "deadline"`, `stopped_reason: "deadline"` and exits 0. Data cursor / RNG are saved as for any checkpoint, so
  the next session resumes bit-identically. See `docs/experiments/SESSION_CHUNKING.md`. `build_cache_v2.py`,
  `calibrate_strength.py` and `generate_examples_v2.py` accept the same flags and stop between items.
* `progress.jsonl` lines beyond the resumed step are moved to `progress_orphaned.jsonl`.
* **Identity check.** The checkpoint stores the training identity (hyper-parameters, caches' index hashes and weights,
  architecture, fill mode, seed). Resuming with a different identity is refused unless `--allow-config-change`.
* The GPU lock `artifacts/gpu.lock` is shared with cache building; a lock whose pid is dead is replaced automatically.
  Use `--break-lock` if the lock is stale but the pid was reused (e.g. after a container restart).

Provenance in `<run-dir>`: `config.json` (resolved config, `git rev-parse HEAD`, dirty flag, sha256 of every
`src/fpcamera/*.py`; written on the first launch), `launches.jsonl` (one line per launch), `progress.jsonl`,
`progress.json` (latest update), `run_status.json` (complete / interrupted / failed, backbone hash check).
Per-update log fields: `loss_a, loss_b, r_fraction, sigma_a, sigma_b, family, u, grad_norm, lr, update_seconds, peak_gib`.

## 6. Commands (Molab)

Check (loads the model, one forward/backward with no step, prints peak memory and time; writes nothing):

```bash
python scripts/train_v2.py --config configs/train_v2.json --run-dir artifacts/v2_full \
  --cache data/v2/cache_1280x704_f121_v2/index.jsonl:1.0 \
  --init-from artifacts/flow_poc_all/adapter_000300.safetensors --check-only
```

Pilot = first 150 updates of the full run (same schedule, same run dir, resumable; mixing the v1 cache at weight 0.25 if wanted):

```bash
python scripts/train_v2.py --config configs/train_v2.json --run-dir artifacts/v2_full \
  --cache data/v2/cache_1280x704_f121_v2/index.jsonl:1.0 \
  --cache data/flow_poc/cache_1280x704_f121_v1/index.jsonl:0.25 \
  --init-from artifacts/flow_poc_all/adapter_000300.safetensors --stop-after-updates 150
```

After the visual check, continue the same run by relaunching without `--stop-after-updates`; it resumes from `latest.json`.
`--stop-after-updates N` stops early without changing the cosine schedule.

From a marimo cell, loop until the process exits with 0 (relaunching is always safe):

```python
import subprocess, sys
cmd = [sys.executable, "scripts/train_v2.py", "--config", "configs/train_v2.json", "--run-dir", "artifacts/v2_full",
       "--cache", "data/v2/cache_1280x704_f121_v2/index.jsonl:1.0",
       "--init-from", "artifacts/flow_poc_all/adapter_000300.safetensors"]
subprocess.run(cmd)   # return code 0 = done, 75 = interrupted with checkpoint (run again), other = failure
```

If the notebook session itself drops, start the identical cell again; the run continues from `latest.json`.

## 7. Expected cost

Measured on the RTX PRO 6000: about 41 s/update (about 32 s warm), about 23-25 GiB peak. 1000 updates is about 9-11 h; the pilot
(first 150 updates) is about 1.7 h. Add about 1 min start-up per (re)launch (model load, backbone hash, zero-init probe).
`peak_gib` and `update_seconds` are in every progress line.

## 8. Overfitting monitor (`scripts/validate_v2.py`)

Held-out losses of saved checkpoints on the **dev split** (28 sources: all moving rows + the static row for pass A) next to a
fixed random subset of 28 **train** sources, evaluated at identical, fixed noise. The Wan transformer is loaded once and the
adapter weights are swapped per checkpoint (no reload). No grad, bf16 autocast as in training.

```bash
python scripts/validate_v2.py --run-dir artifacts/v2_full --cache data/v2/cache_1280x704_f121_v2/index.jsonl \
  --steps 77,250,477 --session-hours 3 --deadline-margin-minutes 15      # or --all-local, or --checkpoint PATH[=LABEL]
```

Outputs in `<run-dir>/validation/` (`--out-dir` to change): `validation.json` (summaries per checkpoint, by family and by
magnitude bucket), `validation.md` (tables), `validation.png` (dev vs train-subset curves), `rows_<step>.json` (per row-draw
losses). Resume: rerun the same command; checkpoints already in `validation.json` are skipped (a changed evaluation setup,
e.g. `--draws`, is refused rather than mixed). `--session-hours/--max-minutes/--deadline-epoch/--deadline-margin-minutes`
behave as in training: no new checkpoint is started if it would not fit (estimate: EMA of measured seconds, initially
`--est-checkpoint-minutes`); exit 0 with `DEADLINE_STOP`. Do not run it while training holds the GPU lock.

**Fixed noise.** For every row the draws `(sigma_A, eps_A, sigma_B, eps_B)` are a function of `(seed, row id, k)` only (CPU
generators), so every checkpoint, and every machine, sees the same inputs; differences between checkpoints are purely the
adapter. The teacher (adapter-off) forward does not depend on the adapter, so it is cached on CPU across checkpoints.

**Metrics** (mean over row-draws; same weighted MSE as training): `L_A`, `L_B`, `L_B|R` (revealed region only, weight R: the
quantity that measures whether holes are being filled with real content), `L_A|valid` (weight 1-R: does the edit stay faithful
where the content is known), `L_B|valid`. Static rows contribute to the A metrics only. Rows whose region is empty are excluded
from that metric's mean (counts in the json under `n`).

**Cost.** Per draw: moving row = A + B forwards (2), static = A (1); the teacher forward is paid once per row-draw in the first
checkpoint and then cached. Defaults: K=1, dev = 28 sources x 2 moving + 28 static, train subset = 28 sources x 1 moving + 28
static (`--train-moving-per-source`), i.e. (56+28) x 2 + 56 = **224 forwards per checkpoint** (+140 teacher forwards, first
checkpoint only), plus about 1 min start-up and cache reads. Training time implies about 18 forward-equivalents per update
(41 s/update; 2 micro-batches x (teacher 1 + A 4 + B 4 with checkpointing)), i.e. about 2.3 s/forward inference, so about
**9 min per checkpoint** (about 14 min for the first). If a forward really takes 4-6 s, expect 15-22 min; read the measured
`seconds` per checkpoint in `validation.json`, then cut with `--train-sources 14` or `--dev-moving-per-source 1`. `--draws 4`
multiplies the cost by 4 (about 35 min). Batch size is 1 (compute-bound; batching does not lower the cost).

**Reading the train/dev gap.** Compare dev and train-subset curves at the same step.
- Both fall together, gap roughly constant: still learning, nothing to act on.
- Train keeps falling, dev flat or rising (gap `dev - train` of `L_B|R` growing from checkpoint to checkpoint): overfitting to
  the training sources; later checkpoints only memorise. `L_B|R` is the most sensitive (hole content is source-specific).
- `L_A|valid` rising on dev while `L_B|R` falls: the generative gain is traded for faithfulness of the known region.
- Use the family / magnitude tables to see where it breaks (e.g. only u>0.7 or only zoom). Differences smaller than the
  checkpoint-to-checkpoint noise on 56 dev rows are not meaningful; compare trends over at least 3 checkpoints.

**Selection / stop rule.** Among checkpoints with no gain regression in the visual evaluation (measured camera gain not below
the earlier acceptable checkpoints), pick the one with the best (lowest) dev `L_B|R`; ties within noise go to the earlier step.
Stop training when dev `L_B|R` has not improved for 3 consecutive evaluated checkpoints while the train subset still improves.
Dev loss is a monitor, not a substitute for the visual check: never select a checkpoint that fails the visual gain check.

## 7. Speed options (lossless)

Same math, only speed/memory changes. None of these is part of the run identity (they are CLI flags, not config keys
that enter `run_identity`), none touches the loss, the sampling, the RNG streams or the checkpoint format, so resuming
with a different setting needs no `--allow-config-change` and the data order / optimizer state resume exactly as before.
All options are opt-in; the defaults reproduce the previous behaviour.

**What "lossless" means here.** `--ckpt-blocks` recomputes the same forward, so it only moves memory and time.
`--attn-backend` and `--compile-blocks` are *kernel-level*: the arithmetic is the same but different kernels run (different
fusion, accumulation order and rounding), so a step is **not bit-identical** to the default path. That is the same standing
as SDPA versus a math kernel, and the differences are at the bf16 kernel-noise level. The speed probe below measures them on
the real model and flags any row whose loss_A, loss_B or global grad norm moves by more than about 1e-2 (relative) as NOT
lossless. Switching kernels mid-run therefore changes the trajectory only at kernel-noise level, never the recipe.

- **Partial activation checkpointing.** v1/v2 checkpoint all 30 blocks (peak about 23 GiB on a 95 GiB GPU), which costs one
  extra forward per block in the backward. `--ckpt-blocks K` checkpoints only the first K blocks (K=30 or more: all, the
  default; 0: none); the config equivalent is `"activation_checkpointing": {"mode": "all"|"none"|"first_k"|"every_n", "k": K, "n": N}`
  (the CLI flag wins; the legacy `"checkpointing": "full"|"none"` still works). Fewer checkpointed blocks = faster, more memory.
  Measured in session 5 (RTX PRO 6000 Blackwell, 97.9 GiB): K=30 23.1 GiB / 41.5 s, K=20 71.7 GiB / 38.5 s (36.8 s per real
  update, about 7.5% faster), K=10 OOM. That is about 4.9 GiB per un-checkpointed block, so K=15 needs roughly 96 GiB in eager
  mode: expect it to be skipped or OOM unless `--compile-blocks` lowers the activation memory.
- **Memory probe.** One forward+backward of both passes (all micro-batches) at 30 / 20 / 15 / 10 / 0 checkpointed blocks,
  printing peak GiB and seconds per row, stopping after the first OOM, and a recommended flag (fastest under
  `--probe-limit-gib`, default 85). With `--attn-backend` / `--compile-blocks` given, the rows run with those settings:

  ```bash
  python scripts/train_v2.py --config configs/train_v2.json --run-dir artifacts/v2_full     --cache data/v2/cache_1280x704_f121_v2/index.jsonl:1.0     --init-from artifacts/flow_poc_all/adapter_000300.safetensors --check-only --memory-probe
  ```

  Then train with e.g. `--ckpt-blocks 20`. The first row includes CUDA warm-up, so treat gaps under ~3% as noise (the speed
  probe below discards a warm-up update per row). Keep `max_peak_gib` (config, default 80) above the chosen setting's real
  peak, since the run stops when it is exceeded.
- **Attention backend: `--attn-backend {auto,cudnn,flash,efficient}`** (training, `train_v2.py`; inference,
  `flow_infer.generate(attn_backend=...)`, `flow_infer.py` / `generate_camera.py` / `calibrate_strength.py` /
  `eval_dev_reveal.py` / `generate_examples_v2.py`; validation, `validate_v2.py`). diffusers' Wan attention calls
  `F.scaled_dot_product_attention` through `dispatch_attention_fn` (its default `native` backend; read from the pinned
  diffusers 0.36 source), which honours torch's global SDPA switches, so `torch.nn.attention.sdpa_kernel` controls it.
  - `auto` (default): nothing is touched (torch's own dispatch, as before; the model load still logs one `sdpa_backends` line).
  - `cudnn`: `sdpa_kernel([CUDNN_ATTENTION, FLASH_ATTENTION, EFFICIENT_ATTENTION], set_priority=True)`, so cuDNN is tried
    first and torch falls back to flash, then memory-efficient, when cuDNN cannot take the inputs. On Hopper torch's cuDNN
    SDPA is often noticeably faster than its FlashAttention-2 backend. The S5 sandbox GPU was a Blackwell RTX PRO 6000 with
    torch 2.11.0+cu130, not a Hopper, so do not assume the gain: measure it with the probe.
  - `flash` / `efficient`: that single kernel only. They fail loudly ("No available kernel") instead of silently timing
    another kernel.
  - Guards: a torch without `SDPBackend.CUDNN_ATTENTION` (or without `sdpa_kernel`) warns and falls back to `auto`; a torch
    without `set_priority` still gets the restriction to the listed kernels.
  - Coverage: in training the context wraps the whole micro-batch (teacher forward, student forward, **and the backward**),
    so the forward recomputation inside activation checkpointing uses the same kernel. In inference it wraps the
    transformer's forward (installed before the camera-guidance wrappers, so every guidance call goes through it); the VAE
    decode is deliberately left on torch's default. The request is logged at startup (`speed_options` and, when active,
    `attn_backend` with the flags seen inside the context) and written to `run.json` (`speed_options`).
- **Block compile: `--compile-blocks`** (training and inference). Regional `torch.compile` of each of the 30 transformer
  blocks (`mode="default"`, `dynamic=False`; Inductor), not of the whole model. The compiled forward is installed as the
  block's `forward` and sits **inside** the activation-checkpoint wrapper: checkpointing stays eager and the backward
  recomputation re-runs the compiled forward (this works for both `--ckpt-blocks K` and the all-blocks default). The LoRA
  wrappers are part of each block and are traced; the adapter on/off state (teacher vs student) and grad mode give separate
  cached graphs per block structure (blocks 0-15 with LoRA, 16-29 without), so Dynamo's recompile limit is raised to 64. The
  `patch_embedding` hook (guide injection) is outside the blocks and stays eager. In inference the blocks are compiled after
  the adapter is attached and restored when `generate()` returns, so compile is paid per call: use it for one long run or a
  fixed-adapter batch, not for many short ones. Compile time is not step time: in training the first update of a launch includes
  it, is logged as `compile_first_update`, followed by `compile_overhead_estimate` (first minus steady update), and is
  excluded from the s/update estimate behind the deadline rule (which reserves 420 s for that first update, so a launch does
  not start a compile it cannot finish; its `progress.jsonl` record carries `"includes_compile": true`). The probe reports the compile overhead and the number of
  updates after which compiling pays for itself (`compile_breakeven_updates`) for each compile row.
  Fallback if compile misbehaves on the real model (graph breaks, recompile storms, OOM): omit the flag; the probe records a
  failing row as `error` and carries on. Compiling only the non-checkpointed blocks is available through
  `speed_options.set_block_compile(model, True, indices=range(K, 30))` but is not a flag, because the checkpoint-outside /
  compile-inside composition is the conservative one (verified on CPU with Dynamo + AOTAutograd in
  `tests/test_attn_compile_options.py`; Inductor and CUDA are not).
- **Speed probe: `--check-only --speed-probe`.** On ONE fixed batch (the sampler is drawn from once, moving requests only so
  both passes run; the trainer step is not advanced, so noise and sigma are identical for every row) it times one full update
  (all micro-batches, both passes) for the grid ckpt K in {20, 15} x attn in {auto, cudnn} x compile in {off, on}. Per row:
  a discarded warm-up update (which carries the one-off compile cost, reported as `compile_overhead_seconds`), then the
  timed update; peak GiB; loss_A, loss_B and grad norm. The baseline row (K=20, auto, no compile, always run first, even if
  not in the grid) is the reference: the relative difference of each metric against it is printed, rows above
  `--probe-equiv-tol` (default 1e-2) are marked **NOT lossless** and never recommended, and the warm-up-versus-timed
  difference of the baseline is reported as the run-to-run noise floor. A row stops cleanly on OOM (rows that could only use
  more memory with the same attention/compile setting are skipped) and a kernel/compile exception is recorded as `error`
  without aborting. The recommendation is the fastest row within `--probe-limit-gib` that passes the equivalence check,
  printed as ready-to-paste flags. Rows run cheap first (all non-compile rows, then compile rows grouped by attention
  backend; Dynamo is reset between attention backends because the SDPA kernel is chosen when the graph is traced) and no new row
  starts after `--probe-budget-minutes` (default 20), which is also the intended cost on an H100-class GPU (about 5 min for
  the four eager rows; the compile rows add compile time, reduced where Inductor's on-disk graph cache applies).
  Pass a trained adapter with `--init-from`: from a zero-initialised adapter (LoRA-B = 0) the LoRA-A gradients vanish and the
  grad-norm check is weak (the probe warns). It takes the GPU lease like any check-only run, so stop training first.

  ```bash
  python scripts/train_v2.py --config configs/train_v2.json --run-dir artifacts/v2_full \
    --cache data/v2/cache_1280x704_f121_v2/index.jsonl:1.0 \
    --init-from artifacts/v2_full/checkpoints/adapter_000415.safetensors \
    --check-only --speed-probe --probe-limit-gib 85
  ```

  Grid knobs: `--probe-ckpt 20,15`, `--probe-attn auto,cudnn` (also `flash`, `efficient`), `--probe-compile off,on`,
  `--probe-budget-minutes 20`, `--probe-timed-updates 1`, `--probe-equiv-tol 0.01`. Then launch with the printed flags, e.g.
  `--ckpt-blocks 20 --attn-backend cudnn --compile-blocks`, and keep `max_peak_gib` above that row's `peak_gib`.
- **Not adopted.** FlashAttention-3 needs a from-source build (Hopper `sm90` only) that is fragile on a non-persistent
  sandbox and does not cover this Blackwell GPU. SageAttention quantises QK^T to int8, which is lossy, so it is not a
  lossless option. No flash-attn install.
- **TF32** (`torch.backends.cuda.matmul.allow_tf32 = True`) is NOT bit-lossless and is not enabled; bf16 matmuls do not use it
  anyway, so little would be gained. Not exposed as a flag.
