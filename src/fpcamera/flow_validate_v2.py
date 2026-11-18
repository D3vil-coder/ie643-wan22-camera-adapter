r"""Overfitting monitor for v2 checkpoints: held-out (dev split) vs train-subset losses at FIXED noise.

For every evaluation row (a moving clip, or a static clip for pass A only) a deterministic set of ``K``
draws ``(sigma_A, eps_A, sigma_B, eps_B)`` is derived from ``(seed, request clip id, k)`` with CPU generators,
so every checkpoint (and every machine) sees bit-identical inputs.  The losses are exactly the training ones
(``weighted_mse`` with the pass-A / pass-B weights) plus region-restricted variants:

  L_A, L_A|valid (weight 1-R), L_B, L_B|R (weight R: revealed region only), L_B|valid.

The pass-A teacher forward (adapter disabled) does not depend on the adapter weights, so its output is cached
across checkpoints within one invocation (CPU, model dtype), saving one of three forwards per draw from the
second checkpoint on.  Everything is no-grad.  Batch size is 1 (the cost is compute-bound; batching rows does
not reduce it and the adapter's guide condition is per sample).
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import re
import time

import torch

from .checkpointing import atomic_write_json, sha256_file
from . import flow_train_v2 as v2

DEV_SPLITS = ("development", "dev")
BUCKETS = ("u<0.4", "0.4-0.7", "u>0.7")
METRICS = ("la", "la_valid", "lb", "lb_R", "lb_valid")
METRIC_LABELS = {"la": "L_A", "la_valid": "L_A|valid", "lb": "L_B", "lb_R": "L_B|R", "lb_valid": "L_B|valid"}


# =============================================================================
# 1. Pure pieces
# =============================================================================
def _maybe(pred, target, w):
    """weighted_mse, or None when the region is empty."""
    if float(w.sum()) <= 0:
        return None
    return float(v2.weighted_mse(pred, target, w))


def pass_a_losses(pred, target, R, alpha_R):
    """Pass-A held-out losses of one draw: ``la`` (training weights) and ``la_valid`` (valid region only, weight 1-R)."""
    return {"la": _maybe(pred, target, v2.pass_a_weight(R, alpha_R)),
            "la_valid": _maybe(pred, target, 1.0 - R)}


def pass_b_losses(pred, target, R, lambda_B, beta):
    """Pass-B held-out losses: ``lb`` (training weights), ``lb_R`` (revealed region only: the hole-filling quality),
    ``lb_valid`` (valid region only).
    """
    return {"lb": _maybe(pred, target, v2.pass_b_weight(R, lambda_B, beta)),
            "lb_R": _maybe(pred, target, R),
            "lb_valid": _maybe(pred, target, 1.0 - R)}


def magnitude_bucket(u, static=False):
    """Bucket a requested strength u for the per-magnitude report: ``u<0.4``, ``0.4-0.7``, ``u>0.7`` or ``static``."""
    if static:
        return "static"
    if u is None:
        return "n/a"
    return "u<0.4" if u < 0.4 else ("0.4-0.7" if u <= 0.7 else "u>0.7")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _agg(rows):
    out = {m: _mean([r.get(m) for r in rows]) for m in METRICS}
    out["n"] = {m: sum(r.get(m) is not None for r in rows) for m in METRICS}
    out["n_rows"] = len(rows)
    return out


def summarize(rows):
    """Overall + by family + by magnitude bucket (mean over all row-draws)."""
    by_family, by_bucket = {}, {}
    for r in rows:
        by_family.setdefault(r["family"], []).append(r)
        by_bucket.setdefault(magnitude_bucket(r["u"], r["static_request"]), []).append(r)
    return {"overall": _agg(rows),
            "by_family": {k: _agg(v) for k, v in sorted(by_family.items())},
            "by_bucket": {k: _agg(v) for k, v in sorted(by_bucket.items())}}


# =============================================================================
# 2. Evaluation set and fixed draws
# =============================================================================
class EvalItem:
    """One evaluation row: a dev/train source's static clip plus the requested (moving or static) clip."""

    def __init__(self, set_name, cache_idx, source_id, static_ref, request_ref, is_static):
        self.set_name, self.cache_idx, self.source_id = set_name, cache_idx, source_id
        self.static_ref, self.request_ref, self.is_static = static_ref, request_ref, is_static

    @property
    def item_id(self):
        """The requested clip's id (also the seed key of its fixed noise draws)."""
        return self.request_ref.clip_id

    def plan(self):
        """The trainer ``Plan`` for this item (never text-dropped)."""
        return v2.Plan(self.cache_idx, self.source_id, self.static_ref, self.request_ref, self.is_static, False)


def build_items(table, set_name, source_ids, *, moving_per_source=0, seed=0, with_static=True):
    """Items for the given sources: every moving row (or a seeded subset of ``moving_per_source``) + the static row."""
    items = []
    for sid in source_ids:
        entry = table.sources[sid]
        moving = list(entry.moving)
        if moving_per_source and moving_per_source < len(moving):
            moving = random.Random(v2.derive_seed(seed, "moving_pick", sid)).sample(moving, moving_per_source)
            moving.sort(key=lambda r: r.clip_id)
        items += [EvalItem(set_name, table.cache_idx, sid, entry.static, r, False) for r in moving]
        if with_static:
            items.append(EvalItem(set_name, table.cache_idx, sid, entry.static, entry.static, True))
    return items


def pick_train_sources(train_table, n, seed):
    """Fixed random subset of ``n`` training sources (seeded), used as the overfitting reference for the dev set."""
    ids = sorted(train_table.sources)
    if n >= len(ids):
        return ids
    return sorted(random.Random(v2.derive_seed(seed, "train_subset")).sample(ids, n))


def fixed_draw(seed, item_id, k, hp, shape, dtype, device):
    """Deterministic (sigma_A, eps_A, sigma_B, eps_B); CPU generators so it is machine independent."""
    g = torch.Generator().manual_seed(v2.derive_seed(seed, "val", item_id, k, "sigma"))
    sigma_a = v2.sample_truncated_normal(g, *hp.sigma_a)
    sigma_b = v2.sample_truncated_normal(g, *hp.sigma_b)

    def eps(tag):
        """Standard-normal noise from a CPU generator seeded by (seed, row id, draw, tag)."""
        gen = torch.Generator().manual_seed(v2.derive_seed(seed, "val", item_id, k, tag))
        return torch.randn(shape, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)

    return sigma_a.to(device), eps("nA"), sigma_b.to(device), eps("nB")


class TeacherCache:
    """CPU cache of teacher (adapter-off) future predictions, bounded in bytes."""

    def __init__(self, max_bytes=16 * 2**30):
        self.max_bytes, self.bytes, self.store = int(max_bytes), 0, {}
        self.hits = self.misses = 0

    def get(self, key, device):
        """Return the cached teacher prediction (moved to ``device``) or None on a miss."""
        t = self.store.get(key)
        if t is None:
            self.misses += 1
            return None
        self.hits += 1
        return t.to(device)

    def put(self, key, t):
        """Cache a teacher prediction on the CPU while the byte budget allows."""
        n = t.numel() * t.element_size()
        if self.bytes + n <= self.max_bytes:
            self.store[key] = t.detach().cpu()
            self.bytes += n


@torch.no_grad()
def evaluate_item(model, adapter, loader, item, hp, *, draws, seed, teacher_cache=None, autocast=None):
    """Held-out pass-A/pass-B losses of one item under ``draws`` fixed (sigma, noise) draws.  No gradients; the teacher
    prediction is cached across checkpoints because it does not depend on the adapter.
    """
    mb = loader(item.plan())
    device = mb.static.device
    R = v2.revealed_mask(mb.condition)
    rows = []
    for k in range(draws):
        sigma_a, noise_a, sigma_b, noise_b = fixed_draw(seed, item.item_id, k, hp, mb.static.shape,
                                                        mb.static.dtype, device)
        noisy = v2._noisy_input(mb.static, mb.first, sigma_a, noise_a)
        ts = v2.token_timestep(mb.static, sigma_a)
        key = (item.item_id, k)
        cached = teacher_cache.get(key, device) if teacher_cache is not None else None
        if cached is None:
            adapter.set_condition(mb.condition)
            with adapter.disabled(), v2._autocast(device, autocast):
                out = v2._call(model, noisy, ts, mb.text)[:, :, 1:]
            if teacher_cache is not None:
                teacher_cache.put(key, out)
            teacher = out.detach().float()
        else:
            teacher = cached.float()
        target, _ = v2.camera_velocity_target(teacher, mb.static[:, :, 1:], mb.requested[:, :, 1:], sigma_a)
        adapter.set_condition(mb.condition)
        with v2._autocast(device, autocast):
            pred = v2._call(model, noisy, ts, mb.text)
        rec = {"set": item.set_name, "source_id": item.source_id, "clip_id": item.item_id, "k": k,
               "family": item.request_ref.family, "u": item.request_ref.magnitude_u,
               "static_request": item.is_static, "sigma_a": float(sigma_a), "sigma_b": None,
               "r_fraction": float(R.mean()),
               "lb": None, "lb_R": None, "lb_valid": None}
        rec.update(pass_a_losses(pred[:, :, 1:], target, R, hp.alpha_R))
        del pred, target, teacher
        if not item.is_static or hp.static_pass_b:
            b = v2.build_pass_b(mb, hp, sigma_b, noise_b, R)
            adapter.set_condition(mb.condition)
            with v2._autocast(device, autocast):
                pred = v2._call(model, b["noisy"], b["timesteps"], mb.text)
            rec.update(pass_b_losses(pred[:, :, 1:], b["target"], R, hp.lambda_B, hp.beta))
            rec["sigma_b"] = float(sigma_b)
            del pred, b
        rows.append(rec)
    return rows


def evaluate_checkpoint(model, adapter, loader, items, hp, *, draws=1, seed=0, teacher_cache=None, autocast=None,
                        progress=None):
    """All items for the adapter weights currently loaded. Returns ``{"summary": {set: ...}, "rows": [...]}``."""
    rows = []
    for i, it in enumerate(items):
        rows += evaluate_item(model, adapter, loader, it, hp, draws=draws, seed=seed,
                              teacher_cache=teacher_cache, autocast=autocast)
        if progress:
            progress(i + 1, len(items))
    sets = sorted({r["set"] for r in rows})
    return {"summary": {s: summarize([r for r in rows if r["set"] == s]) for s in sets}, "rows": rows}


# =============================================================================
# 3. Checkpoint discovery, results file, reports
# =============================================================================
_ADAPTER_RE = re.compile(r"^adapter_(\d+)\.safetensors$")


def discover_checkpoints(run_dir, steps=None, all_local=False, paths=()):
    """-> list of ``(label, path, expected_sha256_or_None)`` sorted by numeric label.

    ``paths`` entries may be ``PATH`` (step parsed from ``adapter_NNNNNN``) or ``PATH=LABEL``.
    Does not touch the checkpoint directory (a concurrent trainer may be writing there).
    """
    ck = Path(run_dir) / "checkpoints"
    found = {}

    def expected_sha(step):
        """adapter sha256 recorded in a checkpoint's manifest (None if unreadable)."""
        m = ck / f"ckpt_{step:06d}.json"
        try:
            return json.loads(m.read_text(encoding="utf8")).get("adapter_sha256")
        except Exception:
            return None

    if all_local:
        for p in sorted(ck.glob("adapter_*.safetensors")):
            m = _ADAPTER_RE.match(p.name)
            if m:
                found[str(int(m.group(1)))] = (p, expected_sha(int(m.group(1))))
    for s in steps or ():
        p = ck / f"adapter_{int(s):06d}.safetensors"
        if not p.is_file():
            raise FileNotFoundError(f"No checkpoint for step {s}: {p}")
        found[str(int(s))] = (p, expected_sha(int(s)))
    for spec in paths:
        path, _, label = str(spec).partition("=")
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(p)
        if not label:
            m = _ADAPTER_RE.match(p.name)
            if not m:
                raise ValueError(f"Cannot infer a step from {p.name}; use PATH=LABEL")
            label = str(int(m.group(1)))
        found[label] = (p, None)
    out = [(label, p, sha) for label, (p, sha) in found.items()]
    out.sort(key=lambda t: (not t[0].isdigit(), int(t[0]) if t[0].isdigit() else 0, t[0]))
    return out


def pending_labels(results, checkpoints):
    """Checkpoints (label, path) not yet present in the results file, i.e. still to evaluate on resume."""
    done = set(results.get("checkpoints", {}))
    return [c for c in checkpoints if c[0] not in done]


def load_results(path, identity):
    """Load ``validation.json`` for resume; refuses (lists the differing keys) if the evaluation setup changed."""
    path = Path(path)
    if path.exists():
        res = json.loads(path.read_text(encoding="utf8"))
        if res.get("identity") != identity:
            diff = sorted(k for k in set(res.get("identity", {})) | set(identity)
                          if res.get("identity", {}).get(k) != identity.get(k))
            raise ValueError(f"{path} was produced with a different evaluation setup (keys: {diff}); "
                             "use another --out-dir or delete it")
        return res
    return {"identity": identity, "checkpoints": {}}


def _f(x, d=4):
    return "-" if x is None else f"{x:.{d}f}"


def _sorted_labels(results):
    return sorted(results["checkpoints"], key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0, s))


def render_markdown(results) -> str:
    """Render the results as the ``validation.md`` tables (dev vs train subset, by family and magnitude)."""
    labels = _sorted_labels(results)
    ident = results["identity"]
    lines = ["# v2 overfitting monitor", "",
             f"Fixed noise: {ident['draws']} draw(s)/row, seed {ident['seed']}; dev sources {ident['n_dev_sources']}, "
             f"train-subset sources {ident['n_train_sources']}. Losses are means over row-draws "
             "(training's weighted MSE; `|R` = revealed region only, `|valid` = valid region only).", ""]
    lines += ["## Dev vs train subset", "",
              "| step | dev L_A | dev L_A\\|valid | dev L_B | dev L_B\\|R | train L_A | train L_A\\|valid | "
              "train L_B | train L_B\\|R | gap L_B\\|R (dev-train) | gap L_A (dev-train) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in labels:
        sm = results["checkpoints"][s]["summary"]
        d, t = sm["dev"]["overall"], sm.get("train_subset", {}).get("overall", {})
        g = lambda a, b: None if a is None or b is None else a - b  # noqa: E731
        lines.append(f"| {s} | {_f(d['la'])} | {_f(d['la_valid'])} | {_f(d['lb'])} | {_f(d['lb_R'])} | "
                     f"{_f(t.get('la'))} | {_f(t.get('la_valid'))} | {_f(t.get('lb'))} | {_f(t.get('lb_R'))} | "
                     f"{_f(g(d['lb_R'], t.get('lb_R')))} | {_f(g(d['la'], t.get('la')))} |")
    for set_name in ("dev", "train_subset"):
        for title, key in (("family", "by_family"), ("magnitude bucket", "by_bucket")):
            lines += ["", f"## {set_name}: by {title}", "",
                      "| step | " + title + " | n rows | L_A | L_A\\|valid | L_B | L_B\\|R |", "|---|---|---|---|---|---|---|"]
            for s in labels:
                grp = results["checkpoints"][s]["summary"].get(set_name, {}).get(key, {})
                for name, v in grp.items():
                    lines.append(f"| {s} | {name} | {v['n_rows']} | {_f(v['la'])} | {_f(v['la_valid'])} | "
                                 f"{_f(v['lb'])} | {_f(v['lb_R'])} |")
    return "\n".join(lines) + "\n"


def plot_results(results, path):
    """Plot dev vs train-subset loss curves over checkpoint steps to ``validation.png`` (matplotlib, Agg)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [s for s in _sorted_labels(results) if s.isdigit()]
    xs = [int(s) for s in labels]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, m in zip(axes.ravel(), ("la", "la_valid", "lb", "lb_R")):
        for set_name, style in (("dev", "o-"), ("train_subset", "s--")):
            ys = [results["checkpoints"][s]["summary"].get(set_name, {}).get("overall", {}).get(m) for s in labels]
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if pts:
                ax.plot(*zip(*pts), style, label=set_name)
        ax.set_title(METRIC_LABELS[m])
        ax.grid(alpha=0.3)
        ax.legend()
    for ax in axes[1]:
        ax.set_xlabel("update")
    fig.suptitle("v2 held-out (dev) vs train-subset loss at fixed noise")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_reports(results, out_dir):
    """Write ``validation.json`` (atomic), ``validation.md`` and ``validation.png`` (plot failure never loses results)."""
    out_dir = Path(out_dir)
    atomic_write_json(out_dir / "validation.json", results)
    (out_dir / "validation.md").write_text(render_markdown(results), encoding="utf8")
    try:
        plot_results(results, out_dir / "validation.png")
    except Exception as exc:  # plotting must never lose results
        print(f"plot failed: {exc!r}", flush=True)


# =============================================================================
# 4. Driver (GPU)
# =============================================================================
def eval_identity(cfg, index_path, dev_ids, train_ids, draws, seed, moving_train, moving_dev):
    """The evaluation setup that must stay constant between resumes (index hash, draws, seed, source lists,
    objective, fill mode, architecture).
    """
    return {"index_sha256": sha256_file(index_path), "draws": int(draws), "seed": int(seed),
            "n_dev_sources": len(dev_ids), "n_train_sources": len(train_ids), "dev_sources": list(dev_ids),
            "train_sources": list(train_ids), "dev_moving_per_source": int(moving_dev),
            "train_moving_per_source": int(moving_train),
            "objective": cfg["objective"], "fill_mode": cfg["fill_mode"], "architecture": cfg["architecture"]}


def run_validation(*, run_dir, index, out_dir=None, checkpoints, config=None, draws=1, seed=20261117,
                   n_train_sources=28, dev_moving_per_source=0, train_moving_per_source=1,
                   teacher_cache_gib=16.0, deadline=None, est_checkpoint_seconds=600.0, progress=print,
                   clock=time.time, attn_backend="auto"):
    """Evaluate each pending checkpoint, saving ``validation.json`` after every one. Returns a status dict."""
    from .session_deadline import Ema, UPLOAD_GRACE_SECONDS, should_stop

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    from . import flow_control
    from .speed_options import attention_backend, resolve_attn_backend

    attn_spec = resolve_attn_backend(attn_backend)
    run_dir = Path(run_dir)
    root = Path(__file__).resolve().parents[2]
    if config is None:
        cfgfile = run_dir / "config.json"
        config = json.loads(cfgfile.read_text(encoding="utf8"))["config"] if cfgfile.exists() else {}
    cfg = v2.normalize_config(config)
    hp = v2.hparams_from_config(cfg)
    index = Path(index)
    dev_table = None
    for split in DEV_SPLITS:
        try:
            dev_table = v2.load_cache_table(index, 0, 1.0, split=split, project_root=root)
            break
        except ValueError:
            continue
    if dev_table is None:
        raise ValueError(f"No development split rows in {index}")
    train_table = v2.load_cache_table(index, 0, 1.0, split="train", project_root=root)
    dev_ids = sorted(dev_table.sources)
    if set(dev_ids) & set(train_table.sources):
        raise ValueError("Dev and train sources overlap")
    train_ids = pick_train_sources(train_table, n_train_sources, seed)
    items = (build_items(dev_table, "dev", dev_ids, moving_per_source=dev_moving_per_source, seed=seed) +
             build_items(train_table, "train_subset", train_ids, moving_per_source=train_moving_per_source, seed=seed))
    out_dir = Path(out_dir) if out_dir else run_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    identity = eval_identity(cfg, index, dev_ids, train_ids, draws, seed, train_moving_per_source,
                             dev_moving_per_source)
    results = load_results(out_dir / "validation.json", identity)
    todo = pending_labels(results, checkpoints)
    progress(json.dumps({"event": "plan", "items": len(items), "draws": draws, "pending": [c[0] for c in todo],
                         "done": sorted(results["checkpoints"])}))
    status = {"status": "complete", "evaluated": [], "out_dir": str(out_dir)}
    if not todo:
        write_reports(results, out_dir)
        return status

    with v2.gpu_lease(root):
        model, versions = v2._load_transformer(out_dir / "library.log")
        model.eval()
        loader = v2.MicrobatchLoader(cfg["video_contract"], cfg["fill_mode"], device="cuda")
        adapter = flow_control.FlowControlLoRA(model, flow_control.FlowControlConfig(**cfg["architecture"]))
        cache = TeacherCache(teacher_cache_gib * 2**30)
        ema = Ema(est_checkpoint_seconds)
        try:
            for label, path, sha in todo:
                if should_stop(clock(), deadline, ema.value, 0.0, UPLOAD_GRACE_SECONDS):
                    progress(f"DEADLINE_STOP evaluated={status['evaluated']} next={label} est_s={ema.value:.0f}")
                    status["status"] = "deadline"
                    break
                if sha and sha256_file(path) != sha:
                    raise RuntimeError(f"Checkpoint sha256 mismatch: {path}")
                from safetensors.torch import load_file

                adapter.load_adapter_state_dict(load_file(str(path), device="cpu"))  # swap weights, no reload
                t0 = clock()
                with attention_backend(attn_spec):  # "auto" = no context = unchanged behaviour
                    res = evaluate_checkpoint(
                        model, adapter, loader, items, hp, draws=draws, seed=seed, teacher_cache=cache,
                        progress=lambda i, n: progress(f"[{label}] {i}/{n} rows") if i % 10 == 0 else None)
                secs = clock() - t0
                ema.update(secs)
                rows_path = out_dir / f"rows_{label}.json"
                atomic_write_json(rows_path, res["rows"])
                results["checkpoints"][label] = {"adapter": str(path), "adapter_sha256": sha or sha256_file(path),
                                                 "seconds": secs, "summary": res["summary"],
                                                 "teacher_cache": {"hits": cache.hits, "misses": cache.misses}}
                results["versions"] = versions
                write_reports(results, out_dir)
                status["evaluated"].append(label)
                d = res["summary"]["dev"]["overall"]
                progress(json.dumps({"event": "checkpoint_done", "step": label, "seconds": round(secs, 1),
                                     "dev_la": d["la"], "dev_lb": d["lb"], "dev_lb_R": d["lb_R"]}))
        finally:
            adapter.close()
    return status
