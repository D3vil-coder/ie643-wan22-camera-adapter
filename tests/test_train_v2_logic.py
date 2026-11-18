"""CPU logic tests for the v2 trainer using a tiny mock Wan transformer + the REAL FlowControlLoRA."""
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpcamera import flow_control, flow_train_v2 as v2  # noqa: E402
from fpcamera.checkpointing import CheckpointManager, sha256_file  # noqa: E402

CONTRACT = {"width": 32, "height": 32, "frames": 9, "fps": 24}  # latent [48,3,2,2]
HID = 16


# ----------------------------------------------------------------- mock backbone
class _Proj(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HID, HID)

    def forward(self, x):
        return torch.nn.functional.gelu(self.proj(x))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = nn.Module()
        self.attn1.to_q, self.attn1.to_k, self.attn1.to_v = (nn.Linear(HID, HID) for _ in range(3))
        self.ffn = nn.Module()
        self.ffn.net = nn.ModuleList([_Proj(), nn.Identity(), nn.Linear(HID, HID)])

    def forward(self, x):
        q, k, v = self.attn1.to_q(x), self.attn1.to_k(x), self.attn1.to_v(x)
        a = torch.softmax(q @ k.transpose(1, 2) / 4, -1) @ v
        x = x + a
        return x + self.ffn.net[2](self.ffn.net[0](x))


class MockWan(nn.Module):
    """Same call surface as WanTransformer3DModel as used by v1."""

    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv3d(48, HID, (1, 2, 2), stride=(1, 2, 2))
        self.blocks = nn.ModuleList([_Block(), _Block()])
        self.t = nn.Linear(1, HID)
        self.txt = nn.Linear(8, HID)
        self.out = nn.Linear(HID, 48 * 4)
        self.config = SimpleNamespace(in_channels=48, patch_size=(1, 2, 2))
        self.calls = []

    def enable_gradient_checkpointing(self):
        pass

    def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict=False):
        self.calls.append({"x": hidden_states.detach().clone(), "t": timestep.detach().clone(),
                           "adapter": getattr(self, "_flow_control_lora").enabled})
        x = self.patch_embedding(hidden_states)
        B, C, T, h, w = x.shape
        tok = x.flatten(2).transpose(1, 2)
        tok = tok + self.t(timestep[..., None] / 1000) + self.txt(encoder_hidden_states).mean(1, keepdim=True)
        for b in self.blocks:
            tok = b(tok)
        o = self.out(tok).reshape(B, T, h, w, 48, 2, 2).permute(0, 4, 1, 2, 5, 3, 6)
        return (o.reshape(B, 48, T, h * 2, w * 2),)


def make_data(n_sources=4, k_moving=2, seed=0):
    """Fake ClipRefs + in-memory tensors keyed by clip id."""
    g = torch.Generator().manual_seed(seed)
    store, sources = {}, {}
    for s in range(n_sources):
        first = torch.randn(48, 1, 2, 2, generator=g)
        text = torch.randn(5, 8, generator=g)
        neg = torch.randn(3, 8, generator=g)

        def clip(cid, family, u, moving):
            lat = torch.cat((first, torch.randn(48, 2, 2, 2, generator=g)), 1)
            hom = torch.eye(3).repeat(9, 1, 1)
            if moving:  # translate right by a growing amount -> revealed band
                for f in range(1, 9):
                    hom[f, 0, 2] = 2.0 * f * u
            store[cid] = {"latents": lat, "homographies": hom, "image_latent": first,
                          "text_embeddings": text, "negative_text_embeddings": neg}
            return v2.ClipRef(0, cid, f"src{s}", "train", family, "right" if moving else None,
                              u if moving else None, cid, cid, None, None, "v2")

        static = clip(f"s{s}_static", "static", 0, False)
        moving = [clip(f"s{s}_m{j}", "pan", 0.5 + 0.5 * j, True) for j in range(k_moving)]
        sources[f"src{s}"] = v2.SourceEntry(f"src{s}", static, moving)
    table = v2.CacheTable(0, "mem", "x", 1.0, sources)
    return table, store


class MemLoader:
    def __init__(self, store):
        self.store = store
        self.fn = v2.make_condition_fn("zeros", CONTRACT)

    def __call__(self, plan):
        sd, rd = self.store[plan.static_ref.clip_id], self.store[plan.request_ref.clip_id]
        static, first = sd["latents"][None], sd["image_latent"][None]
        req = rd["latents"][None]
        text = sd["negative_text_embeddings" if plan.text_dropout else "text_embeddings"][None]
        cond = self.fn(first, rd["homographies"][None])
        r = plan.request_ref
        meta = {"source_id": plan.source_id, "cache_idx": 0, "clip_id": r.clip_id, "family": r.family,
                "direction": r.direction, "u": r.magnitude_u, "static_request": plan.is_static_request,
                "text_dropout": plan.text_dropout}
        return v2.MicroBatch(static, req, first, text, cond, meta)


def build_stack(seed=123, p_static=0.25, hp_over=None, table_store=None):
    torch.manual_seed(0)
    model = MockWan().requires_grad_(False)
    adapter = flow_control.FlowControlLoRA(model, flow_control.FlowControlConfig(hidden=HID, rank=2, alpha=2.0, layers=2))
    g = torch.Generator().manual_seed(7)  # deterministic non-trivial adapter init
    with torch.no_grad():
        for p in adapter.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    table, store = table_store or make_data()
    hp = v2.HybridHParams(p_static=p_static, p_text_dropout=0.2, effective_batch=2, warmup=2, total_updates=20,
                          seed=seed, learning_rate=1e-2, **(hp_over or {}))
    opt = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad], lr=hp.learning_rate,
                            weight_decay=hp.weight_decay)
    sampler = v2.SourceSampler([table], seed=seed, p_static=hp.p_static, p_text_dropout=hp.p_text_dropout)
    trainer = v2.HybridTrainer(model, adapter, opt, hp, sampler, MemLoader(store), device="cpu",
                               frozen_params=(), identity={"k": 1})
    return model, adapter, trainer, (table, store)


def _frozen_params(model, adapter):
    mine = {id(p) for p in adapter.parameters()}
    return [p for p in model.parameters() if id(p) not in mine]


# ----------------------------------------------------------------- math tests
def test_weighted_mse_normalisation():
    pred, tgt = torch.randn(1, 48, 2, 2, 2), torch.randn(1, 48, 2, 2, 2)
    w = torch.rand(1, 1, 2, 2, 2) + 0.1
    got = v2.weighted_mse(pred, tgt, w)
    num = sum(float(w[0, 0, t, y, x]) * float((pred[0, :, t, y, x] - tgt[0, :, t, y, x]).square().sum())
              for t in range(2) for y in range(2) for x in range(2))
    den = float(w.sum()) * 48
    assert math.isclose(float(got), num / den, rel_tol=1e-5)
    assert torch.isclose(v2.weighted_mse(pred, tgt, torch.ones(1, 1, 2, 2, 2)), (pred - tgt).square().mean())
    assert torch.isclose(v2.weighted_mse(pred, tgt, w), v2.weighted_mse(pred, tgt, 7 * w))  # scale invariant


def test_revealed_mask_and_broadcast_weights():
    table, store = make_data()
    ref = list(table.sources.values())[0]
    mb = MemLoader(store)(v2.Plan(0, "src0", ref.static, ref.moving[1], False, False))
    R = v2.revealed_mask(mb.condition)
    assert R.shape == (1, 1, 2, 2, 2)                      # first plane excluded, channel-broadcastable
    assert torch.equal(R, 1 - mb.condition[:, 48:49, 1:])
    assert 0 < R.sum() < R.numel()                          # a partial revealed band
    wa, wb = v2.pass_a_weight(R, 0.1), v2.pass_b_weight(R, 1.0, 4.0)
    assert torch.allclose(wa, torch.where(R > 0, torch.tensor(0.1), torch.tensor(1.0)))
    assert torch.allclose(wb, torch.where(R > 0, torch.tensor(5.0), torch.tensor(1.0)))
    pred = torch.randn(1, 48, 2, 2, 2)
    assert v2.weighted_mse(pred, torch.zeros_like(pred), wa).ndim == 0


def test_pass_a_target_matches_v1_and_formula():
    v1 = pytest.importorskip("fpcamera.flow_train_delta")
    model, adapter, trainer, (table, store) = build_stack()
    ref = list(table.sources.values())[0]
    mb = MemLoader(store)(v2.Plan(0, "src0", ref.static, ref.moving[0], False, False))
    sigma = torch.tensor([0.77])
    noise = torch.randn(mb.static.shape, generator=torch.Generator().manual_seed(1))
    R = v2.revealed_mask(mb.condition)
    a = v2.build_pass_a(model, adapter, mb, trainer.hp, sigma, noise, R)
    teacher = a["teacher_future"]
    ref_target, ref_delta = v1.camera_velocity_target(teacher, mb.static[:, :, 1:], mb.requested[:, :, 1:], sigma)
    assert torch.equal(a["target"], ref_target) and torch.equal(a["delta"], ref_delta)
    assert not model.calls[-1]["adapter"]                     # teacher ran with adapter disabled
    # velocity convention: ideal teacher v=(x-z_s)/s = eps - z_s  =>  target = (x - z_m)/s
    x = a["noisy"][:, :, 1:]
    ideal = (x - mb.static[:, :, 1:]) / sigma
    assert torch.allclose(ideal, noise[:, :, 1:] - mb.static[:, :, 1:], atol=1e-5)
    t, _ = v2.camera_velocity_target(ideal, mb.static[:, :, 1:], mb.requested[:, :, 1:], sigma)
    assert torch.allclose(t, (x - mb.requested[:, :, 1:]) / sigma, atol=1e-5)
    # anchor: plane 0 is the clean image latent and its timestep is 0
    assert torch.equal(a["noisy"][:, :, :1], mb.first)
    ts = a["timesteps"].reshape(1, 3, -1)
    assert (ts[:, 0] == 0).all() and (ts[:, 1:] > 0).all()


def test_pass_a_target_formula_without_v1_import():
    """Same check, independent of importing the v1 module (which needs PIL/diffusers)."""
    teacher, s, m = torch.randn(1, 48, 2, 2, 2), torch.randn(1, 48, 2, 2, 2), torch.randn(1, 48, 2, 2, 2)
    sigma = torch.tensor([0.65])
    t, d = v2.camera_velocity_target(teacher, s, m, sigma)
    assert torch.allclose(t, teacher + (s - m) / 0.65) and torch.allclose(d, (s - m) / 0.65)
    with pytest.raises(ValueError):
        v2.camera_velocity_target(teacher, s, m, torch.tensor([0.0]))


def test_pass_b_target_formula():
    model, adapter, trainer, (table, store) = build_stack()
    ref = list(table.sources.values())[0]
    mb = MemLoader(store)(v2.Plan(0, "src0", ref.static, ref.moving[0], False, False))
    sigma = torch.tensor([0.4])
    noise = torch.randn(mb.requested.shape, generator=torch.Generator().manual_seed(2))
    R = v2.revealed_mask(mb.condition)
    b = v2.build_pass_b(mb, trainer.hp, sigma, noise, R)
    x = b["noisy"]
    assert torch.allclose(x[:, :, 1:], 0.6 * mb.requested[:, :, 1:] + 0.4 * noise[:, :, 1:])
    assert torch.equal(b["target"], (noise - mb.requested)[:, :, 1:])
    # rectified-flow velocity d x / d sigma = eps - z_m
    assert torch.allclose(b["target"], (x[:, :, 1:] - mb.requested[:, :, 1:]) / 0.4, atol=1e-5)
    assert torch.equal(x[:, :, :1], mb.first)


def test_static_request_skips_pass_b():
    for p_static, calls_per_micro, has_b in ((1.0, 2, False), (0.0, 3, True)):
        model, adapter, trainer, _ = build_stack(p_static=p_static)
        rec = trainer.run_update()
        assert all((m["loss_b"] is not None) == has_b for m in rec["micro"])
        assert len(model.calls) == calls_per_micro * trainer.hp.effective_batch   # teacher + A (+ B)
        assert (rec["loss_b"] is not None) == has_b
        assert rec["micro"][0]["static_request"] == (p_static == 1.0)
    _, _, trainer, _ = build_stack(p_static=1.0, hp_over={"static_pass_b": True})
    assert all(m["loss_b"] is not None for m in trainer.run_update()["micro"])


def test_truncated_normal_bounds():
    g = torch.Generator().manual_seed(0)
    x = v2.sample_truncated_normal(g, 0.7, 0.15, 0.2, 1.0, n=20000)
    assert x.min() >= 0.2 and x.max() <= 1.0 and abs(float(x.mean()) - 0.7) < 0.01


def test_update_changes_only_adapter_and_logs_fields():
    model, adapter, trainer, _ = build_stack(p_static=0.0)
    frozen = _frozen_params(model, adapter)
    trainer.frozen_params = tuple(frozen)
    snap = [p.detach().clone() for p in frozen]
    before = [p.detach().clone() for p in adapter.parameters()]
    rec = trainer.run_update()
    for key in ("loss_a", "loss_b", "r_fraction", "sigma_a", "sigma_b", "family", "u", "grad_norm", "lr",
                "update_seconds", "peak_gib"):
        assert key in rec
    assert any(not torch.equal(a, b) for a, b in zip(before, adapter.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(snap, frozen))
    assert all(p.grad is None for p in frozen)
    assert rec["lr"] == pytest.approx(trainer.lr_at(0))


# ----------------------------------------------------------------- determinism / resume
def test_checkpoint_kill_resume_is_bit_identical(tmp_path):
    N, K = 6, 3
    ts = make_data()
    # uninterrupted reference run
    _, ad_ref, tr_ref, _ = build_stack(table_store=ts)
    keys = lambda r: (r["loss_a"], r["loss_b"], r["grad_norm"], r["lr"])  # noqa: E731
    losses_ref = [keys(tr_ref.run_update()) for _ in range(N)]

    # interrupted run: K updates, checkpoint, one more update that is lost, "kill", resume in fresh objects
    mgr = CheckpointManager(tmp_path, every_updates=K, every_seconds=1e9)
    _, ad1, tr1, _ = build_stack(table_store=ts)
    first = [keys(tr1.run_update()) for _ in range(K)]
    assert mgr.due(tr1.step)
    tr1.save_checkpoint(mgr)
    tr1.run_update()
    del ad1, tr1

    loaded = CheckpointManager(tmp_path, every_updates=K, every_seconds=1e9).load_latest()
    assert loaded.step == K
    _, ad2, tr2, _ = build_stack(table_store=ts)
    ad2.load_adapter_state_dict(loaded.adapter_state)
    tr2.load_state_dict(loaded.resume_state)
    assert tr2.step == K
    rest = [keys(tr2.run_update()) for _ in range(N - K)]
    assert first + rest == losses_ref                      # exact float equality, every update
    ref_state, got_state = ad_ref.adapter_state_dict(), ad2.adapter_state_dict()
    for k in ref_state:
        assert torch.equal(ref_state[k], got_state[k]), k


# ----------------------------------------------------------------- checkpoint manager
def _save(mgr, step):
    return mgr.save(step, {"w": torch.full((3,), float(step))}, {"step": step, "x": [step]})


def test_atomic_layout_and_latest_written_last(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_last=3)
    m = _save(mgr, 25)
    ptr = json.loads((tmp_path / "latest.json").read_text())
    assert ptr["step"] == 25 and ptr["adapter_sha256"] == sha256_file(tmp_path / "checkpoints" / m["adapter_file"])
    assert not list(tmp_path.rglob("*.tmp"))


def test_keep_last3_and_every_250(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_last=3, keep_every=250)
    for s in range(25, 626, 25):
        _save(mgr, s)
    assert mgr.manifest_steps() == [250, 500, 575, 600, 625]
    assert len(list((tmp_path / "checkpoints").iterdir())) == 5 * 3
    assert mgr.load_latest().step == 625


def test_corrupt_latest_falls_back(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_last=3)
    for s in (25, 50, 75):
        _save(mgr, s)
    ad = tmp_path / "checkpoints" / "adapter_000075.safetensors"
    ad.write_bytes(ad.read_bytes()[:-7] + b"garbage")        # corrupt newest adapter
    loaded = CheckpointManager(tmp_path).load_latest()
    assert loaded.step == 50 and loaded.skipped and "000075" in loaded.skipped[0][0]
    (tmp_path / "checkpoints" / "resume_000050.pt").write_bytes(b"nope")   # corrupt 50 too
    assert CheckpointManager(tmp_path).load_latest().step == 25
    (tmp_path / "latest.json").write_text("{broken")                          # corrupt pointer
    assert CheckpointManager(tmp_path).load_latest().step == 25


def test_all_corrupt_raises_and_none_when_empty(tmp_path):
    assert CheckpointManager(tmp_path / "a").load_latest() is None
    mgr = CheckpointManager(tmp_path / "b")
    _save(mgr, 25)
    (tmp_path / "b" / "checkpoints" / "resume_000025.pt").write_bytes(b"x")
    with pytest.raises(RuntimeError):
        CheckpointManager(tmp_path / "b").load_latest()


def test_due_policy_and_time_trigger(tmp_path):
    now = [0.0]
    mgr = CheckpointManager(tmp_path, every_updates=25, every_seconds=600, clock=lambda: now[0])
    assert not mgr.due(7) and mgr.due(25)
    now[0] = 601
    assert mgr.due(7)
    _save(mgr, 7)
    assert not mgr.due(7) and not mgr.due(8)


def test_archive_on_no_resume(tmp_path):
    mgr = CheckpointManager(tmp_path)
    _save(mgr, 25)
    dest = mgr.archive_existing()
    assert (dest / "latest.json").exists() and CheckpointManager(tmp_path).load_latest() is None


# ----------------------------------------------------------------- sampler / cache adapters / warm start
def test_sampler_resume_is_exact():
    table, _ = make_data(n_sources=5)
    a = v2.SourceSampler([table], seed=3, p_static=0.3, p_text_dropout=0.2)
    [a.next_plan() for _ in range(7)]
    state = copy.deepcopy(a.state_dict())
    cont = [(p.source_id, p.request_ref.clip_id, p.text_dropout) for p in (a.next_plan() for _ in range(15))]
    b = v2.SourceSampler([table], seed=3, p_static=0.3, p_text_dropout=0.2)
    b.load_state_dict(state)
    assert cont == [(p.source_id, p.request_ref.clip_id, p.text_dropout) for p in (b.next_plan() for _ in range(15))]
    c = v2.SourceSampler([table], seed=9, p_static=0.0, p_text_dropout=0.0)   # an epoch visits each source once
    assert sorted(c.next_plan().source_id for _ in range(5)) == sorted(table.sources)


def _write_cache(tmp, kind):
    from safetensors.torch import save_file
    d = tmp / kind
    (d / "videos").mkdir(parents=True)
    rows = []
    g = torch.Generator().manual_seed(0)
    first = torch.randn(48, 1, 2, 2, generator=g)
    ref = d / "ref.safetensors"
    save_file({"image_latent": first, "text_embeddings": torch.randn(4, 8), "negative_text_embeddings": torch.randn(2, 8)},
              str(ref))
    specs = ([("static", "static", None), ("a", "pan", "left"), ("b", "pan", "right")] if kind == "v2" else
             [("static", "static", None), ("zoom_in", "zoom_in", None), ("zoom_out", "zoom_out", None)])
    for name, fam, dirn in specs:
        f = d / "videos" / f"{name}.safetensors"
        save_file({"latents": torch.cat((first, torch.randn(48, 2, 2, 2, generator=g)), 1),
                   "homographies": torch.eye(3).repeat(9, 1, 1)}, str(f))
        if kind == "v2":
            rows.append({"id": name, "source_id": "s1", "split": "train", "family": fam, "direction": dirn,
                         "magnitude_u": 0.5, "latent_file": f"videos/{name}.safetensors",
                         "reference_file": "ref.safetensors", "sha256": sha256_file(f)})
        else:
            rows.append({"record_id": name, "source_id": "s1", "split": "train", "mode": fam,
                         "normalization": v2.NORMALIZATION, "cache_path": f"videos/{name}.safetensors",
                         "reference_path": "ref.safetensors", "sha256": sha256_file(f),
                         "reference_sha256": sha256_file(ref)})
    (d / "index.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return d / "index.jsonl"


@pytest.mark.parametrize("kind", ["v2", "v1"])
def test_cache_adapter_and_loader(tmp_path, kind):
    idx = _write_cache(tmp_path, kind)
    table = v2.load_cache_table(idx, 0, 1.0)
    entry = table.sources["s1"]
    assert entry.static.family == "static" and len(entry.moving) == 2
    if kind == "v1":
        assert {m.family for m in entry.moving} == {"zoom"} and {m.direction for m in entry.moving} == {"in", "out"}
        assert all(m.magnitude_u is None for m in entry.moving)
    loader = v2.MicrobatchLoader(CONTRACT, "zeros", device="cpu", dtype=torch.float32, text_pad=lambda x: x)
    mb = loader(v2.Plan(0, "s1", entry.static, entry.moving[0], False, False))
    assert mb.static.shape == (1, 48, 3, 2, 2) and mb.condition.shape == (1, 51, 3, 2, 2)
    mb_s = loader(v2.Plan(0, "s1", entry.static, entry.static, True, True))
    assert torch.equal(mb_s.static, mb_s.requested)
    other = _write_cache(tmp_path / "other", "v1" if kind == "v2" else "v2")   # mix both schemas
    t1 = v2.load_cache_table(other, 1, 2.0)
    s = v2.SourceSampler([table, t1], seed=0, p_static=0.1, p_text_dropout=0.1)
    assert {s.next_plan().cache_idx for _ in range(30)} == {0, 1}


def test_condition_fn_fill_mode_passthrough(monkeypatch):
    seen = {}

    def fake(first, hom, *, width, height, frames, fill_mode="zeros"):
        seen["fill_mode"] = fill_mode
        return "ok"

    monkeypatch.setattr(flow_control, "build_image_condition", fake)
    assert v2.make_condition_fn("replicate", CONTRACT)(None, None) == "ok" and seen["fill_mode"] == "replicate"

    def old(first, hom, *, width, height, frames):
        return "old"

    monkeypatch.setattr(flow_control, "build_image_condition", old)
    assert v2.make_condition_fn("zeros", CONTRACT)(None, None) == "old"
    with pytest.raises(RuntimeError):
        v2.make_condition_fn("replicate", CONTRACT)(None, None)


def test_warm_start_strict(tmp_path):
    from safetensors.torch import save_file
    _, adapter, _, _ = build_stack()
    state = {k: v.clone() + 1 for k, v in adapter.adapter_state_dict().items()}
    ok = tmp_path / "ok.safetensors"
    save_file(state, str(ok))
    _, adapter2, _, _ = build_stack()
    rep = v2.load_warm_start(adapter2, ok)
    assert not rep["missing"] and not rep["unexpected"]
    assert torch.equal(adapter2.adapter_state_dict()["guide_projection.bias"], state["guide_projection.bias"])
    bad = dict(state)
    bad.pop("guide_projection.bias")
    bad["extra"] = torch.zeros(1)
    k = next(k for k in bad if k.endswith("lora_A"))
    bad[k] = torch.zeros(1, 1)
    save_file(bad, str(tmp_path / "bad.safetensors"))
    with pytest.raises(ValueError) as e:
        v2.load_warm_start(adapter2, tmp_path / "bad.safetensors")
    msg = str(e.value)
    assert "guide_projection.bias" in msg and "extra" in msg and k in msg


def test_progress_log_truncate(tmp_path):
    log = v2.ProgressLog(tmp_path)
    for s in range(1, 6):
        log.write({"step": s})
    assert log.truncate_to(3) == 2
    assert [json.loads(l)["step"] for l in log.jsonl.read_text().splitlines()] == [1, 2, 3]
    assert (tmp_path / "progress_orphaned.jsonl").exists()


# ----------------------------------------------------------------- session deadline
def test_session_deadline_helpers():
    from fpcamera import session_deadline as sd

    files = {"/proc/stat": "cpu 1\nbtime 1000\n", "/proc/1/stat": "1 (init) S 0 1 1 0 -1 4 0 0 0 0 0 0 0 0 20 0 1 0 500 0"}
    start = sd.sandbox_start_epoch(read=lambda p: files[p], clk_tck=100)
    assert start == 1005.0
    def bad(p):
        raise OSError
    assert sd.sandbox_start_epoch(read=bad, env={"MOLAB_SESSION_START": "42"}) == 42.0
    assert sd.sandbox_start_epoch(read=bad, env={}, now=lambda: 7.0) == 7.0
    assert sd.resolve_deadline() is None
    now = lambda: 10_000.0  # noqa: E731
    # earliest wins, margin subtracted
    d = sd.resolve_deadline(deadline_epoch=20_000, max_minutes=10, session_hours=3, margin_minutes=1,
                            now=now, start_epoch=0.0)
    assert d == 10_000 + 600 - 60
    assert sd.resolve_deadline(session_hours=3, margin_minutes=15, now=now, start_epoch=9_000.0) == 9_000 + 10800 - 900
    assert sd.should_stop(100, 299, 50, 60, 90) and not sd.should_stop(100, 300, 50, 60, 40)
    assert not sd.should_stop(100, None, 1e9)


class _Stop:
    requested = False


def _fake_clock_trainer(trainer, clock, per_update):
    real = trainer.run_update

    def run_update(*a, **k):
        rec = real(*a, **k)
        clock[0] += per_update
        return rec
    trainer.run_update = run_update


def test_deadline_stop_checkpoints_and_resume_is_identical(tmp_path):
    N, per, ckpt = 6, 10.0, 5.0
    ts = make_data()
    keys = lambda r: (r["loss_a"], r["loss_b"], r["grad_norm"], r["lr"])  # noqa: E731
    _, ad_ref, tr_ref, _ = build_stack(table_store=ts)
    ref = [keys(tr_ref.run_update()) for _ in range(N)]

    clock = [1000.0]
    mgr = CheckpointManager(tmp_path, every_updates=1000, every_seconds=1e9, clock=lambda: clock[0])
    _, ad1, tr1, _ = build_stack(table_store=ts)
    _fake_clock_trainer(tr1, clock, per)
    plog = v2.ProgressLog(tmp_path)
    msgs = []
    # deadline leaves room for exactly 3 updates: need now + 10 + 5 + 20 <= deadline
    deadline = 1000.0 + 3 * per + (per + ckpt + 20.0) - 1
    res = v2.train_until(tr1, mgr, plog, stop_step=N, stop=_Stop(), deadline=deadline, clock=lambda: clock[0],
                         progress=msgs.append, est_update=per, est_ckpt=ckpt, grace=20.0)
    assert res["deadline_stop"] and tr1.step == 3
    assert any(m.startswith("DEADLINE_STOP step=3") for m in msgs)
    assert clock[0] + per <= deadline  # never started an update that could overrun
    tr1.save_checkpoint(mgr, "deadline")           # what run_training does after the loop
    first = [json.loads(l) for l in plog.jsonl.read_text().splitlines()]
    assert [r["step"] for r in first] == [1, 2, 3]

    loaded = CheckpointManager(tmp_path).load_latest()
    assert loaded.step == 3 and loaded.manifest["reason"] == "deadline"
    _, ad2, tr2, _ = build_stack(table_store=ts)
    ad2.load_adapter_state_dict(loaded.adapter_state)
    tr2.load_state_dict(loaded.resume_state)
    clock2 = [5000.0]
    mgr2 = CheckpointManager(tmp_path, every_updates=1000, every_seconds=1e9, clock=lambda: clock2[0])
    mgr2.last_saved_step = 3
    rest, got = [], []
    real = tr2.run_update

    def run_update(*a, **k):
        r = real(*a, **k)
        got.append(keys(r))
        return r
    tr2.run_update = run_update
    v2.train_until(tr2, mgr2, plog, stop_step=N, stop=_Stop(), deadline=None, clock=lambda: clock2[0],
                   progress=lambda m: None)
    assert tr2.step == N
    assert [keys(r) for r in []] == [] and got == ref[3:]
    for k, v in ad_ref.adapter_state_dict().items():
        assert torch.equal(v, ad2.adapter_state_dict()[k]), k


def test_deadline_already_passed_does_no_update(tmp_path):
    clock = [100.0]
    mgr = CheckpointManager(tmp_path, clock=lambda: clock[0])
    _, _, tr, _ = build_stack()
    res = v2.train_until(tr, mgr, v2.ProgressLog(tmp_path), stop_step=5, stop=_Stop(), deadline=50.0,
                         clock=lambda: clock[0], progress=lambda m: None)
    assert res["deadline_stop"] and tr.step == 0


def test_identity_change_targeted(tmp_path):
    saved = {"seed": 1, "caches": [{"index_sha256": "a", "weight": 1.0}]}
    new = {"seed": 1, "caches": [{"index_sha256": "b", "weight": 1.0}]}
    assert v2.check_identity_change(saved, saved) == {}
    ch = v2.check_identity_change(saved, new, allow_keys="caches")
    assert list(ch) == ["caches"] and ch["caches"]["old"] != ch["caches"]["new"]
    assert list(v2.check_identity_change(saved, new, allow_all=True)) == ["caches"]
    with pytest.raises(ValueError):
        v2.check_identity_change(saved, new)
    with pytest.raises(ValueError, match="seed"):
        v2.check_identity_change(saved, {**new, "seed": 2}, allow_keys="caches")
    v2.record_identity_change(tmp_path, ch, False)
    rec = json.loads((tmp_path / "launches.jsonl").read_text().splitlines()[-1])
    assert rec["event"] == "identity_change_accepted" and "caches" in rec["changes"]


def test_run_training_does_not_shadow_probe_parameter():
    """Regression: the identity-check clip was a local named ``probe`` that was `del`'d, so
    ``if check_only and probe:`` raised UnboundLocalError under --check-only --memory-probe."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(v2.run_training))
    fn = tree.body[0]
    assert "probe" in [a.arg for a in fn.args.args + fn.args.kwonlyargs]
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == "probe":
            assert isinstance(node.ctx, ast.Load), f"line {node.lineno}: 'probe' parameter is rebound/deleted"
