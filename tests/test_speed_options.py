"""CPU tests for the lossless speed options: partial activation checkpointing, batched frozen forwards."""
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fpcamera import flow_train_v2 as v2  # noqa: E402
from fpcamera.flow_anchor import frozen_camera_difference  # noqa: E402
import test_train_v2_logic as T  # noqa: E402


# ------------------------------------------------------------------ checkpointing selection / identity
def test_resolve_blocks():
    assert v2.resolve_checkpointed_blocks({"mode": "all"}, 30) == list(range(30))
    assert v2.resolve_checkpointed_blocks({"mode": "none"}, 30) == []
    assert v2.resolve_checkpointed_blocks({"mode": "first_k", "k": 10}, 30) == list(range(10))
    assert v2.resolve_checkpointed_blocks({"mode": "first_k", "k": 99}, 30) == list(range(30))
    assert v2.resolve_checkpointed_blocks({"mode": "every_n", "n": 3}, 9) == [0, 3, 6]
    with pytest.raises(ValueError):
        v2.resolve_checkpointed_blocks({"mode": "bogus"}, 3)
    with pytest.raises(ValueError):
        v2.normalize_config({"activation_checkpointing": {"mode": "every_n", "n": 0}})


def test_spec_precedence():
    cfg = v2.normalize_config({})
    assert v2.checkpointing_spec(cfg) == {"mode": "all"}  # default unchanged
    assert v2.checkpointing_spec(v2.normalize_config({"checkpointing": "none"})) == {"mode": "none"}
    cfg = v2.normalize_config({"activation_checkpointing": {"mode": "every_n", "n": 2}})
    assert v2.checkpointing_spec(cfg)["mode"] == "every_n"
    assert v2.checkpointing_spec(cfg, ckpt_blocks=7) == {"mode": "first_k", "k": 7}
    assert v2.checkpointing_spec(cfg, ckpt_blocks=0) == {"mode": "first_k", "k": 0}


def test_identity_unaffected_by_checkpointing():
    tables = [SimpleNamespace(index_sha256="abc", weight=1.0)]
    ids = [
        v2.run_identity(v2.normalize_config(over), tables)
        for over in ({}, {"checkpointing": "none"}, {"activation_checkpointing": {"mode": "first_k", "k": 5}},
                     {"activation_checkpointing": {"mode": "every_n", "n": 2}})
    ]
    assert all(i == ids[0] for i in ids)
    assert not any("heckpoint" in k for k in ids[0])


def test_flag_plumbing():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "train_v2.py").read_text(encoding="utf8")
    for token in ("--ckpt-blocks", "--memory-probe", "ckpt_blocks=args.ckpt_blocks", "probe=args.memory_probe"):
        assert token in src
    import inspect
    sig = inspect.signature(v2.run_training).parameters
    assert {"ckpt_blocks", "probe", "probe_limit_gib"} <= set(sig)


# ------------------------------------------------------------------ checkpointing leaves math unchanged
def _run(spec, updates=2):
    model, adapter, trainer, _ = T.build_stack(p_static=0.0)
    count = {"n": 0}
    for b in model.blocks:
        f = b.forward

        def counted(*a, _f=f, **k):
            count["n"] += 1
            return _f(*a, **k)

        b.forward = counted
    v2.apply_activation_checkpointing(model, spec)
    model.train()
    recs = [trainer.run_update() for _ in range(updates)]
    return recs, [p.detach().clone() for p in adapter.parameters()], count["n"]


def test_checkpointing_does_not_change_losses_or_weights():
    r_none, w_none, n_none = _run({"mode": "none"})
    r_all, w_all, n_all = _run({"mode": "first_k", "k": 2})
    r_ev, w_ev, n_ev = _run({"mode": "every_n", "n": 2})
    assert n_all > n_none and n_none < n_ev < n_all  # recomputation really happens
    for other_r, other_w in ((r_all, w_all), (r_ev, w_ev)):
        for a, b in zip(r_none, other_r):
            assert a["loss"] == pytest.approx(b["loss"], rel=1e-5, abs=1e-6)
            assert a["grad_norm"] == pytest.approx(b["grad_norm"], rel=1e-4, abs=1e-6)
        for a, b in zip(w_none, other_w):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_set_block_checkpointing_is_idempotent_and_reversible():
    model = T.MockWan()
    originals = [b.forward for b in model.blocks]
    v2.set_block_checkpointing(model, [0, 1])
    v2.set_block_checkpointing(model, [0, 1])
    assert all("_unckpt_forward" in b.__dict__ for b in model.blocks)
    v2.set_block_checkpointing(model, [])
    assert all("_unckpt_forward" not in b.__dict__ for b in model.blocks)
    assert [b.forward for b in model.blocks] == originals


# ------------------------------------------------------------------ batched frozen forwards
class _Adapter:
    def __init__(self):
        self.enabled, self.cond = True, None

    def set_condition(self, c):
        self.cond = c

    @contextmanager
    def disabled(self):
        before, self.enabled = self.enabled, False
        try:
            yield self
        finally:
            self.enabled = before


class _Tx(nn.Module):
    """Per-sample attention over tokens + text mean + optional (adapter) condition bias; batch-independent."""

    def __init__(self, adapter):
        super().__init__()
        torch.manual_seed(0)
        self.inp, self.out = nn.Linear(4, 8), nn.Linear(8, 4)
        self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
        self.txt = nn.Linear(6, 8)
        self.adapter = adapter
        self.calls = []

    def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict=False):
        self.calls.append(hidden_states.shape[0])
        x = self.inp(hidden_states) + self.txt(encoder_hidden_states).mean(1, keepdim=True)
        x = x + timestep.float().mean(-1, keepdim=True)[..., None] / 1000
        if self.adapter.enabled and self.adapter.cond is not None:
            x = x + self.adapter.cond
        y, _ = self.attn(x, x, x)
        return (self.out(torch.tanh(x + y)),)


def _drive(batch, reference, guidance=5.0, steps=4, min_sigma=0.3):
    torch.manual_seed(1)
    adapter = _Adapter()
    tx = _Tx(adapter).eval()
    requested, static = torch.randn(1, 1, 8) * 0.3, torch.randn(1, 1, 8) * 0.1
    pos, neg = torch.randn(1, 5, 6), torch.randn(1, 5, 6)
    outs = []
    with torch.no_grad(), frozen_camera_difference(
        tx, adapter, requested, static, camera_gain=1.5, text_guidance=guidance, active_min_sigma=min_sigma,
        reference=reference, batch_forwards=batch,
    ) as state:
        g = torch.Generator().manual_seed(3)
        for i in range(steps):
            lat = torch.randn(1, 7, 4, generator=g)
            ts = torch.full((1, 7), 1000.0 * (1 - i / steps))
            p = tx(hidden_states=lat, timestep=ts, encoder_hidden_states=pos, return_dict=False)[0]
            n = tx(hidden_states=lat, timestep=ts, encoder_hidden_states=neg, return_dict=False)[0]
            outs.append(torch.stack([p, n]))
    return torch.stack(outs), state, tx.calls


@pytest.mark.parametrize("reference", ["static", "teacher"])
def test_batched_forwards_equal_sequential(reference):
    seq, s_seq, calls_seq = _drive(False, reference)
    bat, s_bat, calls_bat = _drive(True, reference)
    torch.testing.assert_close(bat, seq, rtol=0, atol=1e-5)
    assert s_seq["native_calls"] == s_bat["native_calls"]  # protocol assertion in flow_infer still holds
    assert s_seq["active_positive_calls"] == s_bat["active_positive_calls"] > 0
    assert 2 in calls_bat and 2 not in calls_seq
    assert len(calls_bat) < len(calls_seq)
    assert s_bat["batched_forward_calls"] == 3  # step 1 sequential (negative text not yet seen), then 3 steps


def test_batched_cache_not_reused_for_different_inputs():
    """If the negative call does not carry the batched latents the cache is ignored and recomputed."""
    adapter = _Adapter()
    tx = _Tx(adapter).eval()
    pos, neg = torch.randn(1, 5, 6), torch.randn(1, 5, 6)
    with torch.no_grad(), frozen_camera_difference(tx, adapter, torch.zeros(1, 1, 8), None, text_guidance=5.0,
                                                   reference="teacher", camera_gain=0.0) as st:
        for step in range(3):
            lat = torch.randn(1, 7, 4)
            ts = torch.full((1, 7), 800.0)
            tx(hidden_states=lat, timestep=ts, encoder_hidden_states=pos, return_dict=False)
            other = lat + 1.0  # differs from what was batched
            got = tx(hidden_states=other, timestep=ts, encoder_hidden_states=neg, return_dict=False)[0]
            with torch.no_grad():
                ref = _Tx(adapter).eval().forward(other, ts, neg)[0]
            torch.testing.assert_close(got, ref, rtol=0, atol=1e-5)
    assert st["native_calls"] == 6


def test_infer_flag_default_and_off_switch():
    from fpcamera import flow_infer as fi

    base = ["--project-root", ".", "--checkpoint", "c", "--image", "i", "--output", "o"]
    assert fi.parse_args(base).batch_forwards is True
    assert fi.parse_args(base + ["--no-batch-forwards"]).batch_forwards is False
