"""CPU tests for the v2 overfitting monitor (mock transformer from test_train_v2_logic)."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fpcamera import flow_train_v2 as v2, flow_validate_v2 as val  # noqa: E402
from test_train_v2_logic import MemLoader, build_stack, make_data  # noqa: E402


def _setup(n_sources=3):
    ts = make_data(n_sources=n_sources)
    model, adapter, trainer, (table, store) = build_stack(table_store=ts)
    items = val.build_items(table, "dev", sorted(table.sources))
    return model, adapter, trainer.hp, MemLoader(store), items


def _flat(res):
    return [(r["clip_id"], r["k"], r["la"], r["la_valid"], r["lb"], r["lb_R"], r["lb_valid"]) for r in res["rows"]]


def test_items_cover_moving_and_static():
    *_, items = _setup(3)
    assert len(items) == 3 * (2 + 1) and sum(i.is_static for i in items) == 3
    capped = val.build_items(make_data(3)[0], "t", ["src0", "src1"], moving_per_source=1, seed=1)
    assert len(capped) == 2 * 2


def test_deterministic_same_checkpoint():
    model, adapter, hp, loader, items = _setup()
    a = val.evaluate_checkpoint(model, adapter, loader, items, hp, draws=2, seed=5)
    b = val.evaluate_checkpoint(model, adapter, loader, items, hp, draws=2, seed=5)
    assert _flat(a) == _flat(b) and len(a["rows"]) == len(items) * 2
    c = val.evaluate_checkpoint(model, adapter, loader, items, hp, draws=2, seed=6)
    assert _flat(a) != _flat(c)  # the seed controls the draws


def test_draws_depend_only_on_row_and_seed():
    model, adapter, hp, loader, items = _setup()
    shape = (1, 48, 3, 2, 2)
    d1 = val.fixed_draw(1, "x", 0, hp, shape, torch.float32, "cpu")
    d2 = val.fixed_draw(1, "x", 0, hp, shape, torch.float32, "cpu")
    d3 = val.fixed_draw(1, "x", 1, hp, shape, torch.float32, "cpu")
    assert all(torch.equal(a, b) for a, b in zip(d1, d2)) and not torch.equal(d1[1], d3[1])


def test_different_checkpoints_differ_and_teacher_cache_is_exact():
    model, adapter, hp, loader, items = _setup()
    cache = val.TeacherCache()
    r0 = val.evaluate_checkpoint(model, adapter, loader, items, hp, draws=1, seed=5, teacher_cache=cache)
    state = copy.deepcopy(adapter.adapter_state_dict())
    adapter.load_adapter_state_dict({k: v + 0.05 for k, v in state.items()})
    cached = val.evaluate_checkpoint(model, adapter, loader, items, hp, draws=1, seed=5, teacher_cache=cache)
    fresh = val.evaluate_checkpoint(model, adapter, loader, items, hp, draws=1, seed=5)
    assert cache.hits > 0
    assert _flat(cached) == _flat(fresh)                     # cached teacher == recomputed teacher
    assert _flat(cached) != _flat(r0)
    assert cached["summary"]["dev"]["overall"]["lb_R"] != r0["summary"]["dev"]["overall"]["lb_R"]


def test_matches_training_loss_definition():
    model, adapter, hp, loader, items = _setup()
    it = next(i for i in items if not i.is_static)
    rec = val.evaluate_item(model, adapter, loader, it, hp, draws=1, seed=3)[0]
    mb = loader(it.plan())
    R = v2.revealed_mask(mb.condition)
    sa, na, sb, nb = val.fixed_draw(3, it.item_id, 0, hp, mb.static.shape, mb.static.dtype, "cpu")
    a = v2.build_pass_a(model, adapter, mb, hp, sa, na, R)
    adapter.set_condition(mb.condition)
    with torch.no_grad():
        pa = v2._call(model, a["noisy"], a["timesteps"], mb.text)
        la = v2.weighted_mse(pa[:, :, 1:], a["target"], a["weight"])
        b = v2.build_pass_b(mb, hp, sb, nb, R)
        pb = v2._call(model, b["noisy"], b["timesteps"], mb.text)
        lb = v2.weighted_mse(pb[:, :, 1:], b["target"], b["weight"])
    assert rec["la"] == pytest.approx(float(la)) and rec["lb"] == pytest.approx(float(lb))
    assert 0 < rec["r_fraction"] < 1 and rec["lb_R"] is not None


def test_region_restricted_losses_on_synthetic_mask():
    R = torch.zeros(1, 1, 2, 2, 2)
    R[:, :, :, :, 1] = 1.0                                   # right column revealed
    target = torch.zeros(1, 48, 2, 2, 2)
    pred = target.clone()
    pred[:, :, :, :, 1] = 2.0                                # error only inside R
    b = val.pass_b_losses(pred, target, R, 1.0, 4.0)
    assert b["lb_R"] == pytest.approx(4.0) and b["lb_valid"] == pytest.approx(0.0)
    assert b["lb"] == pytest.approx(4.0 * 5 * 4 / (5 * 4 + 1 * 4))   # w=5 on the 4 R cells, w=1 on the 4 valid cells
    a = val.pass_a_losses(pred, target, R, 0.1)
    assert a["la_valid"] == pytest.approx(0.0) and a["la"] == pytest.approx(4 * 0.1 * 4 / (4 * 0.1 + 4))
    pred2 = target.clone()
    pred2[:, :, :, :, 0] = 3.0                               # error only in the valid region
    assert val.pass_b_losses(pred2, target, R, 1.0, 4.0)["lb_R"] == pytest.approx(0.0)
    assert val.pass_a_losses(pred2, target, R, 0.1)["la_valid"] == pytest.approx(9.0)
    none = val.pass_b_losses(pred, target, torch.zeros_like(R), 1.0, 4.0)
    assert none["lb_R"] is None                              # empty region -> excluded, not zero


def test_buckets_families_and_static_rules():
    assert [val.magnitude_bucket(u) for u in (0.1, 0.4, 0.7, 0.71)] == ["u<0.4", "0.4-0.7", "0.4-0.7", "u>0.7"]
    assert val.magnitude_bucket(None) == "n/a" and val.magnitude_bucket(0.9, static=True) == "static"
    model, adapter, hp, loader, items = _setup()
    res = val.evaluate_checkpoint(model, adapter, loader, items, hp, draws=1, seed=1)
    srows = [r for r in res["rows"] if r["static_request"]]
    assert srows and all(r["lb"] is None and r["lb_R"] is None for r in srows)   # static: pass A only
    s = res["summary"]["dev"]
    assert set(s["by_family"]) == {"pan", "static"} and set(s["by_bucket"]) <= {"static", "u<0.4", "0.4-0.7", "u>0.7"}
    assert s["overall"]["n_rows"] == len(items) and s["overall"]["n"]["lb"] == len(items) - 3


def test_discovery_resume_and_reports(tmp_path):
    from safetensors.torch import save_file

    ck = tmp_path / "checkpoints"
    ck.mkdir()
    for s in (77, 250, 477):
        save_file({"w": torch.zeros(1)}, str(ck / f"adapter_{s:06d}.safetensors"))
    got = val.discover_checkpoints(tmp_path, steps=[477, 77])
    assert [g[0] for g in got] == ["77", "477"]
    assert [g[0] for g in val.discover_checkpoints(tmp_path, all_local=True)] == ["77", "250", "477"]
    with pytest.raises(FileNotFoundError):
        val.discover_checkpoints(tmp_path, steps=[5])
    ext = val.discover_checkpoints(tmp_path, paths=[f"{ck / 'adapter_000250.safetensors'}=v1init"])
    assert ext[0][0] == "v1init"

    model, adapter, hp, loader, items = _setup()
    ident = {"draws": 1, "seed": 5, "n_dev_sources": 3, "n_train_sources": 0}
    path = tmp_path / "validation.json"
    res = val.load_results(path, ident)
    summ = val.evaluate_checkpoint(model, adapter, loader, items, hp, seed=5)["summary"]
    summ["train_subset"] = summ["dev"]
    res["checkpoints"]["77"] = {"summary": summ}
    val.write_reports(res, tmp_path)
    again = val.load_results(path, ident)
    assert [c[0] for c in val.pending_labels(again, got)] == ["477"]
    with pytest.raises(ValueError):
        val.load_results(path, {**ident, "draws": 4})
    assert "dev L_B" in (tmp_path / "validation.md").read_text(encoding="utf8")
    assert json.loads(path.read_text())["checkpoints"]["77"]
    pytest.importorskip("matplotlib")
    assert (tmp_path / "validation.png").stat().st_size > 1000
