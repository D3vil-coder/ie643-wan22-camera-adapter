"""CPU tests for the opt-in kernel-level speed options: --attn-backend, --compile-blocks, --speed-probe.

Nothing here needs a GPU: the attention backend selection is checked on torch's global SDPA flags, block compilation is
checked with a counting stand-in for torch.compile plus (one test) the real Dynamo front end with the ``aot_eager``
backend, and the speed-probe loop runs on the mock transformer with injected OOMs / numerical drift / compile errors.
"""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fpcamera import flow_train_v2 as v2  # noqa: E402
from fpcamera import speed_options as so  # noqa: E402
import test_train_v2_logic as T  # noqa: E402

HAS_CUDNN_ENUM = so._sdpa_api()[0] is not None and hasattr(so._sdpa_api()[0], "CUDNN_ATTENTION")


# ------------------------------------------------------------------ attention backend selection
def test_auto_is_a_true_noop(monkeypatch):
    monkeypatch.setattr(so, "_sdpa_api", lambda: (_ for _ in ()).throw(AssertionError("sdpa_kernel must not be touched")))
    spec = so.resolve_attn_backend("auto")
    assert (spec.requested, spec.effective, spec.priority, spec.active) == ("auto", "auto", (), False)
    assert so.resolve_attn_backend(None) == spec
    assert so.resolve_attn_backend(spec) is spec  # an already-resolved spec passes straight through (run_training does this)
    with so.attention_backend(spec) as s:
        assert s is spec
    with so.attention_backend("auto"):
        pass


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        so.resolve_attn_backend("sage")
    assert so.ATTN_BACKENDS == ("auto", "cudnn", "flash", "efficient")


@pytest.mark.skipif(not HAS_CUDNN_ENUM, reason="this torch has no SDPBackend.CUDNN_ATTENTION")
def test_priority_lists():
    assert so.resolve_attn_backend("cudnn").priority == ("CUDNN_ATTENTION", "FLASH_ATTENTION", "EFFICIENT_ATTENTION")
    assert so.resolve_attn_backend("flash").priority == ("FLASH_ATTENTION",)
    assert so.resolve_attn_backend("efficient").priority == ("EFFICIENT_ATTENTION",)
    rec = so.resolve_attn_backend("cudnn").record()
    assert rec["requested"] == "cudnn" and rec["effective"] == "cudnn" and rec["torch"] == torch.__version__


def test_cudnn_falls_back_to_auto_without_the_enum(monkeypatch):
    """torch without SDPBackend.CUDNN_ATTENTION: warn, behave exactly like auto (no context is entered)."""
    class OldBackend:  # an enum from torch < 2.5
        FLASH_ATTENTION, EFFICIENT_ATTENTION, MATH = "flash", "eff", "math"

    def boom(*a, **k):
        raise AssertionError("sdpa_kernel must not be called for a fallback")

    monkeypatch.setattr(so, "_sdpa_api", lambda: (OldBackend, boom))
    with pytest.warns(UserWarning, match="CUDNN_ATTENTION"):
        spec = so.resolve_attn_backend("cudnn")
    assert spec.requested == "cudnn" and spec.effective == "auto" and not spec.active and "CUDNN_ATTENTION" in spec.note
    with so.attention_backend(spec):
        pass
    # flash/efficient still resolve on that torch
    assert so.resolve_attn_backend("flash").priority == ("FLASH_ATTENTION",)


def test_missing_sdpa_kernel_api_falls_back(monkeypatch):
    monkeypatch.setattr(so, "_sdpa_api", lambda: (None, None))
    msgs = []
    spec = so.resolve_attn_backend("flash", warn=msgs.append)
    assert spec.effective == "auto" and not spec.active and msgs and "sdpa_kernel" in msgs[0]


@pytest.mark.skipif(not HAS_CUDNN_ENUM, reason="this torch has no SDPBackend.CUDNN_ATTENTION")
def test_context_sets_and_restores_global_flags():
    before = so.sdpa_flags()
    with so.attention_backend("cudnn") as spec:
        inside = so.sdpa_flags()
        assert spec.active
        assert inside["math"] is False and inside["flash"] and inside["mem_efficient"] and inside["cudnn"]
    assert so.sdpa_flags() == before
    with so.attention_backend("efficient"):
        inside = so.sdpa_flags()
        assert inside["mem_efficient"] and not inside["flash"] and not inside["math"] and not inside["cudnn"]
    assert so.sdpa_flags() == before


def test_context_without_set_priority_still_restricts(monkeypatch):
    """torch < 2.6: sdpa_kernel(backends) has no set_priority; we retry without it."""
    seen = {}

    class B:
        CUDNN_ATTENTION, FLASH_ATTENTION, EFFICIENT_ATTENTION = "c", "f", "e"

    import contextlib

    @contextlib.contextmanager
    def old_kernel(backends):  # no set_priority parameter
        seen["backends"] = list(backends)
        yield

    monkeypatch.setattr(so, "_sdpa_api", lambda: (B, old_kernel))
    with so.attention_backend("cudnn"):
        pass
    assert seen["backends"] == ["c", "f", "e"]


def test_transformer_attention_wraps_forward_and_restores():
    class Tx(torch.nn.Module):
        def forward(self, x):
            return x, so.sdpa_flags()["math"]

    tx = Tx()
    with so.transformer_attention(tx, "auto"):
        assert "forward" not in tx.__dict__
    if not HAS_CUDNN_ENUM:
        pytest.skip("no CUDNN_ATTENTION in this torch")
    with so.transformer_attention(tx, "cudnn"):
        assert "forward" in tx.__dict__
        assert tx(1)[1] is False  # math disabled inside the forward
    assert "forward" not in tx.__dict__ and tx(1)[1] is True  # restored, default flags back
    tx.forward = lambda x: ("custom", None)  # a pre-existing instance forward (e.g. an outer wrapper) is restored
    custom = tx.forward
    with so.transformer_attention(tx, "cudnn"):
        assert tx(1) == ("custom", None)
    assert tx.forward is custom


# ------------------------------------------------------------------ trainer: defaults unchanged, context spans backward
def test_default_trainer_never_enters_an_sdpa_context(monkeypatch):
    _, _, trainer, _ = T.build_stack(p_static=0.0)
    assert trainer.attn_backend.requested == "auto" and not trainer.attn_backend.active
    spec = so.resolve_attn_backend("auto")  # run_training hands the trainer a resolved spec, not a string
    assert v2.HybridTrainer(trainer.model, trainer.adapter, trainer.optimizer, trainer.hp, trainer.sampler,
                            trainer.loader, device="cpu", attn_backend=spec).attn_backend is spec
    monkeypatch.setattr(so, "_sdpa_api", lambda: (_ for _ in ()).throw(AssertionError("sdpa_kernel entered")))
    rec = trainer.run_update()
    assert rec["step"] == 1


@pytest.mark.skipif(not HAS_CUDNN_ENUM, reason="this torch has no SDPBackend.CUDNN_ATTENTION")
def test_attention_context_spans_forward_backward_and_checkpoint_recompute():
    def run(attn):
        model, adapter, trainer, _ = T.build_stack(p_static=0.0)
        trainer.attn_backend = so.resolve_attn_backend(attn)
        seen = []
        for b in model.blocks:
            f = b.forward

            def spy(*a, _f=f, **k):
                seen.append(so.sdpa_flags()["math"])
                return _f(*a, **k)

            b.forward = spy
        v2.apply_activation_checkpointing(model, {"mode": "first_k", "k": 2})
        model.train()
        recs = [trainer.run_update() for _ in range(2)]
        return recs, seen

    base, seen_auto = run("auto")
    got, seen_cudnn = run("cudnn")
    assert all(seen_auto) and not any(seen_cudnn)  # every block call, including the backward-time recomputes
    assert len(seen_cudnn) == len(seen_auto) and len(seen_cudnn) > 2 * 2 * 2  # recompute really happened
    for a, b in zip(base, got):  # the mock has no SDPA, so the numbers are exactly the same
        assert a["loss_a"] == b["loss_a"] and a["loss_b"] == b["loss_b"] and a["grad_norm"] == b["grad_norm"]
    assert so.sdpa_flags()["math"] is True  # restored after each update


def test_microbatches_argument_replays_a_fixed_batch_without_touching_the_sampler():
    _, _, trainer, _ = T.build_stack(p_static=0.0)
    mbs = v2._probe_microbatches(trainer)
    assert len(mbs) == trainer.hp.effective_batch and not any(m.meta["static_request"] for m in mbs)
    state = json.dumps(trainer.sampler.state_dict(), default=str)
    a = trainer.run_update(accumulate_only=True, microbatches=mbs)
    b = trainer.run_update(accumulate_only=True, microbatches=mbs)
    assert json.dumps(trainer.sampler.state_dict(), default=str) == state  # sampler untouched by the replays
    assert trainer.step == 0 and a["grad_norm"] == b["grad_norm"]
    assert [m["loss_a"] for m in a["micro"]] == [m["loss_a"] for m in b["micro"]]


# ------------------------------------------------------------------ run identity / signature / defaults
def test_run_identity_ignores_speed_flags():
    from types import SimpleNamespace

    tables = [SimpleNamespace(index_sha256="abc", weight=1.0)]
    plain = v2.run_identity(v2.normalize_config({}), tables)
    noisy = v2.run_identity(v2.normalize_config({"attn_backend": "cudnn", "compile_blocks": True,
                                                  "speed_probe": {"x": 1},
                                                  "activation_checkpointing": {"mode": "first_k", "k": 5}}), tables)
    assert plain == noisy
    assert not any(any(w in k.lower() for w in ("attn", "compile", "speed", "probe")) for k in plain)


def test_run_training_signature_defaults():
    import inspect

    sig = inspect.signature(v2.run_training).parameters
    assert sig["attn_backend"].default == "auto"
    assert sig["compile_blocks"].default is False
    assert sig["speed_probe_config"].default is None
    assert sig["ckpt_blocks"].default is None and sig["probe"].default is False
    assert 15 in v2.DEFAULT_PROBE_BLOCKS and v2.DEFAULT_PROBE_BLOCKS[:3] == (None, 20, 15)
    assert inspect.signature(v2.HybridTrainer.__init__).parameters["attn_backend"].default == "auto"
    from fpcamera import flow_infer as fi

    gen = inspect.signature(fi.generate).parameters
    assert gen["attn_backend"].default == "auto" and gen["compile_blocks"].default is False


# ------------------------------------------------------------------ CLIs
def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE_ARGS = ["--config", "c.json", "--run-dir", "r"]


def test_train_cli_defaults():
    a = _load_script("train_v2").build_parser().parse_args(BASE_ARGS)
    assert a.attn_backend == "auto" and a.compile_blocks is False and a.speed_probe is False
    assert a.ckpt_blocks is None and a.memory_probe is False
    assert a.probe_ckpt == (20, 15) and a.probe_attn == ("auto", "cudnn") and a.probe_compile == (False, True)
    assert a.probe_budget_minutes == 20.0 and a.probe_timed_updates == 1 and a.probe_equiv_tol == 1e-2


def test_train_cli_parses_and_validates():
    p = _load_script("train_v2").build_parser()
    a = p.parse_args(BASE_ARGS + ["--attn-backend", "cudnn", "--compile-blocks", "--check-only", "--speed-probe",
                                  "--probe-ckpt", "20,15,12", "--probe-attn", "auto,flash,cudnn", "--probe-compile",
                                  "off,on", "--probe-budget-minutes", "9.5", "--probe-timed-updates", "2",
                                  "--probe-equiv-tol", "0.005"])
    assert a.attn_backend == "cudnn" and a.compile_blocks and a.speed_probe and a.check_only
    assert a.probe_ckpt == (20, 15, 12) and a.probe_attn == ("auto", "flash", "cudnn")
    assert a.probe_compile == (False, True) and a.probe_budget_minutes == 9.5 and a.probe_equiv_tol == 0.005
    for bad in (["--attn-backend", "sage"], ["--probe-ckpt", "a,b"], ["--probe-attn", "auto,xla"],
                ["--probe-compile", "maybe"]):
        with pytest.raises(SystemExit):
            p.parse_args(BASE_ARGS + bad)


def test_train_cli_probe_flag_combinations_rejected():
    main = _load_script("train_v2").main
    with pytest.raises(SystemExit):  # --speed-probe needs --check-only, like --memory-probe
        main(BASE_ARGS + ["--speed-probe"])
    with pytest.raises(SystemExit):
        main(BASE_ARGS + ["--memory-probe"])
    with pytest.raises(SystemExit):
        main(BASE_ARGS + ["--check-only", "--speed-probe", "--memory-probe"])
    with pytest.raises(SystemExit):
        main(BASE_ARGS + ["--check-only", "--speed-probe", "--probe-timed-updates", "0"])


def _run_main(monkeypatch, tmp_path, extra):
    mod = _load_script("train_v2")
    cfg = tmp_path / "c.json"
    cfg.write_text("{}", encoding="utf8")
    captured = {}

    def fake_run_training(config, **kw):
        captured.update(kw)
        return {"status": "check_only", "step": 0}

    monkeypatch.setattr(v2, "run_training", fake_run_training)
    assert mod.main(["--config", str(cfg), "--run-dir", str(tmp_path / "r")] + extra) == 0
    return captured


def test_train_cli_plumbs_defaults_unchanged(monkeypatch, tmp_path):
    kw = _run_main(monkeypatch, tmp_path, ["--check-only"])
    assert kw["attn_backend"] == "auto" and kw["compile_blocks"] is False and kw["speed_probe_config"] is None
    assert kw["ckpt_blocks"] is None and kw["probe"] is False and kw["probe_limit_gib"] == 85.0


def test_train_cli_plumbs_options(monkeypatch, tmp_path):
    kw = _run_main(monkeypatch, tmp_path, ["--check-only", "--speed-probe", "--attn-backend", "efficient",
                                           "--compile-blocks", "--probe-limit-gib", "80", "--probe-ckpt", "20"])
    cfg = kw["speed_probe_config"]
    assert kw["attn_backend"] == "efficient" and kw["compile_blocks"] is True
    assert isinstance(cfg, so.SpeedProbeConfig)
    assert (cfg.ckpt_blocks, cfg.limit_gib, cfg.attn_backends, cfg.compile) == ((20,), 80.0, ("auto", "cudnn"),
                                                                                (False, True))


def test_infer_cli_flags():
    from fpcamera import flow_infer as fi

    base = ["--project-root", ".", "--checkpoint", "c", "--image", "i", "--output", "o"]
    a = fi.parse_args(base)
    assert a.attn_backend == "auto" and a.compile_blocks is False
    a = fi.parse_args(base + ["--attn-backend", "cudnn", "--compile-blocks"])
    assert a.attn_backend == "cudnn" and a.compile_blocks is True
    with pytest.raises(SystemExit):
        fi.parse_args(base + ["--attn-backend", "nope"])


def test_other_scripts_expose_the_flags():
    for name in ("generate_camera", "calibrate_strength", "eval_dev_reveal", "generate_examples_v2", "validate_v2"):
        text = (ROOT / "scripts" / f"{name}.py").read_text(encoding="utf8")
        assert "--attn-backend" in text, name
    for name in ("calibrate_strength", "eval_dev_reveal", "generate_examples_v2", "generate_camera"):
        assert "--compile-blocks" in (ROOT / "scripts" / f"{name}.py").read_text(encoding="utf8"), name
    cal = (ROOT / "scripts" / "calibrate_strength.py").read_text(encoding="utf8")
    assert "attn_backend=self.attn_backend" in cal and "compile_blocks=self.compile_blocks" in cal


# ------------------------------------------------------------------ block compilation
@pytest.fixture
def fake_compile(monkeypatch):
    """Counting stand-in for torch.compile: identical numerics, records kwargs and the calls through each block."""
    log = {"compiled": [], "calls": 0}

    def compiler(fn, **kwargs):
        log["compiled"].append(kwargs)

        def compiled(*a, **k):
            log["calls"] += 1
            return fn(*a, **k)

        compiled._wrapped = fn
        return compiled

    monkeypatch.setattr(so, "_torch_compile", compiler)
    return log


def test_set_block_compile_idempotent_and_reversible(fake_compile):
    model = T.MockWan()
    originals = [b.forward for b in model.blocks]
    assert so.set_block_compile(model, True) == [0, 1]
    assert so.set_block_compile(model, True) == [0, 1]
    assert len(fake_compile["compiled"]) == 2  # idempotent
    assert fake_compile["compiled"][0] == {"mode": "default", "dynamic": False}
    assert all("_uncompiled_forward" in b.__dict__ for b in model.blocks)
    assert so.set_block_compile(model, False) == []
    assert [b.forward for b in model.blocks] == originals
    assert all("forward" not in b.__dict__ and "_uncompiled_forward" not in b.__dict__ for b in model.blocks)
    so.set_block_compile(model, True, indices=[1])
    assert sorted(i for i, b in enumerate(model.blocks) if "_uncompiled_forward" in b.__dict__) == [1]


def test_compile_goes_inside_checkpointing_and_wrong_order_is_rejected(fake_compile):
    model = T.MockWan()
    so.set_block_compile(model, True)
    v2.set_block_checkpointing(model, [0])
    assert model.blocks[0].__dict__["_unckpt_forward"]._wrapped  # the checkpoint wrapper holds the compiled forward
    with pytest.raises(RuntimeError, match="activation-checkpoint"):
        so.set_block_compile(model, False)
    v2.configure_blocks(model, 0, True)  # helper unwraps, keeps compile, rewraps nothing
    assert all("_unckpt_forward" not in b.__dict__ and "_uncompiled_forward" in b.__dict__ for b in model.blocks)
    v2.configure_blocks(model, 1, False)
    assert "_uncompiled_forward" not in model.blocks[0].__dict__ and "_unckpt_forward" in model.blocks[0].__dict__


def test_compiled_checkpointed_update_matches_eager_and_recomputes_through_compiled_forward(fake_compile):
    def run(compile_on, k):
        model, adapter, trainer, _ = T.build_stack(p_static=0.0)
        v2.configure_blocks(model, k, compile_on)
        model.train()
        fake_compile["calls"] = 0
        recs = [trainer.run_update() for _ in range(2)]
        return recs, [p.detach().clone() for p in adapter.parameters()], fake_compile["calls"]

    r0, w0, n0 = run(False, 1)
    r1, w1, n1 = run(True, 1)
    assert n0 == 0 and n1 > 0
    for a, b in zip(r0, r1):
        assert a["loss_a"] == pytest.approx(b["loss_a"], rel=1e-6) and a["grad_norm"] == pytest.approx(b["grad_norm"], rel=1e-6)
    for a, b in zip(w0, w1):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
    # block 0 is checkpointed (forward + recompute), block 1 is not: more compiled calls than with no checkpointing
    _, _, n_none = run(True, 0)
    assert n1 > n_none


def test_real_dynamo_aot_eager_composes_with_checkpoint_and_lora():
    """The actual torch.compile front end (Dynamo + AOTAutograd, no C++ toolchain): compiled blocks inside eager
    non-reentrant checkpointing, with the real FlowControlLoRA (adapter on/off flags, fp32 LoRA params, patch hook)."""
    pytest.importorskip("torch._dynamo")
    try:
        import torch._dynamo as dyn  # noqa: F401
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Dynamo unavailable: {e}")

    def run(compile_on):
        model, adapter, trainer, _ = T.build_stack(p_static=0.0)
        v2.configure_blocks(model, 1, compile_on, {"backend": "aot_eager"} if compile_on else None)
        model.train()
        recs = [trainer.run_update() for _ in range(2)]
        return recs, [p.detach().clone() for p in adapter.parameters()]

    try:
        r_eager, w_eager = run(False)
        so.reset_dynamo()
        r_comp, w_comp = run(True)
    except Exception as e:  # environment (e.g. missing compiler stack) rather than our code
        if "backend" in type(e).__name__.lower() or "Unsupported" in type(e).__name__:
            pytest.skip(f"Dynamo cannot run here: {type(e).__name__}")
        raise
    finally:
        so.reset_dynamo()
    for a, b in zip(r_eager, r_comp):
        assert a["loss_a"] == pytest.approx(b["loss_a"], rel=1e-4, abs=1e-6)
        assert a["loss_b"] == pytest.approx(b["loss_b"], rel=1e-4, abs=1e-6)
        assert a["grad_norm"] == pytest.approx(b["grad_norm"], rel=1e-3, abs=1e-6)
    for a, b in zip(w_eager, w_comp):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-5)


def test_configure_dynamo_only_raises_limits():
    import torch._dynamo.config as cfg

    names = [n for n in ("cache_size_limit", "recompile_limit", "accumulated_cache_size_limit",
                         "accumulated_recompile_limit") if isinstance(getattr(cfg, n, None), int)]
    old = {n: getattr(cfg, n) for n in names}
    try:
        so.configure_dynamo(10_000)
        assert all(getattr(cfg, n) >= old[n] for n in names)
        assert any(getattr(cfg, n) > old[n] for n in names) or all(old[n] >= 10_000 for n in names)
        raised = {n: getattr(cfg, n) for n in names}
        assert so.configure_dynamo(1) == {}  # never lowers
        assert {n: getattr(cfg, n) for n in names} == raised
    finally:
        for n, v in old.items():
            setattr(cfg, n, v)


# ------------------------------------------------------------------ train_until: compile time is logged separately
class _Stop:
    requested = False


def test_first_update_compile_time_is_logged_and_kept_out_of_the_ema(tmp_path):
    from fpcamera.checkpointing import CheckpointManager

    def run(first_update_compiles):
        _, _, tr, _ = T.build_stack()
        clock = [1000.0]
        real, durations = tr.run_update, iter([300.0, 40.0, 40.0, 40.0])

        def run_update(*a, **k):
            rec = real(*a, **k)
            clock[0] += next(durations)
            return rec

        tr.run_update = run_update
        msgs = []
        run_dir = tmp_path / str(first_update_compiles)
        run_dir.mkdir()
        mgr = CheckpointManager(run_dir, every_updates=1000, every_seconds=1e9, clock=lambda: clock[0])
        v2.train_until(tr, mgr, v2.ProgressLog(run_dir), stop_step=3, stop=_Stop(),
                       clock=lambda: clock[0], progress=msgs.append, first_update_compiles=first_update_compiles)
        return [json.loads(m) for m in msgs if m.startswith("{")]

    off = run(False)
    assert not any(m.get("event", "").startswith("compile") for m in off)  # default: no extra events
    assert not any("includes_compile" in m for m in off)
    on = run(True)
    assert [m.get("includes_compile") for m in on if "step" in m] == [True, None, None]  # only the first record
    ev = {m["event"]: m for m in on if m.get("event", "").startswith("compile")}
    assert ev["compile_first_update"]["seconds"] == 300.0
    est = ev["compile_overhead_estimate"]
    assert (est["first_update_seconds"], est["steady_update_seconds"], est["compile_seconds_estimate"]) == (300.0, 40.0, 260.0)


# ------------------------------------------------------------------ speed probe: grid, equivalence, recommendation
def test_list_parsers():
    assert so.parse_int_list("20, 15") == (20, 15) and so.parse_int_list("0") == (0,)
    assert so.parse_attn_list("auto,cudnn") == ("auto", "cudnn")
    assert so.parse_onoff_list("off,on") == (False, True) and so.parse_onoff_list("1,0") == (True, False)
    for bad, fn in (("", so.parse_int_list), ("x", so.parse_int_list), ("-1", so.parse_int_list),
                    ("fast", so.parse_attn_list), ("", so.parse_attn_list), ("maybe", so.parse_onoff_list)):
        with pytest.raises(ValueError):
            fn(bad)


def test_default_grid_and_row_order():
    cfg = so.SpeedProbeConfig()
    assert (cfg.ckpt_blocks, cfg.attn_backends, cfg.compile) == ((20, 15), ("auto", "cudnn"), (False, True))
    rows = so.plan_probe_rows(cfg, 30)
    keys = [(r["ckpt_blocks"], r["attn_backend"], r["compile"]) for r in rows]
    assert len(keys) == len(set(keys)) == 8
    assert rows[0]["baseline"] and keys[0] == (20, "auto", False) and not any(r["baseline"] for r in rows[1:])
    assert [k[2] for k in keys] == [False] * 4 + [True] * 4  # cheap rows first, compile rows last
    assert keys[4:] == [(20, "auto", True), (15, "auto", True), (20, "cudnn", True), (15, "cudnn", True)]  # attn-major


def test_baseline_always_present_and_rows_clipped():
    rows = so.plan_probe_rows(so.SpeedProbeConfig(ckpt_blocks=(10,), attn_backends=("cudnn",), compile=(False,)), 30)
    assert [(r["ckpt_blocks"], r["attn_backend"], r["compile"], r["baseline"]) for r in rows] == [
        (20, "auto", False, True), (10, "cudnn", False, False)]
    rows = so.plan_probe_rows(so.SpeedProbeConfig(ckpt_blocks=(99,), attn_backends=("auto",), compile=(False,)), 30)
    assert [(r["ckpt_blocks"], r["baseline"]) for r in rows] == [(20, True), (30, False)]
    with pytest.raises(ValueError):
        so.SpeedProbeConfig(timed_updates=0)
    with pytest.raises(ValueError):
        so.SpeedProbeConfig(equiv_tol=0)


def test_equivalence_flags_above_tolerance():
    base = {"loss_a": 0.10, "loss_b": 0.08, "grad_norm": 2.0}
    ok = so.equivalence({"loss_a": 0.1004, "loss_b": 0.0799, "grad_norm": 2.01}, base)
    assert ok["lossless"] and ok["max_rel_diff"] == pytest.approx(0.005)
    bad = so.equivalence({"loss_a": 0.1, "loss_b": 0.08, "grad_norm": 2.05}, base)  # 2.5% on the grad norm
    assert not bad["lossless"] and bad["rel_diff"]["grad_norm"] == pytest.approx(0.025)
    assert not so.equivalence({"loss_a": float("nan"), "loss_b": 0.08, "grad_norm": 2.0}, base)["lossless"]
    assert not so.equivalence({"loss_a": None, "loss_b": None, "grad_norm": None}, base)["lossless"]
    assert so.equivalence({"loss_a": 0.1, "loss_b": 0.08, "grad_norm": 2.05}, base, tol=0.05)["lossless"]
    assert so.rel_diff(1.0, 0.0) == pytest.approx(1e12) and so.rel_diff(None, 1.0) is None


def test_recommendation_is_fastest_passing_row_within_the_limit():
    def r(k, attn, comp, secs, peak, lossless=True, status="ok", base=False):
        return {"ckpt_blocks": k, "attn_backend": attn, "attn_effective": attn, "compile": comp, "seconds": secs,
                "peak_gib": peak, "lossless": lossless, "status": status, "baseline": base}

    rows = [r(20, "auto", False, 40, 72, base=True), r(20, "cudnn", False, 37, 72), r(15, "cudnn", False, 33, 90),
            r(20, "cudnn", True, 30, 70, lossless=False), r(15, "auto", True, 31, 80, status="oom")]
    best = so.recommend(rows, 85.0)
    assert (best["ckpt_blocks"], best["attn_backend"], best["compile"]) == (20, "cudnn", False)  # 33 s too big, 30 s not lossless
    assert so.flags_for(best) == "--ckpt-blocks 20 --attn-backend cudnn"
    assert so.recommend(rows, 95.0)["seconds"] == 33
    assert so.flags_for(rows[0]) == "--ckpt-blocks 20"
    assert so.flags_for(r(15, "cudnn", True, 1, 1)) == "--ckpt-blocks 15 --attn-backend cudnn --compile-blocks"
    assert so.recommend([r(20, "auto", False, 40, 99, base=True)], 85.0) is None
    # a cudnn request that fell back to auto reports no --attn-backend flag
    fb = r(20, "cudnn", False, 1, 1)
    fb["attn_effective"] = "auto"
    assert so.flags_for(fb) == "--ckpt-blocks 20"


class _Clock:
    def __init__(self, step=1.0):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


def _probe_stack():
    model, adapter, trainer, _ = T.build_stack(p_static=0.25)
    model.train()
    return model, adapter, trainer


def _grid(**kw):
    base = dict(ckpt_blocks=(2, 1), attn_backends=("auto", "cudnn"), compile=(False, True), limit_gib=85.0)
    base.update(kw)
    return so.SpeedProbeConfig(**base)


def test_speed_probe_end_to_end_on_cpu(fake_compile):
    model, adapter, trainer = _probe_stack()
    lines = []
    sampler_before = json.dumps(trainer.sampler.state_dict(), default=str)
    rep = v2.speed_probe(trainer, model, _grid(), progress=lines.append, clock=_Clock(0.5))
    assert [r["status"] for r in rep["rows"]] == ["ok"] * 8
    base = rep["rows"][0]
    assert base["baseline"] and base["speedup"] == 1.0 and base["lossless"] and base["ckpt_blocks"] == 2
    assert base["warmup_seconds"] > 0 and base["seconds"] > 0
    assert all(r["lossless"] for r in rep["rows"])  # same fixed batch and noise: identical numbers on CPU
    for r in rep["rows"]:
        assert r["rel_diff"] and r["max_rel_diff"] <= 1e-4
        assert (r["compile"]) == ("compile_overhead_seconds" in r)
    assert rep["recommended"] and rep["recommended_flags"].startswith("--ckpt-blocks")
    assert rep["adapter_zero_init"] is False and rep["baseline_noise_rel_diff"] is not None
    assert trainer.step == 0 and trainer.inside_update is False  # accumulate-only
    assert json.dumps(trainer.sampler.state_dict(), default=str) != sampler_before  # one batch drawn, once
    events = [json.loads(x) for x in lines if x.startswith("{")]
    assert sum(e["event"] == "speed_probe_row" for e in events) == 8
    assert any("recommended:" in x for x in lines) and any("s/update" in x for x in lines)  # human table printed
    json.dumps(rep, default=str)


def test_speed_probe_same_batch_for_every_row(fake_compile):
    model, adapter, trainer = _probe_stack()
    seen = []
    real = trainer.run_update

    def spy(accumulate_only=False, microbatches=None):
        seen.append((trainer.step, tuple(id(m) for m in microbatches)))
        return real(accumulate_only=accumulate_only, microbatches=microbatches)

    trainer.run_update = spy
    v2.speed_probe(trainer, model, _grid(ckpt_blocks=(1,), compile=(False,)), progress=lambda x: None, clock=_Clock())
    assert len(seen) == 6 and len(set(seen)) == 1  # warm-up + timed update of 3 rows: same step, same micro-batches


def test_speed_probe_flags_numerical_drift_and_does_not_recommend_it(fake_compile, monkeypatch):
    model, adapter, trainer = _probe_stack()
    real = v2._probe_values

    def drift(rec):
        vals = real(rec)
        if any(b.__dict__.get("_uncompiled_forward") for b in model.blocks):  # compiled rows drift by 3% in grad norm
            vals["grad_norm"] *= 1.03
        return vals

    monkeypatch.setattr(v2, "_probe_values", drift)
    grid = dict(ckpt_blocks=(2,), attn_backends=("auto",))
    rep = v2.speed_probe(trainer, model, _grid(**grid), progress=lambda x: None, clock=_Clock())
    by = {(r["ckpt_blocks"], r["attn_backend"], r["compile"]): r for r in rep["rows"]}
    assert set(by) == {(2, "auto", False), (2, "auto", True)}
    assert by[(2, "auto", False)]["lossless"] is True
    for key, r in by.items():
        if key[2]:
            assert r["lossless"] is False and r["max_rel_diff"] == pytest.approx(0.03, rel=1e-3)
    assert not rep["recommended_flags"] or "--compile-blocks" not in rep["recommended_flags"]
    table = "\n".join(so.format_table(rep))
    assert "NOT lossless" in table
    assert rep["equiv_tol"] == 1e-2
    loose = v2.speed_probe(trainer, model, _grid(equiv_tol=0.05, **grid), progress=lambda x: None, clock=_Clock())
    assert all(r["lossless"] for r in loose["rows"])


def test_speed_probe_stops_cleanly_on_oom_and_continues(fake_compile):
    model, adapter, trainer = _probe_stack()
    real = trainer.run_update

    def oom_when_few_checkpointed(*a, **k):
        if sum("_unckpt_forward" in b.__dict__ for b in model.blocks) < 2:  # K=1 "does not fit"
            trainer.inside_update = True
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return real(*a, **k)

    trainer.run_update = oom_when_few_checkpointed
    rep = v2.speed_probe(trainer, model, _grid(ckpt_blocks=(2, 1, 0)), progress=lambda x: None, clock=_Clock())
    st = {(r["ckpt_blocks"], r["attn_backend"], r["compile"]): r["status"] for r in rep["rows"]}
    assert st[(2, "auto", False)] == "ok" and st[(2, "cudnn", True)] == "ok"
    assert st[(1, "auto", False)] == "oom" and st[(0, "auto", False)] == "skipped_after_oom"  # fewer blocks only worse
    assert st[(1, "cudnn", False)] == "oom"  # a different backend is probed independently
    assert st[(1, "auto", True)] == "oom" and st[(0, "auto", True)] == "skipped_after_oom"
    assert trainer.inside_update is False
    assert rep["recommended"]["ckpt_blocks"] == 2  # only fitting rows are candidates


def test_speed_probe_records_errors_and_budget(monkeypatch, fake_compile):
    model, adapter, trainer = _probe_stack()

    def broken_compile(fn, **kw):
        raise RuntimeError("inductor exploded")

    monkeypatch.setattr(so, "_torch_compile", broken_compile)
    rep = v2.speed_probe(trainer, model, _grid(ckpt_blocks=(2,), attn_backends=("auto",)), progress=lambda x: None,
                         clock=_Clock())
    st = {(r["ckpt_blocks"], r["compile"]): r for r in rep["rows"]}
    assert st[(2, False)]["status"] == "ok" and st[(2, True)]["status"] == "error"
    assert "inductor exploded" in st[(2, True)]["error"]
    assert rep["recommended_flags"] == "--ckpt-blocks 2"  # the probe survived and still recommends the baseline

    model, adapter, trainer = _probe_stack()
    rep = v2.speed_probe(trainer, model, _grid(budget_minutes=0.01), progress=lambda x: None, clock=_Clock(1.0))
    sts = [r["status"] for r in rep["rows"]]
    assert sts[0] == "ok" and set(sts[1:]) == {"skipped_budget"}  # the baseline always runs, nothing starts after the budget


def test_speed_probe_without_baseline_leaves_equivalence_unchecked(fake_compile):
    model, adapter, trainer = _probe_stack()
    real = trainer.run_update

    def oom_baseline(*a, **k):
        if all("_unckpt_forward" in b.__dict__ for b in model.blocks):
            raise torch.cuda.OutOfMemoryError("simulated")
        return real(*a, **k)

    trainer.run_update = oom_baseline
    rep = v2.speed_probe(trainer, model, _grid(ckpt_blocks=(1,), compile=(False,)), progress=lambda x: None,
                         clock=_Clock())
    assert rep["rows"][0]["status"] == "oom"
    ok = [r for r in rep["rows"] if r["status"] == "ok"]
    assert ok and all(r["lossless"] is None for r in ok)
    assert rep["recommended"] is None  # nothing can be called lossless without a baseline


def test_adapter_zero_init_warning(fake_compile):
    model, adapter, trainer = _probe_stack()
    with torch.no_grad():
        for n, p in adapter.named_parameters():
            if n.endswith("B") or n.endswith("guide_projection.weight"):
                p.zero_()
    assert v2._adapter_is_zero(adapter)
    lines = []
    rep = v2.speed_probe(trainer, model, _grid(ckpt_blocks=(2,), attn_backends=("auto",), compile=(False,)),
                         progress=lines.append, clock=_Clock())
    assert rep["adapter_zero_init"] and "init-from" in rep["notes"][0]
    assert any("speed_probe_warning" in x for x in lines)


def test_memory_probe_default_list_and_compile_hook(fake_compile, monkeypatch):
    """memory_probe keeps its per-row behaviour; with compile_blocks it compiles each row's blocks."""
    model, adapter, trainer = _probe_stack()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(v2, "peak_gib", lambda d: 1.0)
    rep = v2.memory_probe(trainer, model, settings=(None, 1, 0), progress=lambda x: None, compile_blocks=True)
    assert [r["ckpt_blocks"] for r in rep["rows"]] == [2, 1, 0] and all(r["status"] == "ok" for r in rep["rows"])
    assert len(fake_compile["compiled"]) == 2  # compiled once per block, kept across rows


def test_deadline_rule_reserves_compile_time_for_the_first_update(tmp_path):
    from fpcamera.checkpointing import CheckpointManager

    def run(first_update_compiles):
        _, _, tr, _ = T.build_stack()
        d = tmp_path / str(first_update_compiles)
        d.mkdir()
        mgr = CheckpointManager(d, every_updates=1000, every_seconds=1e9, clock=lambda: 1000.0)
        # now + est_update + est_ckpt + grace = 1000 + 10 + 5 + 20 = 1035 <= 1036: one plain update fits
        res = v2.train_until(tr, mgr, v2.ProgressLog(d), stop_step=1, stop=_Stop(), deadline=1036.0,
                             clock=lambda: 1000.0, progress=lambda m: None, est_update=10.0, est_ckpt=5.0, grace=20.0,
                             first_update_compiles=first_update_compiles, compile_allowance=420.0)
        return res, tr.step

    assert run(False) == ({"mem_stop": False, "deadline_stop": False}, 1)
    res, step = run(True)
    assert res["deadline_stop"] and step == 0  # a 420 s compile allowance does not fit
