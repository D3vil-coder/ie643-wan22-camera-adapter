import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import molab_recover as mr  # noqa: E402


def modes(plan):
    return dict(plan)


def test_all_done_skips_and_optional_off():
    done = {s: True for s in ("probe", "code", "deps", "token", "model", "restore")}
    m = modes(mr.plan_steps(done))
    assert all(m[s] == "skip" for s in done)
    assert m["build"] == "off" and m["train"] == "off"


def test_fresh_runs_everything_and_build_flag():
    m = modes(mr.plan_steps({}, want_build=True))
    assert m["probe"] == m["restore"] == m["build"] == "run" and m["train"] == "off"


def test_partial_and_force_and_only():
    done = {"probe": True, "code": True, "deps": False, "token": True}
    m = modes(mr.plan_steps(done, force=("code",), train_run="v2_full"))
    assert (m["probe"], m["code"], m["deps"], m["token"], m["train"]) == ("skip", "run", "run", "skip", "run")
    m = modes(mr.plan_steps(done, only=["deps", "model"]))
    assert m["deps"] == m["model"] == "run" and m["probe"] == "off"


def test_read_hf_token_strips_space(tmp_path):
    f = tmp_path / "k.env"
    f.write_text("A =1\nHF_WRITE_TOKEN = abc \n")
    assert mr.read_hf_token(str(f)) == "abc"


def test_decide_launch():
    n = "hf_backup.py watch-run"
    assert mr.decide_launch(None, None, [], n)[0] == "launch"
    assert mr.decide_launch(5, "python tools/hf_backup.py watch-run x", [], n)[0] == "skip"
    assert mr.decide_launch(5, None, [], n)[0] == "launch"  # stale pidfile
    assert mr.decide_launch(5, "python other.py", [], n)[0] == "launch"  # pid reused
    assert mr.decide_launch(None, None, [77], n)[0] == "skip"  # fallback scan
    assert mr.decide_launch(5, "other", [77], n)[0] == "skip"


def test_train_cmd_uses_targeted_override():
    src = (Path(mr.__file__)).read_text(encoding="utf8")
    assert "--allow-identity-change caches" in src and "--allow-config-change" not in src.replace(
        "--allow-config-change commit", "")
