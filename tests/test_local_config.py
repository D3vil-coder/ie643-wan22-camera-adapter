"""Machine-specific paths come from the environment (tools/local_config.py), never from a hard-coded laptop path."""
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import local_config as lc  # noqa: E402
import molab_recover as mr  # noqa: E402
import publish_public_hf as pub  # noqa: E402
import pull_milestones as pm  # noqa: E402


def _keys(path, text="HF_WRITE_TOKEN = abc\nHF_DOWNLOAD_TOKEN=def\n"):
    path.write_text(text)
    return path


def test_keys_env_variable_wins_over_repo_default(tmp_path):
    (tmp_path / "keys.env").write_text("x=1")
    other = _keys(tmp_path / "elsewhere.env")
    assert lc.keys_env_path({lc.KEYS_ENV_VAR: str(other)}, root=tmp_path) == other


def test_keys_env_default_is_repo_root_file(tmp_path):
    default = _keys(tmp_path / "keys.env")
    assert lc.keys_env_path({}, root=tmp_path) == default
    assert lc.keys_env_path({lc.KEYS_ENV_VAR: "  "}, root=tmp_path) == default  # blank counts as unset


def test_keys_env_variable_pointing_nowhere_is_an_error(tmp_path):
    _keys(tmp_path / "keys.env")  # a default exists, but an explicit wrong path must not silently fall back
    with pytest.raises(lc.KeysEnvNotFound) as err:
        lc.keys_env_path({lc.KEYS_ENV_VAR: str(tmp_path / "missing.env")}, root=tmp_path)
    assert lc.KEYS_ENV_VAR in str(err.value) and "missing.env" in str(err.value)


def test_keys_env_missing_everywhere_explains_how_to_set_it(tmp_path):
    with pytest.raises(lc.KeysEnvNotFound) as err:
        lc.keys_env_path({}, root=tmp_path)
    msg = str(err.value)
    assert lc.KEYS_ENV_VAR in msg and str(tmp_path / "keys.env") in msg and "HF_WRITE_TOKEN" in msg
    assert isinstance(err.value, FileNotFoundError)


def test_checkpoints_dir_env_and_default(tmp_path):
    assert lc.checkpoints_dir({lc.CHECKPOINTS_DIR_VAR: str(tmp_path / "ck")}, root=tmp_path) == tmp_path / "ck"
    assert lc.checkpoints_dir({}, root=tmp_path) == tmp_path / "artifacts" / "checkpoints" / "v2"


def test_molab_recover_reads_token_from_env_configured_file(tmp_path, monkeypatch):
    monkeypatch.setenv(lc.KEYS_ENV_VAR, str(_keys(tmp_path / "k.env")))
    assert mr.read_hf_token() == "abc"


def test_molab_recover_missing_keys_file_is_a_clear_exit(tmp_path, monkeypatch):
    monkeypatch.setenv(lc.KEYS_ENV_VAR, str(tmp_path / "nope.env"))
    with pytest.raises(SystemExit) as err:
        mr.read_hf_token()
    assert lc.KEYS_ENV_VAR in str(err.value)


def test_molab_recover_token_name_missing_in_file(tmp_path, monkeypatch):
    monkeypatch.setenv(lc.KEYS_ENV_VAR, str(_keys(tmp_path / "k.env", "OTHER=1\n")))
    with pytest.raises(SystemExit) as err:
        mr.read_hf_token()
    assert "HF_WRITE_TOKEN" in str(err.value)


def test_pull_milestones_token_order(tmp_path, monkeypatch):
    monkeypatch.setenv(lc.KEYS_ENV_VAR, str(_keys(tmp_path / "k.env")))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert pm.read_token() == "def"  # HF_DOWNLOAD_TOKEN is preferred over HF_WRITE_TOKEN
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert pm.read_token() == "from-env"


def test_pull_milestones_without_any_keys_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv(lc.KEYS_ENV_VAR, raising=False)
    monkeypatch.setattr(pm, "keys_env_path", lambda: (_ for _ in ()).throw(lc.KeysEnvNotFound("none")))
    assert pm.read_token() is None  # a cached `huggingface-cli login` may still work


def test_pull_milestones_explicit_but_missing_keys_file_exits(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv(lc.KEYS_ENV_VAR, str(tmp_path / "nope.env"))
    with pytest.raises(SystemExit):
        pm.read_token()


def test_publish_default_checkpoint_dir_follows_env(tmp_path, monkeypatch):
    monkeypatch.setenv(lc.CHECKPOINTS_DIR_VAR, str(tmp_path))
    assert pub.default_checkpoint_dir() == tmp_path / "v2_full"
    monkeypatch.delenv(lc.CHECKPOINTS_DIR_VAR)
    assert pub.default_checkpoint_dir() == ROOT / "artifacts" / "checkpoints" / "v2" / "v2_full"


def test_publish_get_token_falls_back_to_hf_token_and_otherwise_explains(tmp_path, monkeypatch):
    monkeypatch.setenv(lc.KEYS_ENV_VAR, str(tmp_path / "nope.env"))
    monkeypatch.setenv("HF_TOKEN", "tok")
    assert pub.get_token() == "tok"
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(SystemExit) as err:
        pub.get_token()
    assert lc.KEYS_ENV_VAR in str(err.value)


# A path that only exists on one laptop must never be baked into code that is published.
_LOCAL_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]|IIT Bombay[\\/]")


@pytest.mark.parametrize("folder", ["tools", "scripts", "src", "notebooks", "docs/submission/build"])
def test_no_hard_coded_laptop_paths_in_code(folder):
    offenders = [str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob("*.py") if _LOCAL_PATH.search(p.read_text(encoding="utf8"))]
    assert offenders == []


_PERSONAL = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]|IIT Bombay[\\/]|[\\/]Downloads[\\/]")


def test_no_personal_paths_in_docs_and_configs():
    files = [ROOT / "README.md", ROOT / "ASSETS.md", *(ROOT / "docs").rglob("*.md"), *(ROOT / "configs").rglob("*.json")]
    offenders = [str(p.relative_to(ROOT)) for p in files if p.is_file() and _PERSONAL.search(p.read_text(encoding="utf8"))]
    assert offenders == []
