import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import publish_public_hf as pub  # noqa: E402
from fpcamera import flow_infer as fi  # noqa: E402

STEP = 415


def _fake_checkpoint_dir(tmp_path, step=STEP):
    d = tmp_path / "ckpt"
    d.mkdir()
    weights = d / f"adapter_{step:06d}.safetensors"
    header = json.dumps({"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}).encode()
    weights.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 24)  # a valid tiny safetensors file
    (d / f"ckpt_{step:06d}.json").write_text(json.dumps({
        "step": step,
        "adapter_file": weights.name,
        "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "architecture": {"kind": "wan_flow_control_lora", "config": {"hidden": 3072}},
    }))
    (d / "config.json").write_text(json.dumps({"config": {
        "fill_mode": "replicate",
        "video_contract": {"width": 1280, "height": 704, "frames": 121, "fps": 24},
        "expected_backbone_sha256": "abc",
        "objective": {"sigma_A": {"lower": 0.6}},
    }}))
    return d, weights


def _fake_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "fpcamera" / "__pycache__").mkdir(parents=True)
    (repo / "src" / "fpcamera" / "provenance").mkdir()
    (repo / "src" / "fpcamera" / "__init__.py").write_text("__version__ = '0'\n")
    (repo / "src" / "fpcamera" / "provenance" / "a.py").write_text("x = 1\n")
    (repo / "src" / "fpcamera" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"junk")
    (repo / "requirements.txt").write_text("numpy\n")
    (repo / "requirements-ui.txt").write_text("marimo\n")
    (repo / "configs").mkdir()
    (repo / "configs" / "strength_calibration_v2.identity.json").write_text('{"version": 1}')
    (repo / "configs" / "inference_v2.json").write_text(json.dumps({"calibration": "configs/x.identity.json", "steps": 50}))
    return repo


@pytest.fixture
def staged(tmp_path, monkeypatch):
    ckpt_dir, weights = _fake_checkpoint_dir(tmp_path)
    repo = _fake_repo(tmp_path)
    monkeypatch.setattr(pub, "source_state", lambda: ("c0ffee" * 6 + "abcd", False))
    out = tmp_path / "stage"
    files, commit, dirty = pub.stage(out, ckpt_dir=ckpt_dir, step=STEP, repo_id="user/repo", repo_root=repo,
                                     expect_sha_prefix=None, secrets=("SECRETVALUE123",))
    return out, files, weights


def test_stage_is_exactly_the_allow_list(staged):
    out, files, _ = staged
    assert {f[0] for f in files} == pub.expected_paths(STEP)
    assert pub.audit_stage(out, STEP) == []


def test_flat_adapter_folder_resolves_replicate_fill(staged):
    """The published layout adapter/{weights,ckpt json,config.json} (no checkpoints/ subfolder) is a valid v2 run dir."""
    out, _, weights = staged
    ckpt = out / "adapter" / weights.name
    meta, source = fi.resolve_checkpoint_meta(ckpt)
    assert source == "v2_manifest" and meta["run_config_found"] is True
    assert meta["completed_updates"] == STEP
    assert meta["checkpoint_sha256"] == pub.sha256_file(ckpt)
    assert fi.resolve_guide_fill_mode(None, meta) == ("replicate", "checkpoint", None)


def test_zip_layout_has_no_pycache_and_a_version(staged):
    out, _, _ = staged
    with zipfile.ZipFile(out / pub.ZIP_NAME) as zf:
        names = zf.namelist()
        assert "VERSION" in names and "requirements.txt" in names and "requirements-ui.txt" in names
        assert "src/fpcamera/__init__.py" in names and "src/fpcamera/provenance/a.py" in names
        assert not [n for n in names if "__pycache__" in n or n.endswith(".pyc")]
        assert zf.read("VERSION").decode().strip().startswith("c0ffee")


def test_staged_inference_config_points_to_the_published_calibration(staged):
    out, _, _ = staged
    cfg = json.loads((out / "configs" / "inference_v2.json").read_text())
    assert cfg["calibration"] == "configs/strength_calibration_v2.json" and cfg["steps"] == 50
    assert json.loads((out / "configs" / "strength_calibration_v2.json").read_text()) == {"version": 1}


def test_zip_is_byte_deterministic(tmp_path):
    repo = _fake_repo(tmp_path)
    files = pub.code_files(repo / "src", repo)
    pub.build_zip(tmp_path / "a.zip", files, "v\n")
    pub.build_zip(tmp_path / "b.zip", files, "v\n")
    assert (tmp_path / "a.zip").read_bytes() == (tmp_path / "b.zip").read_bytes()


def test_stage_rejects_digest_mismatch_and_wrong_prefix(tmp_path, monkeypatch):
    ckpt_dir, weights = _fake_checkpoint_dir(tmp_path)
    repo = _fake_repo(tmp_path)
    monkeypatch.setattr(pub, "source_state", lambda: ("a" * 40, False))
    with pytest.raises(SystemExit, match="expected"):
        pub.stage(tmp_path / "s1", ckpt_dir=ckpt_dir, step=STEP, repo_id="u/r", repo_root=repo, expect_sha_prefix="83df1e3f")
    weights.write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="manifest"):
        pub.stage(tmp_path / "s2", ckpt_dir=ckpt_dir, step=STEP, repo_id="u/r", repo_root=repo, expect_sha_prefix=None)


def test_stage_refuses_dirty_source_tree_for_publish(tmp_path, monkeypatch):
    ckpt_dir, _ = _fake_checkpoint_dir(tmp_path)
    repo = _fake_repo(tmp_path)
    monkeypatch.setattr(pub, "source_state", lambda: ("a" * 40, True))
    with pytest.raises(SystemExit, match="uncommitted"):
        pub.stage(tmp_path / "s", ckpt_dir=ckpt_dir, step=STEP, repo_id="u/r", repo_root=repo,
                  expect_sha_prefix=None, allow_dirty=False)


def test_audit_flags_stray_files_missing_files_and_secrets(staged):
    out, _, _ = staged
    (out / "keys.env").write_text("HF_WRITE_TOKEN = x")
    (out / "clip.mp4").write_bytes(b"x")
    problems = pub.audit_stage(out, STEP)
    assert any("outside the allow-list" in p and "keys.env" in p and "clip.mp4" in p for p in problems)
    (out / "keys.env").unlink(), (out / "clip.mp4").unlink()
    (out / "configs" / "inference_v2.json").unlink()
    assert any("missing files" in p for p in pub.audit_stage(out, STEP))


def test_scan_text_detects_secrets_without_echoing_them():
    token = "hf_" + "A1b2C3d4E5" * 4
    hits = pub.scan_text("README.md", f"token {token}", extra_secrets=("SECRETVALUE123",))
    assert hits == ["README.md: hf token"]
    assert pub.scan_text("a", "see https://sb-abcdef12.sb.molab.run/x") == ["a: molab url"]
    assert pub.scan_text("a", r"C:\Users\someone\Downloads\x.png") == ["a: local user path"]
    assert pub.scan_text("a", "mail me at someone@example.com") == ["a: e-mail"]
    assert pub.scan_text("a", "load from keys.env") == ["a: keys.env"]
    assert pub.scan_text("a", "has SECRETVALUE123 inside", extra_secrets=("SECRETVALUE123",)) == ["a: literal secret value"]
    assert pub.scan_text("a", "@torch.no_grad() def f(): ...") == []


def test_audit_scans_zip_members(staged):
    out, _, _ = staged
    with zipfile.ZipFile(out / pub.ZIP_NAME, "a") as zf:
        zf.writestr("src/fpcamera/leak.py", "T = 'hf_" + "Z" * 30 + "'\n")
    assert any("leak.py: hf token" in p for p in pub.audit_stage(out, STEP))


def test_render_card_fills_every_placeholder():
    assert pub.render_card("a {{X}} b {{X}}", {"X": 1}) == "a 1 b 1"
    with pytest.raises(ValueError, match="unresolved"):
        pub.render_card("a {{Y}}", {"X": 1})


def test_real_template_has_front_matter_and_apache_license():
    text = pub.CARD_TEMPLATE.read_text(encoding="utf8")
    head = text.split("---")[1]
    assert "base_model: Wan-AI/Wan2.2-TI2V-5B-Diffusers" in head and "library_name: diffusers" in head
    assert "license: apache-2.0" in head
    assert "License: Apache-2.0" in text and "not yet specified" not in text
