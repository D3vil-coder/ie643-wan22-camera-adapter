import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import diagnose_v1_borders as dv  # noqa: E402


def res(score, n=50, hf=0.14):
    return {
        "mean_over_frames_with_R_gt_1pct": {"artifact_score": score, "n_frames": n, "hf_ratio": 0.4},
        "x0hat_sigma_0.60_metrics": {"hf_ratio": hf},
    }


def test_verdicts_supported_and_not():
    r = {
        "v1_baseline": res(0.95),
        "fill_replicate": res(0.30),
        "min_sigma_0.3": res(0.90),
        "min_sigma_0.0": res(0.85),
    }
    v = dv.verdicts(r)
    assert v["H1"]["status"] == "SUPPORTED"
    assert v["H3"]["status"] == "NOT SUPPORTED"
    assert v["H2"]["status"].startswith("INDICATOR PRESENT")
    r["fill_replicate"] = res(0.75)  # 21% drop < 25%
    assert dv.verdicts(r)["H1"]["status"] == "NOT SUPPORTED"
    r["min_sigma_0.0"] = res(0.60)  # 37% drop
    assert dv.verdicts(r)["H3"]["status"] == "SUPPORTED"


def test_verdicts_not_applicable_when_no_artifact():
    v = dv.verdicts({"v1_baseline": res(0.17), "fill_replicate": res(0.1)})
    assert all(x["status"] == "NOT APPLICABLE" for x in v.values())


def test_cases_are_zipped_not_crossed(tmp_path):
    for n in ("a.png", "b.png"):
        (tmp_path / n).write_bytes(b"x")
    args = argparse.Namespace(
        images=[f"a={tmp_path / 'a.png'}", f"b={tmp_path / 'b.png'}"],
        modes=["zoom_out", "pan_right"],
        cases=None,
    )
    ids = [c["id"] for c in dv.build_cases(args, tmp_path)]
    assert ids == ["a__zoom_out", "b__pan_right"]
    args = argparse.Namespace(
        images=None, modes=None, cases=[f"a={tmp_path / 'a.png'}:tilt_up", f"{tmp_path / 'b.png'}:zoom_in"]
    )
    assert [c["id"] for c in dv.build_cases(args, tmp_path)] == ["a__tilt_up", "b__zoom_in"]
    args = argparse.Namespace(images=[f"a={tmp_path / 'a.png'}"], modes=["x", "y"], cases=None)
    with pytest.raises(SystemExit):
        dv.build_cases(args, tmp_path)
