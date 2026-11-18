"""Plot latent-space paired-command diagnostics from an audited flow checkpoint.

These are training-target diagnostics, not physical camera motion or generated
video adherence metrics. No model tensors are loaded.

    python scripts/plot_flow_learning.py --checkpoint-dir artifacts/flow_poc_all --output-png learning.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_checkpoint_history(checkpoint_dir: Path) -> tuple[dict, dict]:
    """Read the per-checkpoint diagnostic records from the checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    latest_path = checkpoint_dir / "latest.json"
    if not latest_path.is_file():
        raise FileNotFoundError(latest_path)
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    name = latest.get("checkpoint")
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or not name.endswith(".safetensors")
    ):
        raise ValueError("latest.json checkpoint filename is invalid")
    checkpoint = checkpoint_dir / name
    sidecar_path = checkpoint.with_suffix(".json")
    if not checkpoint.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("latest checkpoint or its sidecar is missing")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    actual_sha = _sha256(checkpoint)
    for label, metadata in (("latest.json", latest), ("checkpoint sidecar", sidecar)):
        if (
            metadata.get("checkpoint") != name
            or metadata.get("checkpoint_sha256") != actual_sha
        ):
            raise ValueError(f"{label} does not identify the current checkpoint bytes")
        if metadata.get("backbone_unchanged") is not True:
            raise ValueError(f"{label} lacks backbone_unchanged=true")
        if metadata.get("completed_updates") != latest.get("completed_updates"):
            raise ValueError(f"{label} completed update count differs")
    history = sidecar.get("history")
    if not isinstance(history, list) or len(history) != int(
        latest["completed_updates"]
    ):
        raise ValueError("Checkpoint sidecar history is absent or incomplete")
    return latest, sidecar


def _analyze(sidecar: dict, *, rolling_window: int = 10) -> dict:
    """Compute rolling statistics of the paired-command diagnostics over the checkpoint history."""
    if rolling_window < 1:
        raise ValueError("rolling_window must be positive")
    history = sidecar["history"]
    updates = []
    losses = []
    raw_pairs = []
    excluded = 0
    source_ids = set()
    declared_sources = 0
    for row in history:
        step = int(row.get("step", 0))
        if step < 1:
            raise ValueError("History contains a nonpositive step")
        updates.append(step)
        loss = row.get("loss")
        losses.append(
            float(loss) if loss is not None and math.isfinite(float(loss)) else None
        )
        declared_sources = max(declared_sources, int(row.get("sources_available", 0)))
        diagnostics = row.get("paired_command_diagnostics") or []
        for pair in diagnostics:
            modes = pair.get("modes", [])
            source_id = pair.get("source_id")
            if isinstance(source_id, str) and source_id:
                source_ids.add(source_id)
            desired = float(pair.get("desired_separation_rms", float("nan")))
            predicted = float(pair.get("prediction_separation_rms", float("nan")))
            projection = pair.get("separation_target_projection")
            if (
                not math.isfinite(desired)
                or desired <= 0
                or modes == ["static", "static"]
            ):
                excluded += 1
                continue
            if not math.isfinite(predicted):
                raise ValueError(f"Nonfinite prediction separation at update {step}")
            if projection is None or not math.isfinite(float(projection)):
                raise ValueError(
                    f"Missing finite separation projection at update {step}"
                )
            ratio = predicted / desired
            p = float(projection)
            cosine = p / ratio if ratio > 0 else None
            normalized_mse = ratio * ratio + 1.0 - 2.0 * p
            raw_pairs.append(
                {
                    "step": step,
                    "source_id": source_id,
                    "modes": modes,
                    "prediction_separation_rms": predicted,
                    "target_separation_rms": desired,
                    "amplitude_ratio": ratio,
                    "projection_coefficient": p,
                    "implied_cosine": cosine,
                    "normalized_command_difference_mse": normalized_mse,
                }
            )
    if not raw_pairs:
        raise ValueError("No nonzero-target moving pairs found in checkpoint history")
    if updates != list(range(1, len(updates) + 1)):
        raise ValueError("History steps are not contiguous from one")

    rolling = []
    for index, pair in enumerate(raw_pairs):
        start = max(0, index - rolling_window + 1)
        window = raw_pairs[start : index + 1]
        rolling.append(
            {
                "step": pair["step"],
                "pair_count": len(window),
                "mean_normalized_command_difference_mse": float(
                    np.mean([x["normalized_command_difference_mse"] for x in window])
                ),
                "mean_amplitude_ratio": float(
                    np.mean([x["amplitude_ratio"] for x in window])
                ),
                "mean_projection_coefficient": float(
                    np.mean([x["projection_coefficient"] for x in window])
                ),
            }
        )

    def summarize(rows: list[dict]) -> dict:
        """Mean separation / response statistics of a group of paired records."""
        return {
            "pair_count": len(rows),
            "mean_prediction_separation_rms": float(
                np.mean([x["prediction_separation_rms"] for x in rows])
            ),
            "mean_target_separation_rms": float(
                np.mean([x["target_separation_rms"] for x in rows])
            ),
            "mean_amplitude_ratio": float(
                np.mean([x["amplitude_ratio"] for x in rows])
            ),
            "mean_projection_coefficient": float(
                np.mean([x["projection_coefficient"] for x in rows])
            ),
            "mean_implied_cosine": float(
                np.mean(
                    [
                        x["implied_cosine"]
                        for x in rows
                        if x["implied_cosine"] is not None
                    ]
                )
            )
            if any(x["implied_cosine"] is not None for x in rows)
            else None,
            "mean_normalized_command_difference_mse": float(
                np.mean([x["normalized_command_difference_mse"] for x in rows])
            ),
        }

    n = min(10, len(raw_pairs))
    raw_loss = [
        {"step": step, "loss": loss}
        for step, loss in zip(updates, losses)
        if loss is not None
    ]
    return {
        "schema_version": 1,
        "status": "training_latent_diagnostics_only",
        "completed_updates": int(sidecar["completed_updates"]),
        "checkpoint": sidecar["checkpoint"],
        "checkpoint_sha256": sidecar["checkpoint_sha256"],
        "backbone_unchanged": sidecar["backbone_unchanged"],
        "source_groups_seen": len(source_ids),
        "source_groups_available": declared_sources,
        "history_pair_count": sum(
            len(row.get("paired_command_diagnostics") or []) for row in history
        ),
        "nonstatic_nonzero_target_pair_count": len(raw_pairs),
        "excluded_zero_target_pair_count": excluded,
        "rolling_window_nonstatic_pairs": rolling_window,
        "first_10_nonstatic_pair_summary": summarize(raw_pairs[:n]),
        "last_10_nonstatic_pair_summary": summarize(raw_pairs[-n:]),
        "definitions": {
            "amplitude_ratio": "prediction_separation_rms / target_separation_rms",
            "projection_coefficient": "dot(prediction difference, target difference) / squared target difference norm, as logged by trainer",
            "implied_cosine": "projection_coefficient / amplitude_ratio; derived only when predicted separation is nonzero",
            "normalized_command_difference_mse": "amplitude_ratio^2 + 1 - 2 * projection_coefficient; 1 is zero-prediction/no-condition baseline and 0 is exact latent target-difference match",
            "rolling": "Trailing arithmetic mean over up to ten nonstatic, nonzero-target pairs; static/static pairs are excluded",
            "raw_loss": "Per-update stochastic training loss is shown separately because sampled scenes and noise levels vary",
            "scope": "These are training latent diagnostics, not physical camera motion, generated-video adherence, or quality acceptance metrics",
        },
        "per_pair": raw_pairs,
        "rolling": rolling,
        "per_update_training_loss": raw_loss,
    }


def _plot(report: dict, output_png: Path) -> None:
    """Draw the learning curves."""
    per_pair = report["per_pair"]
    rolling = report["rolling"]
    x = np.asarray([row["step"] for row in rolling])
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.5, 10.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.05, 0.8]},
    )
    fig.suptitle(
        "Paired camera-control training diagnostics — latent targets",
        fontsize=15,
        weight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.95,
        "Training latent diagnostics only • not video/physical adherence or a success claim",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    ax = axes[0]
    y = np.asarray([row["mean_normalized_command_difference_mse"] for row in rolling])
    ax.plot(
        x,
        y,
        color="#2458A6",
        linewidth=2,
        label=f"Trailing {report['rolling_window_nonstatic_pairs']} moving pairs",
    )
    ax.axhline(
        1.0,
        color="#A66A00",
        linestyle="--",
        linewidth=1.2,
        label="No-condition baseline (1)",
    )
    ax.axhline(
        0.0,
        color="#28784B",
        linestyle=":",
        linewidth=1.4,
        label="Exact latent difference (0)",
    )
    ax.set_ylabel("Normalized command-difference MSE")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="best", ncol=3, fontsize=8)
    ax.set_title("Command separation error", loc="left", fontsize=11)

    ax = axes[1]
    ratio = np.asarray([row["mean_amplitude_ratio"] for row in rolling])
    projection = np.asarray([row["mean_projection_coefficient"] for row in rolling])
    ax.plot(
        x,
        ratio,
        color="#B3482D",
        linewidth=1.8,
        label="Prediction / target separation RMS",
    )
    ax.plot(
        x, projection, color="#7354A3", linewidth=1.8, label="Projection coefficient"
    )
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=1, label="Reference = 1")
    ax.set_ylabel("Amplitude ratio / coefficient")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="best", ncol=2, fontsize=8)
    ax.set_title("Scale and target alignment", loc="left", fontsize=11)

    ax = axes[2]
    loss_rows = report["per_update_training_loss"]
    if loss_rows:
        lx = [row["step"] for row in loss_rows]
        ly = [row["loss"] for row in loss_rows]
        ax.plot(lx, ly, color="#4C7774", linewidth=1.35)
    ax.set_ylabel("Raw update loss")
    ax.set_xlabel("Training update")
    ax.grid(True, alpha=0.22)
    ax.set_title(
        "Stochastic training loss (separate; scenes and noise vary)",
        loc="left",
        fontsize=11,
    )

    final = report["last_10_nonstatic_pair_summary"]
    note = (
        f"{report['source_groups_seen']}/{report['source_groups_available']} source groups seen • "
        f"{report['excluded_zero_target_pair_count']} zero-target pairs excluded • "
        f"last-10 ratio {final['mean_amplitude_ratio']:.3f}, projection {final['mean_projection_coefficient']:.3f}"
    )
    fig.text(0.5, 0.015, note, ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0.04, 0.04, 0.99, 0.91))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """CLI entry; writes the PNG and/or JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts/flow_poc_all",
    )
    parser.add_argument("--output-png", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--rolling-window", type=int, default=10)
    args = parser.parse_args()
    latest, sidecar = _read_checkpoint_history(args.checkpoint_dir)
    report = _analyze(sidecar, rolling_window=args.rolling_window)
    default_root = Path(__file__).resolve().parents[1] / "artifacts/flow_poc"
    output_png = args.output_png or (
        default_root / f"delta_learning_to{latest['completed_updates']}.png"
    )
    output_json = args.output_json or (
        default_root / f"delta_learning_to{latest['completed_updates']}.json"
    )
    output_png = Path(output_png).resolve()
    output_json = Path(output_json).resolve()
    if output_png.exists() or output_json.exists():
        raise FileExistsError("Refusing to overwrite an existing plot or report")
    _plot(report, output_png)
    report["plot_path"] = str(output_png)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "plot": str(output_png),
                "report": str(output_json),
                "updates": report["completed_updates"],
                "source_groups_seen": report["source_groups_seen"],
                "excluded_zero_target_pairs": report["excluded_zero_target_pair_count"],
                "first_10": report["first_10_nonstatic_pair_summary"],
                "last_10": report["last_10_nonstatic_pair_summary"],
                "scope": report["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
