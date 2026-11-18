"""Paired held-out denoising diagnostic; never a generated-motion acceptance test."""

from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from safetensors.torch import load_file
from .flow_cache import load_cached
from .flow_control import FlowControlConfig, FlowControlLoRA, build_image_condition
from .flow_geometry import camera_path
from .flow_train import _load_transformer, _input, token_timestep
from .runtime import gpu_lease, file_sha256, write_json
from .wan import hash_module


def _tensor_rms(value):
    return float(value.detach().float().square().mean().sqrt().item())


def _paired_prediction_diagnostics(predictions):
    """Reduce future-latent predictions to pairwise RMS scalars only."""
    correct = predictions["correct"].float()
    opposite = predictions["opposite"].float()
    disabled = predictions["disabled"].float()
    disabled_rms = _tensor_rms(disabled)
    correct_opposite = _tensor_rms(correct - opposite)
    correct_disabled = _tensor_rms(correct - disabled)
    return {
        "prediction_rms_future": {
            label: _tensor_rms(prediction) for label, prediction in predictions.items()
        },
        "prediction_delta_rms_future": {
            "correct_vs_opposite": correct_opposite,
            "correct_vs_disabled": correct_disabled,
        },
        "prediction_delta_rms_relative_to_disabled_rms": {
            "correct_vs_opposite": correct_opposite / disabled_rms
            if disabled_rms > 0
            else None,
            "correct_vs_disabled": correct_disabled / disabled_rms
            if disabled_rms > 0
            else None,
        },
        "prediction_delta_comparison_domain": "future latent frames only; BF16 predictions cast to FP32 for scalar reductions",
    }


def _adapter_parameter_summary(adapter):
    """Scalar summaries for guide projection and LoRA matrices."""
    summary = {
        "guide_projection_weight_rms": _tensor_rms(adapter.guide_projection.weight),
        "guide_projection_bias_rms": _tensor_rms(adapter.guide_projection.bias),
    }
    for factor in ("lora_A", "lora_B"):
        values = [
            _tensor_rms(getattr(wrapper, factor))
            for wrapper in adapter._wrappers.values()
        ]
        summary[f"{factor}_matrix_rms"] = {
            "count": len(values),
            "mean": float(sum(values) / len(values)) if values else None,
            "min": float(min(values)) if values else None,
            "max": float(max(values)) if values else None,
        }
    return summary


def _clean_motion_diagnostics(predictions, noisy, clean, static, sigma):
    """Measure motion-specific errors separately from whole-video velocity MSE.

    The static reference is the separately encoded, same-source static target.
    These are teacher-noised reconstruction diagnostics, not generated videos.
    At a shared noisy input, z0_hat = x_sigma - sigma * velocity_prediction.
    """
    target = clean.float()
    static = static.float()
    moving_delta = target[:, :, 1:] - static[:, :, 1:]
    energy = moving_delta.square().sum()
    target_dt = target[:, :, 1:] - target[:, :, :-1]
    reconstructed = {}
    output = {
        "target_motion_vs_static_rms": _tensor_rms(moving_delta),
        "target_temporal_difference_rms": _tensor_rms(target_dt),
        "per_condition": {},
        "interpretation": "Teacher-noised clean reconstruction; input already contains target information. Not motion acceptance or a standalone cause diagnosis.",
    }
    for label, prediction in predictions.items():
        future = noisy[:, :, 1:].float() - sigma * prediction.float()
        estimate = torch.cat((target[:, :, :1], future), dim=2)
        reconstructed[label] = future
        temporal_error = (estimate[:, :, 1:] - estimate[:, :, :-1]) - target_dt
        progress = (
            ((future - static[:, :, 1:]) * moving_delta).sum() / energy
            if energy > 0
            else None
        )
        output["per_condition"][label] = {
            "clean_reconstruction_mse_future": float(
                (future - target[:, :, 1:]).square().mean()
            ),
            "temporal_difference_mse": float(temporal_error.square().mean()),
            "motion_direction_projection_relative_to_static_target": float(progress)
            if progress is not None
            else None,
        }
    if energy > 0:
        output["correct_minus_opposite_motion_projection"] = float(
            (
                (reconstructed["correct"] - reconstructed["opposite"]) * moving_delta
            ).sum()
            / energy
        )
    else:
        output["correct_minus_opposite_motion_projection"] = None
    return output


def validate(root, checkpoint, index, output, source_limit=2):
    root, checkpoint, index, output = map(Path, (root, checkpoint, index, output))
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(checkpoint.with_suffix(".json").read_text())
    if file_sha256(checkpoint) != meta["checkpoint_sha256"] or not meta.get(
        "backbone_unchanged"
    ):
        raise ValueError("Checkpoint integrity/backbone audit missing")
    if file_sha256(index) != meta["identity"]["index_sha256"]:
        raise ValueError("Validation index changed since training")
    rows = [json.loads(x) for x in index.read_text().splitlines() if x.strip()]
    dev = [
        r
        for r in rows
        if r["split"] == "development" and r["mode"] in ("zoom_in", "zoom_out")
    ]
    sources = sorted({r["source_id"] for r in dev})[:source_limit]
    dev = [r for r in dev if r["source_id"] in sources]
    static_rows = {
        r["source_id"]: r
        for r in rows
        if r["split"] == "development" and r["mode"] == "static"
    }
    if not dev:
        raise ValueError("No development clips")
    if any(r["source_id"] not in static_rows for r in dev):
        raise ValueError("Motion diagnostics require a same-source static target")
    if {r["source_sha256"] for r in dev} & {
        r["source_sha256"] for r in rows if r["split"] == "train"
    }:
        raise ValueError("Development source leakage")
    results = []
    started = time.monotonic()
    with gpu_lease(root):
        model, versions = _load_transformer(output.with_suffix(".library.log"))
        original = tuple(model.parameters())
        before = hash_module(model, parameters=original)
        if before != meta["backbone_sha256_after"]:
            raise ValueError("Wrong frozen backbone")
        adapter = FlowControlLoRA(
            model, FlowControlConfig(**meta["architecture"]["config"])
        )
        adapter.load_adapter_state_dict(load_file(str(checkpoint), device="cpu"))
        adapter_parameters = _adapter_parameter_summary(adapter)
        try:
            for number, row in enumerate(dev):
                item = load_cached(row, root)
                clean, first, text, correct = _input(item)
                contract = item["contract"]
                del item
                static_item = load_cached(static_rows[row["source_id"]], root)
                static = static_item["latents"][None].to(
                    device="cuda", dtype=torch.bfloat16
                )
                if static.shape != clean.shape or not torch.equal(
                    static[:, :, :1], clean[:, :, :1]
                ):
                    raise ValueError("Static/moving reference target mismatch")
                del static_item
                opposite = "zoom_out" if row["mode"] == "zoom_in" else "zoom_in"
                wrong_path, _ = camera_path(opposite, contract)
                wrong = build_image_condition(
                    first,
                    wrong_path,
                    width=contract.width,
                    height=contract.height,
                    frames=contract.frames,
                )
                # sigma=1 removes future target information from the noisy input;
                # it directly probes conditioning at the start of generation.
                for level in (0.5, 0.8, 1.0):
                    sigma = torch.tensor([level], device="cuda")
                    generator = torch.Generator(device="cuda").manual_seed(711 + number)
                    noise = torch.randn(
                        clean.shape,
                        device="cuda",
                        dtype=clean.dtype,
                        generator=generator,
                    )
                    noisy = (1 - level) * clean + level * noise
                    noisy = torch.cat((first, noisy[:, :, 1:]), dim=2)
                    target = noise.float() - clean.float()
                    losses = {}
                    predictions = {}
                    for label, condition in (
                        ("correct", correct),
                        ("opposite", wrong),
                        ("disabled", None),
                    ):
                        adapter.set_condition(condition)
                        with (
                            torch.no_grad(),
                            torch.autocast("cuda", dtype=torch.bfloat16),
                        ):
                            if label == "disabled":
                                with adapter.disabled():
                                    prediction = model(
                                        hidden_states=noisy,
                                        timestep=token_timestep(clean, sigma),
                                        encoder_hidden_states=text,
                                        return_dict=False,
                                    )[0]
                            else:
                                prediction = model(
                                    hidden_states=noisy,
                                    timestep=token_timestep(clean, sigma),
                                    encoder_hidden_states=text,
                                    return_dict=False,
                                )[0]
                            future_prediction = prediction[:, :, 1:].detach()
                            loss = (
                                (future_prediction.float() - target[:, :, 1:])
                                .square()
                                .mean()
                            )
                        if not torch.isfinite(loss):
                            raise RuntimeError("Nonfinite validation loss")
                        losses[label] = float(loss)
                        predictions[label] = future_prediction
                        del prediction, future_prediction, loss
                    paired_prediction = _paired_prediction_diagnostics(predictions)
                    clean_motion = _clean_motion_diagnostics(
                        predictions, noisy, clean, static, level
                    )
                    results.append(
                        {
                            "source_id": row["source_id"],
                            "mode": row["mode"],
                            "sigma": level,
                            "losses": losses,
                            "correct_minus_opposite": losses["correct"]
                            - losses["opposite"],
                            "correct_minus_disabled": losses["correct"]
                            - losses["disabled"],
                            "paired_prediction_delta": paired_prediction,
                            "clean_motion_diagnostics": clean_motion,
                        }
                    )
                    del predictions, paired_prediction, clean_motion
                    print(json.dumps(results[-1]), flush=True)
                del clean, first, text, correct, wrong, noisy, target, noise, static
        finally:
            adapter.close()
        after = hash_module(model, parameters=original)
        if after != before:
            raise RuntimeError("Frozen backbone changed")
    record = {
        "status": "denoising_diagnostic_only_generated_motion_unproven",
        "results": results,
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "completed_updates": meta["completed_updates"],
        "source_groups": len(sources),
        "seconds": time.monotonic() - started,
        "backbone_unchanged": True,
        "adapter_parameter_summary": adapter_parameters,
        "diagnostic_scope": "Paired denoising/prediction deltas and scalar adapter norms only; not camera-motion acceptance.",
        "versions": versions,
    }
    write_json(output, record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("project-root", "checkpoint", "index", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-limit", type=int, default=2)
    args = parser.parse_args()
    validate(
        args.project_root, args.checkpoint, args.index, args.output, args.source_limit
    )
