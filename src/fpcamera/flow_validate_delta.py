"""Validate teacher-relative camera edits against one shared frozen teacher.

This paired denoising diagnostic is objective-specific and is not an RGB video
quality or camera-control acceptance test.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import uuid

import torch
from safetensors.torch import load_file

from .flow_cache import NORMALIZATION, load_cached
from .flow_control import FlowControlConfig, FlowControlLoRA
from .flow_train_delta import (_input, _load_transformer, camera_velocity_target,
                               token_timestep)
from .runtime import file_sha256, gpu_lease
from .wan import MODEL, REVISION, hash_module


SIGMAS = (0.7, 0.9, 1.0)
MODES = ("zoom_in", "zoom_out")
NATIVE_CONTRACT = {"width": 1280, "height": 704, "frames": 121, "fps": 24}


def _mse(left, right) -> float:
    return float((left.float() - right.float()).square().mean().item())


def _rms(value) -> float:
    return float(value.float().square().mean().sqrt().item())


def _pair_separation(predicted_plus, predicted_minus, delta_plus, delta_minus) -> dict:
    predicted = predicted_plus.float() - predicted_minus.float()
    desired = delta_plus.float() - delta_minus.float()
    dot = (predicted * desired).sum()
    predicted_norm = predicted.square().sum().sqrt()
    desired_norm = desired.square().sum().sqrt()
    energy = desired.square().sum()
    return {
        "prediction_separation_rms_future": _rms(predicted),
        "desired_separation_rms_future": _rms(desired),
        "projection_coefficient": float(dot / energy) if energy > 0 else None,
        "cosine_similarity": float(dot / (predicted_norm * desired_norm))
        if predicted_norm > 0 and desired_norm > 0 else None,
        "future_latent_only": True,
    }


def _select_sources(rows, source_limit: int = 2) -> tuple[list[str], dict[str, dict[str, dict]]]:
    development = [row for row in rows if row.get("split") == "development"]
    grouped: dict[str, dict[str, dict]] = {}
    for row in development:
        mode = row.get("mode")
        if mode in ("static", *MODES):
            by_mode = grouped.setdefault(row["source_id"], {})
            if mode in by_mode:
                raise ValueError(f"Duplicate development row for {row['source_id']}/{mode}")
            by_mode[mode] = row
    complete = [source for source, modes in sorted(grouped.items())
                if all(mode in modes for mode in ("static", *MODES))]
    if len(complete) < source_limit:
        raise ValueError(f"Need {source_limit} development sources with static/zoom_in/zoom_out; found {len(complete)}")
    sources = complete[:source_limit]
    return sources, {source: grouped[source] for source in sources}


def _write_new_json(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)  # Atomic create-if-absent; never replace an existing report.
    finally:
        temp.unlink(missing_ok=True)


def validate(root: Path, checkpoint: Path, index: Path, output: Path) -> dict:
    root, checkpoint, index, output = map(Path, (root, checkpoint, index, output))
    if output.exists():
        raise FileExistsError(output)
    meta = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    checkpoint_sha = file_sha256(checkpoint)
    if checkpoint_sha != meta.get("checkpoint_sha256"):
        raise ValueError("Delta checkpoint SHA256 does not match its metadata")
    if int(meta.get("completed_updates", -1)) < 1 or meta.get("backbone_unchanged") is not True:
        raise ValueError("Delta checkpoint lacks a completed-update/backbone audit")
    identity = meta.get("identity", {})
    if identity.get("objective") != "teacher_relative_camera_velocity_on_shared_static_noisy_input":
        raise ValueError("Checkpoint objective is not the teacher-relative camera-velocity recipe")
    if identity.get("model") != MODEL or identity.get("revision") != REVISION:
        raise ValueError("Delta checkpoint was trained against a different Wan model revision")
    index_sha = file_sha256(index)
    if index_sha != identity.get("index_sha256"):
        raise ValueError("Validation index differs from the checkpoint training index")

    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_hashes = {row["source_sha256"] for row in rows if row.get("split") == "train"}
    sources, paired = _select_sources(rows, source_limit=2)
    if any(paired[source]["static"]["source_sha256"] in train_hashes for source in sources):
        raise ValueError("Development sources overlap training source hashes")
    for source, mode_rows in paired.items():
        if len({mode_rows[mode].get("caption") for mode in ("static", *MODES)}) != 1:
            raise ValueError(f"Paired validation captions differ for source {source}")
        if len({mode_rows[mode]["source_sha256"] for mode in ("static", *MODES)}) != 1:
            raise ValueError(f"Paired validation source hashes differ for source {source}")
        for mode in ("static", *MODES):
            contract = {key: mode_rows[mode]["video_contract"].get(key)
                        for key in NATIVE_CONTRACT}
            if contract != NATIVE_CONTRACT:
                raise ValueError(f"Non-native cache contract for {source}/{mode}: {contract}")
            if mode_rows[mode].get("normalization") != NORMALIZATION:
                raise ValueError(f"Unexpected latent normalization for {source}/{mode}")

    started = time.monotonic()
    results = []
    adapter_parameters = None
    versions = None
    forward_counts = {"teacher": 0, "student_static": 0, "student_zoom_in": 0,
                      "student_zoom_out": 0}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    with gpu_lease(root):
        model, versions = _load_transformer(output.with_suffix(".library.log"))
        model.eval()
        original = tuple(model.parameters())
        before = hash_module(model, parameters=original)
        if before != meta.get("backbone_sha256_after"):
            raise ValueError("Loaded frozen Wan backbone differs from delta checkpoint audit")
        adapter = FlowControlLoRA(model, FlowControlConfig(**meta["architecture"]["config"]))
        adapter.load_adapter_state_dict(load_file(str(checkpoint), device="cpu"))
        adapter_parameters = sum(parameter.numel() for parameter in adapter.parameters()
                                 if parameter.requires_grad)
        try:
            for source_index, source in enumerate(sources):
                static_item = load_cached(paired[source]["static"], root)
                static, first, text, static_condition = _input(static_item)
                contract = static_item["contract"]
                if (static.shape[2] != 31 or first.shape[2] != 1 or
                        not torch.equal(static[:, :, :1], first)):
                    raise ValueError("Expected native 121-frame latent and single-frame I2V anchor")
                moving = {}
                for mode in MODES:
                    moving_item = load_cached(paired[source][mode], root)
                    clean, moving_first, moving_text, condition = _input(moving_item)
                    if clean.shape != static.shape or not torch.equal(moving_first, first):
                        raise ValueError(f"Static/moving latent shapes or first-image anchors differ for {source}/{mode}")
                    if moving_item["contract"].record() != contract.record():
                        raise ValueError(f"Static/moving video contracts differ for {source}/{mode}")
                    if not torch.equal(moving_text, text):
                        raise ValueError(f"Static/moving text conditioning differs for {source}/{mode}")
                    moving[mode] = {"clean": clean, "condition": condition}
                    del moving_item, moving_first, moving_text

                for sigma_value in SIGMAS:
                    sigma = torch.tensor([sigma_value], device="cuda", dtype=torch.float32)
                    generator = torch.Generator(device="cuda").manual_seed(
                        711 + 31 * source_index + SIGMAS.index(sigma_value))
                    noise = torch.randn(static.shape, device="cuda", dtype=static.dtype, generator=generator)
                    noisy = (1 - sigma[:, None, None, None, None]) * static + sigma[:, None, None, None, None] * noise
                    noisy = torch.cat((first, noisy[:, :, 1:]), dim=2)
                    timesteps = token_timestep(static, sigma)
                    if torch.count_nonzero(timesteps.reshape(1, 31, 22, 40)[:, 0]).item() != 0:
                        raise RuntimeError("First Wan latent plane is not anchored at timestep zero")

                    adapter.set_condition(static_condition)
                    with torch.no_grad(), adapter.disabled(), torch.autocast("cuda", dtype=torch.bfloat16):
                        teacher = model(hidden_states=noisy, timestep=timesteps,
                            encoder_hidden_states=text, return_dict=False)[0][:, :, 1:].detach().float()
                    forward_counts["teacher"] += 1

                    predictions = {}
                    for mode, condition in (("static", static_condition),
                                            ("zoom_in", moving["zoom_in"]["condition"]),
                                            ("zoom_out", moving["zoom_out"]["condition"])):
                        adapter.set_condition(condition)
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            prediction = model(hidden_states=noisy, timestep=timesteps,
                                encoder_hidden_states=text, return_dict=False)[0][:, :, 1:].detach().float()
                        predictions[mode] = prediction
                        forward_counts[f"student_{mode}"] += 1

                    targets = {}
                    deltas = {}
                    for mode in MODES:
                        targets[mode], deltas[mode] = camera_velocity_target(
                            teacher, static[:, :, 1:], moving[mode]["clean"][:, :, 1:], sigma)
                    command_metrics = {}
                    for mode, opposite in (("zoom_in", "zoom_out"), ("zoom_out", "zoom_in")):
                        target = targets[mode]
                        # The inference anchor cancels the static-conditioned student,
                        # rather than adding the student's common appearance correction.
                        anchored_edit = predictions[mode] - predictions["static"]
                        desired_edit = deltas[mode]
                        dot = (anchored_edit * desired_edit).sum()
                        predicted_energy = anchored_edit.square().sum()
                        desired_energy = desired_edit.square().sum()
                        best_gain = max(0., float(dot / predicted_energy)) if predicted_energy > 0 else 0.
                        command_metrics[mode] = {
                            "correct_mse": _mse(predictions[mode], target),
                            "reversed_mse": _mse(predictions[opposite], target),
                            "no_edit_teacher_mse": _mse(teacher, target),
                            "teacher_relative_target_rms": _rms(target),
                            "requested_edit_delta_rms": _rms(deltas[mode]),
                            "anchored_edit_mse": _mse(anchored_edit, desired_edit),
                            "anchored_edit_rms": _rms(anchored_edit),
                            "anchored_edit_target_projection": float(dot / desired_energy) if desired_energy > 0 else None,
                            "anchored_edit_target_cosine": float(dot / (predicted_energy * desired_energy).sqrt())
                            if predicted_energy > 0 and desired_energy > 0 else None,
                            "diagnostic_best_nonnegative_scalar_gain": best_gain,
                            "diagnostic_best_scalar_mse": _mse(best_gain * anchored_edit, desired_edit),
                            "scalar_gain_scope": "latent diagnostic only; not an inference setting or RGB acceptance result",
                        }
                    results.append({
                        "source_id": source,
                        "source_sha256": paired[source]["static"]["source_sha256"],
                        "sigma": sigma_value,
                        "static_identity_mse": _mse(predictions["static"], teacher),
                        "commands": command_metrics,
                        "paired_prediction_separation": _pair_separation(
                            predictions["zoom_in"], predictions["zoom_out"],
                            deltas["zoom_in"], deltas["zoom_out"]),
                        "noise_seed": 711 + 31 * source_index + SIGMAS.index(sigma_value),
                        "noisy_input": "shared static future noise; first latent plane replaced with first-image anchor",
                    })
                    del noise, noisy, timesteps, teacher, predictions, targets, deltas, sigma
                for item in moving.values():
                    del item["clean"], item["condition"]
                del static_item, static, first, text, static_condition, moving, contract
            after = hash_module(model, parameters=original)
            if after != before:
                raise RuntimeError("Frozen Wan weights changed during delta validation")
        finally:
            adapter.close()

    if forward_counts != {"teacher": 6, "student_static": 6,
                          "student_zoom_in": 6, "student_zoom_out": 6}:
        raise RuntimeError(f"Expected exactly 24 forwards for 2 sources x 3 sigmas x 4 roles; got {forward_counts}")
    record = {
        "status": "teacher_relative_denoising_diagnostic_only_not_rgb_acceptance",
        "objective": identity["objective"],
        "checkpoint_sha256": checkpoint_sha,
        "completed_updates": meta["completed_updates"],
        "checkpoint_backbone_sha256_before": meta.get("backbone_sha256_before"),
        "checkpoint_backbone_sha256_after": meta.get("backbone_sha256_after"),
        "backbone_sha256_validation_before": before,
        "backbone_sha256_validation_after": after,
        "backbone_unchanged": True,
        "model": MODEL, "revision": REVISION, "versions": versions,
        "index_path": str(index), "index_sha256": index_sha,
        "development_sources": sources,
        "source_hashes": {source: paired[source]["static"]["source_sha256"] for source in sources},
        "cache_rows": {source: {mode: {"record_id": paired[source][mode]["record_id"],
                                         "cache_sha256": paired[source][mode]["sha256"]}
                                  for mode in ("static", *MODES)} for source in sources},
        "sigma_levels": list(SIGMAS),
        "forward_counts": forward_counts,
        "expected_forward_count": 24,
        "adapter_trainable_parameters": adapter_parameters,
        "implementation_sha256": {name:file_sha256(Path(__file__).with_name(name))
            for name in ('flow_validate_delta.py','flow_train_delta.py','flow_control.py','flow_geometry.py','wan.py')},
        "results": results,
        "diagnostic_scope": "Four student/teacher predictions on shared static noisy input per source and sigma; latent objective diagnosis only, not RGB preservation or generated camera-motion acceptance.",
        "seconds": time.monotonic() - started,
    }
    _write_new_json(output, record)
    record["output"] = str(output)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = validate(args.project_root, args.checkpoint, args.index, args.output)
    print(json.dumps({"status": record["status"], "completed_updates": record["completed_updates"],
                      "development_sources": record["development_sources"],
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
