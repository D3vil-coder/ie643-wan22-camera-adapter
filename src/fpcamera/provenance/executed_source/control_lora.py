"""Spatial camera conditioning with Control-LoRA on frozen Wan blocks.

Targets the Diffusers Wan layout verified in ``transformer_wan.py``:
``blocks[i].attn1.{to_q,to_k,to_v}``, ``blocks[i].ffn.net[0].proj`` and
``blocks[i].ffn.net[2]``. The frozen Linear modules stay registered inside
wrappers with the same Parameter objects; only low-rank deltas train.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ControlLoRAConfig:
    hidden: int = 3072
    rank: int = 32
    alpha: float = 32.0
    layers: int = 16
    ray_channels: int = 6
    temporal_kernel: int = 4
    min_sigma: float = 0.0


class ControlLoRALinear(nn.Module):
    """Frozen linear plus a zero-start low-rank delta."""

    def __init__(self, base: nn.Linear, rank: int = 32, alpha: float = 32.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Control-LoRA target must be nn.Linear, got {type(base).__name__}")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features,
                                               device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank,
                                               device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.enabled = True

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        result = self.base(x)
        if self.enabled:
            # Keep adapter parameters/optimizer state in FP32 while using the
            # same BF16 matmul policy as the frozen CUDA backbone.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                delta = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scaling
            result = result + delta.to(dtype=result.dtype)
        return result


class ControlLoRA(nn.Module):
    """Attach camera patch features and LoRA deltas to a frozen Wan transformer.

    ``set_condition`` receives ``(B,6,F,Htoken,Wtoken)`` rays at the backbone's
    spatial token resolution. A causal, left-replicated temporal Conv3d maps
    frames to latent-time patches. The first latent-time feature is then zeroed
    to preserve the image-conditioning frame. Wan's patch embedding output is
    modified only by a zero-initialized projection, so attachment is exact
    identity at initialization.
    """

    def __init__(self, transformer: nn.Module, config: ControlLoRAConfig | None = None):
        super().__init__()
        self.config = config or ControlLoRAConfig()
        if self.config.min_sigma != 0:
            raise ValueError('Control-LoRA camera conditioning must remain active across the full noise schedule')
        blocks = getattr(transformer, "blocks", None)
        patch_embedding = getattr(transformer, "patch_embedding", None)
        if blocks is None or patch_embedding is None:
            raise TypeError("Expected Diffusers Wan transformer with blocks and patch_embedding")
        if len(blocks) < self.config.layers:
            raise ValueError(f"Need at least {self.config.layers} Wan blocks; found {len(blocks)}")
        if not isinstance(patch_embedding, nn.Conv3d):
            raise TypeError("Wan patch_embedding must be Conv3d")
        if patch_embedding.out_channels != self.config.hidden:
            raise ValueError(f"Wan hidden width is {patch_embedding.out_channels}, expected {self.config.hidden}")
        if getattr(transformer, "_fpcamera_control_lora", None) is not None:
            raise RuntimeError("Control-LoRA is already attached to this transformer")
        for block in list(blocks)[:self.config.layers]:
            attention = getattr(block, "attn1", None)
            if getattr(attention, "fused_projections", False):
                unfuse = getattr(attention, "unfuse_projections", None)
                if not callable(unfuse):
                    raise TypeError("Cannot attach Control-LoRA: fused Wan attention cannot be unfused")
                unfuse()
        transformer.requires_grad_(False)
        transformer.eval()
        # Keep the external backbone out of this controller's state_dict: a
        # normal controller checkpoint should contain only camera/LoRA weights.
        object.__setattr__(self, "transformer", transformer)
        self.camera_projection = nn.Conv3d(
            self.config.ray_channels, self.config.hidden,
            kernel_size=(self.config.temporal_kernel, 1, 1),
            stride=(self.config.temporal_kernel, 1, 1), bias=True,
        ).to(device=patch_embedding.weight.device, dtype=torch.float32)
        # Zero-initialized camera projection with a learnable scale factor.
        # The scale starts at 1.0 (γ=0) so the first forward is identical to
        # the frozen backbone. γ grows during training to amplify the camera
        # signal above the noise floor of the frozen 3072-dim features.
        nn.init.zeros_(self.camera_projection.weight)
        nn.init.zeros_(self.camera_projection.bias)
        self._gamma = nn.Parameter(torch.tensor(0.0, device=patch_embedding.weight.device, dtype=torch.float32))
        self._rays: torch.Tensor | None = None
        self._active = True
        self._wrapped: list[ControlLoRALinear] = []
        self._target_wrappers: dict[str, ControlLoRALinear] = {}
        self._adapter_parameters = nn.ParameterDict()
        self._original_parameter_refs = tuple(transformer.parameters())
        self._install_lora()
        self._hook = patch_embedding.register_forward_hook(self._add_camera_features)
        object.__setattr__(transformer, "_fpcamera_control_lora", self)

    def _wrap(self, parent: nn.Module, name: str, path: str):
        module = getattr(parent, name, None)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected Wan Linear at {type(parent).__name__}.{name}; got {type(module).__name__}")
        wrapper = ControlLoRALinear(module, self.config.rank, self.config.alpha)
        setattr(parent, name, wrapper)
        self._wrapped.append(wrapper)
        self._target_wrappers[path] = wrapper
        index = len(self._target_wrappers) - 1
        self._adapter_parameters[f"a{index:03d}A"] = wrapper.lora_A
        self._adapter_parameters[f"a{index:03d}B"] = wrapper.lora_B

    def _install_lora(self):
        for index in range(self.config.layers):
            block = self.transformer.blocks[index]
            attention = getattr(block, "attn1", None)
            if attention is None or not all(hasattr(attention, n) for n in ("to_q", "to_k", "to_v")):
                raise TypeError(f"Wan block {index} lacks unfused attn1.to_q/to_k/to_v Linear modules")
            for name in ("to_q", "to_k", "to_v"):
                self._wrap(attention, name, f"blocks.{index}.attn1.{name}")
            ffn = getattr(block, "ffn", None)
            try:
                first = ffn.net[0].proj
                last = ffn.net[2]
            except (AttributeError, IndexError, TypeError) as exc:
                raise TypeError(f"Wan block {index} FFN layout is not net[0].proj/net[2]") from exc
            self._wrap(ffn.net[0], "proj", f"blocks.{index}.ffn.net.0.proj")
            ffn.net[2] = ControlLoRALinear(last, self.config.rank, self.config.alpha)
            self._wrapped.append(ffn.net[2])
            path = f"blocks.{index}.ffn.net.2"
            self._target_wrappers[path] = ffn.net[2]
            slot = len(self._target_wrappers) - 1
            self._adapter_parameters[f"a{slot:03d}A"] = ffn.net[2].lora_A
            self._adapter_parameters[f"a{slot:03d}B"] = ffn.net[2].lora_B

    def _add_camera_features(self, _module, _inputs, output):
        if not self._active or self._rays is None:
            return output
        if output.ndim != 5:
            raise ValueError(f"Expected Wan patch embedding B,C,T,H,W output, got {tuple(output.shape)}")
        rays = self._rays.to(device=output.device, dtype=output.dtype)
        expected = (output.shape[0], self.config.ray_channels, None, output.shape[3], output.shape[4])
        if rays.ndim != 5 or rays.shape[0] != expected[0] or rays.shape[1] != expected[1] or rays.shape[3:] != expected[3:]:
            raise ValueError(f"Camera rays must be (B,6,F,{output.shape[3]},{output.shape[4]}), got {tuple(rays.shape)}")
        # Replicate the first RGB-frame ray three times on the left. For 97
        # frames, stride 4 produces 25 temporal feature slices, matching TI2V
        # latent time (25). Reject any other mismatch rather than interpolate.
        rays = F.pad(rays, (0, 0, 0, 0, self.config.temporal_kernel - 1, 0), mode="replicate")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=output.is_cuda):
            features = self.camera_projection(rays.float())
        features = features.to(dtype=output.dtype)
        if features.shape[2:] != output.shape[2:]:
            raise ValueError(f"Causal camera grid {tuple(features.shape[2:])} != Wan patch grid {tuple(output.shape[2:])}")
        features = features.clone()
        features[:, :, 0] = 0
        # Scale the camera features so the signal doesn't drown in frozen 3072-dim features.
        gamma = torch.sigmoid(self._gamma) * 10.0  # Maps γ ∈ (-∞,∞) → scale ∈ (0, 10)
        return output + gamma * features

    def set_condition(self, rays: torch.Tensor | None):
        """Set rays or clear conditioning with ``None``."""
        self._rays = rays

    @property
    def rays(self):
        """Current camera map, for integration with existing guidance wrappers."""
        return self._rays

    @contextlib.contextmanager
    def disabled(self):
        """Disable both pose injection and all LoRA deltas for a frozen baseline."""
        previous = self._active
        states = [module.enabled for module in self._wrapped]
        self._active = False
        for module in self._wrapped:
            module.enabled = False
        try:
            yield self
        finally:
            self._active = previous
            for module, state in zip(self._wrapped, states):
                module.enabled = state

    def architecture(self) -> dict:
        return {
            "kind": "wan_control_lora",
            "config": asdict(self.config),
            "target_paths": [
                f"blocks.{i}.attn1.{name}" for i in range(self.config.layers)
                for name in ("to_q", "to_k", "to_v")
            ] + [
                path for i in range(self.config.layers)
                for path in (f"blocks.{i}.ffn.net.0.proj", f"blocks.{i}.ffn.net.2")
            ],
            "camera_feature_path": "zero_camera_projection_6_to_hidden(causal_temporal_stride4)->patch_embedding_output",
        }

    def original_parameter_refs_unchanged(self) -> bool:
        """Check the frozen Wan Parameter objects remain the ones captured at attach."""
        current_ids = {id(p) for p in self.transformer.parameters()}
        return all(id(p) in current_ids for p in self._original_parameter_refs)

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def adapter_state_dict(self):
        """Return only adapter tensors, without frozen backbone weights."""
        state = {"camera_projection.weight": self.camera_projection.weight.detach().clone(),
                 "camera_projection.bias": self.camera_projection.bias.detach().clone(),
                 "camera_gamma": self._gamma.detach().clone()}
        for path, wrapper in self._target_wrappers.items():
            state[f"{path}.lora_A"] = wrapper.lora_A.detach().clone()
            state[f"{path}.lora_B"] = wrapper.lora_B.detach().clone()
        return state

    def load_adapter_state_dict(self, state, strict: bool = True):
        expected = self.adapter_state_dict()
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        if strict and (missing or unexpected):
            # Allow backward compat: gamma might not exist in old checkpoints
            expected_no_gamma = {k: v for k, v in expected.items() if k != "camera_gamma"}
            actual_no_gamma = {k: v for k, v in state.items() if k != "camera_gamma"}
            if strict and (set(expected_no_gamma) != set(actual_no_gamma)):
                raise RuntimeError(f"Control-LoRA state mismatch: missing={missing}, unexpected={unexpected}")
            missing = [k for k in missing if k != "camera_gamma"]
        with torch.no_grad():
            for key in set(expected).intersection(state):
                if key == "camera_projection.weight":
                    dst = self.camera_projection.weight
                elif key == "camera_projection.bias":
                    dst = self.camera_projection.bias
                elif key == "camera_gamma":
                    dst = self._gamma
                else:
                    path, name = key.rsplit(".lora_", 1)
                    dst = getattr(self._target_wrappers[path], f"lora_{name}")
                src = state[key]
                if dst.shape != src.shape:
                    raise RuntimeError(f"Shape mismatch for {key}: {tuple(src.shape)} != {tuple(dst.shape)}")
                dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
        return {"missing_keys": missing, "unexpected_keys": unexpected}

    def close(self):
        self._hook.remove()
        for path, wrapper in self._target_wrappers.items():
            parent = self.transformer
            parts = path.split(".")
            for part in parts[:-1]:
                parent = parent[int(part)] if isinstance(parent, (nn.ModuleList, nn.Sequential)) else getattr(parent, part)
            name = parts[-1]
            if name == "proj" and isinstance(parent, nn.Module):
                setattr(parent, name, wrapper.base)
            elif isinstance(parent, (nn.ModuleList, nn.Sequential)):
                parent[int(name)] = wrapper.base
            else:
                setattr(parent, name, wrapper.base)
        if getattr(self.transformer, "_fpcamera_control_lora", None) is self:
            delattr(self.transformer, "_fpcamera_control_lora")


def attach(transformer: nn.Module, config: ControlLoRAConfig | None = None) -> ControlLoRA:
    """Convenience constructor; return the controller for checkpointing/integration."""
    return ControlLoRA(transformer, config)
