"""The v2 adapter: a guide projection plus rank-16 LoRA on the frozen Wan TI2V-5B transformer.

Pipeline position: this is the model component trained by ``flow_train_v2`` and loaded by
``flow_infer``.  ``build_image_condition`` turns *one* first-image latent and the requested
camera homographies into a 51-channel "guide" (48 warped image-feature channels + 1 validity
channel (validity = 1 - R) + 2 displacement channels).  ``FlowControlLoRA`` attaches to a transformer
and (a) adds ``guide_projection(guide)`` to the patch-embedding output (token features) and
(b) wraps 16 blocks' attn1 q/k/v and FFN linears with zero-start LoRA deltas
(``control_lora.ControlLoRALinear``).  Everything is zero-initialised, so attaching the adapter
is an exact identity on the frozen model (checked by the trainer at every launch).

The condition is computed only from one image latent and supplied homographies; no target-video
tensor is accepted by the conditioning API, so inference cannot leak future frames.

Main entry points: ``build_image_condition``, ``FlowControlConfig``, ``FlowControlLoRA``
(``set_condition``, ``disabled``, ``adapter_state_dict`` / ``load_adapter_state_dict``, ``close``).
No CLI; exercised by ``tests/test_condition_fill.py`` and the trainer / inference tests.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .control_lora import ControlLoRALinear


@dataclass(frozen=True)
class FlowControlConfig:
    """Adapter architecture: transformer width, LoRA rank/alpha (alpha == rank gives scale 1), number of
    leading blocks adapted (16 of 30), and the pinned Wan VAE strides (temporal 4, spatial 16) that the guide
    must match.
    """

    hidden: int = 3072
    rank: int = 16
    alpha: float = 16.0
    layers: int = 16
    image_channels: int = 48
    vae_temporal_stride: int = 4
    vae_spatial_stride: int = 16


FILL_MODES = ("zeros", "replicate")


def build_image_condition(
    image_latent, homographies, *, width, height, frames, fill_mode="zeros"
):
    """48 guide +1 validity +2 forward-displacement-at-destination channels.

    Feature sampling uses normalized endpoint coordinates / align_corners=True.
    This is a motion prior; a warped latent is not assumed equal to VAE(target).

    ``fill_mode`` controls the 48 guide channels where the validity channel is 0
    (revealed regions that fall outside the first image):
      - "zeros" (v1, default): zero padding and guide * valid, i.e. latent 0.
      - "replicate": ``padding_mode="border"`` -- invalid positions carry the
        nearest edge feature; the guide is NOT multiplied by valid. The validity
        and displacement channels are identical in both modes.
    """
    if fill_mode not in FILL_MODES:
        raise ValueError(f"fill_mode must be one of {FILL_MODES}, got {fill_mode!r}")
    if image_latent.ndim != 5 or image_latent.shape[1:3] != (48, 1):
        raise ValueError("Expected first-image latent [B,48,1,H/16,W/16]")
    if (frames - 1) % 4 or width % 32 or height % 32:
        raise ValueError("Invalid Wan TI2V output contract")
    b, c, _, lh, lw = image_latent.shape
    if (height, width) != (lh * 16, lw * 16):
        raise ValueError("Image latent spatial shape and canvas disagree")
    device = image_latent.device
    matrices = torch.as_tensor(homographies, device=device, dtype=torch.float32)
    if matrices.ndim == 3:
        matrices = matrices[None]
    if matrices.shape != (b, frames, 3, 3) or not torch.isfinite(matrices).all():
        raise ValueError("Expected finite homographies [B,RGB_frames,3,3]")
    if not torch.allclose(
        matrices[:, 0], torch.eye(3, device=device)[None].expand(b, -1, -1), atol=1e-5
    ):
        raise ValueError("The first camera mapping must be identity")
    # Wan's VAE compresses time by 4: latent frame t corresponds to RGB frame 4t, so keep every 4th homography
    # (frame 0 is the first-image plane).
    matrices = matrices[:, ::4]
    t = matrices.shape[1]
    yy, xx = torch.meshgrid(
        torch.linspace(0, height - 1, lh, device=device),
        torch.linspace(0, width - 1, lw, device=device),
        indexing="ij",
    )
    q = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
    # inverse warp: for each destination latent cell q, find where it came from in the FIRST view by applying
    # H_t^-1 (H maps first view -> frame t).  p are those source coordinates (pixels of the first image).
    mapped = torch.einsum("btij,hwj->bthwi", torch.linalg.inv(matrices), q)
    positive = mapped[..., 2] > 1e-6
    p = mapped[..., :2] / mapped[..., 2:].clamp_min(1e-6)
    denominator = torch.tensor([width - 1, height - 1], device=device)
    pn = p / denominator
    # validity = 1 - R: the source point lies inside the first image (and in front of the camera).  Cells where it is
    # 0 are *revealed* regions -- content that does not exist in the conditioning image and must be generated.
    valid = positive & (pn >= 0).all(dim=-1) & (pn <= 1).all(dim=-1)
    grid = (2 * pn - 1).reshape(b * t, lh, lw, 2)
    reference = image_latent[:, :, 0].float()[:, None].expand(-1, t, -1, -1, -1)
    guide = F.grid_sample(
        reference.reshape(b * t, c, lh, lw),
        grid,
        mode="bilinear",
        # replicate fill: outside the first image 'border' padding copies the nearest edge feature instead of latent 0
        # (which decodes to mid-gray and let v1 'output the guide' for holes).  The validity channel still marks these
        # cells, so the network knows the feature is extrapolated.
        padding_mode="zeros" if fill_mode == "zeros" else "border",
        align_corners=True,
    )
    guide = guide.reshape(b, t, c, lh, lw).permute(0, 2, 1, 3, 4)
    if fill_mode == "zeros":
        guide = guide * valid[:, None]
    # 2 extra channels: forward displacement at the destination (destination minus source, normalised by the
    # image size) -- a direct motion cue the guide features alone do not provide.
    displacement = ((q[..., :2][None, None] - p) / denominator).permute(0, 4, 1, 2, 3)
    # Identity has a closed-form exact answer. Avoid resampling roundoff in the
    # first-frame guide and in genuine static-camera training examples.
    identity = (matrices - torch.eye(3, device=device)).abs().amax(dim=(-1, -2)) < 1e-7
    guide = torch.where(
        identity[:, None, :, None, None],
        image_latent.float().expand(-1, -1, t, -1, -1),
        guide,
    )
    valid = valid | identity[:, :, None, None]
    displacement = torch.where(
        identity[:, None, :, None, None], torch.zeros_like(displacement), displacement
    )
    result = torch.cat((guide, valid[:, None].float(), displacement), dim=1)
    if (
        result.shape != (b, 51, (frames - 1) // 4 + 1, lh, lw)
        or not torch.isfinite(result).all()
    ):
        raise RuntimeError("Invalid camera-condition result")
    return result


class FlowControlLoRA(nn.Module):
    """Camera adapter attached to a frozen Wan transformer.

    Trainable state: ``guide_projection`` (Conv3d 51 -> 3072, kernel/stride (1,2,2), same token grid as Wan's
    patch embedding) and the LoRA A/B matrices on attn1 q/k/v + both FFN linears of the first ``layers`` blocks.
    The guide is injected by a forward hook on ``patch_embedding``; the first latent plane (the clean conditioning
    image) is never modified.  The frozen parameters stay registered and untouched
    (``_original_parameter_refs``).  Always call ``close()`` to restore the transformer.
    """

    def __init__(self, transformer, config: FlowControlConfig | None = None):
        super().__init__()
        self.config = config or FlowControlConfig()
        if (
            self.config.image_channels,
            self.config.vae_temporal_stride,
            self.config.vae_spatial_stride,
        ) != (48, 4, 16):
            raise ValueError("Only the pinned Wan TI2V VAE is supported")
        if (
            getattr(transformer, "_fpcamera_control_lora", None) is not None
            or getattr(transformer, "_flow_control_lora", None) is not None
        ):
            raise ValueError("A controller is already attached")
        patch = transformer.patch_embedding
        if (
            not isinstance(patch, nn.Conv3d)
            or patch.out_channels != self.config.hidden
            or tuple(patch.kernel_size) != (1, 2, 2)
        ):
            raise ValueError("Unexpected Wan patch embedding")
        if (
            not 1 <= self.config.layers <= len(transformer.blocks)
            or self.config.rank <= 0
        ):
            raise ValueError("Invalid LoRA layers/rank")
        transformer.requires_grad_(False)
        object.__setattr__(self, "transformer", transformer)
        self._original_parameter_refs = tuple(transformer.parameters())
        self.guide_projection = nn.Conv3d(
            51, self.config.hidden, (1, 2, 2), stride=(1, 2, 2)
        ).to(device=patch.weight.device, dtype=torch.float32)
        # zero init => attaching the adapter is an exact identity on the frozen model (LoRA-B is zero-init as well).
        nn.init.zeros_(self.guide_projection.weight)
        nn.init.zeros_(self.guide_projection.bias)
        self.lora_weights = nn.ParameterDict()
        self._wrappers = {}
        self._condition = None
        self.enabled = True
        for i, block in enumerate(transformer.blocks[: self.config.layers]):
            attn = block.attn1
            if getattr(attn, "fused_projections", False):
                attn.unfuse_projections()
            targets = [
                (attn, name, f"blocks.{i}.attn1.{name}")
                for name in ("to_q", "to_k", "to_v")
            ]
            targets += [
                (block.ffn.net[0], "proj", f"blocks.{i}.ffn.net.0.proj"),
                (block.ffn.net, "2", f"blocks.{i}.ffn.net.2"),
            ]
            for parent, name, path in targets:
                base = parent[int(name)] if name.isdigit() else getattr(parent, name)
                wrapper = ControlLoRALinear(base, self.config.rank, self.config.alpha)
                if name.isdigit():
                    parent[int(name)] = wrapper
                else:
                    setattr(parent, name, wrapper)
                n = len(self._wrappers)
                self.lora_weights[f"layer{n:03d}A"] = wrapper.lora_A
                self.lora_weights[f"layer{n:03d}B"] = wrapper.lora_B
                self._wrappers[path] = wrapper
        self._hook = patch.register_forward_hook(self._inject)
        object.__setattr__(transformer, "_flow_control_lora", self)

    def set_condition(self, condition):
        """Set the 51-channel guide ``[B,51,T,H,W]`` used by subsequent forwards (validated finite).  Must be called
        before every forward unless the adapter is disabled; the trainer and the anchored sampler call it to switch
        between the requested and the static guide.
        """
        if condition is not None and (
            condition.ndim != 5
            or condition.shape[1] != 51
            or not torch.isfinite(condition).all()
        ):
            raise ValueError("Expected finite [B,51,Tlatent,Hlatent,Wlatent] guide")
        self._condition = condition

    def _inject(self, module, args, output):
        if not self.enabled:
            return output
        if self._condition is None:
            raise ValueError("Set a flow condition or explicitly disable the adapter")
        condition = self._condition.to(device=output.device, dtype=torch.float32)
        if condition.shape[0] == 1 and output.shape[0] > 1:
            condition = condition.expand(output.shape[0], -1, -1, -1, -1)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=output.is_cuda
        ):
            features = self.guide_projection(condition)
        if features.shape != output.shape:
            raise ValueError(
                f"Condition/backbone token grids disagree: {tuple(features.shape)} versus {tuple(output.shape)}"
            )
        # zero the first latent plane's guide features: the conditioning image plane must stay untouched (identity
        # anchor), the guide only acts on the 30 future planes.
        features = torch.cat(
            (torch.zeros_like(features[:, :, :1]), features[:, :, 1:]), dim=2
        )
        return output + features.to(output.dtype)

    @contextmanager
    def disabled(self):
        """Context manager that turns the whole adapter off (guide hook + every LoRA wrapper), restoring the
        previous states on exit.  Inside it the model is exactly the frozen Wan, i.e. the *teacher*.
        """
        before = self.enabled
        states = [w.enabled for w in self._wrappers.values()]
        self.enabled = False
        for wrapper in self._wrappers.values():
            wrapper.enabled = False
        try:
            yield self
        finally:
            self.enabled = before
            for wrapper, state in zip(self._wrappers.values(), states):
                wrapper.enabled = state

    def architecture(self):
        """Serialisable architecture record (config, condition layout, LoRA target paths) stored in checkpoints."""
        return {
            "kind": "wan_flow_control_lora",
            "config": asdict(self.config),
            "condition": "first_image_warp48_validity1_forward_displacement_at_destination2",
            "target_paths": list(self._wrappers),
        }

    def adapter_state_dict(self):
        """Detached copy of all adapter tensors, keyed ``guide_projection.{weight,bias}`` and
        ``<path>.lora_{A,B}`` (the layout of the published safetensors, 14.26M parameters).
        """
        state = {
            "guide_projection.weight": self.guide_projection.weight.detach().clone(),
            "guide_projection.bias": self.guide_projection.bias.detach().clone(),
        }
        for path, wrapper in self._wrappers.items():
            state[f"{path}.lora_A"] = wrapper.lora_A.detach().clone()
            state[f"{path}.lora_B"] = wrapper.lora_B.detach().clone()
        return state

    def load_adapter_state_dict(self, state):
        """Load a state dict with strict key set, shape and finiteness checks (raises ``ValueError``)."""
        expected = self.adapter_state_dict()
        if set(state) != set(expected):
            raise ValueError("Flow adapter state keys do not match the architecture")
        with torch.no_grad():
            for key, value in state.items():
                if (
                    not torch.isfinite(value).all()
                    or value.shape != expected[key].shape
                ):
                    raise ValueError(f"Invalid checkpoint tensor: {key}")
                if key.startswith("guide_projection."):
                    dst = getattr(self.guide_projection, key.split(".")[-1])
                else:
                    path, name = key.rsplit(".", 1)
                    dst = getattr(self._wrappers[path], name)
                dst.copy_(value.to(dst))

    def close(self):
        """Remove the hook and put the original Linear modules back, leaving the transformer as it was."""
        self._hook.remove()
        for path, wrapper in self._wrappers.items():
            parent = self.transformer
            parts = path.split(".")
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            name = parts[-1]
            if name.isdigit():
                parent[int(name)] = wrapper.base
            else:
                setattr(parent, name, wrapper.base)
        if getattr(self.transformer, "_flow_control_lora", None) is self:
            delattr(self.transformer, "_flow_control_lora")
