# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Native inference transformer for the released SANA-Video 2.0 5B.

Adapted from NVlabs/Sana at e93c883e10730ee5a4a6edf1cbcf501dc4ef753b.
Parallel execution and the video denoising pipeline are separate integrations.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import SanaVideo2Block
from .components import CaptionEmbedder, PatchEmbedMS3D, RMSNorm, T2IFinalLayer, TimestepEmbedder, WanRotaryPosEmbed


def get_softmax_layer_indices(depth: int, softmax_ratio: float = 0.25) -> list[int]:
    """Distribute dense-attention anchor layers uniformly through the network."""
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}.")
    if not 0.0 < softmax_ratio <= 1.0:
        raise ValueError(f"softmax_ratio must be in (0, 1], got {softmax_ratio}.")
    anchor_count = max(1, int(depth * softmax_ratio))
    step = depth / anchor_count
    return [int((index + 1) * step) - 1 for index in range(anchor_count)]


class DepthRMSNorm(nn.Module):
    """Parameter-free RMS normalization over the hidden dimension."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class BlockAttentionResidual(nn.Module):
    """Shared block-level Attention Residual aggregation."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn_proj = nn.Linear(hidden_size, 1, bias=False)
        self.mlp_proj = nn.Linear(hidden_size, 1, bias=False)
        self.final_proj = nn.Linear(hidden_size, 1, bias=False)
        self.key_norm = DepthRMSNorm(eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.attn_proj.weight)
        nn.init.zeros_(self.mlp_proj.weight)
        nn.init.zeros_(self.final_proj.weight)

    def attend(
        self,
        projection: nn.Linear,
        block_representations: list[torch.Tensor],
        partial_block: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aggregate depth representations using an autograd-safe stack."""
        sources = block_representations + ([partial_block] if partial_block is not None else [])
        if len(sources) == 1:
            return sources[0]

        values = torch.stack(sources, dim=0)
        keys = self.key_norm(values)
        query = projection.weight.squeeze(0)
        logits = torch.einsum("d,nbtd->nbt", query, keys)
        weights = F.softmax(logits, dim=0)
        return torch.einsum("nbt,nbtd->btd", weights, values)

    def attend_buffer(
        self,
        projection: nn.Linear,
        value_buffer: torch.Tensor,
        key_buffer: torch.Tensor,
        active_count: int,
        partial_block: torch.Tensor | None,
    ) -> torch.Tensor:
        """Inference-only aggregation using preallocated value and key buffers."""
        if partial_block is not None:
            value_buffer[active_count] = partial_block
            key_buffer[active_count] = self.key_norm(partial_block.unsqueeze(0)).squeeze(0)
            source_count = active_count + 1
        else:
            source_count = active_count
        if source_count == 1:
            return value_buffer[0]

        values = value_buffer[:source_count]
        keys = key_buffer[:source_count]
        query = projection.weight.squeeze(0)
        logits = torch.einsum("d,nbtd->nbt", query, keys)
        weights = F.softmax(logits, dim=0)
        return torch.einsum("nbt,nbtd->btd", weights, values)


@dataclass(frozen=True)
class SanaVideo2TransformerConfig:
    in_channels: int = 128
    hidden_size: int = 2560
    depth: int = 32
    num_heads: int = 20
    caption_channels: int = 2304
    model_max_length: int = 300
    linear_head_dim: int = 128
    softmax_head_dim: int = 256
    softmax_ratio: float = 0.25
    mlp_ratio: float = 4.0
    attn_res_block_size: int = 8
    patch_size: tuple[int, int, int] = (1, 1, 1)
    qk_norm: bool = True
    cross_norm: bool = True
    y_norm: bool = True
    y_norm_scale_factor: float = 0.01
    norm_eps: float = 1e-5
    fp32_attention: bool = True
    timestep_norm_scale_factor: float = 1.0

    def __post_init__(self):
        for name in (
            "in_channels",
            "hidden_size",
            "depth",
            "num_heads",
            "caption_channels",
            "model_max_length",
            "linear_head_dim",
            "softmax_head_dim",
            "attn_res_block_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if tuple(self.patch_size) != (1, 1, 1):
            raise ValueError("SANA-Video 2.0 release requires patch_size=(1, 1, 1)")
        for dim in (self.linear_head_dim, self.softmax_head_dim, self.num_heads):
            if self.hidden_size % dim:
                raise ValueError("hidden_size must divide evenly into self- and cross-attention heads")
        if any(dim < 6 or dim % 2 for dim in (self.linear_head_dim, self.softmax_head_dim)):
            raise ValueError("RoPE head dimensions must be even and at least 6")
        get_softmax_layer_indices(self.depth, self.softmax_ratio)
        if any(
            not math.isfinite(value) or value <= 0
            for value in (self.mlp_ratio, self.norm_eps, self.timestep_norm_scale_factor)
        ):
            raise ValueError("MLP ratio, normalization epsilon and timestep scale must be positive")
        if not self.fp32_attention:
            raise ValueError("The release numerical contract requires fp32_attention=True")

    @classmethod
    def from_dict(cls, values: Mapping):
        """Parse a native config, or the model/vae/text_encoder release YAML sections.

        Training-only fields in a release YAML do not configure this inference
        module. Architecture-affecting options outside the release are rejected.
        """
        if not isinstance(values.get("model"), Mapping):
            unknown = set(values) - {f.name for f in fields(cls)}
            if unknown:
                raise ValueError(f"Unknown transformer config fields: {sorted(unknown)}")
            return cls(**values)
        model = values["model"]
        required = {"model": "SanaVideo2_5B", "ffn_type": "SwiGLU", "use_pe": True, "pos_embed_type": "wan_rope"}
        for key, expected in required.items():
            if model.get(key, expected) != expected:
                raise ValueError(f"Unsupported SANA-Video 2.0 {key}: {model[key]!r}")
        # Explicitly ignore only training/runtime metadata; never silently drop
        # an unfamiliar architecture option (e.g. rope_fhw_dim).
        metadata = {"image_size", "mixed_precision", "load_from", "multi_scale", "class_dropout_prob"}
        unknown = set(model) - {f.name for f in fields(cls)} - set(required) - metadata
        if unknown:
            raise ValueError(f"Unsupported model config fields: {sorted(unknown)}")
        parsed = {f.name: model[f.name] for f in fields(cls) if f.name in model}
        text = values.get("text_encoder", {})
        for key in ("caption_channels", "model_max_length", "y_norm", "y_norm_scale_factor"):
            if key in text:
                parsed[key] = text[key]
        if "vae_latent_dim" in values.get("vae", {}):
            parsed["in_channels"] = values["vae"]["vae_latent_dim"]
        if values.get("scheduler", {}).get("pred_sigma", False):
            raise ValueError("The released flow model requires pred_sigma=False")
        return cls(**parsed)

    @classmethod
    def from_file(cls, path: str | Path):
        import yaml

        with open(path) as stream:
            return cls.from_dict(yaml.safe_load(stream))


@dataclass(frozen=True)
class SanaVideo2LoadReport:
    """Coverage is reported only after exact key and shape validation succeeds."""

    loaded_keys: tuple[str, ...]
    ignored_keys: tuple[str, ...]
    loaded_numel: int
    source_dtypes: tuple[str, ...]


class SanaVideo2TransformerModel(nn.Module):
    def __init__(self, config: SanaVideo2TransformerConfig | Mapping | None = None):
        super().__init__()
        if config is None:
            config = SanaVideo2TransformerConfig()
        elif isinstance(config, Mapping):
            config = SanaVideo2TransformerConfig.from_dict(config)
        self.config = config
        c = config
        self.x_embedder = PatchEmbedMS3D(c.in_channels, c.hidden_size, c.patch_size)
        self.t_embedder = TimestepEmbedder(c.hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(c.hidden_size, 6 * c.hidden_size))
        self.y_embedder = CaptionEmbedder(c.caption_channels, c.hidden_size)
        if c.y_norm:
            self.attention_y_norm = RMSNorm(c.hidden_size, c.y_norm_scale_factor, c.norm_eps)
        self.softmax_layer_indices = get_softmax_layer_indices(c.depth, c.softmax_ratio)
        self.block_attention_types = [
            "softmax" if i in self.softmax_layer_indices else "linear" for i in range(c.depth)
        ]
        self.rope_linear = WanRotaryPosEmbed(c.linear_head_dim, c.patch_size, 1024)
        self.rope_softmax = WanRotaryPosEmbed(c.softmax_head_dim, c.patch_size, 1024)
        self.blocks = nn.ModuleList(
            [
                SanaVideo2Block(
                    c.hidden_size,
                    c.num_heads,
                    kind,
                    c.linear_head_dim,
                    c.softmax_head_dim,
                    mlp_ratio=c.mlp_ratio,
                    qk_norm=c.qk_norm,
                    cross_norm=c.cross_norm,
                )
                for kind in self.block_attention_types
            ]
        )
        self.attn_res = BlockAttentionResidual(c.hidden_size)
        self.final_layer = T2IFinalLayer(c.hidden_size, c.patch_size, c.in_channels)

    @property
    def dtype(self):
        return self.x_embedder.proj.weight.dtype

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ):
        """Predict flow from B,C,F,H,W latents and B,L,C text embeddings.

        Timesteps are B or B,1,F,1,1 (TI2V). All depth state and video dimensions
        are local to this call; no state is reused across requests or timesteps.
        """
        c = self.config
        if hidden_states.ndim != 5 or hidden_states.shape[1] != c.in_channels:
            raise ValueError("Latents must have shape (batch, in_channels, frames, height, width)")
        batch, _, frames, height, width = hidden_states.shape
        if batch < 1 or min(frames, height, width) < 1 or max(frames, height, width) > 1024:
            raise ValueError("Latent axes must be in [1, 1024] and batch must be positive")
        y = encoder_hidden_states
        if (
            y.ndim != 3
            or y.shape[0] != batch
            or y.shape[2] != c.caption_channels
            or not 1 <= y.shape[1] <= c.model_max_length
        ):
            raise ValueError("Text embeddings must have shape (batch, tokens <= model_max_length, caption_channels)")
        mask = encoder_attention_mask
        if mask is not None:
            if mask.dtype != torch.bool or mask.shape != y.shape[:2]:
                raise ValueError("Text mask must be bool with shape (batch, text_tokens)")
            if not mask.any(dim=1).all():
                raise ValueError("Each sample must contain at least one unmasked text token")
        if timestep.shape == (batch, 1, frames, 1, 1):
            timestep = timestep.reshape(batch, 1, frames)
        elif timestep.shape != (batch,):
            raise ValueError("Timesteps must have shape (batch,) or (batch, 1, frames, 1, 1)")
        timestep = (
            timestep.float() / c.timestep_norm_scale_factor
            if c.timestep_norm_scale_factor != 1.0
            else timestep.long().float()
        )
        x = self.x_embedder(hidden_states.to(self.dtype))
        thw = (frames, height, width)
        ropes = {"linear": self.rope_linear(thw, x.device), "softmax": self.rope_softmax(thw, x.device)}
        t = self.t_embedder(timestep.flatten())
        t0 = self.t_block(t).unflatten(0, timestep.shape)
        t = t.unflatten(0, timestep.shape)
        y = self.y_embedder(y.to(self.dtype))
        if c.y_norm:
            y = self.attention_y_norm(y)
        # Match the release inference buffer layout as well as its arithmetic.
        # In BF16, recomputing norms on a differently strided stack can change
        # rounding and compound through the learned depth projections.
        group_count = math.ceil(c.depth / c.attn_res_block_size)
        values = x.new_empty(group_count + 2, batch, x.shape[1], c.hidden_size)
        keys = torch.empty_like(values)
        values[0] = x
        keys[0] = self.attn_res.key_norm(x.unsqueeze(0)).squeeze(0)
        active_count = 1
        partial = None
        for index, block in enumerate(self.blocks):
            hidden = self.attn_res.attend_buffer(self.attn_res.attn_proj, values, keys, active_count, partial)
            delta = block.forward_attn_sublayer(
                hidden, y, t0, mask=mask, rotary_emb=ropes[self.block_attention_types[index]]
            )
            partial = delta if partial is None else partial + delta
            hidden = self.attn_res.attend_buffer(self.attn_res.mlp_proj, values, keys, active_count, partial)
            partial = partial + block.forward_mlp_sublayer(hidden, t0)
            if (index + 1) % c.attn_res_block_size == 0 or index + 1 == c.depth:
                values[active_count] = partial
                keys[active_count] = self.attn_res.key_norm(partial.unsqueeze(0)).squeeze(0)
                active_count += 1
                partial = None
        x = self.attn_res.attend_buffer(self.attn_res.final_proj, values, keys, active_count, None)
        x = self.final_layer(x, t)
        return x.transpose(1, 2).reshape(batch, c.in_channels, frames, height, width)

    def load_weights(self, weights) -> SanaVideo2LoadReport:
        """Load upstream keys verbatim; no lossy renaming or partial loading.

        Only ``pos_embed`` (unused legacy positional buffer) and
        ``y_embedder.y_embedding`` (training caption dropout buffer) are ignored.
        Unconditional text conditioning is a pipeline input, not this buffer.
        """
        expected = self.state_dict()
        mapped = {}
        ignored = []
        seen = set()
        errors = []
        for key, tensor in weights:
            if key in seen:
                errors.append(f"duplicate key {key}")
                continue
            seen.add(key)
            if not isinstance(tensor, torch.Tensor):
                errors.append(f"non-tensor value for {key}")
                continue
            if key in ("pos_embed", "y_embedder.y_embedding"):
                valid = (
                    tensor.ndim == 3 and tensor.shape[0] == 1 and tensor.shape[-1] == self.config.hidden_size
                    if key == "pos_embed"
                    else tensor.shape == (self.config.model_max_length, self.config.caption_channels)
                )
                if not valid:
                    errors.append(f"invalid ignored buffer shape {key}: {tuple(tensor.shape)}")
                ignored.append(key)
                continue
            if key not in expected:
                errors.append(f"unexpected key {key}")
            elif tensor.shape != expected[key].shape:
                errors.append(
                    f"shape mismatch {key}: checkpoint {tuple(tensor.shape)}, model {tuple(expected[key].shape)}"
                )
            elif not tensor.is_floating_point():
                errors.append(f"non-floating weight {key}: {tensor.dtype}")
            else:
                mapped[key] = tensor
        missing = set(expected) - set(mapped)
        if missing:
            errors.append(f"missing keys: {sorted(missing)}")
        if errors:
            raise ValueError("Invalid SANA-Video 2.0 checkpoint:\n" + "\n".join(errors))
        if any(t.is_meta for t in expected.values()):
            raise ValueError("Materialize the model with to_empty() before loading weights")
        self.load_state_dict(mapped, strict=True)
        return SanaVideo2LoadReport(
            tuple(sorted(mapped)),
            tuple(sorted(ignored)),
            sum(t.numel() for t in mapped.values()),
            tuple(sorted({str(t.dtype) for t in mapped.values()})),
        )

    def load_checkpoint(self, path: str | Path) -> SanaVideo2LoadReport:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("state_dict"), Mapping):
            raise ValueError("Expected a SANA .pth checkpoint containing state_dict")
        return self.load_weights(checkpoint["state_dict"].items())
