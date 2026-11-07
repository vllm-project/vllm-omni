# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Causal Wan2.1 text-to-video DiT for Self-Forcing checkpoints.

The backbone is the LingBot World causal Wan block without its camera path, so
the self- and cross-attention modules (direct and paged KV) are shared with
``lingbot_world.transformer`` instead of being re-implemented here.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from typing_extensions import Self
from vllm.model_executor.layers.conv import Conv3dLayer
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.layers.norm import LayerNorm
from vllm_omni.diffusion.layers.prope import PRoPECameras, apply_prope
from vllm_omni.diffusion.models.lingbot_world.transformer import (
    LingBotAttentionCache,
    LingBotCrossAttention,
    LingBotSelfAttention,
    LingBotTransformerCache,
    _rope_axis,
    _sinusoidal_embedding,
    allocate_lingbot_cache,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import (
    ARDiffusionPagedLayerContext,
    ARDiffusionPagedLayerInputs,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

# Fixed geometry of the released Self-Forcing generator (Wan2.1-T2V-1.3B).
SELF_FORCING_CHECKPOINT_CONTRACT: dict[str, Any] = {
    "patch_size": (1, 2, 2),
    "num_attention_heads": 12,
    "attention_head_dim": 128,
    "in_channels": 16,
    "out_channels": 16,
    "text_dim": 4096,
    "freq_dim": 256,
    "ffn_dim": 8960,
    "num_layers": 30,
    "cross_attn_norm": True,
    "eps": 1e-6,
    "rope_max_seq_len": 1024,
}

# Causal rollout settings the checkpoint was distilled with; configurable for
# checkpoints of the same backbone that use another block or window size.
SELF_FORCING_ROLLOUT_DEFAULTS: dict[str, int] = {
    "num_frames_per_block": 3,
    "window_frames": 21,
    "sink_frames": 0,
}


@dataclass(frozen=True)
class SelfForcingTransformerConfig:
    patch_size: tuple[int, int, int]
    num_attention_heads: int
    attention_head_dim: int
    in_channels: int
    out_channels: int
    text_dim: int
    freq_dim: int
    ffn_dim: int
    num_layers: int
    cross_attn_norm: bool
    eps: float
    rope_max_seq_len: int
    num_frames_per_block: int
    window_frames: int
    sink_frames: int
    use_prope: bool


def _prefixed(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


class SelfForcingPRoPESelfAttention(LingBotSelfAttention):
    """Self-attention plus a camera-relative PRoPE attention over a second K/V cache.

    The PRoPE path transforms the normalized q/k/v before RoPE and adds its
    output through ``prope_o``; without cameras the layer is plain self-attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(dim, num_heads, eps=eps, quant_config=quant_config, prefix=prefix)
        if self.ulysses_world_size > 1:
            raise ValueError("PRoPE self-attention does not support Ulysses sequence parallelism.")
        self.prope_o = RowParallelLinear(
            dim,
            dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=_prefixed(prefix, "prope_o"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cache: LingBotAttentionCache | ARDiffusionPagedLayerInputs,
        current_start: int,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        sink_tokens: int,
        update_cache: bool = True,
        prope_cache: LingBotAttentionCache | ARDiffusionPagedLayerInputs | None = None,
        cameras: PRoPECameras | None = None,
    ) -> torch.Tensor:
        if cameras is None:
            return super().forward(
                hidden_states,
                cache=cache,
                current_start=current_start,
                rotary_emb=rotary_emb,
                sink_tokens=sink_tokens,
                update_cache=update_cache,
            )
        if prope_cache is None:
            raise ValueError("PRoPE self-attention with cameras requires a PRoPE cache.")
        query, key, value = self.project_qkv(hidden_states)
        prope_query = apply_prope(query, cameras.query)
        prope_key = apply_prope(key, cameras.key)
        prope_value = apply_prope(value, cameras.key)
        if rotary_emb is not None:
            cos, sin = rotary_emb
            query = self.rotary_embedding(query, cos, sin)
            key = self.rotary_embedding(key, cos, sin)
        output = self.attend(
            query,
            key,
            value,
            cache=cache,
            current_start=current_start,
            sink_tokens=sink_tokens,
            update_cache=update_cache,
        )
        prope_output = self.attend(
            prope_query,
            prope_key,
            prope_value,
            cache=prope_cache,
            current_start=current_start,
            sink_tokens=sink_tokens,
            update_cache=update_cache,
        )
        return self.o(output.flatten(2, 3)) + self.prope_o(apply_prope(prope_output, cameras.output).flatten(2, 3))


class SelfForcingAttentionBlock(nn.Module):
    """Wan2.1 block with block-causal self-attention and cached text cross-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        ffn_dim: int,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        use_prope: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.norm1 = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self_attention_class = SelfForcingPRoPESelfAttention if use_prope else LingBotSelfAttention
        self.self_attn = self_attention_class(
            dim,
            num_heads,
            eps=eps,
            quant_config=quant_config,
            prefix=_prefixed(prefix, "self_attn"),
        )
        self.cross_attn = LingBotCrossAttention(
            dim,
            num_heads,
            eps=eps,
            quant_config=quant_config,
            prefix=_prefixed(prefix, "cross_attn"),
        )
        self.norm2 = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = LayerNorm(dim, eps=eps) if cross_attn_norm else nn.Identity()
        self.ffn = nn.Sequential(
            ColumnParallelLinear(
                dim,
                ffn_dim,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=_prefixed(prefix, "ffn.0"),
            ),
            nn.GELU(approximate="tanh"),
            RowParallelLinear(
                ffn_dim,
                dim,
                bias=True,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=_prefixed(prefix, "ffn.2"),
            ),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / math.sqrt(dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        timestep_projection: torch.Tensor,
        *,
        self_cache: LingBotAttentionCache | ARDiffusionPagedLayerInputs,
        cross_cache: LingBotAttentionCache | None,
        current_start: int,
        sink_tokens: int,
        update_cache: bool,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        prope_cache: LingBotAttentionCache | ARDiffusionPagedLayerInputs | None = None,
        cameras: PRoPECameras | None = None,
    ) -> tuple[torch.Tensor, LingBotAttentionCache]:
        batch_size, token_count, dim = hidden_states.shape
        if (
            timestep_projection.ndim != 4
            or timestep_projection.shape[0] != batch_size
            or timestep_projection.shape[2:] != (6, dim)
        ):
            raise ValueError("Self-Forcing timestep projection must have shape (batch, frames, 6, dim).")
        num_frames = timestep_projection.shape[1]
        if num_frames == 0 or token_count % num_frames:
            raise ValueError("Self-Forcing timestep projection must align with hidden tokens.")
        tokens_per_frame = token_count // num_frames
        modulation = self.modulation.unsqueeze(1) + timestep_projection.float()
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = modulation.chunk(6, dim=2)

        hidden_grid = hidden_states.unflatten(1, (num_frames, tokens_per_frame))
        normalized = self.norm1(hidden_states.float()).unflatten(1, (num_frames, tokens_per_frame))
        normalized = (normalized * (1 + scale_msa) + shift_msa).flatten(1, 2).to(hidden_states.dtype)
        if cameras is None:
            attention_output = self.self_attn(
                normalized,
                cache=self_cache,
                current_start=current_start,
                rotary_emb=rotary_emb,
                sink_tokens=sink_tokens,
                update_cache=update_cache,
            )
        else:
            attention_output = self.self_attn(
                normalized,
                cache=self_cache,
                current_start=current_start,
                rotary_emb=rotary_emb,
                sink_tokens=sink_tokens,
                update_cache=update_cache,
                prope_cache=prope_cache,
                cameras=cameras,
            )
        hidden_grid = hidden_grid + attention_output.unflatten(1, (num_frames, tokens_per_frame)) * gate_msa
        hidden_states = hidden_grid.flatten(1, 2).to(hidden_states.dtype)

        attention_output, cross_cache = self.cross_attn(
            self.norm3(hidden_states),
            encoder_hidden_states,
            cache=cross_cache,
        )
        hidden_states = hidden_states + attention_output

        hidden_grid = hidden_states.unflatten(1, (num_frames, tokens_per_frame))
        normalized = self.norm2(hidden_states.float()).unflatten(1, (num_frames, tokens_per_frame))
        normalized = (normalized * (1 + scale_ffn) + shift_ffn).flatten(1, 2).to(hidden_states.dtype)
        ffn_output = self.ffn(normalized).unflatten(1, (num_frames, tokens_per_frame))
        hidden_states = (hidden_grid + ffn_output * gate_ffn).flatten(1, 2).to(hidden_states.dtype)
        return hidden_states, cast(LingBotAttentionCache, cross_cache)


class SelfForcingHead(nn.Module):
    """Output projection with per-frame shift/scale modulation."""

    def __init__(self, dim: int, out_channels: int, patch_size: tuple[int, int, int], eps: float) -> None:
        super().__init__()
        self.norm = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_channels * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / math.sqrt(dim))

    def forward(self, hidden_states: torch.Tensor, timestep_embedding: torch.Tensor) -> torch.Tensor:
        num_frames = timestep_embedding.shape[1]
        modulation = self.modulation.unsqueeze(1) + timestep_embedding.unsqueeze(2).float()
        shift, scale = modulation.chunk(2, dim=2)
        normalized = self.norm(hidden_states.float()).unflatten(1, (num_frames, -1))
        normalized = (normalized * (1 + scale) + shift).flatten(1, 2).to(hidden_states.dtype)
        return self.head(normalized)


class SelfForcingTransformer3DModel(nn.Module):
    """Block-causal Wan2.1-T2V-1.3B transformer used by Self-Forcing.

    Each forward denoises exactly one AR block of ``num_frames_per_block``
    latent frames against the K/V of previously committed blocks. ``cache``
    holds either request-local contiguous K/V or the runner-bound paged
    contexts of an AR-Diffusion session.
    """

    _repeated_blocks = ["SelfForcingAttentionBlock"]
    packed_modules_mapping = {"qkv": ["q", "k", "v"]}
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 12,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 8960,
        num_layers: int = 30,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        rope_max_seq_len: int = 1024,
        num_frames_per_block: int = 3,
        window_frames: int = 21,
        sink_frames: int = 0,
        use_prope: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if len(patch_size) != 3 or any(size <= 0 for size in patch_size):
            raise ValueError("patch_size must contain three positive integers.")
        if patch_size[0] != 1:
            raise ValueError("Self-Forcing requires temporal patch size 1.")
        if attention_head_dim % 2:
            raise ValueError("attention_head_dim must be even for RoPE.")
        if num_frames_per_block <= 0 or window_frames <= 0 or sink_frames < 0:
            raise ValueError("num_frames_per_block and window_frames must be positive and sink_frames non-negative.")
        if window_frames < sink_frames + num_frames_per_block:
            raise ValueError("window_frames must fit all sink frames and one complete current block.")
        dim = num_attention_heads * attention_head_dim
        self.dim = dim
        self.config = SelfForcingTransformerConfig(
            patch_size=tuple(patch_size),
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            rope_max_seq_len=rope_max_seq_len,
            num_frames_per_block=num_frames_per_block,
            window_frames=window_frames,
            sink_frames=sink_frames,
            use_prope=use_prope,
        )

        self.patch_embedding = Conv3dLayer(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )
        self.blocks = nn.ModuleList(
            [
                SelfForcingAttentionBlock(
                    dim,
                    num_attention_heads,
                    ffn_dim=ffn_dim,
                    cross_attn_norm=cross_attn_norm,
                    eps=eps,
                    use_prope=use_prope,
                    quant_config=quant_config,
                    prefix=_prefixed(prefix, f"blocks.{index}"),
                )
                for index in range(num_layers)
            ]
        )
        self.head = SelfForcingHead(dim, out_channels, patch_size, eps)

        temporal_dim = attention_head_dim - 4 * (attention_head_dim // 6)
        spatial_dim = 2 * (attention_head_dim // 6)
        for axis, axis_dim in (("temporal", temporal_dim), ("height", spatial_dim), ("width", spatial_dim)):
            cosine, sine = _rope_axis(rope_max_seq_len, axis_dim)
            self.register_buffer(f"_rope_{axis}_cosine", cosine, persistent=False)
            self.register_buffer(f"_rope_{axis}_sine", sine, persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype used by the transformer parameters."""
        return next(self.parameters()).dtype

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> Self:
        """Build the transformer after checking the released checkpoint geometry."""
        class_name = config.get("_class_name")
        if class_name is not None and class_name != cls.__name__:
            raise ValueError(
                f"_class_name must be {cls.__name__!r} for a Self-Forcing transformer, got {class_name!r}."
            )
        kwargs: dict[str, Any] = {}
        for name, expected in SELF_FORCING_CHECKPOINT_CONTRACT.items():
            actual = config.get(name, expected)
            if name == "patch_size":
                actual = tuple(actual)
            if actual != expected:
                raise ValueError(f"Self-Forcing checkpoint config {name} must be {expected!r}, got {actual!r}.")
            kwargs[name] = actual
        for name, default in SELF_FORCING_ROLLOUT_DEFAULTS.items():
            kwargs[name] = int(config.get(name, default))
        use_prope = config.get("use_prope", False)
        if not isinstance(use_prope, bool):
            raise ValueError(f"Self-Forcing checkpoint config use_prope must be a bool, got {use_prope!r}.")
        return cls(**kwargs, use_prope=use_prope, quant_config=quant_config, prefix=prefix)

    def tokens_per_frame(self, latent_height: int, latent_width: int) -> int:
        """Post-patch tokens in one latent frame of the given latent size."""
        _, patch_height, patch_width = self.config.patch_size
        if latent_height % patch_height or latent_width % patch_width:
            raise ValueError("latent height/width must align with the Self-Forcing patch size.")
        return (latent_height // patch_height) * (latent_width // patch_width)

    def allocate_cache(
        self,
        *,
        latent_height: int,
        latent_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LingBotTransformerCache:
        """Allocate one request-local cache covering the attention window."""
        max_tokens = self.config.window_frames * self.tokens_per_frame(latent_height, latent_width)
        return allocate_lingbot_cache(
            batch_size=1,
            num_layers=self.config.num_layers,
            max_tokens=max_tokens,
            num_local_heads=self.blocks[0].self_attn.num_sp_heads,
            head_dim=self.config.attention_head_dim,
            device=device,
            dtype=dtype,
        )

    def allocate_prope_cache(
        self,
        *,
        latent_height: int,
        latent_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[LingBotAttentionCache]:
        """Allocate the request-local self-attention K/V of the PRoPE branch."""
        if not self.config.use_prope:
            raise ValueError("Only a PRoPE checkpoint has a PRoPE cache.")
        cache = self.allocate_cache(latent_height=latent_height, latent_width=latent_width, device=device, dtype=dtype)
        return cast(list[LingBotAttentionCache], cache.self_attention)

    def _rotary_embedding(
        self,
        *,
        frames: int,
        height: int,
        width: int,
        start_frame: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if height > self.config.rope_max_seq_len or width > self.config.rope_max_seq_len:
            raise ValueError("Spatial RoPE positions exceed rope_max_seq_len.")
        if start_frame + frames <= self._rope_temporal_cosine.shape[0]:
            temporal_cosine = self._rope_temporal_cosine[start_frame : start_frame + frames]
            temporal_sine = self._rope_temporal_sine[start_frame : start_frame + frames]
        else:
            # Unbounded sessions keep absolute temporal positions past the table.
            temporal_dim = self._rope_temporal_cosine.shape[1] * 2
            temporal_cosine, temporal_sine = _rope_axis(frames, temporal_dim, start=start_frame)
            temporal_cosine = temporal_cosine.to(self._rope_temporal_cosine)
            temporal_sine = temporal_sine.to(self._rope_temporal_sine)

        def expand(temporal: torch.Tensor, spatial_height: torch.Tensor, spatial_width: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                (
                    temporal.view(frames, 1, 1, -1).expand(frames, height, width, -1),
                    spatial_height[:height].view(1, height, 1, -1).expand(frames, height, width, -1),
                    spatial_width[:width].view(1, 1, width, -1).expand(frames, height, width, -1),
                ),
                dim=-1,
            ).reshape(frames * height * width, -1)

        cosine = expand(temporal_cosine, self._rope_height_cosine, self._rope_width_cosine)
        sine = expand(temporal_sine, self._rope_height_sine, self._rope_width_sine)
        return cosine.to(device=device, dtype=dtype), sine.to(device=device, dtype=dtype)

    def _timestep_embeddings(
        self,
        timestep: torch.Tensor,
        *,
        batch_size: int,
        frames: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if timestep.ndim == 0:
            timestep = timestep.reshape(1)
        if timestep.ndim == 1:
            if timestep.shape[0] != batch_size:
                raise ValueError("A rank-1 timestep must contain one value per batch item.")
            timestep = timestep.unsqueeze(1).expand(batch_size, frames)
        elif timestep.shape != (batch_size, frames):
            raise ValueError(f"timestep must be scalar, [batch], or [batch, frames] = {(batch_size, frames)}.")
        frequency_embedding = _sinusoidal_embedding(self.config.freq_dim, timestep.reshape(-1)).to(dtype=dtype)
        timestep_embedding = self.time_embedding(frequency_embedding).unflatten(0, (batch_size, frames))
        timestep_projection = self.time_projection(timestep_embedding).unflatten(2, (6, self.dim))
        return timestep_embedding, timestep_projection

    def _unpatchify(self, hidden_states: torch.Tensor, *, frames: int, height: int, width: int) -> torch.Tensor:
        _, patch_height, patch_width = self.config.patch_size
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], frames, height, width, 1, patch_height, patch_width, self.config.out_channels
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    @staticmethod
    def _paged_layer_inputs(
        contexts: list[ARDiffusionPagedLayerContext], *, tokens: int, device: torch.device
    ) -> list[ARDiffusionPagedLayerInputs]:
        forward_context = contexts[0].forward_ctx
        if forward_context.seq_len != tokens:
            raise RuntimeError(
                "Self-Forcing paged context token count does not match this block: "
                f"{forward_context.seq_len} != {tokens}."
            )
        forward_context.prepare(device=device, action_len=0, query_len=tokens)
        return [layer_context.to_layer_inputs() for layer_context in contexts]

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        cache: LingBotTransformerCache,
        start_frame: int,
        update_cache: bool,
        prope_cache: list[LingBotAttentionCache] | list[ARDiffusionPagedLayerContext] | None = None,
        cameras: PRoPECameras | None = None,
    ) -> torch.Tensor:
        """Predict flow for one AR block.

        Args:
            hidden_states (torch.Tensor): noisy latents ``[batch, channels, frames, height, width]``.
            timestep (torch.Tensor): scalar, ``[batch]`` or ``[batch, frames]`` timesteps.
            encoder_hidden_states (torch.Tensor): UMT5 embeddings ``[batch, tokens, text_dim]``.
            cache (LingBotTransformerCache): per-layer self- and cross-attention caches.
            start_frame (int): absolute latent frame index of the block's first frame.
            update_cache (bool): persist this block's K/V in a request-local cache.
            prope_cache (list | None): per-layer PRoPE self-attention caches; required with ``cameras``.
            cameras (PRoPECameras | None): this block's per-frame camera transforms, for PRoPE checkpoints.

        Returns:
            torch.Tensor: flow prediction with the shape of ``hidden_states``.
        """
        if hidden_states.ndim != 5 or hidden_states.shape[1] != self.config.in_channels:
            raise ValueError(f"hidden_states must be [batch, {self.config.in_channels}, frames, height, width].")
        if isinstance(start_frame, bool) or not isinstance(start_frame, int) or start_frame < 0:
            raise ValueError("start_frame must be a non-negative integer latent-frame offset.")
        batch_size, _, frames, height, width = hidden_states.shape
        if frames != self.config.num_frames_per_block:
            raise ValueError(
                f"Each Self-Forcing forward must contain exactly {self.config.num_frames_per_block} latent frames, "
                f"got {frames}."
            )
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[2] != self.config.text_dim:
            raise ValueError(f"encoder_hidden_states must be [batch, tokens, {self.config.text_dim}].")
        if len(cache.self_attention) != self.config.num_layers or len(cache.cross_attention) != self.config.num_layers:
            raise ValueError("Self-Forcing cache layer counts must match num_layers.")
        if cameras is not None:
            if not self.config.use_prope:
                raise ValueError("Camera conditioning requires a PRoPE checkpoint.")
            if len(cameras) != frames:
                raise ValueError(f"cameras must describe the block's {frames} latent frames, got {len(cameras)}.")
            if prope_cache is None or len(prope_cache) != self.config.num_layers:
                raise ValueError("Camera conditioning requires one PRoPE cache per layer.")

        _, patch_height, patch_width = self.config.patch_size
        patched_height = height // patch_height
        patched_width = width // patch_width
        tokens_per_frame = self.tokens_per_frame(height, width)
        current_start = start_frame * tokens_per_frame
        sink_tokens = self.config.sink_frames * tokens_per_frame
        rotary_emb = self._rotary_embedding(
            frames=frames,
            height=patched_height,
            width=patched_width,
            start_frame=start_frame,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        timestep_embedding, timestep_projection = self._timestep_embeddings(
            timestep,
            batch_size=batch_size,
            frames=frames,
            dtype=hidden_states.dtype,
        )
        projected_text = (
            self.text_embedding(encoder_hidden_states)
            if any(value is None for value in cache.cross_attention)
            else None
        )
        if cache.self_attention and isinstance(cache.self_attention[0], ARDiffusionPagedLayerContext):
            if batch_size != 1:
                raise RuntimeError("Self-Forcing AR-Diffusion paged attention requires batch_size=1.")
            cache.self_attention = self._paged_layer_inputs(
                cast(list[ARDiffusionPagedLayerContext], cache.self_attention),
                tokens=frames * tokens_per_frame,
                device=hidden_states.device,
            )
        if cameras is not None and isinstance(cast(list, prope_cache)[0], ARDiffusionPagedLayerContext):
            prope_cache = self._paged_layer_inputs(
                cast(list[ARDiffusionPagedLayerContext], prope_cache),
                tokens=frames * tokens_per_frame,
                device=hidden_states.device,
            )

        for index, block in enumerate(self.blocks):
            hidden_states, cross_cache = block(
                hidden_states,
                projected_text if cache.cross_attention[index] is None else None,
                timestep_projection,
                self_cache=cache.self_attention[index],
                cross_cache=cache.cross_attention[index],
                current_start=current_start,
                sink_tokens=sink_tokens,
                update_cache=update_cache,
                rotary_emb=rotary_emb,
                prope_cache=None if cameras is None else cast(list, prope_cache)[index],
                cameras=cameras,
            )
            cache.cross_attention[index] = cross_cache

        hidden_states = self.head(hidden_states, timestep_embedding)
        return self._unpatchify(hidden_states, frames=frames, height=patched_height, width=patched_width)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load original Wan parameter names, packing self-attention q/k/v."""
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for checkpoint_name, loaded_weight in weights:
            name = checkpoint_name
            shard_id = None
            for projection in ("q", "k", "v"):
                marker = f".self_attn.{projection}."
                if marker in name:
                    name = name.replace(marker, ".self_attn.qkv.")
                    shard_id = projection
                    break
            if name not in params:
                raise KeyError(f"Unexpected Self-Forcing transformer weight name: {checkpoint_name}")
            param = params[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is None:
                weight_loader(param, loaded_weight)
            else:
                weight_loader(param, loaded_weight, shard_id)
            loaded.add(checkpoint_name)
            loaded.add(name)
        return loaded
