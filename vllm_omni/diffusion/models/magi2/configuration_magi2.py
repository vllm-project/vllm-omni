# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Native configuration for the released MAGI-2 Preview and refiner checkpoints.

The defaults mirror ``sand-ai/MAGI-2-preview``.  Keeping the architecture in
vLLM-Omni makes model construction independent of SandAI's Python package and
also gives tests a small-config entry point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class Magi2MHCConfig:
    enabled: bool = True
    num_streams: int = 4
    alpha_init: float = 0.01
    sinkhorn_iterations: int = 20
    sinkhorn_epsilon: float = 1e-12


@dataclass(frozen=True)
class Magi2MoEConfig:
    num_heads: int = 12
    num_experts: int = 256
    top_k: int = 6
    expert_intermediate_size: int = 1280
    shared_expert_intermediate_size: int = 1280
    modality_shared_expert_intermediate_size: int = 1280
    score_function: str = "sigmoid"
    normalize_routing_weights: bool = True
    routing_scale: float = 4.9
    layers: tuple[int, ...] = tuple(range(2, 38))

    def __post_init__(self) -> None:
        if self.num_heads <= 0 or self.num_experts <= 0:
            raise ValueError("MAGI-2 MoE head and expert counts must be positive")
        if not 0 < self.top_k <= self.num_experts:
            raise ValueError("MAGI-2 MoE top_k must be in [1, num_experts]")
        if self.score_function != "sigmoid":
            raise ValueError("The released MAGI-2 checkpoint uses sigmoid routing")


@dataclass(frozen=True)
class Magi2PreviewConfig:
    num_layers: int = 40
    hidden_size: int = 3072
    head_dim: int = 128
    num_query_groups: int = 24
    video_in_channels: int = 48
    audio_in_channels: int = 64
    text_in_channels: int = 5120
    intermediate_factor: float = 4.0
    multimodal_layers: tuple[int, ...] = (0, 1, 38, 39)
    params_dtype: torch.dtype = torch.bfloat16
    attention_softcap: float = -1.0
    attention_sink_tokens: int = 1
    attention_gating: bool = True
    mhc: Magi2MHCConfig = field(default_factory=Magi2MHCConfig)
    moe: Magi2MoEConfig = field(default_factory=Magi2MoEConfig)

    @property
    def num_attention_heads(self) -> int:
        if self.hidden_size % self.head_dim:
            raise ValueError("hidden_size must be divisible by head_dim")
        return self.hidden_size // self.head_dim

    @property
    def num_heads_q(self) -> int:
        return self.num_attention_heads

    @property
    def num_heads_kv(self) -> int:
        return self.num_query_groups

    @property
    def virtual_width(self) -> int:
        return self.hidden_size * self.mhc.num_streams

    def validate(self) -> None:
        if not self.mhc.enabled:
            raise ValueError("The released MAGI-2 Preview checkpoint requires mHC")
        if not self.attention_gating:
            raise ValueError("The released MAGI-2 Preview checkpoint requires attention gating")
        if self.attention_sink_tokens < 1:
            raise ValueError("The released MAGI-2 Preview checkpoint requires an attention sink")
        if self.hidden_size % self.moe.num_heads:
            raise ValueError("hidden_size must be divisible by the number of MoE heads")
        layer_ids = set(range(self.num_layers))
        if not set(self.multimodal_layers).issubset(layer_ids):
            raise ValueError("multimodal layer index is outside the transformer")
        if not set(self.moe.layers).issubset(layer_ids):
            raise ValueError("MoE layer index is outside the transformer")


@dataclass(frozen=True)
class Magi2GenerationConfig:
    duration_seconds: float = 10.0
    fps: float = 12.5
    output_frames: int = 125
    preview_steps: int = 100
    shift: float = 7.0
    video_guidance_scale: float = 5.0
    audio_guidance_scale: float = 7.0
    audio_latent_fps: float = 25.0
    video_vae_stride: tuple[int, int, int] = (8, 16, 16)
    patch_size: tuple[int, int, int] = (1, 1, 1)
    video_latent_channels: int = 48
    audio_latent_channels: int = 64
    audio_sample_rate: int = 44_100


@dataclass(frozen=True)
class Magi2RefinerWindowConfig:
    """Block-grid local attention of the refiner.

    Video tokens are grouped into ``block_size`` (t, h, w) blocks.  A query
    block attends every key block within ``radius`` blocks on each axis plus
    all text tokens; text queries attend every token.
    """

    block_size: tuple[int, int, int] = (8, 4, 4)
    radius: tuple[int, int, int] = (2, 2, 2)

    def __post_init__(self) -> None:
        if len(self.block_size) != 3 or len(self.radius) != 3:
            raise ValueError("MAGI-2 refiner window block_size and radius must have three (t, h, w) entries")
        if any(size <= 0 for size in self.block_size):
            raise ValueError(f"MAGI-2 refiner window block sizes must be positive, got {self.block_size}")
        if any(r < 0 for r in self.radius):
            raise ValueError(f"MAGI-2 refiner window radii must be non-negative, got {self.radius}")

    @property
    def dense_query_chunk(self) -> int:
        """Query chunk length of the dense text rows: one full block."""
        return math.prod(self.block_size)


@dataclass(frozen=True)
class Magi2RefinerConfig:
    """Architecture of the 1080p refiner DiT (dense, no mHC, MoE or sinks)."""

    num_layers: int = 30
    hidden_size: int = 4096
    head_dim: int = 128
    num_query_groups: int = 8
    video_in_channels: int = 48
    audio_in_channels: int = 64
    text_in_channels: int = 5120
    multimodal_layers: tuple[int, ...] = (0, 1, 28, 29)
    params_dtype: torch.dtype = torch.bfloat16
    attention_gating: bool = True
    norm_eps: float = 1e-6
    rope_temperature: float = 10000.0
    swiglu7_alpha: float = 1.702
    swiglu7_limit: float = 7.0
    window: Magi2RefinerWindowConfig = field(default_factory=Magi2RefinerWindowConfig)

    @property
    def num_attention_heads(self) -> int:
        if self.hidden_size % self.head_dim:
            raise ValueError("hidden_size must be divisible by head_dim")
        return self.hidden_size // self.head_dim

    @property
    def num_heads_q(self) -> int:
        return self.num_attention_heads

    @property
    def num_heads_kv(self) -> int:
        return self.num_query_groups

    @property
    def intermediate_size(self) -> int:
        """SwiGLU7 width; ``up_gate_proj`` produces twice this many features."""
        return int(self.hidden_size * 4 * 2 / 3) // 4 * 4

    @property
    def num_rope_bands(self) -> int:
        return self.head_dim // 8

    @property
    def rotary_dim(self) -> int:
        """Rotated dims per head: sin and cos of three axes, one per band."""
        return 6 * self.num_rope_bands

    def num_modalities(self, layer_idx: int) -> int:
        """Modality experts of a layer: video, audio and text in the multimodal layers."""
        return 3 if layer_idx in self.multimodal_layers else 1

    def validate(self) -> None:
        if self.num_heads_q % self.num_heads_kv:
            raise ValueError("query heads must be a multiple of key/value heads")
        if self.rotary_dim > self.head_dim or self.rotary_dim % 2:
            raise ValueError("MAGI-2 refiner rotary dims must be even and fit in head_dim")
        if not set(self.multimodal_layers).issubset(range(self.num_layers)):
            raise ValueError("multimodal layer index is outside the transformer")


@dataclass(frozen=True)
class Magi2RefinerGenerationConfig:
    """Sampling contract of the refiner stage that follows a 540p Preview."""

    num_inference_steps: int = 5
    shift: float = 5.0
    video_guidance_scale: float = 2.0
    noise_index: int = 220
    zero_snr_linear_start: float = 0.00085
    zero_snr_linear_end: float = 0.012
    zero_snr_num_timesteps: int = 1000
    video_vae_stride: tuple[int, int, int] = (4, 16, 16)
    fps: float = 25.0
    output_frames: int = 249


MAGI2_PREVIEW_CONFIG = Magi2PreviewConfig()
MAGI2_GENERATION_CONFIG = Magi2GenerationConfig()
MAGI2_REFINER_CONFIG = Magi2RefinerConfig()
MAGI2_REFINER_GENERATION_CONFIG = Magi2RefinerGenerationConfig()
