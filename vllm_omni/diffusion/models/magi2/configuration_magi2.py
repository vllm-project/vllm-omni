# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Native configuration for the released MAGI-2 Preview checkpoint.

The defaults mirror ``sand-ai/MAGI-2-preview``.  Keeping the architecture in
vLLM-Omni makes model construction independent of SandAI's Python package and
also gives tests a small-config entry point.
"""

from __future__ import annotations

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


MAGI2_PREVIEW_CONFIG = Magi2PreviewConfig()
MAGI2_GENERATION_CONFIG = Magi2GenerationConfig()
