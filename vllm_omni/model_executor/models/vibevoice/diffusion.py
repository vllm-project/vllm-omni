# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VibeVoice diffusion head and per-audio-token latent sampling.

Module and parameter names follow the native HF checkpoint
(``vibevoice/VibeVoice-1.5B-hf``) so weights load without renaming. A fresh
DPM solver is built per audio token, so no mutable solver state is shared.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.vibevoice_acoustic_tokenizer.modeling_vibevoice_acoustic_tokenizer import (
    VibeVoiceAcousticTokenizerRMSNorm,
)

if TYPE_CHECKING:
    from diffusers import DPMSolverMultistepScheduler

    from vllm_omni.transformers_utils.configs.vibevoice import VibeVoiceDiffusionHeadConfig

# Official solver contract from generation_config.json of the native
# checkpoint; the checkpoint stores it outside config.json.
VIBEVOICE_DDPM_NUM_STEPS = 1000
VIBEVOICE_DDPM_BETA_SCHEDULE = "squaredcos_cap_v2"
VIBEVOICE_DDPM_PREDICTION_TYPE = "v_prediction"
VIBEVOICE_DEFAULT_GUIDANCE_SCALE = 1.3
VIBEVOICE_DEFAULT_NUM_DIFFUSION_STEPS = 10


class VibeVoiceDiffusionHeadSinusoidalEmbedding(nn.Module):
    def __init__(self, config: VibeVoiceDiffusionHeadConfig) -> None:
        super().__init__()
        dim = config.frequency_embedding_size // 2
        freq = torch.exp(-math.log(config.diffusion_max_period) * torch.arange(dim, dtype=torch.float32) / dim)
        self.frequency_embedding_size = config.frequency_embedding_size
        self.register_buffer("freq", freq, persistent=False)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        args = timesteps[:, None].float() * self.freq[None].to(timesteps.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = nn.functional.pad(embedding, (0, 1))
        return embedding


class VibeVoiceDiffusionHeadMLP(nn.Module):
    def __init__(self, config: VibeVoiceDiffusionHeadConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.frequency_embedding_size, config.hidden_size, bias=False)
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden_states)))


class VibeVoiceMLP(nn.Module):
    def __init__(self, config: VibeVoiceDiffusionHeadConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class VibeVoiceDiffusionHeadAdaLayerNorm(nn.Module):
    def __init__(self, config: VibeVoiceDiffusionHeadConfig) -> None:
        super().__init__()
        self.ffn = VibeVoiceMLP(config)
        self.norm = VibeVoiceAcousticTokenizerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.act_fn = ACT2FN[config.hidden_act]
        self.linear = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)

    def forward(self, hidden_states: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.linear(self.act_fn(condition)).chunk(3, dim=-1)
        modulated = self.norm(hidden_states) * (1 + scale) + shift
        return hidden_states + gate * self.ffn(modulated)


class VibeVoiceDiffusionHeadFinalLayer(nn.Module):
    def __init__(self, config: VibeVoiceDiffusionHeadConfig, output_size: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.linear_2 = nn.Linear(config.hidden_size, output_size, bias=False)
        self.rms_norm_eps = config.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.linear_1(self.act_fn(condition)).chunk(2, dim=-1)
        normed = nn.functional.rms_norm(hidden_states, (hidden_states.shape[-1],), eps=self.rms_norm_eps)
        return self.linear_2(normed * (1 + scale) + shift)


class VibeVoiceDiffusionHead(nn.Module):
    """One learned denoising prediction at a single timestep."""

    def __init__(self, config: VibeVoiceDiffusionHeadConfig) -> None:
        super().__init__()
        self.noisy_images_proj = nn.Linear(config.latent_size, config.hidden_size, bias=False)
        self.cond_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.timestep_embedding = VibeVoiceDiffusionHeadSinusoidalEmbedding(config)
        self.timestep_proj = VibeVoiceDiffusionHeadMLP(config)
        self.layers = nn.ModuleList(VibeVoiceDiffusionHeadAdaLayerNorm(config) for _ in range(config.num_hidden_layers))
        self.final_layer = VibeVoiceDiffusionHeadFinalLayer(config, output_size=config.latent_size)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.noisy_images_proj(noisy_latents)
        timestep_embedding = self.timestep_embedding(timesteps).to(condition.dtype)
        timestep_features = self.timestep_proj(timestep_embedding)
        condition = self.cond_proj(condition) + timestep_features
        for layer in self.layers:
            hidden_states = layer(hidden_states, condition)
        return self.final_layer(hidden_states, condition)


def create_dpm_scheduler() -> DPMSolverMultistepScheduler:
    """Return a fresh official-equivalent DPM solver; solver state is mutable."""
    from diffusers import DPMSolverMultistepScheduler

    return DPMSolverMultistepScheduler(
        num_train_timesteps=VIBEVOICE_DDPM_NUM_STEPS,
        beta_schedule=VIBEVOICE_DDPM_BETA_SCHEDULE,
        prediction_type=VIBEVOICE_DDPM_PREDICTION_TYPE,
    )


__all__ = [
    "VIBEVOICE_DEFAULT_GUIDANCE_SCALE",
    "VIBEVOICE_DEFAULT_NUM_DIFFUSION_STEPS",
    "VibeVoiceDiffusionHead",
    "create_dpm_scheduler",
]
