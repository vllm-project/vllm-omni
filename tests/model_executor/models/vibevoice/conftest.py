# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.vibevoice.vibevoice import VibeVoiceModel


class _NoisyTower(torch.nn.Module):
    """Stands in for the sampled VAE: latents = waveform + randn noise, drawn like the real encoder."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def encode(self, values, sample=True):
        return SimpleNamespace(latents=(values + torch.randn_like(values)).transpose(1, 2))  # (B, tokens, 1)


@pytest.fixture
def reference_encoder():
    """Build ``VibeVoiceModel.reference_embeddings`` over a stub tower on ``device`` (2 samples per token)."""

    def build(device="cpu"):
        stub = SimpleNamespace(
            samples_per_token=2,
            audio_tower=_NoisyTower().to(device),
            latent_bias_factor=0.0,
            latent_scaling_factor=1.0,
            multi_modal_projector=lambda x: x,
        )
        stub._encode_references = lambda rows, counts: VibeVoiceModel._encode_references(stub, rows, counts)
        return lambda *args: VibeVoiceModel.reference_embeddings(stub, *args)

    return build
