# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for VibeVoice waveform decode and semantic feedback."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.vibevoice.audio_decode import (
    VibeVoiceAudioTokenDecoder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Cache:
    def __init__(self) -> None:
        self.calls = 0


class _FakeAudioTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.last_latent: torch.Tensor | None = None
        self.last_cache = None

    def decode(self, latents, padding_cache=None, use_cache=False):
        assert use_cache is True
        self.last_latent = latents.clone()
        self.last_cache = padding_cache
        cache = padding_cache or _Cache()
        cache.calls += 1
        audio = latents.sum(dim=-1, keepdim=True).repeat(1, 1, 4)
        return SimpleNamespace(audio=audio, padding_cache=cache)


class _FakeSemanticEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.last_audio: torch.Tensor | None = None
        self.last_cache = None

    def forward(self, audio, padding_cache=None, use_cache=False):
        assert use_cache is True
        self.last_audio = audio.clone()
        self.last_cache = padding_cache
        cache = padding_cache or _Cache()
        cache.calls += 1
        mean = audio.mean(dim=-1, keepdim=False).unsqueeze(-1)
        latents = torch.cat([mean, mean + 1, mean + 2], dim=-1)
        return SimpleNamespace(latents=latents, padding_cache=cache)


class _FakeAcousticProjector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def forward(self, latent):
        return torch.cat([latent, latent], dim=-1)


class _FakeSemanticConnector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def forward(self, latent):
        return torch.cat([latent, latent[..., :1]], dim=-1)


def _decoder() -> VibeVoiceAudioTokenDecoder:
    config = SimpleNamespace(
        hidden_size=4,
        audio_config=SimpleNamespace(
            hidden_size=2,
            decoder_config=SimpleNamespace(
                channels=1,
                upsampling_ratios=[2, 2],
            ),
        ),
        semantic_model_config=SimpleNamespace(hidden_size=3),
    )
    return VibeVoiceAudioTokenDecoder.from_model_config(config)


def _modules():
    return (
        _FakeAudioTower(),
        _FakeSemanticEncoder(),
        _FakeAcousticProjector(),
        _FakeSemanticConnector(),
    )


def test_audio_token_decoder_applies_official_formula_and_threads_caches() -> None:
    decoder = _decoder()
    audio_tower, semantic_encoder, acoustic_projector, semantic_connector = _modules()
    latent = torch.tensor([[[2.0, 4.0]]])
    original_latent = latent.clone()
    scale = torch.tensor(2.0)
    bias = torch.tensor(-1.0)

    first = decoder.decode_audio_token(
        audio_tower=audio_tower,
        semantic_encoder=semantic_encoder,
        acoustic_projector=acoustic_projector,
        semantic_connector=semantic_connector,
        latent_scaling_factor=scale,
        latent_bias_factor=bias,
        audio_latent=latent,
    )

    # Decoder uses inverse-normalized latent; feedback projector uses the
    # original diffusion latent.
    assert torch.equal(audio_tower.last_latent, torch.tensor([[[2.0, 3.0]]]))
    assert torch.equal(first.audio, torch.full((1, 1, 4), 5.0))
    assert torch.equal(semantic_encoder.last_audio, first.audio)
    assert torch.equal(first.semantic_latent, torch.tensor([[[5.0, 6.0, 7.0]]]))
    assert torch.equal(first.next_embedding, torch.tensor([[[7.0, 10.0, 9.0, 9.0]]]))
    assert first.acoustic_cache is not first.semantic_cache
    assert first.acoustic_cache.calls == 1
    assert first.semantic_cache.calls == 1
    assert torch.equal(latent, original_latent)

    second = decoder.decode_audio_token(
        audio_tower=audio_tower,
        semantic_encoder=semantic_encoder,
        acoustic_projector=acoustic_projector,
        semantic_connector=semantic_connector,
        latent_scaling_factor=scale,
        latent_bias_factor=bias,
        audio_latent=latent,
        acoustic_cache=first.acoustic_cache,
        semantic_cache=first.semantic_cache,
    )
    assert audio_tower.last_cache is first.acoustic_cache
    assert semantic_encoder.last_cache is first.semantic_cache
    assert second.acoustic_cache is first.acoustic_cache
    assert second.semantic_cache is first.semantic_cache
    assert second.acoustic_cache.calls == 2
    assert second.semantic_cache.calls == 2


def test_audio_token_decoder_config_uses_decoder_upsampling_product() -> None:
    decoder = _decoder()
    assert decoder.latent_size == 2
    assert decoder.semantic_size == 3
    assert decoder.condition_size == 4
    assert decoder.audio_channels == 1
    assert decoder.samples_per_token == 4


@pytest.mark.parametrize(
    ("latent", "message"),
    [
        (torch.zeros(1, 2), "audio_latent must have shape"),
        (torch.zeros(0, 1, 2), "batch cannot be empty"),
        (torch.zeros(1, 2, 2), "audio_latent must have shape"),
        (torch.zeros(1, 1, 3), "audio_latent must have shape"),
        (torch.zeros(1, 1, 2, dtype=torch.long), "must be a floating-point"),
    ],
)
def test_audio_token_decoder_rejects_invalid_latent_contracts(
    latent: torch.Tensor,
    message: str,
) -> None:
    audio_tower, semantic_encoder, acoustic_projector, semantic_connector = _modules()
    with pytest.raises((TypeError, ValueError), match=message):
        _decoder().decode_audio_token(
            audio_tower=audio_tower,
            semantic_encoder=semantic_encoder,
            acoustic_projector=acoustic_projector,
            semantic_connector=semantic_connector,
            latent_scaling_factor=torch.tensor(1.0),
            latent_bias_factor=torch.tensor(0.0),
            audio_latent=latent,
        )


def test_audio_token_decoder_rejects_missing_causal_caches() -> None:
    class _NoCacheAudioTower(_FakeAudioTower):
        def decode(self, latents, padding_cache=None, use_cache=False):
            output = super().decode(latents, padding_cache, use_cache)
            output.padding_cache = None
            return output

    _, semantic_encoder, acoustic_projector, semantic_connector = _modules()
    with pytest.raises(ValueError, match="did not return a causal padding cache"):
        _decoder().decode_audio_token(
            audio_tower=_NoCacheAudioTower(),
            semantic_encoder=semantic_encoder,
            acoustic_projector=acoustic_projector,
            semantic_connector=semantic_connector,
            latent_scaling_factor=torch.tensor(1.0),
            latent_bias_factor=torch.tensor(0.0),
            audio_latent=torch.zeros(1, 1, 2),
        )
