# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for MiniCPM-o 4.5 streaming audio cache compatibility."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.models.whisper.modeling_whisper import WhisperConfig

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMWhisperEncoder,
    _get_audio_cache_length,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_audio_cache_length_uses_current_cache_api() -> None:
    dynamic_cache = SimpleNamespace(get_seq_length=lambda: 7)
    encoder_decoder_cache = SimpleNamespace(self_attention_cache=dynamic_cache)

    assert _get_audio_cache_length(encoder_decoder_cache) == 7


def test_audio_cache_length_supports_legacy_cache() -> None:
    key = torch.zeros(1, 1, 5, 2)
    value = torch.zeros_like(key)

    assert _get_audio_cache_length(((key, value),)) == 5


def _tiny_encoder_config(attn_implementation: str) -> WhisperConfig:
    config = WhisperConfig(
        num_mel_bins=16,
        d_model=32,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_dim=64,
        max_source_positions=100,
        dropout=0.0,
        attention_dropout=0.0,
    )
    config._attn_implementation = attn_implementation
    return config


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_encoder_reuses_cache_across_streaming_chunks(attn_implementation: str) -> None:
    """Encode two chunks against the installed Whisper attention and cache classes.

    Exercise the real cache API that ``get_audio_embedding_streaming`` relies
    on, including cache identity and history reuse across chunks.
    """
    torch.manual_seed(0)
    config = _tiny_encoder_config(attn_implementation)
    encoder = MiniCPMWhisperEncoder(config).eval()

    first_frames, second_frames = 20, 14
    first_len = (first_frames - 1) // 2 + 1
    second_len = (second_frames - 1) // 2 + 1
    first_mel = torch.randn(1, config.num_mel_bins, first_frames)
    second_mel = torch.randn(1, config.num_mel_bins, second_frames)

    with torch.no_grad():
        first = encoder(
            first_mel,
            attention_mask=torch.zeros(1, 1, first_len, first_len),
            past_key_values=None,
            use_cache=True,
            output_hidden_states=True,
        )
        cache = first.past_key_values
        assert cache is not None
        assert _get_audio_cache_length(cache) == first_len

        second = encoder(
            second_mel,
            attention_mask=torch.zeros(1, 1, second_len, first_len + second_len),
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
        )
        assert second.past_key_values is cache
        assert _get_audio_cache_length(cache) == first_len + second_len

        stateless = encoder(
            second_mel,
            attention_mask=torch.zeros(1, 1, second_len, second_len),
            past_key_values=None,
            use_cache=True,
            output_hidden_states=True,
        )

    assert torch.isfinite(second.last_hidden_state).all()
    assert not torch.allclose(second.last_hidden_state, stateless.last_hidden_state)
