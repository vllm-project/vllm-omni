# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MiniCPM-o 4.5 streaming audio cache compatibility."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.whisper.modeling_whisper import WhisperConfig

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    WHISPER_ATTENTION_CLASSES,
    MiniCPMO45OmniLLMForConditionalGeneration,
    MiniCPMWhisperEncoder,
    MiniCPMWhisperEncoderLayer,
    MultiModalProjector,
    _get_audio_cache_length,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _LegacyWhisperAttention(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.seen_cache = None

    def forward(self, hidden_states, past_key_value=None, **kwargs):
        self.seen_cache = past_key_value
        return hidden_states, None, past_key_value


class _CurrentWhisperAttention(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.seen_cache = None

    def forward(self, hidden_states, past_key_values=None, **kwargs):
        self.seen_cache = past_key_values
        return hidden_states, None


@pytest.mark.parametrize(
    ("implementation", "attention_cls"),
    [
        ("test_legacy_cache", _LegacyWhisperAttention),
        ("test_current_cache", _CurrentWhisperAttention),
    ],
)
def test_whisper_attention_preserves_streaming_cache(
    monkeypatch: pytest.MonkeyPatch,
    implementation: str,
    attention_cls: type[nn.Module],
) -> None:
    monkeypatch.setitem(WHISPER_ATTENTION_CLASSES, implementation, attention_cls)
    config = WhisperConfig(
        d_model=4,
        encoder_attention_heads=1,
        encoder_ffn_dim=8,
        encoder_layers=1,
    )
    config._attn_implementation = implementation
    layer = MiniCPMWhisperEncoderLayer(config, layer_idx=0)
    cache = object()

    outputs = layer(
        torch.randn(1, 2, config.d_model),
        attention_mask=None,
        layer_head_mask=None,
        past_key_values=cache,
        use_cache=True,
    )

    assert layer.self_attn.seen_cache is cache
    assert outputs[-1] is cache


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

    Unlike the stub-based tests above, this drives the real transformers cache
    API that ``get_audio_embedding_streaming`` relies on, so it fails whenever a
    removed cache method or renamed attention kwarg breaks streaming reuse.
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
        assert _get_audio_cache_length(second.past_key_values) == first_len + second_len

        stateless = encoder(
            second_mel,
            attention_mask=torch.zeros(1, 1, second_len, second_len),
            past_key_values=None,
            use_cache=True,
            output_hidden_states=True,
        )

    assert torch.isfinite(second.last_hidden_state).all()
    assert not torch.allclose(second.last_hidden_state, stateless.last_hidden_state)


class _StreamingAPMHarness(nn.Module):
    """Bind the production streaming method onto a minimal audio stack.

    ``get_audio_embedding_streaming`` only touches the APM encoder, the
    projector/pooler, and ``audio_past_key_values``; binding the unmodified
    method onto this harness drives the exact production code path on CPU.
    """

    get_audio_embedding_streaming = MiniCPMO45OmniLLMForConditionalGeneration.get_audio_embedding_streaming
    _get_feat_extract_output_lengths = MiniCPMO45OmniLLMForConditionalGeneration._get_feat_extract_output_lengths

    def __init__(self, max_source_positions: int = 1500, pool_step: int = 5):
        super().__init__()
        config = WhisperConfig(
            num_mel_bins=16,
            d_model=32,
            encoder_layers=2,
            encoder_attention_heads=2,
            encoder_ffn_dim=128,
            max_source_positions=max_source_positions,
            dropout=0.0,
            attention_dropout=0.0,
        )
        config._attn_implementation = "sdpa"
        self.apm = MiniCPMWhisperEncoder(config)
        self.audio_avg_pooler = nn.AvgPool1d(pool_step, stride=pool_step)
        self.audio_projection_layer = MultiModalProjector(in_dim=32, out_dim=32)
        self.audio_encoder_layer = -1
        self.audio_past_key_values = None
        self.config = SimpleNamespace(audio_pool_step=pool_step)


def test_streaming_audio_cache_resets_once_at_apm_position_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """47 streamed one-second units cross the 1500-position APM boundary once.

    Official MiniCPM-o 4.5 (checkpoint modeling_minicpmo.py and the unified
    duplex demo) hard-resets ``audio_past_key_values`` at the same boundary:
    the APM's learned absolute positions cap the audio encoder at ~30s of
    context, after which the cache rebuilds from scratch. This locks the
    official alignment in: exactly one reset, the cache never exceeds the
    position limit, and every post-reset chunk still yields the expected
    finite embeddings.
    """
    torch.manual_seed(0)
    model = _StreamingAPMHarness().eval()
    reset_warnings: list[str] = []
    monkeypatch.setattr(
        "vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm.logger.warning",
        lambda *args, **kwargs: reset_warnings.append(str(args)),
    )

    chunk_mel_frames = 104  # 2 prefix + 100 core + 2 suffix mel frames per 1s unit
    per_chunk_tokens = 10  # 100 core conv frames, pool_step 5
    num_chunks = 47  # >45s of streamed audio

    cache_lengths: list[int] = []
    with torch.no_grad():
        for chunk_idx in range(num_chunks):
            data = {
                "audio_features": torch.randn(1, 16, chunk_mel_frames),
                "audio_feature_lens": [torch.tensor([chunk_mel_frames])],
            }
            embeds = model.get_audio_embedding_streaming(
                data,
                use_extra_context=True,
                prefix_extra_frames=0 if chunk_idx == 0 else 2,
                suffix_extra_frames=2,
            )
            cache_lengths.append(_get_audio_cache_length(model.audio_past_key_values))

            assert len(embeds) == 1
            chunk = embeds[0][0]
            assert chunk.shape[0] == per_chunk_tokens
            assert torch.isfinite(chunk).all()

    # Chunk 0 appends 51 net frames, later chunks 50 each: the boundary fires
    # when the accumulated cache plus the incoming chunk would reach 1500
    # (1451 + 50), i.e. once, at the ~30s mark. The cache then rebuilds from
    # that chunk and keeps growing for the remaining ~17s.
    assert cache_lengths[0] == 51
    assert cache_lengths[28] == 51 + 28 * 50
    assert cache_lengths[29] == 50
    assert cache_lengths[-1] == 50 + (num_chunks - 30) * 50
    assert max(cache_lengths) < 1500
    assert len(reset_warnings) == 1
