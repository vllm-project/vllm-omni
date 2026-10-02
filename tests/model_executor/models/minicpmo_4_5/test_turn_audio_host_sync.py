# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Turn-mode (full-utterance) Stage-0 audio: ``get_audio_hidden_states`` over a ragged batch.

The assembly loop reads the pooled lengths back once (``.tolist()``) instead of
slicing with a device scalar per row; each row must keep its predicted length.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.whisper.modeling_whisper import WhisperConfig

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMForConditionalGeneration,
    MiniCPMWhisperEncoder,
    MultiModalProjector,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

N_MELS = 16
POOL = 5


class _Thinker:
    """The audio half of the thinker: real modules, the real method under test."""

    get_audio_hidden_states = MiniCPMO45OmniLLMForConditionalGeneration.get_audio_hidden_states
    subsequent_chunk_mask = MiniCPMO45OmniLLMForConditionalGeneration.subsequent_chunk_mask
    _get_feat_extract_output_lengths = MiniCPMO45OmniLLMForConditionalGeneration._get_feat_extract_output_lengths

    def __init__(self, *, attn_implementation: str = "sdpa", chunk_length: float = 0.0) -> None:
        torch.manual_seed(0)
        config = WhisperConfig(
            num_mel_bins=N_MELS,
            d_model=32,
            encoder_layers=2,
            encoder_attention_heads=2,
            encoder_ffn_dim=128,  # the projection takes ffn // 4 == d_model, matching the checkpoint
            max_source_positions=200,
            dropout=0.0,
            attention_dropout=0.0,
        )
        config._attn_implementation = attn_implementation
        self.apm = MiniCPMWhisperEncoder(config).eval()
        self.audio_projection_layer = MultiModalProjector(in_dim=config.encoder_ffn_dim // 4, out_dim=24)
        self.audio_avg_pooler = nn.AvgPool1d(POOL, stride=POOL)
        self.audio_encoder_layer = -1
        self.config = SimpleNamespace(audio_pool_step=POOL, audio_chunk_length=chunk_length)


def _feats(frames: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((N_MELS, frames), generator=generator)


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_batched_multi_request_matches_per_row_reference(attn_implementation: str) -> None:
    """Several ragged-length audios in one call (mirrors several concurrent
    requests grouped into one encoder call) assemble to rows of the length
    ``_get_feat_extract_output_lengths`` predicts."""
    thinker = _Thinker(attn_implementation=attn_implementation)
    lens = [131, 214, 98, 172]
    feats = [_feats(n, seed=i) for i, n in enumerate(lens)]
    data = {
        "audio_features": feats,
        "audio_feature_lens": [torch.tensor([n]) for n in lens],
    }
    with torch.no_grad():
        outputs = thinker.get_audio_hidden_states(data)

    assert len(outputs) == len(lens)
    for out, n in zip(outputs, lens, strict=True):
        _, expected_len = thinker._get_feat_extract_output_lengths(torch.tensor([n]))
        assert out.shape[0] == int(expected_len.item())
