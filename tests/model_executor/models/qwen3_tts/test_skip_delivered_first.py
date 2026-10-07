# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Code2Wav drops only frames explicitly marked as delivered by the Talker."""

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

UPSAMPLE = 4


def _decoder(skip: bool):
    decoder = Qwen3TTSTokenizerV2Decoder.__new__(Qwen3TTSTokenizerV2Decoder)
    decoder.__dict__["capture_first_audio_state_only"] = skip
    decoder.__dict__["total_upsample"] = UPSAMPLE
    calls = []

    def full(codes, caches):
        calls.append("full")
        frames = int(codes.shape[-1])
        return torch.arange(frames * UPSAMPLE, dtype=torch.float32).view(1, 1, -1)

    def state_only(codes, caches):
        calls.append("state")
        return torch.zeros(1)

    decoder.__dict__["_decode_xvec_first_chunk"] = full
    decoder.__dict__["_decode_xvec_first_chunk_state_only"] = state_only
    return decoder, calls


def test_one_frame_first_chunk_only_builds_state():
    decoder, calls = _decoder(skip=True)
    out = decoder._decode_stream_first_chunk(torch.zeros(1, 16, 1, dtype=torch.long), {"skip_first_audio": True})
    assert out.shape[-1] == 0 and calls == ["state"]


def test_longer_first_chunk_keeps_all_but_the_delivered_frame():
    decoder, calls = _decoder(skip=True)
    out = decoder._decode_stream_first_chunk(
        torch.zeros(1, 16, 3, dtype=torch.long),
        {"skip_first_audio": True},
    )
    assert calls == ["full"]
    assert torch.equal(out, torch.arange(UPSAMPLE, 3 * UPSAMPLE, dtype=torch.float32).view(1, 1, -1))


def test_without_skip_the_whole_first_chunk_is_decoded():
    decoder, calls = _decoder(skip=False)
    out = decoder._decode_stream_first_chunk(
        torch.zeros(1, 16, 3, dtype=torch.long),
        {"skip_first_audio": False},
    )
    assert calls == ["full"] and out.shape[-1] == 3 * UPSAMPLE


def test_request_without_first_audio_uses_regular_decoder_even_when_graph_skip_is_enabled():
    decoder, calls = _decoder(skip=True)
    out = decoder._decode_stream_first_chunk(torch.zeros(1, 16, 1, dtype=torch.long), {})
    assert calls == ["full"] and out.shape[-1] == UPSAMPLE
