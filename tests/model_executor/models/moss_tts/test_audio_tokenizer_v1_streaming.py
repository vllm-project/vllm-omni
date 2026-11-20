# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer import (
    MossAudioTokenizerConfig,
    MossAudioTokenizerModel,
    _Attention,
    _StreamingExecutionContext,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_codec() -> MossAudioTokenizerModel:
    torch.manual_seed(0)
    config = MossAudioTokenizerConfig(
        sampling_rate=100,
        downsample_rate=1,
        causal_transformer_context_duration=0.08,
        encoder_kwargs=[{"module_type": "PatchedPretransform", "patch_size": 1}],
        decoder_kwargs=[
            {
                "module_type": "Transformer",
                "input_dimension": 4,
                "output_dimension": 1,
                "d_model": 4,
                "num_heads": 2,
                "num_layers": 2,
                "dim_feedforward": 8,
                "causal": True,
                "norm": "layer_norm",
                "positional_embedding": "rope",
                "max_period": 10_000,
                "gating": "none",
                "layer_scale": 0.01,
                "conv_layout": True,
            }
        ],
        quantizer_kwargs={
            "input_dim": 4,
            "rvq_dim": 4,
            "output_dim": 4,
            "num_quantizers": 2,
            "codebook_size": 8,
            "codebook_dim": 2,
            "quantizer_type": "rlfq",
        },
    )
    return MossAudioTokenizerModel(config).eval()


def _codes(values: list[list[int]]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long)


def _reference_audio(codec: MossAudioTokenizerModel, codes: torch.Tensor) -> torch.Tensor:
    output = codec.batch_decode([codes], num_quantizers=int(codes.shape[0]))
    assert output.audio is not None
    return output.audio[0]


def _stream(
    codec: MossAudioTokenizerModel,
    codes: torch.Tensor,
    *,
    slot: int,
) -> torch.Tensor:
    output = codec.decode_streaming_batch(
        codes.unsqueeze(1),
        torch.tensor([codes.shape[1]], dtype=torch.long),
        torch.tensor([slot], dtype=torch.long),
        torch.tensor([True]),
    )
    assert output.audio is not None
    return output.audio[0]


def test_streaming_chunks_match_full_decode() -> None:
    codec = _make_codec()
    codes = _codes([[1, 2, 3, 4], [4, 3, 2, 1]])
    expected = _reference_audio(codec, codes)

    codec.initialize_decoder_state_pool(state_capacity=2)
    first = _stream(codec, codes[:, :2], slot=1)
    second = _stream(codec, codes[:, 2:], slot=1)

    torch.testing.assert_close(torch.cat([first, second], dim=-1), expected, atol=1e-5, rtol=1e-5)


def test_streaming_attention_ring_wrap_matches_full_attention() -> None:
    torch.manual_seed(0)
    attention = _Attention(
        embed_dim=8,
        num_heads=2,
        causal=True,
        max_period=10_000,
        context=8,
    ).eval()
    hidden_states = torch.randn(1, 12, 8)
    expected = attention(hidden_states)

    attention.initialize_streaming_state(torch.zeros(1, dtype=torch.long))
    execution_context = _StreamingExecutionContext(
        state_slot_ids=torch.tensor([0]),
        valid_rows=torch.tensor([True]),
    )
    chunks = [
        attention(hidden_states[:, start : start + 3], execution_context)
        for start in range(0, hidden_states.shape[1], 3)
    ]

    torch.testing.assert_close(torch.cat(chunks, dim=1), expected, atol=1e-6, rtol=1e-6)


def test_streaming_slots_keep_independent_state() -> None:
    codec = _make_codec()
    first_codes = _codes([[1, 2, 3, 4], [4, 3, 2, 1]])
    second_codes = _codes([[5, 6, 7, 1], [1, 7, 6, 5]])
    first_expected = _reference_audio(codec, first_codes)
    second_expected = _reference_audio(codec, second_codes)

    codec.initialize_decoder_state_pool(state_capacity=3)
    chunks: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in (0, 2):
        batch = torch.stack([first_codes[:, start : start + 2], second_codes[:, start : start + 2]], dim=1)
        output = codec.decode_streaming_batch(
            batch,
            torch.tensor([2, 2], dtype=torch.long),
            torch.tensor([2, 0], dtype=torch.long),
            torch.tensor([True, True]),
        )
        assert output.audio is not None
        chunks.append((output.audio[0], output.audio[1]))

    torch.testing.assert_close(
        torch.cat([chunk[0] for chunk in chunks], dim=-1),
        first_expected,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        torch.cat([chunk[1] for chunk in chunks], dim=-1),
        second_expected,
        atol=1e-5,
        rtol=1e-5,
    )


def test_reset_slot_removes_previous_request_state() -> None:
    codec = _make_codec()
    old_codes = _codes([[1, 2], [4, 3]])
    new_codes = _codes([[7, 6], [2, 5]])
    expected = _reference_audio(codec, new_codes)

    codec.initialize_decoder_state_pool(state_capacity=1)
    _stream(codec, old_codes, slot=0)
    codec.reset_decoder_state_slots(torch.tensor([0], dtype=torch.long))

    torch.testing.assert_close(_stream(codec, new_codes, slot=0), expected, atol=1e-5, rtol=1e-5)


def test_invalid_padding_row_does_not_advance_state() -> None:
    codec = _make_codec()
    codes = _codes([[1, 2], [4, 3]])
    expected = _reference_audio(codec, codes)

    codec.initialize_decoder_state_pool(state_capacity=1, scratch_capacity=1)
    padding_output = codec.decode_streaming_batch(
        codes.unsqueeze(1),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([False]),
    )
    assert padding_output.audio_lengths is not None
    assert int(padding_output.audio_lengths[0]) == 0

    torch.testing.assert_close(_stream(codec, codes, slot=1), expected, atol=1e-5, rtol=1e-5)


def test_streaming_requires_initialized_state_pool() -> None:
    codec = _make_codec()
    codes = _codes([[1], [2]])

    with pytest.raises(RuntimeError, match="state pool is not initialized"):
        _stream(codec, codes, slot=0)


def test_closed_state_pool_can_be_reinitialized() -> None:
    codec = _make_codec()
    codes = _codes([[1], [2]])

    codec.initialize_decoder_state_pool(state_capacity=1)
    codec.close_decoder_state_pool()
    codec.initialize_decoder_state_pool(state_capacity=1)

    assert _stream(codec, codes, slot=0).numel() > 0
