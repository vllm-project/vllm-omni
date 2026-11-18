# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Tests for batched audio input parsing, decoding and context-frame cutting.

These functions parse the [ctx_frames, context_length, ...tokens] wire format
produced by generator2tokenizer_async_chunk and cut leading context samples
from decoded audio arrays.
"""

import functools

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@functools.lru_cache(maxsize=1)
def _voxtral_tts_parsers():
    # ``vllm_omni...voxtral_tts`` does ``from vllm.model_executor.models.utils import ...``.
    # If that module was first pulled in under a different import context in the
    # same process, ``direct_register_custom_op`` for ``sequence_parallel_chunk_impl``
    # can run twice.  Match ``tests.model_executor.helpers`` + pin ``utils`` first.
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

    bootstrap_vllm_layer_custom_op_modules()
    import vllm.model_executor.models.utils  # noqa: F401

    from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts import (
        apply_ctx_frames_cutting,
        parse_batched_audio_input,
    )

    return parse_batched_audio_input, apply_ctx_frames_cutting


NUM_CODEBOOKS = 37


# ──────────────────────────────────────────────────────────────────
# parse_batched_audio_input tests
# ──────────────────────────────────────────────────────────────────


def _make_input_ids(*requests: tuple[int, int, torch.Tensor]) -> torch.Tensor:
    """Build a flat input_ids tensor from (ctx_frames, context_length, tokens) tuples."""
    parts = []
    for ctx_frames, context_length, tokens in requests:
        parts.append(torch.tensor([ctx_frames, context_length]))
        parts.append(tokens.flatten())
    return torch.cat(parts)


def test_parse_single_request():
    """Single request is parsed correctly."""
    parse_batched_audio_input, _ = _voxtral_tts_parsers()
    ctx_frames = 3
    context_length = 5
    total_frames = ctx_frames + context_length
    tokens = torch.arange(total_frames * NUM_CODEBOOKS).reshape(total_frames, NUM_CODEBOOKS)
    input_ids = _make_input_ids((ctx_frames, context_length, tokens))

    all_audio_tokens, all_ctx_frames = parse_batched_audio_input(input_ids, num_codebooks=NUM_CODEBOOKS)

    assert len(all_audio_tokens) == 1
    assert len(all_ctx_frames) == 1
    assert all_ctx_frames[0] == ctx_frames
    assert all_audio_tokens[0].shape == (total_frames, NUM_CODEBOOKS)
    torch.testing.assert_close(all_audio_tokens[0], tokens)


def test_parse_multiple_requests():
    """Multiple requests in a batch are parsed correctly."""
    parse_batched_audio_input, _ = _voxtral_tts_parsers()
    req1 = (2, 4, torch.ones((6, NUM_CODEBOOKS), dtype=torch.long))
    req2 = (0, 3, torch.full((3, NUM_CODEBOOKS), 7, dtype=torch.long))
    input_ids = _make_input_ids(req1, req2)

    all_audio_tokens, all_ctx_frames = parse_batched_audio_input(input_ids, num_codebooks=NUM_CODEBOOKS)

    assert len(all_audio_tokens) == 2
    assert all_ctx_frames == [2, 0]
    assert all_audio_tokens[0].shape == (6, NUM_CODEBOOKS)
    assert all_audio_tokens[1].shape == (3, NUM_CODEBOOKS)
    assert torch.all(all_audio_tokens[0] == 1)
    assert torch.all(all_audio_tokens[1] == 7)


def test_parse_zero_ctx_frames():
    """ctx_frames=0 means no context, just chunk data."""
    parse_batched_audio_input, _ = _voxtral_tts_parsers()
    ctx_frames = 0
    context_length = 10
    tokens = torch.randint(0, 100, (context_length, NUM_CODEBOOKS))
    input_ids = _make_input_ids((ctx_frames, context_length, tokens))

    all_audio_tokens, all_ctx_frames = parse_batched_audio_input(input_ids, num_codebooks=NUM_CODEBOOKS)

    assert all_ctx_frames[0] == 0
    assert all_audio_tokens[0].shape == (context_length, NUM_CODEBOOKS)


def test_parse_fails_on_misaligned_input():
    """Input not divisible by num_codebooks should raise AssertionError."""
    parse_batched_audio_input, _ = _voxtral_tts_parsers()
    # 2 header elements + 5 data elements (not divisible by 37)
    input_ids = torch.tensor([0, 1] + [0] * 5)
    with pytest.raises(AssertionError, match="divisible by"):
        parse_batched_audio_input(input_ids, num_codebooks=NUM_CODEBOOKS)


def test_parse_custom_num_codebooks():
    """Works with a non-default num_codebooks value."""
    parse_batched_audio_input, _ = _voxtral_tts_parsers()
    cb = 4
    ctx_frames = 1
    context_length = 2
    total_frames = ctx_frames + context_length
    tokens = torch.arange(total_frames * cb).reshape(total_frames, cb)
    input_ids = _make_input_ids((ctx_frames, context_length, tokens))

    all_audio_tokens, all_ctx_frames = parse_batched_audio_input(input_ids, num_codebooks=cb)

    assert all_audio_tokens[0].shape == (total_frames, cb)
    torch.testing.assert_close(all_audio_tokens[0], tokens)


# ──────────────────────────────────────────────────────────────────
# apply_ctx_frames_cutting tests
# ──────────────────────────────────────────────────────────────────


def test_cut_removes_leading_samples():
    """Context frames are removed from the front of the audio array."""
    _, apply_ctx_frames_cutting = _voxtral_tts_parsers()
    downsample_factor = 240
    ctx_frames = 5
    total_samples = 2400  # 10 frames * 240
    audio = torch.arange(total_samples, dtype=torch.float32)

    result = apply_ctx_frames_cutting([audio], [ctx_frames], downsample_factor)

    expected_cut = ctx_frames * downsample_factor  # 1200
    assert len(result) == 1
    assert result[0].shape[0] == total_samples - expected_cut
    torch.testing.assert_close(result[0], audio[expected_cut:])


def test_cut_zero_ctx_frames_unchanged():
    """ctx_frames=0 leaves the audio unchanged."""
    _, apply_ctx_frames_cutting = _voxtral_tts_parsers()
    downsample_factor = 240
    audio = torch.randn(2400)

    result = apply_ctx_frames_cutting([audio], [0], downsample_factor)

    assert result[0].shape == audio.shape
    torch.testing.assert_close(result[0], audio)


def test_cut_multiple_requests():
    """Each request in the batch gets its own ctx_frames cut."""
    _, apply_ctx_frames_cutting = _voxtral_tts_parsers()
    downsample_factor = 100
    audio1 = torch.arange(1000, dtype=torch.float32)  # 10 frames
    audio2 = torch.arange(500, dtype=torch.float32)  # 5 frames

    result = apply_ctx_frames_cutting([audio1, audio2], [3, 0], downsample_factor)

    assert len(result) == 2
    assert result[0].shape[0] == 1000 - 3 * 100  # 700
    assert result[1].shape[0] == 500  # unchanged
    torch.testing.assert_close(result[0], audio1[300:])
    torch.testing.assert_close(result[1], audio2)


def test_cut_all_frames_returns_empty():
    """Cutting all frames returns an empty tensor."""
    _, apply_ctx_frames_cutting = _voxtral_tts_parsers()
    downsample_factor = 100
    ctx_frames = 5
    audio = torch.randn(500)  # exactly 5 frames

    result = apply_ctx_frames_cutting([audio], [ctx_frames], downsample_factor)

    assert result[0].numel() == 0


@pytest.fixture
def audio_tokenizer(monkeypatch: pytest.MonkeyPatch):
    _voxtral_tts_parsers()
    from vllm_omni.model_executor.models.voxtral_tts import voxtral_tts_audio_tokenizer as module

    monkeypatch.setattr(module.current_omni_platform, "device_type", "cpu")

    # Keep the real quantizer, decoder convolutions and batching path without
    # allocating the checkpoint's encoder, embedding table or transformers.
    tokenizer = module.VoxtralTTSAudioTokenizer.__new__(module.VoxtralTTSAudioTokenizer)
    nn.Module.__init__(tokenizer)
    tokenizer.patch_size = 1
    tokenizer._sampling_rate = 24000
    tokenizer._frame_rate = 24000
    tokenizer.quantizer = module.MistralAudioCodebook(
        module.AudioTokenizerArgs(
            semantic_codebook_size=4,
            semantic_dim=2,
            acoustic_codebook_size=3,
            acoustic_dim=1,
        )
    )
    tokenizer.decoder_blocks = nn.ModuleList(
        [module.CausalConv1d(3, 3, kernel_size=1, use_weight_norm=False, use_bias=False, pad_mode="constant")]
    )
    tokenizer.output_proj = module.CausalConv1d(
        3, 1, kernel_size=1, use_weight_norm=False, use_bias=False, pad_mode="constant"
    )
    with torch.no_grad():
        tokenizer.quantizer.semantic_codebook.embedding.fill_(0.125)
        tokenizer.decoder_blocks[0].conv.weight.copy_(torch.eye(3).unsqueeze(-1))
        tokenizer.output_proj.conv.weight.fill_(0.25)
    return tokenizer.eval()


@pytest.mark.parametrize(
    "model_dtype", [torch.float16, torch.bfloat16, torch.float32], ids=["float16", "bfloat16", "float32"]
)
@pytest.mark.parametrize("frame_counts", [(116,), (380, 0, 7, 3)], ids=["warmup", "mixed_chunks"])
def test_batch_decode_matches_decoder_dtype(
    audio_tokenizer, model_dtype: torch.dtype, frame_counts: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
):
    audio_tokenizer.to(dtype=model_dtype)
    decode = audio_tokenizer.decode

    def checked_decode(codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        assert dtype == model_dtype
        return decode(codes, dtype=dtype)

    monkeypatch.setattr(audio_tokenizer, "decode", checked_decode)
    codes_list = []
    expected = []
    for frames in frame_counts:
        acoustic_codes = torch.arange(frames) % 3
        # Two special-token slots precede the semantic/acoustic codebooks.
        codes = torch.stack([torch.zeros_like(acoustic_codes), acoustic_codes], dim=1) + 2
        codes_list.append(torch.cat([codes, torch.ones((1, 2), dtype=torch.long)]))
        # The semantic features sum to 0.25; acoustic levels are -1, 0, 1.
        expected.append(torch.tensor([-0.1875, 0.0625, 0.3125])[acoustic_codes])

    with torch.inference_mode():
        waveforms = audio_tokenizer.decode_helper_batch_async(codes_list)

    assert len(waveforms) == len(expected)
    for waveform, reference in zip(waveforms, expected):
        assert waveform.dtype == torch.float32
        assert waveform.device.type == "cpu"
        torch.testing.assert_close(waveform, reference, rtol=0, atol=0)
