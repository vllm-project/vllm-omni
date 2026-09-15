# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for parse_batched_audio_input and apply_ctx_frames_cutting.

These functions parse the [ctx_frames, context_length, ...tokens] wire format
produced by generator2tokenizer_async_chunk and cut leading context samples
from decoded audio arrays.
"""

import functools

import pytest
import torch

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


# ──────────────────────────────────────────────────────────────────
# decode_helper_batch_async dtype regression test (#4838)
# ──────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def _voxtral_tts_audio_tokenizer_cls():
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

    bootstrap_vllm_layer_custom_op_modules()
    import vllm.model_executor.models.utils  # noqa: F401

    from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts_audio_tokenizer import (
        VoxtralTTSAudioTokenizer,
    )

    return VoxtralTTSAudioTokenizer


def _make_bare_audio_tokenizer(param_dtype: torch.dtype):
    """Build a VoxtralTTSAudioTokenizer without running __init__.

    __init__ requires a full VllmConfig + checkpoint, which is too heavy
    for a CPU unit test. Instead we bypass it and set only the state
    needed to exercise decode_helper_batch_async.
    """
    cls = _voxtral_tts_audio_tokenizer_cls()
    tokenizer = cls.__new__(cls)
    torch.nn.Module.__init__(tokenizer)
    tokenizer.register_parameter("_dummy_param", torch.nn.Parameter(torch.zeros(1, dtype=param_dtype)))
    tokenizer._sampling_rate = 24000
    tokenizer._frame_rate = 24000 / 240  # -> downsample_factor == 240
    return tokenizer


def test_dtype_property_reflects_param_dtype():
    """VoxtralTTSAudioTokenizer.dtype reflects the actual loaded parameter dtype."""
    tokenizer = _make_bare_audio_tokenizer(torch.float16)
    assert tokenizer.dtype == torch.float16


def test_decode_helper_batch_async_uses_resolved_dtype_not_hardcoded(monkeypatch):
    """Regression test for #4838.

    decode_helper_batch_async previously hardcoded dtype=torch.bfloat16
    when calling self.decode(...). This mismatched the tokenizer's actual
    (e.g. float16) parameter dtype on GPUs without bfloat16 support, such
    as Tesla T4, causing:
        RuntimeError: expected scalar type BFloat16 but found Half

    This verifies decode_helper_batch_async passes the tokenizer's
    resolved `dtype` property to decode(), instead of a hardcoded value.
    """
    tokenizer = _make_bare_audio_tokenizer(torch.float16)
    seen_dtypes = []

    def fake_decode(codes, dtype=torch.float32):
        seen_dtypes.append(dtype)
        b, _, t = codes.shape
        return torch.zeros(b, 1, t * tokenizer.downsample_factor)

    monkeypatch.setattr(tokenizer, "decode", fake_decode)

    num_codebooks = 4
    non_eoa_row = torch.zeros(num_codebooks, dtype=torch.long)
    eoa_row = torch.zeros(num_codebooks, dtype=torch.long)
    eoa_row[0] = 1
    codes = torch.stack([non_eoa_row, eoa_row])  # [T=2, K]

    tokenizer.decode_helper_batch_async([codes])

    assert seen_dtypes == [torch.float16]
