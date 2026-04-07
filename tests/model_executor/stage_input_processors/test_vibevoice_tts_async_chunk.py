# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for VibeVoice TTS stage input processors.

Tests generator2tokenizer (non-streaming) and
generator2tokenizer_async_chunk (streaming) for continuous latent transport.
"""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.vibevoice_tts import (
    generator2tokenizer,
    generator2tokenizer_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LATENT_DIM = 64


# ── Helpers for generator2tokenizer (non-streaming) ──


def _make_stage(engine_outputs):
    return SimpleNamespace(engine_outputs=engine_outputs)


def _make_output(audio_tensors):
    """Build a generator output with multimodal_output["audio"]."""
    return SimpleNamespace(
        outputs=[SimpleNamespace(multimodal_output={"audio": audio_tensors})],
    )


# ── Tests for generator2tokenizer ──


class TestGenerator2Tokenizer:
    def test_single_output_produces_prompt(self):
        """Single output with latent frames produces correct prompt."""
        frames = [torch.randn(1, 1, LATENT_DIM) for _ in range(5)]
        stage = _make_stage([_make_output(frames)])

        result = generator2tokenizer([stage], engine_input_source=[0])

        assert len(result) == 1
        ids = result[0]["prompt_token_ids"]
        # Header: [ctx_frames=0, num_frames=5] + 5*64 values
        assert len(ids) == 2 + 5 * LATENT_DIM
        assert ids[0] == 0.0  # ctx_frames
        assert ids[1] == 5.0  # num_frames

    def test_empty_engine_input_source_raises(self):
        """Empty engine_input_source raises ValueError."""
        stage = _make_stage([_make_output([torch.randn(1, 1, 4)])])
        with pytest.raises(ValueError, match="cannot be empty"):
            generator2tokenizer([stage], engine_input_source=[])

    def test_invalid_stage_id_raises(self):
        """Stage id beyond list length raises IndexError."""
        stage = _make_stage([_make_output([torch.randn(1, 1, 4)])])
        with pytest.raises(IndexError, match="Invalid stage_id"):
            generator2tokenizer([stage], engine_input_source=[5])

    def test_no_outputs_yet_raises(self):
        """Stage with engine_outputs=None raises RuntimeError."""
        stage = _make_stage(engine_outputs=None)
        with pytest.raises(RuntimeError, match="no outputs yet"):
            generator2tokenizer([stage], engine_input_source=[0])


# ── Helpers for generator2tokenizer_async_chunk ──


def _req(external_req_id: str, *, finished: bool):
    return SimpleNamespace(
        external_req_id=external_req_id,
        is_finished=lambda: finished,
    )


def _make_transfer_manager(
    codec_chunk_frames=25,
    codec_left_context_frames=25,
    codec_chunk_frames_at_begin=5,
):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": codec_chunk_frames,
                    "codec_left_context_frames": codec_left_context_frames,
                    "codec_chunk_frames_at_begin": codec_chunk_frames_at_begin,
                }
            }
        ),
    )


# ── Tests for generator2tokenizer_async_chunk ──


def test_empty_chunk_when_not_finished():
    """Returns None when no audio data and not finished."""
    tm = _make_transfer_manager()
    request = _req("rid-empty", finished=False)

    payload = generator2tokenizer_async_chunk(
        transfer_manager=tm,
        pooling_output={"audio": torch.zeros((0,))},
        request=request,
    )

    assert payload is None


def test_eof_marker_when_finished_with_no_frames():
    """Emits EOF marker with empty codes when finished with no frames."""
    tm = _make_transfer_manager()
    request = _req("rid-eof", finished=True)

    payload = generator2tokenizer_async_chunk(
        transfer_manager=tm,
        pooling_output=None,
        request=request,
    )

    assert payload == {
        "code_predictor_codes": [],
        "finished": torch.tensor(True, dtype=torch.bool),
    }


def test_flush_tail_when_finished():
    """Emits remaining frames on finish."""
    tm = _make_transfer_manager()
    rid = "rid-tail"
    # Pre-populate with 24 frames of LATENT_DIM floats
    tm.code_prompt_token_ids[rid] = [
        [float(i)] * LATENT_DIM for i in range(24)
    ]
    request = _req(rid, finished=True)

    payload = generator2tokenizer_async_chunk(
        transfer_manager=tm,
        pooling_output=None,
        request=request,
    )

    assert payload is not None
    assert payload["finished"].item() is True
    codes = payload["code_predictor_codes"]
    assert len(codes) >= 2  # header: ctx_frames + context_length
    ctx_frames = int(codes[0])
    context_length = int(codes[1])
    flat_values = codes[2:]
    total_window_frames = ctx_frames + context_length
    assert len(flat_values) == total_window_frames * LATENT_DIM


def test_small_initial_chunks():
    """Uses codec_chunk_frames_at_begin for initial frames."""
    tm = _make_transfer_manager(
        codec_chunk_frames=25,
        codec_left_context_frames=25,
        codec_chunk_frames_at_begin=5,
    )
    rid = "rid-begin"
    request = _req(rid, finished=False)

    # Feed 5 frames one by one
    for i in range(5):
        pooling_output = {"audio": torch.tensor([float(i + 1)] * LATENT_DIM)}
        payload = generator2tokenizer_async_chunk(
            transfer_manager=tm,
            pooling_output=pooling_output,
            request=request,
        )

    assert payload is not None
    codes = payload["code_predictor_codes"]
    ctx_frames = int(codes[0])
    context_length = int(codes[1])
    assert ctx_frames == 0
    assert context_length == 5


def test_no_emission_between_boundaries():
    """No chunk emitted between chunk boundaries when not finished."""
    tm = _make_transfer_manager(
        codec_chunk_frames=25,
        codec_left_context_frames=25,
        codec_chunk_frames_at_begin=5,
    )
    rid = "rid-mid"
    request = _req(rid, finished=False)

    # Feed 3 frames (3 % 5 != 0)
    for i in range(3):
        pooling_output = {"audio": torch.tensor([float(i)] * LATENT_DIM)}
        payload = generator2tokenizer_async_chunk(
            transfer_manager=tm,
            pooling_output=pooling_output,
            request=request,
        )

    assert payload is None


def test_context_handling_format():
    """Verifies [ctx_frames, context_length, ...values] format."""
    tm = _make_transfer_manager(
        codec_chunk_frames=10,
        codec_left_context_frames=5,
        codec_chunk_frames_at_begin=5,
    )
    rid = "rid-ctx"
    request = _req(rid, finished=False)

    latent_dim = 4
    for i in range(5):
        pooling_output = {"audio": torch.tensor([float(i + 10)] * latent_dim)}
        payload = generator2tokenizer_async_chunk(
            transfer_manager=tm,
            pooling_output=pooling_output,
            request=request,
        )

    assert payload is not None
    codes = payload["code_predictor_codes"]
    ctx_frames = int(codes[0])
    context_length = int(codes[1])
    flat_values = codes[2:]
    assert ctx_frames >= 0
    assert context_length > 0
    total_window_frames = ctx_frames + context_length
    assert len(flat_values) == total_window_frames * latent_dim


def test_values_are_floats():
    """Emitted values are floats (not ints like Voxtral codes)."""
    tm = _make_transfer_manager(
        codec_chunk_frames=25,
        codec_left_context_frames=25,
        codec_chunk_frames_at_begin=5,
    )
    rid = "rid-float"
    request = _req(rid, finished=False)

    for i in range(5):
        pooling_output = {"audio": torch.tensor([0.123 * (i + 1)] * 4)}
        payload = generator2tokenizer_async_chunk(
            transfer_manager=tm,
            pooling_output=pooling_output,
            request=request,
        )

    assert payload is not None
    codes = payload["code_predictor_codes"]
    # Verify non-header values are floats, not rounded ints
    flat_values = codes[2:]
    has_non_integer = any(v != int(v) for v in flat_values)
    assert has_non_integer, "Expected float values in latent codes"


def test_none_pooling_output_not_finished_returns_none():
    """None pooling_output when not finished returns None."""
    tm = _make_transfer_manager()
    request = _req("rid-none", finished=False)

    payload = generator2tokenizer_async_chunk(
        transfer_manager=tm,
        pooling_output=None,
        request=request,
    )

    assert payload is None
