# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.voxtral_tts import (
    generator2tokenizer,
    generator2tokenizer_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---- Helpers for generator2tokenizer (non-streaming) ----


def _make_stage(engine_outputs):
    return SimpleNamespace(engine_outputs=engine_outputs)


def _make_output(audio_tensors):
    """Build a single generator output with multimodal_output["audio"] = audio_tensors."""
    return SimpleNamespace(
        outputs=[SimpleNamespace(multimodal_output={"audio": audio_tensors})],
    )


# ---- Tests for generator2tokenizer ----


class TestGenerator2Tokenizer:
    def test_single_output_single_chunk(self):
        """Single output with one audio tensor produces correct prompt_token_ids."""
        audio = torch.tensor([10, 20, 30, 40])
        stage = _make_stage([_make_output([audio])])

        result = generator2tokenizer(stage.engine_outputs)

        assert len(result) == 1
        assert result[0]["prompt_token_ids"] == [10, 20, 30, 40]
        assert result[0]["multi_modal_data"] is None

    def test_2d_audio_tensors_flattened(self):
        """2D audio tensors (e.g., multi-codebook frames) are flattened."""
        audio = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
        stage = _make_stage([_make_output([audio])])

        result = generator2tokenizer(stage.engine_outputs)

        assert result[0]["prompt_token_ids"] == [1, 2, 3, 4, 5, 6]

    def test_no_outputs_yet_raises(self):
        """Stage with engine_outputs=None raises RuntimeError."""
        stage = _make_stage(engine_outputs=None)
        with pytest.raises(TypeError):
            generator2tokenizer(stage.engine_outputs)


# ---- Helpers for generator2tokenizer_async_chunk (streaming) ----


def _req(external_req_id: str, *, finished: bool):
    return SimpleNamespace(
        external_req_id=external_req_id,
        is_finished=lambda: finished,
    )


def _make_transfer_manager(codec_chunk_frames=25, codec_left_context_frames=25, codec_chunk_frames_at_begin=5):
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


def test_empty_chunk_when_not_finished():
    """Returns None when no audio data and not finished."""
    transfer_manager = _make_transfer_manager()
    request = _req("rid-empty", finished=False)

    payload = generator2tokenizer_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output={"audio": torch.zeros((0,))},
        request=request,
    )

    assert payload is None


def test_flush_tail_when_finished():
    """Emits remaining frames on finish with finished=True."""
    transfer_manager = _make_transfer_manager()
    request_id = "rid-tail"
    # Pre-populate with 24 frames of 8-element codes
    transfer_manager.code_prompt_token_ids[request_id] = [[1, 2, 3, 4, 5, 6, 7, 8] for _ in range(24)]
    request = _req(request_id, finished=True)

    payload = generator2tokenizer_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,  # e.g. EOS step with no audio
        request=request,
    )

    assert payload is not None
    assert payload.meta.finished.item() is True
    codes = payload.codes.audio
    # Format: [ctx_frames, context_length, ...flat_codes]
    assert len(codes) >= 2  # At least ctx_frames + context_length header
    ctx_frames = codes[0]
    context_length = codes[1]
    flat_codes = codes[2:]
    # Total frames in window = ctx_frames + context_length
    total_window_frames = ctx_frames + context_length
    assert len(flat_codes) == total_window_frames * 8


def test_eof_marker_when_finished_with_no_frames():
    """Emits EOF marker with empty codes when finished with no accumulated frames."""
    transfer_manager = _make_transfer_manager()
    request = _req("rid-eof", finished=True)

    payload = generator2tokenizer_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
    )

    assert payload.codes.audio.tolist() == []
    assert payload.meta.finished.item() is True


def test_normal_chunk_emission():
    """Emits chunks at codec_chunk_frames boundary."""
    transfer_manager = _make_transfer_manager(
        codec_chunk_frames=25,
        codec_left_context_frames=25,
        codec_chunk_frames_at_begin=5,
    )
    request_id = "rid-chunk"
    request = _req(request_id, finished=False)

    # Feed 25 frames one by one
    for i in range(25):
        pooling_output = {"audio": torch.tensor([float(i)] * 4)}
        payload = generator2tokenizer_async_chunk(
            transfer_manager=transfer_manager,
            pooling_output=pooling_output,
            request=request,
        )

    # A chunk should be emitted
    assert payload is not None
    codes = payload.codes.audio
    ctx_frames = codes[0]
    context_length = codes[1]
    assert ctx_frames == 20  # 25 - 5(chunk_size_at_begin)
    assert context_length == 5  # 25 - 20(chunk_size_at_begin * 4)


def test_small_initial_chunks():
    """Uses codec_chunk_frames_at_begin for initial frames when length <= chunk_size."""
    transfer_manager = _make_transfer_manager(
        codec_chunk_frames=25,
        codec_left_context_frames=25,
        codec_chunk_frames_at_begin=5,
    )
    request_id = "rid-begin"
    request = _req(request_id, finished=False)

    # Feed 5 frames (should trigger emission because codec_chunk_frames_at_begin=5)
    for i in range(5):
        pooling_output = {"audio": torch.tensor([float(i + 1)] * 3)}
        payload = generator2tokenizer_async_chunk(
            transfer_manager=transfer_manager,
            pooling_output=pooling_output,
            request=request,
        )

    assert payload is not None
    codes = payload.codes.audio
    ctx_frames = codes[0]
    context_length = codes[1]
    assert ctx_frames == 0
    assert context_length == 5  # chunk_size_at_begin


def test_no_emission_between_boundaries():
    """No chunk emitted when not at chunk boundary and not finished."""
    transfer_manager = _make_transfer_manager(
        codec_chunk_frames=25,
        codec_left_context_frames=25,
        codec_chunk_frames_at_begin=5,
    )
    request_id = "rid-mid"
    request = _req(request_id, finished=False)

    # Feed 3 frames (3 % 5 != 0, should not emit)
    for i in range(3):
        pooling_output = {"audio": torch.tensor([float(i)] * 4)}
        payload = generator2tokenizer_async_chunk(
            transfer_manager=transfer_manager,
            pooling_output=pooling_output,
            request=request,
        )

    assert payload is None


def test_context_handling_format():
    """Verifies [ctx_frames, context_length, ...codes] format."""
    transfer_manager = _make_transfer_manager(
        codec_chunk_frames=10,
        codec_left_context_frames=5,
        codec_chunk_frames_at_begin=5,
    )
    request_id = "rid-ctx"
    request = _req(request_id, finished=False)

    # Feed 5 frames to trigger a chunk (chunk_size_at_begin=5)
    for i in range(5):
        pooling_output = {"audio": torch.tensor([float(i + 10)] * 2)}  # codebook_dim=2
        payload = generator2tokenizer_async_chunk(
            transfer_manager=transfer_manager,
            pooling_output=pooling_output,
            request=request,
        )

    assert payload is not None
    codes = payload.codes.audio
    # codes is a 1-D long tensor: [ctx_frames, context_length, ...flat_codes]
    ctx_frames = int(codes[0].item())
    context_length = int(codes[1].item())
    flat_codes = codes[2:]
    assert ctx_frames >= 0
    assert context_length > 0
    # flat_codes = total_window_frames * codebook_dim
    total_window_frames = ctx_frames + context_length
    assert flat_codes.numel() == total_window_frames * 2  # codebook_dim=2


def test_none_pooling_output_not_finished_returns_none():
    """None pooling_output when not finished returns None."""
    transfer_manager = _make_transfer_manager()
    request = _req("rid-none", finished=False)

    payload = generator2tokenizer_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
    )

    assert payload is None
