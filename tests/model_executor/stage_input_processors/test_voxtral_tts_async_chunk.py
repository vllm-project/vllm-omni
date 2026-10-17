# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.voxtral_tts import (
    generator2tokenizer_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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
        multimodal_output={"codes": {"audio": torch.zeros((0,))}},
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
        multimodal_output=None,  # e.g. EOS step with no audio
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
        multimodal_output=None,
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
        multimodal_output = {"codes": {"audio": torch.tensor([float(i)] * 4)}}
        payload = generator2tokenizer_async_chunk(
            transfer_manager=transfer_manager,
            multimodal_output=multimodal_output,
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
        multimodal_output = {"codes": {"audio": torch.tensor([float(i + 1)] * 3)}}
        payload = generator2tokenizer_async_chunk(
            transfer_manager=transfer_manager,
            multimodal_output=multimodal_output,
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
        multimodal_output = {"codes": {"audio": torch.tensor([float(i)] * 4)}}
        payload = generator2tokenizer_async_chunk(
            transfer_manager=transfer_manager,
            multimodal_output=multimodal_output,
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
        multimodal_output = {"codes": {"audio": torch.tensor([float(i + 10)] * 2)}}  # codebook_dim=2
        payload = generator2tokenizer_async_chunk(
            transfer_manager=transfer_manager,
            multimodal_output=multimodal_output,
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


def test_none_multimodal_output_not_finished_returns_none():
    """None multimodal_output when not finished returns None."""
    transfer_manager = _make_transfer_manager()
    request = _req("rid-none", finished=False)

    payload = generator2tokenizer_async_chunk(
        transfer_manager=transfer_manager,
        multimodal_output=None,
        request=request,
    )

    assert payload is None


def _compute_mm_logits_payload(batch: int = 1, codebooks: int = 8):
    """Mirror voxtral_tts_audio_generation.compute_mm_logits() L1075:
    audio_codes [B, C] -> unsqueeze(1) -> [B, 1, C] -> split -> list of (1, 1, C).
    """
    audio_codes = torch.arange(batch * codebooks, dtype=torch.float32).reshape(batch, codebooks)
    audio_list = list(torch.split(audio_codes.unsqueeze(1), 1, dim=0))
    return {"codes": {"audio": audio_list}}


def test_3d_per_step_frame_list_accepted():
    """Voxtral P1 regression: compute_mm_logits 3-D per-step [B, 1, C]
    payload must not raise ValueError; the last frame is extracted as 1-D."""
    from vllm_omni.model_executor.stage_input_processors import _common

    transfer_manager = _make_transfer_manager()
    request_id = "rid-3d"
    request = _req(request_id, finished=False)

    payload = generator2tokenizer_async_chunk(
        transfer_manager=transfer_manager,
        multimodal_output=_compute_mm_logits_payload(1, 8),
        request=request,
    )

    # The frame is stored as the flat 8-element codec row.
    assert transfer_manager.code_prompt_token_ids[request_id][-1] == [float(i) for i in range(8)]
    # One frame < chunk_size_at_begin=5: no emission yet.
    assert payload is None
    # Direct helper call: 1-D float frame, precision preserved (to_long=False).
    frame = _common.extract_last_codec_frame(
        {"codes": {"audio": torch.zeros(1, 1, 8)}},
        to_long=False,
    )
    assert frame is not None
    assert frame.shape == (8,)
    assert frame.dtype == torch.float32


def test_3d_direct_tensor_accepted():
    """A bare 3-D [1, 1, C] tensor is normalized via squeeze(1) and the
    2-D last-row semantics apply unchanged."""
    from vllm_omni.model_executor.stage_input_processors import _common

    frame = _common.extract_last_codec_frame(
        {"codes": {"audio": torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)}},
        to_long=False,
    )
    assert frame is not None
    assert frame.shape == (8,)
    assert frame.dtype == torch.float32
    assert frame.tolist() == [float(i) for i in range(8)]


def test_3d_per_step_chunk_emission():
    """Feeding 5 per-step 3-D frames reaches the codec_chunk_frames_at_begin
    boundary and emits the standard [ctx_frames, context_length, ...flat] payload."""
    transfer_manager = _make_transfer_manager(
        codec_chunk_frames=25,
        codec_left_context_frames=25,
        codec_chunk_frames_at_begin=5,
    )
    request_id = "rid-3d-chunk"
    request = _req(request_id, finished=False)

    payload = None
    for _ in range(5):
        payload = generator2tokenizer_async_chunk(
            transfer_manager=transfer_manager,
            multimodal_output=_compute_mm_logits_payload(1, 8),
            request=request,
        )

    assert payload is not None
    codes = payload.codes.audio
    assert int(codes[0].item()) == 0  # ctx_frames
    assert int(codes[1].item()) == 5  # context_length == chunk_size_at_begin
    assert codes[2:].numel() == 5 * 8  # 5 frames x 8 codebooks


def test_3d_non_singleton_frame_dim_raises():
    """Defensive contract: non per-step 3-D layouts (dim1 != 1) still raise
    ValueError instead of being silently flattened."""
    transfer_manager = _make_transfer_manager()
    request = _req("rid-3d-non-singleton", finished=False)

    with pytest.raises(ValueError):
        generator2tokenizer_async_chunk(
            transfer_manager=transfer_manager,
            multimodal_output={"codes": {"audio": torch.zeros(1, 2, 3)}},
            request=request,
        )
