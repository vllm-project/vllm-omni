# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.audio8_tts import (
    extract_last_frame,
    slow_ar_to_codec_decoder_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NUM_CODEBOOKS = 3


def _make_transfer_manager(*, chunk_frames=2, left_context=1, initial_chunk_frames=0):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        put_req_chunk=defaultdict(int),
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": chunk_frames,
                    "codec_left_context_frames": left_context,
                    "initial_codec_chunk_frames": initial_chunk_frames,
                }
            }
        ),
    )


def _make_request(req_id="req", *, finished=False):
    return SimpleNamespace(
        external_req_id=req_id,
        additional_information=None,
        is_finished=lambda: finished,
    )


def _frame(value: int) -> torch.Tensor:
    return torch.full((NUM_CODEBOOKS,), value, dtype=torch.long)


def _step(transfer_manager, request, frame_value: int | None, *, finished: bool = False):
    multimodal_output = {} if frame_value is None else {"audio_codes": _frame(frame_value).unsqueeze(0)}
    return slow_ar_to_codec_decoder_async_chunk(transfer_manager, multimodal_output, request, is_finished=finished)


def test_codes_accumulate_across_steps_and_chunk_is_withheld_until_full():
    """Codes must accumulate; emitting per-step would starve the codec of the
    context it needs and resetting would truncate the utterance."""
    transfer_manager = _make_transfer_manager(chunk_frames=2, left_context=1)
    request = _make_request()

    assert _step(transfer_manager, request, 1) is None
    payload = _step(transfer_manager, request, 2)

    assert payload is not None
    assert len(transfer_manager.code_prompt_token_ids["req"]) == 2
    # [num_codebooks, frames], codebook-major.
    assert tuple(payload.codes.audio.shape) == (NUM_CODEBOOKS, 2)
    assert payload.codes.audio[0].tolist() == [1, 2]
    # Nothing precedes the first chunk, so there is no left context yet.
    assert payload.meta.left_context_size == 0


def test_second_chunk_carries_left_context_and_reports_its_size():
    transfer_manager = _make_transfer_manager(chunk_frames=2, left_context=1)
    request = _make_request()
    for value in (1, 2, 3):
        _step(transfer_manager, request, value)
    payload = _step(transfer_manager, request, 4)

    assert payload is not None
    # 1 context frame + 2 new frames.
    assert tuple(payload.codes.audio.shape) == (NUM_CODEBOOKS, 3)
    assert payload.codes.audio[0].tolist() == [2, 3, 4]
    assert payload.meta.left_context_size == 1


def test_partial_tail_is_flushed_when_the_request_finishes():
    transfer_manager = _make_transfer_manager(chunk_frames=4, left_context=2)
    request = _make_request()
    _step(transfer_manager, request, 1)
    _step(transfer_manager, request, 2)
    payload = _step(transfer_manager, request, 3, finished=True)

    assert payload is not None
    assert payload.codes.audio[0].tolist() == [1, 2, 3]
    assert bool(payload.meta.finished) is True


def test_finished_request_without_any_frame_emits_a_terminal_payload():
    transfer_manager = _make_transfer_manager()
    payload = slow_ar_to_codec_decoder_async_chunk(
        transfer_manager, None, _make_request(finished=True), is_finished=True
    )
    assert payload is not None
    assert payload.codes.audio.numel() == 0
    assert bool(payload.meta.finished) is True


def test_terminal_call_on_a_boundary_does_not_re_emit_the_last_chunk():
    """A finished call carrying no new frame, arriving when the stream sits
    exactly on a chunk boundary, must close the stream rather than re-ship the
    last window. Before the guard, the final chunk_frames frames shipped twice."""
    transfer_manager = _make_transfer_manager(chunk_frames=2, left_context=1)
    request = _make_request()
    for value in (1, 2, 3, 4):  # length ends exactly on a chunk boundary
        _step(transfer_manager, request, value)
    # Separate terminal call, no new frame: the boundary chunk is already sent.
    payload = _step(transfer_manager, request, None, finished=True)
    assert payload is not None
    assert payload.codes.audio.numel() == 0
    assert bool(payload.meta.finished) is True


def test_final_frame_on_a_boundary_still_emits_when_it_arrives_finished():
    """If the last frame both lands on a boundary and carries finished, its chunk
    has not been shipped yet, so it must still be emitted (the guard must key on
    'no new frame this call', not merely on being finished at a boundary)."""
    transfer_manager = _make_transfer_manager(chunk_frames=2, left_context=1)
    request = _make_request()
    _step(transfer_manager, request, 1)
    payload = _step(transfer_manager, request, 2, finished=True)
    assert payload is not None
    assert payload.codes.audio[0].tolist() == [1, 2]
    assert bool(payload.meta.finished) is True


def test_initial_chunk_shortens_time_to_first_audio():
    transfer_manager = _make_transfer_manager(chunk_frames=4, left_context=2, initial_chunk_frames=1)
    request = _make_request()
    payload = _step(transfer_manager, request, 1)

    assert payload is not None
    assert payload.codes.audio[0].tolist() == [1]
    assert payload.meta.left_context_size == 0


def test_prefill_placeholder_frames_are_not_forwarded():
    """Prefill emits an all-zero placeholder frame per position; forwarding it
    would prepend silence and desynchronise the chunk boundaries."""
    assert extract_last_frame({"audio_codes": torch.zeros((3, NUM_CODEBOOKS), dtype=torch.long)}) is None
    assert extract_last_frame({}) is None
    assert extract_last_frame({"audio_codes": torch.zeros((0, NUM_CODEBOOKS), dtype=torch.long)}) is None


def test_explicit_validity_flag_overrides_the_all_zero_heuristic():
    frame = torch.zeros((1, NUM_CODEBOOKS), dtype=torch.long)
    out = extract_last_frame({"audio_codes": frame, "audio_code_valid": torch.tensor([True])})
    assert out is not None and out.tolist() == [0] * NUM_CODEBOOKS


def test_invalid_chunk_config_fails_loudly():
    transfer_manager = _make_transfer_manager(chunk_frames=0)
    request = _make_request()
    with pytest.raises(ValueError, match="codec chunk config"):
        _step(transfer_manager, request, 1)
