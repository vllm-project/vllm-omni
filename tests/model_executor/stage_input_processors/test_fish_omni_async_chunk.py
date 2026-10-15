# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drama 3 frame stream -> DAC windows (ramp, left context, final flush)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.fish_omni import (
    SEMANTIC_BEGIN_ID,
    fish_omni_to_dac_decoder_async_chunk,
    fish_omni_to_dac_decoder_async_chunk_batch,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

EOS = 248046
CODEBOOKS = 10


def _manager(ramp="2,3", chunk=4, left=2):
    return SimpleNamespace(
        connector=SimpleNamespace(
            config={"extra": {"codec_chunk_ramp": ramp, "codec_chunk_frames": chunk, "codec_left_context_frames": left}}
        ),
        request_payload={},
    )


def _request(req_id="r"):
    return SimpleNamespace(external_req_id=req_id, is_finished=lambda: False)


def _frame(index: int, main: int | None = None) -> dict:
    codes = torch.arange(CODEBOOKS) + 100 * index
    row = torch.cat((torch.tensor([SEMANTIC_BEGIN_ID + index if main is None else main]), codes))
    return {"codes.audio": row.unsqueeze(0)}


def _window(payload) -> tuple[list[int], int]:
    """(first code 0 of each frame, left context) of an emitted window."""
    codes = payload.codes.audio
    assert codes.dtype == torch.int64 and codes.ndim == 2 and codes.shape[0] == CODEBOOKS
    return (codes[0] // 100).tolist(), int(payload.meta.left_context_size)


def test_windows_follow_the_ramp_with_left_context():
    manager, request = _manager(), _request()
    windows = []
    for index in range(12):
        payload = fish_omni_to_dac_decoder_async_chunk(manager, _frame(index), request)
        if payload is not None:
            windows.append(_window(payload))
    # Ramp 2, 3, then chunks of 4, each with up to 2 earlier frames.
    assert windows == [
        ([0, 1], 0),
        ([0, 1, 2, 3, 4], 2),
        ([3, 4, 5, 6, 7, 8], 2),
    ]
    final = fish_omni_to_dac_decoder_async_chunk(manager, _frame(12, main=EOS), request, is_finished=True)
    assert _window(final) == ([7, 8, 9, 10, 11], 2)
    assert bool(final.meta.finished)
    assert fish_omni_to_dac_decoder_async_chunk(manager, None, request, is_finished=True) is None


def test_end_of_speech_frame_carries_no_audio():
    manager, request = _manager(ramp="1"), _request()
    assert fish_omni_to_dac_decoder_async_chunk(manager, _frame(0, main=EOS), request) is None
    final = fish_omni_to_dac_decoder_async_chunk(manager, None, request, is_finished=True)
    assert final.codes.audio.shape == (CODEBOOKS, 0) and bool(final.meta.finished)


def test_a_one_frame_window_has_byte_viewable_codes():
    # A request that yields a single frame sends a [codebooks, 1] window; the
    # shared-memory connector views the codes as bytes, which needs stride 1.
    manager, request = _manager(ramp="2"), _request()
    assert fish_omni_to_dac_decoder_async_chunk(manager, _frame(0), request) is None
    final = fish_omni_to_dac_decoder_async_chunk(manager, None, request, is_finished=True)
    codes = final.codes.audio
    assert codes.shape == (CODEBOOKS, 1) and codes.stride(-1) == 1
    assert codes.view(torch.uint8).numel() == CODEBOOKS * 8


def test_stream_keeps_only_frames_later_windows_read():
    manager, request = _manager(ramp="2", chunk=2, left=1), _request()
    for index in range(40):
        fish_omni_to_dac_decoder_async_chunk(manager, _frame(index), request)
    state = manager.request_payload["r"]["_fish_omni_stream"]
    assert len(state["codes"]) <= 1 + 2


def test_batch_matches_per_request_calls():
    single, batched = _manager(), _manager()
    requests = [_request("a"), _request("b"), _request("c")]
    for index in range(10):
        outputs = [_frame(index), _frame(index, main=EOS) if index == 7 else _frame(index), {}]
        finished = [False, index == 7, index == 9]
        expected = [
            fish_omni_to_dac_decoder_async_chunk(single, output, request, done)
            for output, request, done in zip(outputs, requests, finished)
        ]
        got = fish_omni_to_dac_decoder_async_chunk_batch(batched, outputs, requests, finished)
        for want, have in zip(expected, got):
            assert (want is None) == (have is None)
            if want is not None:
                assert np.array_equal(want.codes.audio.numpy(), have.codes.audio.numpy())
                assert bool(want.meta.finished) == bool(have.meta.finished)


@pytest.mark.parametrize("left", [0, 2])
def test_windows_carry_their_index_in_the_stream(left):
    """chunk_seq numbers a stream's windows from 0: the codec starts its history at 0 even
    when windows carry no left context."""
    manager, request = _manager(left=left), _request()
    seqs = []
    for index in range(12):
        payload = fish_omni_to_dac_decoder_async_chunk(manager, _frame(index), request)
        if payload is not None:
            seqs.append((payload.meta.chunk_seq, int(payload.meta.left_context_size)))
    final = fish_omni_to_dac_decoder_async_chunk(manager, None, request, is_finished=True)
    seqs.append((final.meta.chunk_seq, int(final.meta.left_context_size)))
    assert [seq for seq, _ in seqs] == list(range(len(seqs)))
    assert all(context == (0 if left == 0 or seq == 0 else left) for seq, context in seqs)
