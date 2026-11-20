# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.model_executor.stage_input_processors.qwen3_omni as q3
from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import ramp_decode_windows

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_Q = 2
_RAMP = [1, 2, 4, 8, 16, 25]


def _manager(**extra):
    config = {"codec_chunk_frames": 25, "codec_left_context_frames": 25, "initial_codec_chunk_frames": 4, **extra}
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        put_req_chunk=defaultdict(int),
        ramp_chunk_count=defaultdict(int),
        connector=SimpleNamespace(config={"extra": config}),
    )


def _request(request_id="req"):
    return SimpleNamespace(external_req_id=request_id, sampling_params=SimpleNamespace(stop_token_ids=[]))


def _frame(index: int) -> dict:
    return {
        "codes": {"audio": torch.tensor([[index, 1000 + index]], dtype=torch.long)},
        "meta": {"codec_frame_valid": torch.ones(1, dtype=torch.int8)},
    }


def _stream(manager, num_frames, finish_after_last=True):
    request = _request()
    chunks = []

    def emit(payload):
        if payload is None:
            return
        codes = payload.codes.audio.reshape(_Q, -1)
        chunks.append((codes[0].tolist(), int(payload.meta.left_context_size), bool(payload.meta.finished)))
        # The transport counts every enqueued chunk.
        manager.put_req_chunk[request.external_req_id] += 1
        manager.ramp_chunk_count[request.external_req_id] += 1

    for index in range(num_frames):
        last = finish_after_last and index == num_frames - 1
        emit(q3.talker2code2wav_async_chunk(manager, _frame(index), request, is_finished=last))
    return chunks


def test_ramp_decode_windows_include_left_context():
    assert ramp_decode_windows(_RAMP, 25) == [1, 3, 7, 15, 31, 50]
    assert ramp_decode_windows([1, 2], 0) == [1, 2]


def test_ramp_chunks_grow_and_resend_left_context():
    chunks = _stream(_manager(codec_chunk_ramp=_RAMP), 56 + 25, finish_after_last=False)

    new_frames = [len(window) - left for window, left, _finished in chunks]
    assert new_frames == [1, 2, 4, 8, 16, 25, 25]
    assert [left for _window, left, _finished in chunks] == [0, 1, 3, 7, 15, 25, 25]
    assert [len(window) for window, _left, _finished in chunks] == [1, 3, 7, 15, 31, 50, 50]
    emitted = 0
    for window, left, _finished in chunks:
        # Each window ends at the newest frame and re-sends `left` earlier ones.
        assert window == list(range(emitted - left, emitted + len(window) - left))
        emitted += len(window) - left
    # The first chunk is the stream's first frame alone (no context).
    assert chunks[0][0] == [0]


def test_ramp_flushes_partial_tail_with_left_context():
    chunks = _stream(_manager(codec_chunk_ramp=_RAMP), 10)

    assert [len(window) - left for window, left, _finished in chunks] == [1, 2, 4, 3]
    window, left, finished = chunks[-1]
    assert finished and left == 7 and window == list(range(0, 10))


def test_ramp_finish_on_chunk_boundary_leaves_the_marker_to_the_transport():
    manager = _manager(codec_chunk_ramp=_RAMP)
    request = _request()
    for index in range(3):
        payload = q3.talker2code2wav_async_chunk(manager, _frame(index), request)
        if payload is not None:
            manager.ramp_chunk_count[request.external_req_id] += 1
    # Frames 0 and 1-2 went out as chunks 0 and 1; the finish step has no frame.
    finish = {"codes": {"audio": torch.zeros((1, _Q), dtype=torch.long)}, "meta": {"codec_frame_valid": torch.zeros(1)}}
    assert q3.talker2code2wav_async_chunk(manager, finish, request, is_finished=True) is None


def test_without_ramp_the_initial_chunk_setting_is_unchanged():
    chunks = _stream(_manager(), 29 + 3)

    assert [(len(window), left) for window, left, _finished in chunks] == [(4, 0), (29, 4), (28, 25)]


def test_first_audio_marker_only_reaches_initial_codec_chunk():
    manager = _manager(codec_chunk_ramp=[1, 1])
    for index in range(2):
        frame = _frame(index)
        frame["meta"]["first_audio"] = torch.tensor([True])
        payload = q3.talker2code2wav_async_chunk(manager, frame, _request())
        assert bool(payload.meta.first_audio) is (index == 0)
        manager.put_req_chunk["req"] += 1
        manager.ramp_chunk_count["req"] += 1
