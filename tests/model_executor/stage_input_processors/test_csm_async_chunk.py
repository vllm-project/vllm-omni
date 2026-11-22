# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the CSM talker-to-codec handoff: every playable frame is sent exactly once."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from tests.worker_v2.test_omni_data_plane import _complete, _make_plane, _new_request
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.model_executor.stage_input_processors.csm import talker2code2wav_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]

NUM_CODEBOOKS = 32
STOP_TOKEN = 2050


def _manager(**extra) -> SimpleNamespace:
    """The sender state a chunk transport hands to the processor."""
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        connector=SimpleNamespace(config={"extra": extra}),
    )


def _request(request_id: str = "req", *, finished: bool = False) -> SimpleNamespace:
    return SimpleNamespace(external_req_id=request_id, is_finished=lambda: finished)


def _frames(count: int, *, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, 2048, (count, NUM_CODEBOOKS), generator=generator)


def _step(frame: torch.Tensor, *, valid: bool = True) -> dict:
    """One decode step's output for one request."""
    return {
        "codes": {"audio": frame.reshape(1, -1)},
        "meta": {"codec_frame_valid": torch.tensor([int(valid)], dtype=torch.int8)},
    }


def _sent_frames(payload) -> torch.Tensor:
    return payload.codes.audio.reshape(NUM_CODEBOOKS, -1).T


@pytest.mark.parametrize(
    ("total", "chunk", "initial", "expected_sizes"),
    [
        (12, 5, 1, [1, 5, 5, 1]),
        (11, 5, 1, [1, 5, 5, 0]),
        (7, 3, 3, [3, 3, 1]),
        (4, 5, None, [4]),
        (3, 2, 9, [2, 1]),
    ],
)
def test_each_frame_is_sent_once_in_order(total, chunk, initial, expected_sizes) -> None:
    extra = {"codec_chunk_frames": chunk, "codec_left_context_frames": 0}
    if initial is not None:
        extra["initial_codec_chunk_frames"] = initial
    manager = _manager(**extra)
    frames = _frames(total)

    payloads = [talker2code2wav_async_chunk(manager, _step(frame), _request()) for frame in frames]
    # The transport reports the end of the request without a new frame.
    payloads.append(talker2code2wav_async_chunk(manager, None, _request(finished=True)))
    payloads = [payload for payload in payloads if payload is not None]

    assert [_sent_frames(payload).shape[0] for payload in payloads] == expected_sizes
    assert torch.equal(torch.cat([_sent_frames(payload) for payload in payloads]), frames)
    assert [bool(payload.meta.finished) for payload in payloads] == [False] * (len(payloads) - 1) + [True]
    assert all(payload.codes.audio.dtype == torch.long for payload in payloads)
    assert all(payload.meta.request_id == "req" for payload in payloads)
    assert manager.code_prompt_token_ids["req"] == []


def test_steps_without_a_playable_frame_add_nothing() -> None:
    manager = _manager(codec_chunk_frames=1)
    request = _request()
    frame = _frames(1)[0]
    stop_frame = torch.zeros(NUM_CODEBOOKS, dtype=torch.long)
    stop_frame[-1] = 5

    # The step after a stop frame, and the overrun step of async scheduling.
    assert talker2code2wav_async_chunk(manager, _step(frame, valid=False), request) is None
    # The frame where the reference implementation cuts the audio.
    assert talker2code2wav_async_chunk(manager, _step(torch.zeros(NUM_CODEBOOKS, dtype=torch.long)), request) is None
    # A wake-up without model output.
    assert talker2code2wav_async_chunk(manager, {}, request) is None
    assert talker2code2wav_async_chunk(manager, None, request) is None
    assert manager.code_prompt_token_ids["req"] == []

    # A stop frame whose last codebook is not zero is still audio.
    payload = talker2code2wav_async_chunk(manager, _step(stop_frame), request)
    assert torch.equal(_sent_frames(payload), stop_frame[None])


def test_a_prefill_step_carries_its_frame_in_the_last_row() -> None:
    manager = _manager(codec_chunk_frames=1)
    frame = _frames(1)[0]
    rows = torch.zeros((4, NUM_CODEBOOKS), dtype=torch.long)
    rows[-1] = frame
    output = {"codes": {"audio": rows}, "meta": {"codec_frame_valid": torch.tensor([0, 0, 0, 1], dtype=torch.int8)}}

    payload = talker2code2wav_async_chunk(manager, output, _request())

    assert torch.equal(_sent_frames(payload), frame[None])


def test_a_request_without_frames_ends_with_an_empty_finished_chunk() -> None:
    manager = _manager(codec_chunk_frames=5)

    payload = talker2code2wav_async_chunk(manager, None, _request(finished=True))

    assert payload.codes.audio.shape == (0,) and payload.codes.audio.dtype == torch.long
    assert bool(payload.meta.finished)


def test_the_last_frame_can_arrive_with_the_finish() -> None:
    manager = _manager(codec_chunk_frames=5, initial_codec_chunk_frames=5)
    frames = _frames(2)
    assert talker2code2wav_async_chunk(manager, _step(frames[0]), _request()) is None

    payload = talker2code2wav_async_chunk(manager, _step(frames[1]), _request(), is_finished=True)

    assert torch.equal(_sent_frames(payload), frames)
    assert bool(payload.meta.finished)


def test_requests_keep_separate_buffers() -> None:
    manager = _manager(codec_chunk_frames=2, initial_codec_chunk_frames=2)
    first, second = _frames(3, seed=1), _frames(3, seed=2)
    sent: dict[str, list[torch.Tensor]] = {"a": [], "b": []}

    for step in range(3):
        for request_id, frames in (("a", first), ("b", second)):
            payload = talker2code2wav_async_chunk(manager, _step(frames[step]), _request(request_id))
            if payload is not None:
                sent[request_id].append(_sent_frames(payload))
    for request_id in sent:
        payload = talker2code2wav_async_chunk(manager, None, _request(request_id, finished=True))
        sent[request_id].append(_sent_frames(payload))

    assert torch.equal(torch.cat(sent["a"]), first)
    assert torch.equal(torch.cat(sent["b"]), second)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"codec_chunk_frames": 0}, "Invalid codec chunk config"),
        ({"codec_chunk_frames": 5, "initial_codec_chunk_frames": -1}, "Invalid codec chunk config"),
        ({"codec_chunk_frames": 5, "codec_left_context_frames": 25}, "keeps its decoder state"),
    ],
)
def test_unsupported_connector_settings_are_rejected(extra, message) -> None:
    with pytest.raises(ValueError, match=message):
        talker2code2wav_async_chunk(_manager(**extra), _step(_frames(1)[0]), _request())


def test_native_data_plane_sends_each_frame_once_and_nothing_after_the_stop_token() -> None:
    """Through the Model Runner V2 data plane, including the steps async scheduling runs past the stop."""
    plane = _make_plane(replace_send=True)
    plane._custom_process_func = talker2code2wav_async_chunk
    plane._omni_connector = SharedMemoryConnector(
        {"extra": {"codec_chunk_frames": 3, "initial_codec_chunk_frames": 1, "codec_left_context_frames": 0}}
    )
    plane.request_payload = {}
    plane._code_prompt_token_ids = defaultdict(list)
    payloads = []

    def send_chunks(entries, **_kwargs):
        for snapshot, model_output in entries:
            payload = plane._build_custom_process_payload(
                snapshot.external_req_id, snapshot, model_output, propagate_errors=True
            )
            if payload is not None:
                payloads.append(payload)
                plane._put_req_chunk[snapshot.external_req_id] += 1
        return len(entries)

    plane.send_chunks = send_chunks
    request = _new_request()
    request.resumable = False
    request.sampling_params = SimpleNamespace(stop_token_ids=[STOP_TOKEN])
    plane.register_request(request)
    frames = _frames(8)
    steps = [(_step(frame), int(frame[0])) for frame in frames]
    # A request that ends on its own spends two tokens beyond its audio frames: the all-zero
    # end frame, which is not audio, and the forced stop token, which publishes no frame.
    end_frame = torch.zeros(NUM_CODEBOOKS, dtype=torch.long)
    steps.append((_step(end_frame), 0))
    steps.append((_step(end_frame, valid=False), STOP_TOKEN))
    assert len(steps) == frames.shape[0] + 2
    # The two steps after the stop token were already scheduled.
    steps.extend((_step(frame), STOP_TOKEN) for frame in _frames(2, seed=9))
    try:
        for output, token in steps:
            plane.reserve_outputs([request.req_id])
            _complete(plane, [output], token=token)
        plane.request_terminal({request.req_id})
    finally:
        plane._omni_connector.close()

    assert [_sent_frames(payload).shape[0] for payload in payloads] == [1, 3, 3, 1]
    assert torch.equal(torch.cat([_sent_frames(payload) for payload in payloads]), frames)
    assert [bool(payload.meta.finished) for payload in payloads] == [False, False, False, True]
    assert plane.record.cleaned == [request.req_id]
