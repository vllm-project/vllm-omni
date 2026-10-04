# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MOSS raw codec rows are buffered into first/regular chunks in codebook-major order."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.moss_tts import (
    _MOSS_AUDIO_PAD_CODE,
    talker2codec_raw_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _manager(first=1, regular=3):
    config = {"extra": {"codec_chunk_frames": regular, "initial_codec_chunk_frames": first}}
    return SimpleNamespace(connector=SimpleNamespace(config=config))


def _step(manager, frames, finished=False):
    request = SimpleNamespace(request_id="r", external_req_id="r")
    payload = talker2codec_raw_async_chunk(manager, {"codes": {"audio": frames}}, request, is_finished=finished)
    if payload is not None and payload.codes is not None:
        manager.put_req_chunk["r"] += 1
        manager.ramp_chunk_count["r"] += 1
    return payload


def test_first_regular_and_terminal_chunks():
    manager = _manager()
    rows = torch.arange(6 * 4, dtype=torch.int32).reshape(6, 4)
    first = _step(manager, rows[:1])
    assert first.meta.codec_chunk_frames == 1
    assert first.codes.audio.dtype == torch.int64
    assert torch.equal(first.codes.audio, rows[0].long())

    assert _step(manager, rows[1:2]) is None
    padded = torch.full((1, 4), _MOSS_AUDIO_PAD_CODE, dtype=torch.int32)
    assert _step(manager, padded) is None  # PAD rows are not audio
    assert _step(manager, rows[2:3]) is None
    regular = _step(manager, rows[3:4])
    assert regular.meta.codec_chunk_frames == 3
    assert torch.equal(regular.codes.audio, rows[1:4].long().T.reshape(-1))

    tail = _step(manager, rows[4:6], finished=True)
    assert tail.meta.codec_chunk_frames == 2
    assert torch.equal(tail.codes.audio, rows[4:6].long().T.reshape(-1))
    assert bool(tail.meta.finished)
    assert "r" not in manager.code_prompt_token_ids


def test_buffered_rows_do_not_alias_producer_storage():
    manager = _manager(first=2, regular=2)
    frames = torch.tensor([[7, 8, 9]])
    assert _step(manager, frames) is None
    frames.fill_(0)
    chunk = _step(manager, torch.tensor([[1, 2, 3]]))
    assert torch.equal(chunk.codes.audio, torch.tensor([7, 1, 8, 2, 9, 3]))


@pytest.mark.parametrize("total", [1, 3, 5, 10, 13, 28, 31])
def test_ramp_preserves_every_frame_and_terminal(total):
    manager = _manager(first=1, regular=15)
    manager.connector.config["extra"]["codec_chunk_ramp"] = [1, 4, 8, 15]
    rows = torch.arange(total * 4, dtype=torch.int32).reshape(total, 4)
    packets = []
    for index in range(total):
        packet = _step(manager, rows[index : index + 1], finished=index == total - 1)
        if packet is not None:
            packets.append(packet)
    rebuilt = torch.cat([packet.codes.audio.reshape(4, -1).T for packet in packets])
    assert torch.equal(rebuilt, rows.long())
    assert all(not bool(packet.meta.finished) for packet in packets[:-1])
    assert bool(packets[-1].meta.finished)
    assert "r" not in manager.code_prompt_token_ids
    if total == 31:
        assert [packet.meta.codec_chunk_frames for packet in packets] == [1, 4, 8, 15, 3]
