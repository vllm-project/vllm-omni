# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MOSS raw codec rows are buffered into first/regular chunks in codebook-major order."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.model_executor.stage_input_processors.moss_tts import (
    _MOSS_AUDIO_PAD_CODE,
    talker2codec,
    talker2codec_raw_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _manager(first=1, regular=3):
    config = {"extra": {"codec_chunk_frames": regular, "initial_codec_chunk_frames": first}}
    return SimpleNamespace(connector=SimpleNamespace(config=config))


def _step(manager, frames, finished=False):
    request = SimpleNamespace(
        request_id="r", external_req_id="r", additional_information=getattr(manager, "info", None)
    )
    payload = talker2codec_raw_async_chunk(manager, {"codes": {"audio": frames}}, request, is_finished=finished)
    if payload is not None and payload.codes is not None:
        manager.put_req_chunk["r"] += 1
        manager.ramp_chunk_count["r"] += 1
    return payload


@pytest.mark.parametrize("continuation", [False, True])
def test_first_regular_and_terminal_chunks(continuation):
    manager = _manager()
    reference = torch.full((2, 4), 9)
    manager.info = serialize_additional_information(
        {"codes": {"ref": reference}, "meta": {"ref_code_len": 2 if continuation else 0}}
    )
    rows = torch.arange(6 * 4, dtype=torch.int32).reshape(6, 4)
    first = _step(manager, rows[:1])
    expected = torch.cat((reference, rows[:1])) if continuation else rows[:1]
    assert first.meta.codec_chunk_frames == len(expected)
    assert first.meta.left_context_size == (2 if continuation else 0)
    assert first.codes.audio.dtype == torch.int64
    assert torch.equal(first.codes.audio, expected.long().T.reshape(-1))

    assert _step(manager, rows[1:2]) is None
    padded = torch.full((1, 4), _MOSS_AUDIO_PAD_CODE, dtype=torch.int32)
    assert _step(manager, padded) is None  # PAD rows are not audio
    assert _step(manager, rows[2:3]) is None
    regular = _step(manager, rows[3:4])
    assert regular.meta.codec_chunk_frames == 3
    assert regular.meta.left_context_size == 0
    assert torch.equal(regular.codes.audio, rows[1:4].long().T.reshape(-1))

    tail = _step(manager, rows[4:6], finished=True)
    assert tail.meta.codec_chunk_frames == 2
    assert torch.equal(tail.codes.audio, rows[4:6].long().T.reshape(-1))
    assert bool(tail.meta.finished)
    assert "r" not in manager.code_prompt_token_ids
    assert not manager.request_payload


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


@pytest.mark.parametrize("continuation", [False, True])
def test_whole_sequence_bridge_uses_completion_payload_and_mode(continuation):
    codes = torch.tensor([[1024, 1024], [5, 6]])
    output = SimpleNamespace(outputs=[SimpleNamespace(multimodal_output={"codes": {"audio": codes}})])
    info = {
        "codes": {"ref": torch.tensor([[1024, 1024], [1, 2], [3, 4]])},
        "meta": {"ref_code_len": 2 if continuation else 0},
    }
    # StageEngineCoreClient passes outputs, original prompt, then the multimodal flag.
    result = talker2codec([output], {"additional_information": info}, False)[0]
    assert result["prompt_token_ids"] == ([1, 3, 5, 2, 4, 6] if continuation else [5, 6])
    assert result["additional_information"]["meta"]["left_context_size"] == (2 if continuation else 0)
