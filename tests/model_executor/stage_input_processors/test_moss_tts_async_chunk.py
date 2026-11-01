# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Progressive Local codec chunks preserve frames and agree with graph shapes."""

from collections import defaultdict
from dataclasses import dataclass, field
from unittest.mock import Mock

import pytest
import torch
from transformers import PretrainedConfig
from vllm.config import CompilationConfig, ModelConfig, SchedulerConfig, VllmConfig

from vllm_omni.data_entry_keys import OmniPayloadStruct
from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import MossTTSCodecDecoder
from vllm_omni.model_executor.stage_input_processors.moss_tts import talker2codec_raw_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _Connector:
    config: dict[str, dict[str, object]]


@dataclass
class _TransferManager:
    connector: _Connector
    code_prompt_token_ids: dict[str, list[object]] = field(default_factory=lambda: defaultdict(list))
    request_payload: dict[str, object] = field(default_factory=dict)
    put_req_chunk: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    ramp_chunk_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class _Request:
    external_req_id: str


def _manager(ramp: object = None, *, first: int = 1) -> _TransferManager:
    extra = {"initial_codec_chunk_frames": first, "codec_chunk_frames": 15, "codec_chunk_ramp": ramp}
    return _TransferManager(_Connector({"extra": extra}))


def _emit(
    manager: _TransferManager, req_id: str, value: int | None, finished: bool = False
) -> OmniPayloadStruct | None:
    output = None if value is None else {"codes": {"audio": torch.full((1, 4), value)}}
    payload = talker2codec_raw_async_chunk(manager, output, _Request(req_id), finished)
    if payload is not None:
        manager.put_req_chunk[req_id] += 1
        manager.ramp_chunk_count[req_id] += 1
    return payload


@pytest.mark.parametrize(
    "ramp,first,expected",
    [
        (None, 1, [1, 15, 15, 9]),
        ([1, 2, 4, 8, 15], 1, [1, 2, 4, 8, 15, 10]),
        ([1, 4, 15], 7, [1, 4, 15, 15, 5]),
        ([4, 8, 15], 1, [4, 8, 15, 13]),
        ([1, 20, 15], 1, [1, 20, 15, 4]),
        ("1,4,15", 1, [1, 4, 15, 15, 5]),
        ([1, 0, 15], 1, [1, 15, 15, 9]),
        ([1], 1, [1, 15, 15, 9]),
        ("invalid", 1, [1, 15, 15, 9]),
    ],
)
def test_order_and_final_flush(ramp: object, first: int, expected: list[int]) -> None:
    manager = _manager(ramp, first=first)
    packets = []
    for index in range(40):
        packet = _emit(manager, "a", index, finished=index == 39)
        if packet is not None:
            packets.append(packet)
    assert [p.meta.codec_chunk_frames for p in packets] == expected
    actual = torch.cat([p.codes.audio.reshape(4, -1).T for p in packets])
    assert torch.equal(actual, torch.arange(40).unsqueeze(1).expand(-1, 4))
    assert all(not bool(p.meta.finished) for p in packets[:-1])
    assert bool(packets[-1].meta.finished)
    assert "a" not in manager.code_prompt_token_ids


def test_requests_progress_independently_and_empty_finish() -> None:
    manager = _manager([1, 2, 4, 8, 15])
    for name in ("a", "b"):
        first = _emit(manager, name, 0)
        assert first is not None
        assert first.meta.codec_chunk_frames == 1
    assert _emit(manager, "a", 1) is None
    final_a = _emit(manager, "a", None, finished=True)
    assert final_a is not None
    assert final_a.meta.codec_chunk_frames == 1
    assert bool(final_a.meta.finished)
    final_b = _emit(manager, "b", None, finished=True)
    assert final_b is not None
    assert final_b.meta.codec_chunk_frames == 0
    assert bool(final_b.meta.finished)
    assert final_b.codes.audio.numel() == 0
    assert not manager.code_prompt_token_ids


@pytest.mark.parametrize(
    "ramp,first,expected,max_step,effective_first",
    [
        (None, 1, [1, 15], 15, 1),
        ([1, 2, 4, 8, 15], 1, [1, 2, 4, 8, 15], 15, 1),
        ([1, 20, 15], 1, [1, 15, 20], 20, 1),
        ([4, 8, 15], 1, [4, 8, 15], 15, 4),
        ([1, 4, 15], 7, [1, 4, 15], 15, 1),
        ([1, 0, 15], 1, [1, 15], 15, 1),
    ],
)
def test_graph_shapes_follow_sender(
    ramp: object, first: int, expected: list[int], max_step: int, effective_first: int
) -> None:
    connector = _manager(ramp, first=first).connector.config
    config = Mock(
        spec=VllmConfig,
        model_config=Mock(
            spec=ModelConfig,
            hf_config=PretrainedConfig(model_type="moss_tts_local"),
            async_chunk=True,
            use_v2_model_runner=True,
            enforce_eager=False,
            stage_connector_config=connector,
        ),
        scheduler_config=Mock(spec=SchedulerConfig, max_num_seqs=8),
        compilation_config=Mock(spec=CompilationConfig, cudagraph_capture_sizes=[1, 2, 4, 8]),
    )
    codec = MossTTSCodecDecoder(vllm_config=config)
    assert codec._streaming_graph_frame_sizes == expected
    assert codec._stream_max_step_frames == max_step
    assert codec._initial_stream_chunk_frames == effective_first
