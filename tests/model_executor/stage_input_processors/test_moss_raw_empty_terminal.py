# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Empty MOSS terminals must not execute prewarmed or previous codec tokens."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.core.sched.omni_scheduling_coordinator import OmniSchedulingCoordinator
from vllm_omni.data_entry_keys import to_dict
from vllm_omni.distributed.omni_connectors.model_runner.omni_connector_payload_transport import (
    _OmniConnectorPayloadTransportMixin as Transport,
)
from vllm_omni.model_executor.stage_input_processors.moss_tts import (
    _MOSS_AUDIO_PAD_CODE,
    talker2codec_raw_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request(tokens):
    return SimpleNamespace(
        request_id="r",
        external_req_id="r",
        prompt_token_ids=list(tokens),
        num_prompt_tokens=len(tokens),
        _all_token_ids=list(tokens),
        _output_token_ids=[],
        num_computed_tokens=0,
    )


def _request_metadata(payload, model_mode="generation"):
    transport = Transport.__new__(Transport)
    transport._async_chunk = True
    transport._model_mode = model_mode
    update = transport._extract_scheduling_metadata_update(payload)
    return {"r": update} if update is not None else {}


@pytest.mark.parametrize("after_audio", [False, True])
def test_empty_terminal_clears_native_codec_prompt(after_audio):
    manager = SimpleNamespace(
        connector=SimpleNamespace(config={"extra": {"initial_codec_chunk_frames": 1, "codec_chunk_frames": 3}})
    )
    source = _request([0] * 88)
    previous = [0] * 88
    if after_audio:
        packet = talker2codec_raw_async_chunk(manager, {"codes": {"audio": torch.ones(1, 12)}}, source)
        manager.put_req_chunk["r"] += 1
        previous = packet.codes.audio.tolist()
    terminal = talker2codec_raw_async_chunk(
        manager,
        {"codes": {"audio": torch.full((1, 12), _MOSS_AUDIO_PAD_CODE)}},
        source,
        is_finished=True,
    )
    payload = to_dict(terminal)
    assert not Transport._payload_is_consumable(payload)
    metadata = _request_metadata(payload)
    receiver = _request(previous)
    coordinator = OmniSchedulingCoordinator(stage_id=1)
    coordinator.update_request_metadata({"r": receiver}, metadata)
    assert receiver.prompt_token_ids == []
    assert receiver._all_token_ids == []
    assert receiver.num_prompt_tokens == receiver.num_computed_tokens == 0
    assert "r" in coordinator.input_terminal_req_ids
    assert "r" not in manager.code_prompt_token_ids


@pytest.mark.parametrize("empty", [[], torch.empty(0, dtype=torch.long)])
def test_explicit_empty_codes_clear_generation_prompt(empty):
    request = _request([17, 18, 19])
    request._output_token_ids = [23]
    request.num_computed_tokens = 3
    coordinator = OmniSchedulingCoordinator(stage_id=1)
    coordinator.update_request_metadata(
        {"r": request}, _request_metadata({"codes": {"audio": empty}, "meta": {"finished": True}})
    )
    assert request.prompt_token_ids == request._all_token_ids == request._output_token_ids == []
    assert request.num_prompt_tokens == request.num_computed_tokens == 0
    assert "r" in coordinator.input_terminal_req_ids


@pytest.mark.parametrize("payload", [{"meta": {"finished": True}}, {"codes": {"audio": None}}])
def test_absent_codes_preserve_generation_prompt(payload):
    request = _request([17, 18, 19])
    coordinator = OmniSchedulingCoordinator(stage_id=1)
    coordinator.update_request_metadata({"r": request}, _request_metadata(payload))
    assert request.prompt_token_ids == request._all_token_ids == [17, 18, 19]


def test_ar_prompt_is_not_replaced_by_empty_audio_codes():
    request = _request([17, 18, 19])
    coordinator = OmniSchedulingCoordinator(stage_id=1)
    coordinator.update_request_metadata(
        {"r": request}, _request_metadata({"codes": {"audio": []}, "meta": {"finished": True}}, model_mode="ar")
    )
    assert request.prompt_token_ids == request._all_token_ids == [17, 18, 19]
