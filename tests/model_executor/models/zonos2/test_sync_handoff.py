# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sync transport must deliver the payload that releases the DAC input gate."""

import pytest
import torch
from vllm import SamplingParams
from vllm.outputs import RequestOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_scheduling_coordinator import (
    OmniSchedulingCoordinator,
    uses_full_payload_input_coordinator,
)
from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.distributed.omni_connectors.model_runner.omni_connector_runtime import (
    should_accumulate_full_payload_output,
)
from vllm_omni.model_executor.models.zonos2.pipeline import ZONOS2_PIPELINE
from vllm_omni.request import OmniRequest
from vllm_omni.worker_v2.omni_data_plane import OmniRunnerDataPlane, _NativeRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_sync_producer_delivers_before_dac_is_scheduled():
    producer, consumer = ZONOS2_PIPELINE.stages
    assert uses_full_payload_input_coordinator(consumer)
    _, hook = OmniRunnerDataPlane._load_custom_func(producer)
    assert should_accumulate_full_payload_output(producer, hook), "Sync DAC waits for an unregistered producer"

    manager = OmniRunnerDataPlane.__new__(OmniRunnerDataPlane)
    manager._custom_process_func = hook
    manager._pending_full_payload_send = {}
    state = _NativeRequestState("r", "external-r", [0])
    raw = torch.arange(27 * 9).reshape(27, 9) % 1024
    for row in raw:
        manager.accumulate_full_payload_output(
            "r", flatten_payload({"codes": {"audio": row[None]}, "meta": {"eos_frame": torch.tensor([17])}}), state
        )
    accumulated, _ = manager._materialize_full_payload_entry(manager._pending_full_payload_send.pop("r"))
    state.finished = True
    payload = manager._build_custom_process_payload(
        "r", state.snapshot(include_token_history=True), accumulated, propagate_errors=True
    )
    assert torch.equal(payload.codes.audio, raw.T)
    assert payload.meta.num_processed_tokens == 17
    assert payload.meta.last_chunk and bool(payload.meta.finished)

    request = OmniRequest("r", [0], SamplingParams(max_tokens=1), pooling_params=None, external_req_id="external-r")
    waiting = create_request_queue(SchedulingPolicy.FCFS)
    waiting.add_request(request)
    coordinator = OmniSchedulingCoordinator(stage_id=1, async_chunk=False)
    coordinator.process_pending_full_payload_inputs(waiting, set())
    assert request.status == RequestStatus.WAITING_FOR_INPUT
    assert not waiting
    assert [handle.request_id for handle in coordinator.pending_input_registrations] == ["r"]
    coordinator.process_pending_full_payload_inputs(waiting, {request.request_id})
    assert request.status == RequestStatus.WAITING
    assert list(waiting) == [request]


def test_sync_prompt_is_token_only_and_skips_unfinished_sources():
    from importlib import import_module

    path = ZONOS2_PIPELINE.stages[1].sync_process_input_func
    module, name = path.rsplit(".", 1)
    convert = getattr(import_module(module), name)
    pending = RequestOutput("pending", None, [], None, [], False)
    finished = RequestOutput("finished", None, [], None, [], True)
    [prompt] = convert([pending, finished])
    assert prompt["prompt_token_ids"] == [0]
    assert not prompt.get("additional_information"), "Codec data belongs to the connector"


@pytest.mark.parametrize("eos", [torch.tensor([]), torch.tensor([-1]), None])
def test_sync_payload_without_natural_eos_keeps_all_raw_frames(eos):
    from vllm_omni.model_executor.stage_input_processors.zonos2 import talker2dac_full_payload

    raw = torch.zeros((3, 9), dtype=torch.long)
    payload = talker2dac_full_payload(None, {"codes.audio": raw, "meta.eos_frame": eos}, None)
    assert payload.meta.num_processed_tokens == 3
    assert torch.equal(payload.codes.audio, raw.T)


def test_sync_payload_rejects_missing_codes():
    from vllm_omni.model_executor.stage_input_processors.zonos2 import talker2dac_full_payload

    with pytest.raises(ValueError, match="requires accumulated audio codes"):
        talker2dac_full_payload(None, {}, None)
