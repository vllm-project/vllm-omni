# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""AR prompt replacement through native admission, stops, and stale output drain."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Omni patches Request and StreamingUpdate before these native imports bind them.
# isort: off
import vllm_omni  # noqa: F401
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.v1.engine import FinishReason
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from tests.core.sched.test_omni_ar_scheduler_streaming import (
    _make_live_session_scheduler,
    _make_request,
    _make_update,
    _run_idle_step,
)
# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _scheduler(*, max_model_len: int = 64):
    sched = _make_live_session_scheduler(max_model_len=max_model_len)
    sched._pooling_output_decoder = None
    sched.aux_output_connector = None
    sched._native_data_plane = False
    sched.structured_output_manager = MagicMock()
    sched.structured_output_manager.accept_tokens.return_value = True
    sched._process_kv_transfer_trigger = MagicMock(return_value=False)
    sched._update_request_with_output = MagicMock(wraps=sched._update_request_with_output)
    sched._free_request = MagicMock(wraps=sched._free_request)
    return sched


def _owner(sched, *, request_id: str = "ar-owner", running: bool = False):
    request = _make_request()
    request.request_id = request_id
    request.external_req_id = "public-session"
    request.session_id = "public-session"
    request.resumable = True
    request.streaming_queue = deque()
    request.model_intermediate_buffer = {"meta": {"session_epoch": 7}, "old": [1]}
    request.additional_information = {"old": "metadata"}
    request.num_computed_tokens = len(request.prompt_token_ids)
    request.status = RequestStatus.RUNNING if running else RequestStatus.WAITING_FOR_STREAMING_REQ
    sched.requests[request_id] = request
    if running:
        sched.running.append(request)
    else:
        sched.kv_holding_waiting.add_request(request)
        sched.deferred_waiting.add(request)
        sched.num_waiting_for_streaming_input += 1
    return request


def _replacement(*, prompt_len: int = 4, flag_in: str = "model_intermediate_buffer") -> StreamingUpdate:
    update = _make_update(list(range(20, 20 + prompt_len)))
    update.max_tokens = 17
    update.sampling_params = SamplingParams(max_tokens=17)
    update.model_intermediate_buffer = {"meta": {"session_epoch": 7}, "conditioning": [[2, 3]]}
    update.additional_information = {"new": "metadata"}
    getattr(update, flag_in).setdefault("meta", {})["replace_streaming_prompt"] = True
    return update


def _queue_replacement(sched, owner, update):
    """Native add_request must enqueue a typed update while this owner is RUNNING."""
    incoming = Request(
        request_id=owner.request_id,
        prompt_token_ids=update.prompt_token_ids,
        sampling_params=update.sampling_params,
        pooling_params=None,
        arrival_time=update.arrival_time,
        block_hasher=None,
        additional_information=update.additional_information,
        model_intermediate_buffer=update.model_intermediate_buffer,
        resumable=True,
    )
    VLLMScheduler.add_request(sched, incoming)
    assert sched.requests[owner.request_id] is owner
    assert len(owner.streaming_queue) == 1
    assert owner.prompt_token_ids == [1, 2, 3]
    queued = owner.streaming_queue[0]
    assert isinstance(queued, StreamingUpdate)
    assert queued.model_intermediate_buffer is update.model_intermediate_buffer
    return queued


def _output_step(sched, owner, *, token: int, num_scheduled: int = 1):
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={owner.request_id: num_scheduled},
        scheduled_spec_decode_tokens={},
        num_invalid_spec_tokens=0,
    )
    runner_output = SimpleNamespace(
        sampled_token_ids=[[token]],
        logprobs=None,
        prompt_logprobs_dict={},
        prompt_token_id_logprobs_dict={},
        pooler_output=None,
        multimodal_outputs=None,
        inter_stage_outputs=None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        req_id_to_index={owner.request_id: 0},
        routed_experts=None,
    )
    return sched.update_from_output(scheduler_output, runner_output)


@pytest.mark.parametrize("flag_in", ["model_intermediate_buffer", "additional_information"])
def test_inactive_replacement_rebuilds_prompt_and_metadata_on_same_owner(flag_in):
    sched = _scheduler(max_model_len=8)
    owner = _owner(sched)
    owner.append_output_token_ids([7, 8, 9])
    owner.num_computed_tokens = 6
    owner.spec_token_ids = [99]
    old_sampling_params = owner.sampling_params
    update = _replacement(prompt_len=7, flag_in=flag_in)

    # Old prompt + retained output + seven new tokens would overflow. A full
    # replacement fits and reserves one sample position.
    sched._update_request_as_session(owner, update)

    assert sched.requests[owner.request_id] is owner
    assert owner.request_id == "ar-owner"
    assert owner.external_req_id == owner.session_id == "public-session"
    assert owner.resumable
    assert owner.prompt_token_ids == list(range(20, 27))
    assert list(owner.all_token_ids) == list(range(20, 27))
    assert list(owner.output_token_ids) == []
    assert owner.num_prompt_tokens == 7
    assert owner.num_computed_tokens == 0
    assert owner.spec_token_ids == []
    assert owner._omni_segment_generation == 1
    assert owner.model_intermediate_buffer is update.model_intermediate_buffer
    assert owner.model_intermediate_buffer["meta"]["session_epoch"] == 7
    assert owner.additional_information is update.additional_information
    assert owner.sampling_params is update.sampling_params
    assert owner.sampling_params is not old_sampling_params
    assert owner.max_tokens == 17
    assert owner.arrival_time == update.arrival_time
    assert owner.status == RequestStatus.WAITING
    assert list(sched.waiting) == [owner]
    assert list(sched.kv_holding_waiting) == []
    assert owner not in sched.deferred_waiting
    assert sched.num_waiting_for_streaming_input == 0
    sched._free_request_blocks.assert_called_once_with(owner)
    sched.encoder_cache_manager.free.assert_called_once_with(owner)
    sched._free_request.assert_not_called()


@pytest.mark.parametrize(
    "pending",
    ["running", "prefill", "num_in_flight_tokens", "num_output_placeholders", "num_stale_output_tokens", "chunk"],
)
def test_unsafe_replacement_rejects_without_half_modifying_prompt(pending):
    sched = _scheduler()
    owner = _owner(sched, running=pending == "running")
    owner.append_output_token_ids([7, 8])
    owner.num_computed_tokens = 5
    owner.spec_token_ids = [99]
    if pending == "prefill":
        sched._inflight_prefills.add(owner)
    elif pending == "chunk":
        owner.status = RequestStatus.WAITING_FOR_CHUNK
    elif pending != "running":
        setattr(owner, pending, 1)
    before = (
        list(owner.prompt_token_ids),
        list(owner.all_token_ids),
        list(owner.output_token_ids),
        owner.num_computed_tokens,
        owner.model_intermediate_buffer,
        owner.additional_information,
        owner.sampling_params,
        owner.max_tokens,
        owner.arrival_time,
        list(owner.spec_token_ids),
    )

    sched._update_request_as_session(owner, _replacement())

    after = (
        list(owner.prompt_token_ids),
        list(owner.all_token_ids),
        list(owner.output_token_ids),
        owner.num_computed_tokens,
        owner.model_intermediate_buffer,
        owner.additional_information,
        owner.sampling_params,
        owner.max_tokens,
        owner.arrival_time,
        list(owner.spec_token_ids),
    )
    assert after == before
    assert owner.status == RequestStatus.FINISHED_ERROR
    assert owner.request_id not in sched.requests
    assert "streaming_prompt_replacement_rejected" in sched._streaming_context_overflow[owner.request_id][1]
    assert owner.request_id not in sched._new_prompt_len_snapshot
    sched._free_request.assert_called_once_with(owner, delay_free_blocks=False)
    assert not hasattr(owner, "_omni_segment_generation")


def test_native_queued_completed_stop_replaces_and_drains_two_old_frames():
    sched = _scheduler()
    owner = _owner(sched, running=True)
    owner.max_tokens = 1
    owner.num_computed_tokens = 5
    owner.num_output_placeholders = 2
    owner.num_in_flight_tokens = 3  # completed frame, then two old lookahead frames
    owner.spec_token_ids = [90, 91]
    update = _replacement()
    _queue_replacement(sched, owner, update)

    outputs = _output_step(sched, owner, token=42)

    # The real native append/check_stop reaches LENGTH before the queued update
    # is applied; no manually assigned finished status or stopped-result mock.
    sched._update_request_with_output.assert_called_once_with(owner, [42])
    assert any(out.finish_reason == FinishReason.LENGTH for out in outputs[owner.client_index].outputs)
    assert sched.requests[owner.request_id] is owner
    assert owner.status == RequestStatus.WAITING
    assert owner not in sched.running
    assert list(sched.waiting) == [owner]
    assert owner.prompt_token_ids == update.prompt_token_ids
    assert owner.num_computed_tokens == 0
    assert owner.max_tokens == 17
    assert owner.num_stale_output_tokens == owner.num_in_flight_tokens == 2
    assert owner.drop_stale_output
    assert owner.num_output_placeholders == 0
    assert not owner.streaming_queue
    assert getattr(sched, "_completed_streaming_segment_stop", None) is None
    sched._free_request.assert_not_called()
    sched._free_request_blocks.assert_called_once_with(owner)

    assert _output_step(sched, owner, token=80) == {}
    assert _output_step(sched, owner, token=81) == {}
    assert owner.num_stale_output_tokens == owner.num_in_flight_tokens == 0
    assert list(owner.output_token_ids) == []
    assert sched._update_request_with_output.call_count == 1
    sched.waiting.remove_requests((owner,))
    owner.status = RequestStatus.RUNNING
    sched.running.append(owner)
    owner.num_computed_tokens = owner.num_prompt_tokens
    owner.num_in_flight_tokens = 1

    outputs = _output_step(sched, owner, token=55)

    assert list(owner.output_token_ids) == [55]
    assert outputs[owner.client_index].outputs[0].new_token_ids == [55]
    assert sched._update_request_with_output.call_count == 2
    assert owner.num_in_flight_tokens == 0
    sched._free_request.assert_not_called()


@pytest.mark.parametrize("prompt_len", [8, 9])
@pytest.mark.parametrize("queued", [False, True])
def test_replacement_capacity_error_is_local_and_frees_queued_owner_once(prompt_len, queued):
    sched = _scheduler(max_model_len=8)
    owner = _owner(sched, running=queued)
    peer = _owner(sched, request_id="independent-owner")
    peer_before = (list(peer.prompt_token_ids), peer.status, peer.model_intermediate_buffer)
    update = _replacement(prompt_len=prompt_len)
    old_payload = owner.model_intermediate_buffer
    if queued:
        owner.max_tokens = 1
        owner.num_in_flight_tokens = 1
        _queue_replacement(sched, owner, update)
        outputs = _output_step(sched, owner, token=42)
    else:
        sched._update_request_as_session(owner, update)
        outputs = _run_idle_step(sched)

    assert owner.status == RequestStatus.FINISHED_ERROR
    if queued:
        assert owner.resumable is False
    assert owner.request_id not in sched.requests
    assert owner.prompt_token_ids == [1, 2, 3]
    assert owner.model_intermediate_buffer is old_payload
    assert not hasattr(owner, "_omni_segment_generation")
    assert all(request is not owner for request in sched.waiting)
    assert all(request is not owner for request in sched.kv_holding_waiting)
    assert sched.requests[peer.request_id] is peer
    assert (list(peer.prompt_token_ids), peer.status, peer.model_intermediate_buffer) == peer_before
    assert peer in sched.kv_holding_waiting
    assert sched.num_waiting_for_streaming_input == 1
    sched._free_request.assert_called_once()
    assert sched._free_request.call_args.args[0] is owner
    sched._free_request_blocks.assert_called_once_with(owner)
    errors = [out for out in outputs[owner.client_index].outputs if out.finish_reason == FinishReason.ERROR]
    assert len(errors) == 1
    assert errors[0].request_id == owner.request_id
    assert f"{prompt_len} tokens" in errors[0].stop_reason
    assert "max_model_len 8" in errors[0].stop_reason
    assert _run_idle_step(sched) == {}


@pytest.mark.parametrize(("sample_room", "prompt_len", "accepted"), [(2, 6, True), (2, 7, False)])
def test_replacement_reserves_every_token_sampled_per_step(sample_room, prompt_len, accepted):
    sched = _scheduler(max_model_len=8)
    sched.num_sampled_tokens_per_step = sample_room
    owner = _owner(sched)
    update = _replacement(prompt_len=prompt_len)

    sched._update_request_as_session(owner, update)

    if accepted:
        assert owner.status == RequestStatus.WAITING
        assert owner.prompt_token_ids == update.prompt_token_ids
        assert owner.num_computed_tokens == 0
        sched._free_request.assert_not_called()
    else:
        assert owner.status == RequestStatus.FINISHED_ERROR
        assert owner.prompt_token_ids == [1, 2, 3]
        sched._free_request.assert_called_once()
