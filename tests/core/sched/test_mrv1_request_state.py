# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MRV1 token history survives the native scheduler's pause/resend protocol."""

from unittest.mock import Mock

import pytest
import torch
from vllm import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.request import Request

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.worker import gpu_model_runner
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
from vllm_omni.worker.v1_compat import input_batch
from vllm_omni.worker.v1_compat.input_batch import CachedRequestState, InputBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _runner(monkeypatch, request, *, cached=True):
    monkeypatch.setattr(input_batch, "PIN_MEMORY", False)
    monkeypatch.setattr(gpu_model_runner, "get_pp_group", lambda: Mock(is_last_rank=True))
    runner = object.__new__(OmniGPUModelRunner)
    runner.input_batch = InputBatch(
        max_num_reqs=1,
        max_model_len=16,
        max_num_batched_tokens=16,
        device=torch.device("cpu"),
        vocab_size=32,
        block_sizes=[1],
        max_num_blocks_per_req=[16],
    )
    state = CachedRequestState(
        req_id=request.request_id,
        prompt_token_ids=[1, 2],
        mm_features=[],
        sampling_params=request.sampling_params,
        generator=None,
        block_ids=([1, 2, 3, 4],),
        num_computed_tokens=2,
        output_token_ids=[3],
    )
    runner.requests = {request.request_id: state} if cached else {}
    if cached:
        runner.input_batch.add_request(state)
    runner.model = torch.nn.Module()
    runner.model_intermediate_buffer = {}
    runner.num_prompt_logprobs = {}
    runner.encoder_cache = {}
    runner.late_interaction_runner = Mock()
    runner.speculative_config = None
    runner.use_async_scheduling = True
    runner.use_async_spec_decode = False
    runner.is_pooling_model = False
    runner.uses_mrope = False
    runner._may_reorder_batch = lambda output: None
    return runner


def _scheduler(request, *, use_v2=False):
    scheduler = object.__new__(OmniARScheduler)
    scheduler.requests = {request.request_id: request}
    scheduler.prev_step_scheduled_req_ids = {request.request_id}
    scheduler.use_v2_model_runner = use_v2
    scheduler.chunk_transfer_adapter = None
    return scheduler


def _cached_output(request):
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {request.request_id: 1}
    output.total_num_scheduled_tokens = 1
    output.scheduled_cached_reqs = CachedRequestData(
        req_ids=[request.request_id],
        new_token_ids=[],
        new_block_ids=[None],
        num_computed_tokens=[3],
        num_output_tokens=[2],
    )
    return output


def test_async_cached_request_recovers_history_after_unscheduled_step(monkeypatch):
    request = Request("paused", [1, 2], SamplingParams(max_tokens=8), pooling_params=None)
    request.append_output_token_ids([3, 4])
    scheduler = _scheduler(request)
    runner = _runner(monkeypatch, request)

    paused = SchedulerOutput.make_empty()
    scheduler._postprocess_omni_schedule_output(paused)
    runner._update_states(paused)
    assert request.request_id not in runner.input_batch.req_id_to_index

    resumed = _cached_output(request)
    scheduler._postprocess_omni_schedule_output(resumed)
    runner._update_states(resumed)

    assert runner.requests[request.request_id].output_token_ids == [3, 4]
    assert runner.input_batch.token_ids_cpu[0, :4].tolist() == [1, 2, 3, 4]
    assert resumed.scheduled_cached_reqs.all_token_ids[request.request_id] == [1, 2, 3, 4]
    # The snapshot must not alias later scheduler-side token updates.
    request.append_output_token_ids(5)
    assert resumed.scheduled_cached_reqs.all_token_ids[request.request_id] == [1, 2, 3, 4]

    consecutive = _cached_output(request)
    scheduler._postprocess_omni_schedule_output(consecutive)
    assert consecutive.scheduled_cached_reqs.all_token_ids == {}


def test_native_v2_cached_output_keeps_its_upstream_contract():
    request = Request("native", [1, 2], SamplingParams(max_tokens=8), pooling_params=None)
    scheduler = _scheduler(request, use_v2=True)
    output = _cached_output(request)
    cached = output.scheduled_cached_reqs
    scheduler._postprocess_omni_schedule_output(output)
    assert output.scheduled_cached_reqs is cached


@pytest.mark.parametrize("cached", [False, True], ids=["worker-state-dropped", "worker-state-retained"])
@pytest.mark.parametrize("replacement", [False, True], ids=["preempted-full-resend", "streaming-prompt-replacement"])
def test_full_request_resend_preserves_generated_suffix(monkeypatch, cached, replacement):
    prompt = [5, 6, 7] if replacement else [1, 2]
    expected_output = [] if replacement else [3, 4]
    request = Request("resend", prompt, SamplingParams(max_tokens=8), pooling_params=None)
    request.append_output_token_ids(expected_output)
    runner = _runner(monkeypatch, request, cached=cached)
    if cached and not replacement:
        runner.requests[request.request_id].prev_num_draft_len = 2
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [NewRequestData.from_request(request, ([8, 9, 10, 11],), request.prefill_token_ids)]
    output.num_scheduled_tokens = {request.request_id: 1}
    output.total_num_scheduled_tokens = 1

    runner._update_states(output)

    state = runner.requests[request.request_id]
    assert state.prompt_token_ids == prompt
    assert state.output_token_ids == expected_output
    if not replacement:
        assert state.prev_num_draft_len == 0
    assert state.block_ids == ([8, 9, 10, 11],)
    expected_tokens = prompt + expected_output
    assert runner.input_batch.token_ids_cpu[0, : len(expected_tokens)].tolist() == expected_tokens
