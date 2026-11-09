# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preemption of a MiniCPM-o length stop that waits for its terminal forward.

The real _preempt_request resets the request; the stale terminal output and
the recompute chunks then run through the real update path. Admission and
schedule() increments are inputs; block return is recorded.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.request import RequestStatus

import vllm_omni.core.sched.omni_ar_scheduler as ar_sched_mod
from tests.core.sched.test_minicpmo_mrv2_terminal_cancel import _book_terminal_forward, _cancellable
from tests.core.sched.test_minicpmo_mrv2_terminal_row import _step, _StepCompleteError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _rows(ids, first_position):
    return {
        "latent": torch.zeros(len(ids), 2),
        "latent_input_ids": torch.tensor(ids).reshape(-1, 1),
        "latent_positions": torch.arange(first_position, first_position + len(ids)).reshape(-1, 1),
    }


def _deliver(scheduler, request, scheduled, sampled, payload):
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={request.request_id: scheduled},
        scheduled_spec_decode_tokens={},
        num_invalid_spec_tokens=0,
        finished_req_ids=set(),
    )
    runner_output = SimpleNamespace(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[list(sampled)],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        multimodal_outputs=[payload],
        inter_stage_outputs=[payload],
        num_nans_in_logits=None,
        kv_connector_output=None,
        ec_connector_output=None,
        cudagraph_stats=None,
    )
    with pytest.raises(_StepCompleteError):
        scheduler.update_from_output(scheduler_output, runner_output)


def _readmit(scheduler, request):
    # schedule() takes the preempted request back into the running queue.
    scheduler.waiting.remove_requests([request])
    request.status = RequestStatus.RUNNING
    scheduler.running = [request]


def _book_chunk(request, chunk, placeholder):
    # schedule()'s increments for one recompute chunk.
    request.num_computed_tokens += chunk
    request.num_in_flight_tokens += chunk
    request.num_output_placeholders += placeholder


def _preempt_terminal_forward(monkeypatch):
    scheduler, request, _ = _cancellable(monkeypatch)
    scheduler._free_request_blocks = lambda req: None
    scheduler._inflight_prefills = set()
    scheduler.log_stats = False
    scheduler.reset_preempted_req_ids = set()
    request.num_preemptions = 0
    _step(scheduler, request, [], [12])
    assert request.request_id in scheduler._minicpmo45_terminal_rows
    _book_terminal_forward(scheduler, request)

    scheduler.running.remove(request)
    scheduler._preempt_request(request, 0.0)
    assert request.status == RequestStatus.PREEMPTED and request.num_computed_tokens == 0

    # The stale terminal output leaves the stop pending.
    _deliver(scheduler, request, 1, [999], _rows([12], 3))
    assert not request.is_finished() and request.num_output_placeholders == 0
    _readmit(scheduler, request)
    return scheduler, request


def test_preemption_while_the_terminal_forward_is_in_flight(monkeypatch):
    scheduler, request = _preempt_terminal_forward(monkeypatch)
    _book_chunk(request, 2, 0)
    request.spec_token_ids = [-1, -1, -1]
    assert scheduler._hold_minicpmo45_terminal_steps() == [] and request.spec_token_ids == []
    _deliver(scheduler, request, 2, [], _rows([1, 2], 0))
    assert not request.is_finished() and request.num_output_placeholders == 0
    assert request.request_id in scheduler._minicpmo45_terminal_rows

    # The chunk that reaches the final token supplies its own row and restores the stop.
    _book_chunk(request, 2, 1)
    _deliver(scheduler, request, 2, [999], _rows([11, 12], 2))
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert request.output_token_ids == [11, 12]
    assert request.num_output_placeholders == 0
    assert request.request_id not in scheduler._minicpmo45_terminal_rows


def test_overlapping_recompute_chunks_after_preemption(monkeypatch):
    scheduler, request = _preempt_terminal_forward(monkeypatch)
    # Async scheduling books the final chunk before the partial chunk's output.
    _book_chunk(request, 2, 0)
    _book_chunk(request, 2, 1)
    assert request.num_computed_tokens == request.num_tokens

    _deliver(scheduler, request, 2, [], _rows([1, 2], 0))
    assert not request.is_finished() and request.num_output_placeholders == 1
    assert request.request_id in scheduler._minicpmo45_terminal_rows
    # The final chunk is still in flight, so nothing else may be scheduled.
    request.next_decode_eligible_step = 0
    held = scheduler._hold_minicpmo45_terminal_steps()
    assert held == [(request, 0)] and request.next_decode_eligible_step == ar_sched_mod._HELD_DECODE_STEP
    scheduler._release_minicpmo45_terminal_holds(held)
    assert request.next_decode_eligible_step == 0

    _deliver(scheduler, request, 2, [999], _rows([11, 12], 2))
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert request.output_token_ids == [11, 12]
    assert request.num_output_placeholders == 0
    assert request.request_id not in scheduler._minicpmo45_terminal_rows


def test_stale_output_commits_the_final_token_after_preemption(monkeypatch):
    scheduler, request, _ = _cancellable(monkeypatch)
    scheduler._free_request_blocks = lambda req: None
    scheduler._inflight_prefills = set()
    scheduler.log_stats = False
    scheduler.reset_preempted_req_ids = set()
    request.num_preemptions = 0
    # The decode step that samples the final token is in flight when the request is preempted.
    _book_chunk(request, 1, 1)
    scheduler.running.remove(request)
    scheduler._preempt_request(request, 0.0)

    # Its stale output commits the final token; the recompute will take it as input.
    _deliver(scheduler, request, 1, [12], _rows([11], 2))
    assert request.status == RequestStatus.PREEMPTED and request.output_token_ids == [11, 12]
    assert request.request_id in scheduler._minicpmo45_terminal_rows

    _readmit(scheduler, request)
    _book_chunk(request, 2, 0)
    _book_chunk(request, 2, 1)
    _deliver(scheduler, request, 2, [], _rows([1, 2], 0))
    assert not request.is_finished() and request.num_output_placeholders == 1
    _deliver(scheduler, request, 2, [999], _rows([11, 12], 2))
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert request.output_token_ids == [11, 12]
    assert request.num_output_placeholders == 0
    assert request.request_id not in scheduler._minicpmo45_terminal_rows


def test_negative_in_flight_count_is_logged(monkeypatch):
    scheduler, request, _ = _cancellable(monkeypatch)
    scheduler._free_request_blocks = lambda req: None
    scheduler._inflight_prefills = set()
    scheduler.log_stats = False
    warnings = []
    monkeypatch.setattr(ar_sched_mod.logger, "warning", lambda *args, **kwargs: warnings.append(args))
    _step(scheduler, request, [], [12])
    _book_terminal_forward(scheduler, request)
    # An accounting bug elsewhere leaves the counter one short.
    request.num_in_flight_tokens -= 1

    _deliver(scheduler, request, 1, [999], _rows([12], 3))
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert request.num_output_placeholders == 0
    assert [w[1] for w in warnings if "negative" in w[0]] == [request.request_id]
