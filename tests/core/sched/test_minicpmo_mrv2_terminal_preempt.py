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


def test_preemption_while_the_terminal_forward_is_in_flight(monkeypatch):
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

    # The stale terminal output and a partial recompute leave the stop pending.
    _deliver(scheduler, request, 1, [999], _rows([12], 3))
    assert not request.is_finished() and request.num_output_placeholders == 0
    _readmit(scheduler, request)
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
