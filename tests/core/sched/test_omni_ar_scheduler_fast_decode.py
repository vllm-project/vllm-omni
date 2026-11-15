# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The plain-decode short path of update_from_output publishes what the general path does."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.engine import FinishReason
from vllm.v1.request import RequestStatus

from tests.core.sched.test_omni_ar_scheduler_logprobs import (
    _bind_request_lifecycle,
    _make_scheduler_stub,
    _Request,
)
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

EOS = 99
AUDIO = torch.arange(6, dtype=torch.float32)


def _run(spec_decode_tokens: dict) -> tuple[list, dict[str, _Request], SimpleNamespace]:
    requests = [_Request(f"r{i}") for i in range(4)]
    for request in requests:
        request.sampling_params = SimpleNamespace(num_logprobs=None)
        request.num_computed_tokens = 10
        request.num_in_flight_tokens = 1
    # r3 is still being decoded but its output belongs to a later segment.
    requests[3].num_stale_output_tokens = 1
    requests[3].drop_stale_output = True
    scheduler = _make_scheduler_stub(requests)
    calls: list[str] = []

    def update_request(request, token_ids):
        calls.append(request.request_id)
        if token_ids[-1] == EOS:
            request.status = RequestStatus.FINISHED_STOPPED
            return token_ids, True
        return token_ids, False

    _bind_request_lifecycle(scheduler, update_request=update_request)
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={request.request_id: 1 for request in requests},
        scheduled_spec_decode_tokens=spec_decode_tokens,
        num_invalid_spec_tokens=0,
    )
    model_runner_output = SimpleNamespace(
        sampled_token_ids=[[5], [EOS], [6], [7]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        multimodal_outputs=[{"model_outputs": AUDIO}, {}, {}, {}],
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        req_id_to_index={request.request_id: i for i, request in enumerate(requests)},
        routed_experts=None,
    )
    outputs = OmniARScheduler.update_from_output(scheduler, scheduler_output, model_runner_output)
    return (
        list(outputs[0].outputs),
        {request.request_id: request for request in requests},
        SimpleNamespace(scheduler=scheduler, calls=calls),
    )


def _fields(output) -> tuple:
    return (
        output.request_id,
        list(output.new_token_ids),
        output.finish_reason,
        output.stop_reason,
        output.events,
        output.prefill_stats,
        output.kv_transfer_params,
        output.num_generation_tokens,
        output.new_prompt_len_snapshot,
        output.is_segment_finished,
        output.num_nans_in_logits,
        output.pooling_output,
        output.pooling_output_payload,
        None if output.multimodal_output is None else {k: id(v) for k, v in output.multimodal_output.items()},
    )


def test_fast_decode_matches_the_general_path():
    fast_outputs, fast_requests, fast = _run({})
    # Any scheduled spec token sends the whole step down the general path.
    general_outputs, general_requests, general = _run({"elsewhere": [1]})

    assert [_fields(o) for o in fast_outputs] == [_fields(o) for o in general_outputs]
    assert fast.calls == general.calls == ["r0", "r1", "r2"]
    for req_id in fast_requests:
        a, b = fast_requests[req_id], general_requests[req_id]
        assert (a.status, a.output_token_ids, a.num_in_flight_tokens, a.num_stale_output_tokens) == (
            b.status,
            b.output_token_ids,
            b.num_in_flight_tokens,
            b.num_stale_output_tokens,
        )
    assert fast.scheduler.finished_req_ids == general.scheduler.finished_req_ids == {"r1"}
    by_id = {o.request_id: o for o in fast_outputs}
    assert by_id["r1"].finish_reason is FinishReason.STOP
    assert by_id["r0"].multimodal_output["model_outputs"] is AUDIO
    assert "r3" not in by_id
