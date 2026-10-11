# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
from vllm.v1.engine import FinishReason

from tests.core.sched.test_omni_ar_scheduler_logprobs import (
    _bind_request_lifecycle,
    _make_scheduler_stub,
    _Request,
)
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_model_input_failure_terminates_only_bad_request_without_tokens():
    bad, good = _Request("bad"), _Request("good")
    bad.streaming_prompt_continuous = True
    good.sampling_params.num_logprobs = None
    scheduler = _make_scheduler_stub([bad, good])
    updates = []

    def update(request, tokens):
        updates.append(request.request_id)
        return tokens, False

    _bind_request_lifecycle(scheduler, update_request=update)
    scheduled = SimpleNamespace(
        num_scheduled_tokens={"bad": 1, "good": 1}, scheduled_spec_decode_tokens={}, num_invalid_spec_tokens=0
    )
    runner = SimpleNamespace(
        sampled_token_ids=[[7], [8]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        multimodal_outputs=None,
        inter_stage_outputs=None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        req_id_to_index={"bad": 0, "good": 1},
        routed_experts=None,
        model_input_errors={"bad": "invalid audio unit"},
    )
    outputs = OmniARScheduler.update_from_output(scheduler, scheduled, runner)
    by_id = {output.request_id: output for output in outputs[0].outputs}
    assert by_id["bad"].finish_reason is FinishReason.ERROR
    assert by_id["bad"].new_token_ids == []
    assert by_id["good"].new_token_ids == [8]
    assert updates == ["good"]
    assert not bad.resumable and not bad.streaming_prompt_continuous
    assert "bad" not in scheduler.requests and "good" in scheduler.requests
