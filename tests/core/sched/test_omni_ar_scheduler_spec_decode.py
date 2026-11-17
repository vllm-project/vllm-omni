# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU contract tests for Omni speculative-decoding bookkeeping."""

from types import SimpleNamespace

import pytest

from tests.core.sched.test_omni_ar_scheduler_logprobs import (
    _bind_request_lifecycle,
    _make_scheduler_stub,
    _Request,
)
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _runner_output(sampled_token_ids: list[list[int]]) -> SimpleNamespace:
    req_ids = [f"r{i}" for i in range(len(sampled_token_ids))]
    return SimpleNamespace(
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        multimodal_outputs=[{} for _ in req_ids],
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        routed_experts=None,
    )


def test_spec_decode_bookkeeping_keeps_per_request_accepted_counts() -> None:
    """Requests in one batch may accept different numbers of draft tokens."""
    requests = [_Request("r0"), _Request("r1")]
    for request in requests:
        request.sampling_params = SimpleNamespace(num_logprobs=None)
        request.num_computed_tokens = 10
        request.num_in_flight_tokens = 0

    scheduler = _make_scheduler_stub(requests)
    _bind_request_lifecycle(scheduler, update_request=lambda _request, token_ids: (token_ids, False))
    stats = []
    scheduler.make_spec_decoding_stats = lambda _stats, **kwargs: (stats.append(kwargs), _stats)[1]

    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"r0": 4, "r1": 2},
        scheduled_spec_decode_tokens={"r0": [1, 2, 3], "r1": [4, 5, 6]},
        num_invalid_spec_tokens=0,
    )
    outputs = OmniARScheduler.update_from_output(
        scheduler,
        scheduler_output,
        _runner_output([[10, 11, 12, 13], [20, 21]]),
    )

    assert [(item["request_id"], item["num_accepted_tokens"]) for item in stats] == [
        ("r0", 3),
        ("r1", 1),
    ]
    assert [output.new_token_ids for output in outputs[0].outputs] == [[10, 11, 12, 13], [20, 21]]


@pytest.mark.parametrize(
    ("prompt_len", "computed_after_schedule", "expected_computed", "expected_placeholders"),
    [
        pytest.param(3, 14, 10, 0, id="decode-rolls-back-base-and-drafts"),
        pytest.param(20, 12, 9, 1, id="prefill-rolls-back-drafts-only"),
    ],
)
def test_kstep_empty_row_rolls_back_optimistic_progress(
    prompt_len: int,
    computed_after_schedule: int,
    expected_computed: int,
    expected_placeholders: int,
) -> None:
    """A scheduled K-step with no output must not leave phantom progress."""
    request = _Request("r0")
    request.sampling_params = SimpleNamespace(num_logprobs=None)
    request.prompt_token_ids = [0] * prompt_len
    request.num_computed_tokens = computed_after_schedule
    request.num_output_placeholders = 4

    scheduler = _make_scheduler_stub([request])
    scheduler._talker_kstep_armed = lambda: True
    _bind_request_lifecycle(scheduler, update_request=lambda _request, token_ids: (token_ids, False))

    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"r0": 4},
        scheduled_spec_decode_tokens={"r0": [1, 2, 3]},
        num_invalid_spec_tokens=0,
    )
    OmniARScheduler.update_from_output(scheduler, scheduler_output, _runner_output([[]]))

    assert request.num_computed_tokens == expected_computed
    assert request.num_output_placeholders == expected_placeholders


def test_kstep_empty_row_without_draft_does_not_roll_back() -> None:
    """An empty no-draft row can be a normal prefill continuation."""
    request = _Request("r0")
    request.sampling_params = SimpleNamespace(num_logprobs=None)
    request.prompt_token_ids = [0] * 20
    request.num_computed_tokens = 12
    request.num_output_placeholders = 4

    scheduler = _make_scheduler_stub([request])
    scheduler._talker_kstep_armed = lambda: True
    _bind_request_lifecycle(scheduler, update_request=lambda _request, token_ids: (token_ids, False))

    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"r0": 4},
        scheduled_spec_decode_tokens={},
        num_invalid_spec_tokens=0,
    )
    OmniARScheduler.update_from_output(scheduler, scheduler_output, _runner_output([[]]))

    assert request.num_computed_tokens == 12
    assert request.num_output_placeholders == 4
