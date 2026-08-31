# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Only the hidden-state rows a request emitted may go downstream.

A speculative step forwards one real position plus the drafts, and rejection
sampling keeps a prefix of them. Slicing the Omni payload by the scheduled
token count also ships the rows past that prefix -- hidden states for tokens
the model never emitted. Pure CPU: no model, no accelerator.
"""

import pytest

from vllm_omni.worker.sampling_utils import accepted_hidden_rows

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _rows(
    *,
    num_scheduled_tokens: int,
    drafts: dict[str, list[int]],
    accepted: list[list[int]],
    req_id: str = "r0",
    req_index: int = 0,
) -> int:
    return accepted_hidden_rows(
        req_id=req_id,
        req_index=req_index,
        num_scheduled_tokens=num_scheduled_tokens,
        scheduled_spec_decode_tokens=drafts,
        valid_sampled_token_ids=accepted,
    )


def test_partial_rejection_drops_the_remainder():
    # 1 real position + 3 drafts scheduled, 2 tokens accepted.
    assert _rows(num_scheduled_tokens=4, drafts={"r0": [11, 12, 13]}, accepted=[[101, 102]]) == 2


def test_full_acceptance_keeps_every_row():
    assert _rows(num_scheduled_tokens=4, drafts={"r0": [11, 12, 13]}, accepted=[[101, 102, 103, 104]]) == 4


def test_total_rejection_keeps_only_the_real_position():
    assert _rows(num_scheduled_tokens=4, drafts={"r0": [11, 12, 13]}, accepted=[[101]]) == 1


def test_prefill_step_is_untouched():
    # No drafts scheduled: the step carries the prompt and the downstream stage
    # needs all of it, however many tokens were sampled.
    assert _rows(num_scheduled_tokens=16, drafts={}, accepted=[[101]]) == 16
    assert _rows(num_scheduled_tokens=16, drafts={"r0": []}, accepted=[[101]]) == 16


def test_other_requests_drafts_do_not_shorten_this_one():
    assert _rows(num_scheduled_tokens=16, drafts={"r1": [11]}, accepted=[[101]]) == 16


def test_missing_accepted_bookkeeping_is_untouched():
    # Async scheduling fills valid_sampled_token_ids in after the payload is
    # built; an absent or empty entry must not shorten the slice.
    assert _rows(num_scheduled_tokens=4, drafts={"r0": [11, 12, 13]}, accepted=[]) == 4
    assert _rows(num_scheduled_tokens=4, drafts={"r0": [11, 12, 13]}, accepted=[[]]) == 4


def test_accepted_count_never_lengthens_the_slice():
    assert _rows(num_scheduled_tokens=2, drafts={"r0": [11]}, accepted=[[101, 102, 103]]) == 2


def test_request_index_selects_its_own_row():
    drafts = {"r0": [11], "r1": [21]}
    accepted = [[101, 102], [201]]
    assert _rows(num_scheduled_tokens=2, drafts=drafts, accepted=accepted, req_id="r0", req_index=0) == 2
    assert _rows(num_scheduled_tokens=2, drafts=drafts, accepted=accepted, req_id="r1", req_index=1) == 1
