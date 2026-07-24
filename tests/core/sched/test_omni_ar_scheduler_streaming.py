# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for Omni AR streaming-session async placeholder handling."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Imports must run in this order: vllm_omni applies patches to vllm.v1.request before
# Request / StreamingUpdate are bound in this module. Ruff isort would reorder them.
# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_scheduler(*, stage_id: int = 0) -> OmniARScheduler:
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._new_prompt_len_snapshot = {}
    sched.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=stage_id))
    sched.num_waiting_for_streaming_input = 0
    sched.log_stats = False
    sched.chunk_transfer_adapter = None
    sched.skipped_waiting = set()
    return sched


def _make_request() -> Request:
    return Request(
        request_id="req-ar-streaming-test",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )


def _make_update(prompt_token_ids: list[int] | None = None) -> StreamingUpdate:
    return StreamingUpdate(
        mm_features=None,
        prompt_token_ids=[10, 20] if prompt_token_ids is None else prompt_token_ids,
        max_tokens=32,
        arrival_time=200.0,
        sampling_params=SamplingParams(max_tokens=16),
    )


def _run_resumable_segment_stop(session: Request) -> None:
    sched = MagicMock()
    sched.requests = {session.request_id: session}
    sched.perf_metrics = None
    sched.structured_output_manager.should_advance.return_value = False
    sched._update_request_with_output.return_value = ([42], True)
    sched._handle_stopped_request.return_value = False
    sched.chunk_transfer_adapter = None
    sched.running = [session]
    sched.waiting_for_transfer_free = set()
    sched.transfer_triggered_requests = set()
    sched.active_kv_transfers = set()
    sched.pending_stop_after_extraction = set()
    sched.connector = None
    sched.kv_cache_manager.take_events.return_value = None
    sched.finished_req_ids_dict = {}
    sched.make_stats.return_value = None

    scheduler_output = MagicMock(spec=SchedulerOutput)
    scheduler_output.num_scheduled_tokens = {session.request_id: 1}
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.num_invalid_spec_tokens = 0

    model_runner_output = MagicMock(spec=ModelRunnerOutput)
    model_runner_output.sampled_token_ids = [[42]]
    model_runner_output.logprobs = None
    model_runner_output.prompt_logprobs_dict = {}
    model_runner_output.pooler_output = None
    model_runner_output.num_nans_in_logits = None
    model_runner_output.kv_connector_output = None
    model_runner_output.cudagraph_stats = None
    model_runner_output.req_id_to_index = {session.request_id: 0}
    model_runner_output.routed_experts = None

    OmniARScheduler.update_from_output(sched, scheduler_output, model_runner_output)


@pytest.mark.parametrize("outstanding_async_tokens", [0, 1, 2])
def test_resumable_segment_stop_reconciles_async_placeholders(
    outstanding_async_tokens: int,
) -> None:
    """A segment stop discards and rolls back only in-flight async tokens."""
    session = _make_request()
    session.status = RequestStatus.RUNNING
    session.resumable = True
    session.append_output_token_ids([7, 8])
    session.num_computed_tokens = session.num_tokens + outstanding_async_tokens
    session.num_output_placeholders = outstanding_async_tokens
    session.spec_token_ids = [-1] * outstanding_async_tokens

    _run_resumable_segment_stop(session)

    assert session.async_tokens_to_discard == outstanding_async_tokens
    assert session.num_computed_tokens == session.num_tokens
    assert session.num_output_placeholders == 0
    assert session.spec_token_ids == []
    assert session._output_token_ids == []


def test_stage0_streaming_update_discards_outstanding_async_placeholder_token() -> None:
    sched = _make_scheduler(stage_id=0)
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = 6
    session.num_output_placeholders = 1
    session.spec_token_ids = [-1]

    sched._update_request_as_session(session, _make_update([10, 20]))

    assert session.async_tokens_to_discard == 1
    assert session.num_output_placeholders == 0
    assert session.spec_token_ids == []
    # The async placeholder makes token 9 unconfirmed, so only 7 and 8 are
    # carried into the next streaming prompt before the new chunk tokens.
    assert session.prompt_token_ids == [1, 2, 3, 7, 8, 10, 20]
    assert list(session._all_token_ids) == [1, 2, 3, 7, 8, 10, 20]
    assert session._output_token_ids == []
    assert session.num_prompt_tokens == 7
    assert sched._new_prompt_len_snapshot[session.request_id] == 2


def test_stage0_streaming_update_keeps_all_computed_tokens_without_placeholder() -> None:
    sched = _make_scheduler(stage_id=0)
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = 6
    session.num_output_placeholders = 0

    sched._update_request_as_session(session, _make_update([10, 20]))

    assert getattr(session, "async_tokens_to_discard", 0) == 0
    assert session.num_output_placeholders == 0
    assert session.prompt_token_ids == [1, 2, 3, 7, 8, 9, 10, 20]
    assert list(session._all_token_ids) == [1, 2, 3, 7, 8, 9, 10, 20]
    assert session._output_token_ids == []
    assert session.num_prompt_tokens == 8
    assert sched._new_prompt_len_snapshot[session.request_id] == 2


def test_explicit_streaming_payload_replaces_placeholder_prompt() -> None:
    sched = _make_scheduler(stage_id=1)
    sched.chunk_transfer_adapter = SimpleNamespace(
        receives_chunks=False,
        segment_finished_requests=set(),
    )
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    update = _make_update([10, 20])
    update.additional_information = {
        "tts_token_ids": [10, 20],
        "meta": {"replace_streaming_prompt": True},
    }

    sched._update_request_as_session(session, update)

    assert session.prompt_token_ids == [10, 20]
    assert session.additional_information == update.additional_information
    assert session.status == RequestStatus.WAITING
