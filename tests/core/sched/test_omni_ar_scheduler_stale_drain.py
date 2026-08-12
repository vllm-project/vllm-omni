# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Investigation tests for the num_stale_output_tokens drain (59b9e719).

The seed-TTS WER regression appeared with the vLLM 0.27 rebase (nightly
2946 -> 2949: mean WER 0.2695 -> 0.3936, median 0.0 in both), whose prime
suspect is the reimplemented async-token discard: vLLM 0.26 drained
``async_tokens_to_discard`` one arriving frame at a time and suppressed
only the token append; the 0.27 adaptation seeds
``num_stale_output_tokens`` with the discarded *placeholder count* and
drains it by each arriving frame's ``num_tokens_scheduled``, dropping the
whole per-request update. Whenever seed and arrivals disagree, the counter
outlives the old segment and the drain consumes *valid* frames of the new
segment — a hole in the talker's codec stream (garbled audio, text/audio
mismatch) — or underflows its own assert on the new segment's prefill
frame (engine-core death mid-step).

The fix seeds the counter from ``num_in_flight_tokens`` — the same
scheduled-token units the drain subtracts — so every pre-discard frame
drains exactly its own contribution and new-segment frames can never be
swallowed. These tests pin that contract one frame at a time; the
delivered / swallowed observable is whether ``_update_request_with_output``
runs for the frame.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Imports must run in this order: vllm_omni applies patches to vllm.v1.request
# before Request / StreamingUpdate are bound in this module.
# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_session() -> Request:
    session = Request(
        request_id="req-stale-drain-test",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=32),
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )
    session.status = RequestStatus.RUNNING
    if not hasattr(session, "num_stale_output_tokens"):
        # vLLM 0.27 defines this Request field; 0.26 (local dev) does not.
        session.num_stale_output_tokens = 0
    if not hasattr(session, "num_in_flight_tokens"):
        session.num_in_flight_tokens = 0
    return session


def _replace_streaming_session(session: Request) -> None:
    """Run the real stage-0 streaming replacement (the discard/seed site)."""
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._new_prompt_len_snapshot = {}
    sched.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=0))
    sched.num_waiting_for_streaming_input = 0
    sched.log_stats = False
    sched.chunk_transfer_adapter = None
    sched.skipped_waiting = set()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    update = StreamingUpdate(
        mm_features=None,
        prompt_token_ids=[10, 20],
        max_tokens=32,
        arrival_time=200.0,
        sampling_params=SamplingParams(max_tokens=16),
    )
    sched._update_request_as_session(session, update)
    session.status = RequestStatus.RUNNING


def _make_drain_sched(session: Request) -> MagicMock:
    sched = MagicMock()
    sched.requests = {session.request_id: session}
    sched.perf_metrics = None
    sched.structured_output_manager.should_advance.return_value = False
    sched._update_request_with_output.return_value = ([42], False)
    sched._process_kv_transfer_trigger.return_value = False
    sched.chunk_transfer_adapter = MagicMock()
    sched.running = [session]
    sched.waiting_for_transfer_free = set()
    sched.transfer_triggered_requests = set()
    sched.active_kv_transfers = set()
    sched.pending_stop_after_extraction = set()
    sched.connector = None
    sched.kv_cache_manager.take_events.return_value = None
    sched.finished_req_ids_dict = {}
    sched.make_stats.return_value = None
    return sched


def _run_step(sched: MagicMock, session: Request, *, num_scheduled: int, token: int) -> bool:
    """Feed one model-runner frame through update_from_output.

    Returns True when the frame's tokens were delivered (the append ran),
    False when the frame was swallowed by the stale drain.
    """
    scheduler_output = MagicMock(spec=SchedulerOutput)
    scheduler_output.num_scheduled_tokens = {session.request_id: num_scheduled}
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.num_invalid_spec_tokens = 0

    model_runner_output = MagicMock(spec=ModelRunnerOutput)
    model_runner_output.sampled_token_ids = [[token]]
    model_runner_output.logprobs = None
    model_runner_output.prompt_logprobs_dict = {}
    model_runner_output.pooler_output = None
    model_runner_output.num_nans_in_logits = None
    model_runner_output.kv_connector_output = None
    model_runner_output.cudagraph_stats = None
    model_runner_output.req_id_to_index = {session.request_id: 0}
    model_runner_output.routed_experts = None
    model_runner_output.inter_stage_outputs = None

    sched._update_request_with_output.reset_mock()
    OmniARScheduler.update_from_output(sched, scheduler_output, model_runner_output)
    return sched._update_request_with_output.called


def test_exact_drain_delivers_new_segment_frame() -> None:
    """One in-flight decode frame at replacement: its late output drains the
    counter exactly and the new segment's first frame is delivered."""
    session = _make_session()
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = 6
    session.num_output_placeholders = 1
    session.num_in_flight_tokens = 1

    _replace_streaming_session(session)
    assert session.num_stale_output_tokens == 1

    sched = _make_drain_sched(session)
    assert _run_step(sched, session, num_scheduled=1, token=42) is False  # late frame dropped
    assert session.num_stale_output_tokens == 0
    assert _run_step(sched, session, num_scheduled=1, token=43) is True  # new segment survives


def test_two_in_flight_frames_drain_without_swallowing() -> None:
    """Two in-flight decode frames: both late frames drop, the new segment's
    first valid frame is delivered. A placeholder-based seed swallowed it
    whenever fewer frames arrived than placeholders counted (the driver of
    the seed-TTS WER regression, nightly 2946 -> 2949: 0.2695 -> 0.3936)."""
    session = _make_session()
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = 7
    session.num_output_placeholders = 2
    session.num_in_flight_tokens = 2

    _replace_streaming_session(session)
    assert session.num_stale_output_tokens == 2

    sched = _make_drain_sched(session)
    assert _run_step(sched, session, num_scheduled=1, token=42) is False  # late frame 1
    assert _run_step(sched, session, num_scheduled=1, token=44) is False  # late frame 2
    assert session.num_stale_output_tokens == 0

    delivered = _run_step(sched, session, num_scheduled=1, token=43)  # new segment, frame 1
    assert delivered is True, (
        "valid new-segment frame was swallowed by leftover stale count "
        f"(num_stale_output_tokens left at {session.num_stale_output_tokens})"
    )


def test_spec_placeholder_asymmetry_does_not_swallow_valid_frame() -> None:
    """Placeholder counts can exceed scheduled counts (e.g. a draft
    invalidated before execution): 1 in-flight MTP frame scheduled 2 tokens
    but contributed 3 placeholders. Seeding from num_in_flight_tokens keeps
    seed == drain, so the new segment's valid frame is delivered."""
    session = _make_session()
    session.append_output_token_ids([7])
    session.num_computed_tokens = 7
    session.num_output_placeholders = 3  # 1 sampled + 2 spec placeholders
    session.num_in_flight_tokens = 2  # the frame actually scheduled 1 + 1 tokens

    _replace_streaming_session(session)
    assert session.num_stale_output_tokens == 2

    sched = _make_drain_sched(session)
    assert _run_step(sched, session, num_scheduled=2, token=42) is False  # late MTP frame
    assert session.num_stale_output_tokens == 0
    delivered = _run_step(sched, session, num_scheduled=1, token=43)  # new segment, frame 1
    assert delivered is True, (
        "valid new-segment frame swallowed after spec placeholder asymmetry "
        f"(num_stale_output_tokens left at {session.num_stale_output_tokens})"
    )


def test_in_flight_prefill_chunk_drains_exactly_without_underflow() -> None:
    """An in-flight prefill chunk carries no placeholders but 5 scheduled
    tokens. The placeholder-based seed missed it entirely (its late output
    was appended to the rolled-back session) and any leftover counter
    underflowed 'assert num_stale_output_tokens >= 0' on the chunk's arrival,
    killing the engine-core step. Seeding from num_in_flight_tokens drops the
    late chunk and drains to exactly zero."""
    session = _make_session()
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = 7
    session.num_output_placeholders = 0  # prefill chunk: no placeholders
    session.num_in_flight_tokens = 5  # the chunk's scheduled tokens

    _replace_streaming_session(session)
    assert session.num_stale_output_tokens == 5

    sched = _make_drain_sched(session)
    assert _run_step(sched, session, num_scheduled=5, token=42) is False  # late prefill chunk dropped
    assert session.num_stale_output_tokens == 0
    assert _run_step(sched, session, num_scheduled=1, token=43) is True  # new segment survives
