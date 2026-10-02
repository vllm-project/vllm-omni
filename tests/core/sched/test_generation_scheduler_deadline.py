# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``OmniGenerationScheduler`` with deadline batching (``additional_config.codec_deadline``)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm_omni.core.sched.codec_deadline import CodecDeadlineConfig, CodecDeadlinePolicy, StreamLedger

import vllm_omni.core.sched.omni_generation_scheduler as scheduler_module
from tests.core.sched.test_generation_scheduler_restore import (
    FakeAdapter,
    _chunk_request,
    _make_generation_scheduler,
)
from vllm_omni.core.sched.omni_generation_scheduler import (
    OmniGenerationScheduler,
    _build_codec_deadline,
    _codec_deadline_plan,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ReadyAdapter(FakeAdapter):
    """Chunk adapter whose queued requests all hold a loaded chunk."""

    def __init__(self) -> None:
        super().__init__()
        self.requests_with_ready_chunks: set[str] = set()

    def process_pending_chunks(self, waiting, running, scheduler_requests=None):
        pass

    def restore_queues(self, waiting, running, scheduler_requests=None):
        self.restore_called = True


def _meta(chunk_seq: int) -> dict:
    return {
        "meta": {
            "chunk_seq": chunk_seq,
            "cache_epoch": 0,
            "last_chunk": False,
            "turn_end": False,
            "codec_chunk_frames": 25,
            "duplex_epoch": 0,
            "tts_is_last_chunk": True,
        }
    }


def _deadline_scheduler(monkeypatch, requests, *, policy: CodecDeadlinePolicy | None):
    monkeypatch.setattr(scheduler_module, "create_request_queue", create_request_queue)
    scheduler = _make_generation_scheduler(requests[0])
    scheduler.max_num_scheduled_tokens = 64
    scheduler.max_num_running_reqs = 8
    scheduler.chunk_transfer_adapter = _ReadyAdapter()
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.requests = {}
    for request in requests:
        scheduler.waiting.add_request(request)
        scheduler.requests[request.request_id] = request
        scheduler.chunk_transfer_adapter.requests_with_ready_chunks.add(request.request_id)
    scheduler._codec_deadline = policy
    return scheduler


def _requests():
    return [
        _chunk_request("cont", prompt_token_ids=[1, 2], additional_information=_meta(3)),
        _chunk_request("first", prompt_token_ids=[1, 2], additional_information=_meta(0)),
    ]


def test_without_a_policy_both_ready_chunks_are_scheduled(monkeypatch) -> None:
    scheduler = _deadline_scheduler(monkeypatch, _requests(), policy=None)
    output = OmniGenerationScheduler.schedule(scheduler)
    assert set(output.num_scheduled_tokens) == {"cont", "first"}


def test_a_continuation_with_slack_is_held_and_released_by_its_hold_bound(monkeypatch) -> None:
    policy = CodecDeadlinePolicy(CodecDeadlineConfig(enabled=True))
    now = scheduler_module.time.monotonic()
    policy.ledgers["cont"] = StreamLedger(key=(0, 0), anchor=now - 5, play_end=now + 30, audio_s=30.0, d1=1.0)
    requests = _requests()
    scheduler = _deadline_scheduler(monkeypatch, requests, policy=policy)

    output = OmniGenerationScheduler.schedule(scheduler)
    # The first chunk goes alone; the continuation stays queued, ready mark kept.
    assert set(output.num_scheduled_tokens) == {"first"}
    assert [request.request_id for request in scheduler.waiting] == ["cont"]
    assert "cont" in scheduler.chunk_transfer_adapter.requests_with_ready_chunks
    assert 0.0 < scheduler.next_release_in() <= 0.35
    assert policy.dispatched["first"][0].chunk_seq == 0

    # The first chunk ran (the adapter clears its ready mark); past its max
    # hold the continuation is released.
    requests[1].num_computed_tokens = 2
    scheduler.chunk_transfer_adapter.requests_with_ready_chunks.discard("first")
    policy.ready_since["cont"] -= 1.0
    output = OmniGenerationScheduler.schedule(scheduler)
    assert set(output.num_scheduled_tokens) == {"cont"}
    assert policy.dispatched["cont"][1] is True  # it was held
    assert scheduler.next_release_in() is None


def test_held_running_requests_are_skipped_in_place(monkeypatch) -> None:
    policy = CodecDeadlinePolicy(CodecDeadlineConfig(enabled=True))
    now = scheduler_module.time.monotonic()
    policy.ledgers["cont"] = StreamLedger(key=(0, 0), anchor=now - 5, play_end=now + 30, audio_s=30.0, d1=1.0)
    requests = _requests()
    scheduler = _deadline_scheduler(monkeypatch, requests[1:], policy=policy)
    scheduler.running = [requests[0]]
    scheduler.requests["cont"] = requests[0]
    scheduler.chunk_transfer_adapter.requests_with_ready_chunks.add("cont")
    output = OmniGenerationScheduler.schedule(scheduler)
    assert set(output.num_scheduled_tokens) == {"first"}
    assert requests[0] in scheduler.running


def test_plan_is_none_for_mock_or_missing_policies() -> None:
    assert _codec_deadline_plan(MagicMock(), 0.0) is None
    assert _codec_deadline_plan(SimpleNamespace(), 0.0) is None
    assert OmniGenerationScheduler.next_release_in(MagicMock()) is None
    assert OmniGenerationScheduler.next_release_in(SimpleNamespace()) is None


def test_free_request_forgets_the_stream(monkeypatch) -> None:
    policy = CodecDeadlinePolicy(CodecDeadlineConfig(enabled=True))
    policy.ledgers["cont"] = StreamLedger(key=(0, 0), anchor=0.0, play_end=1.0, audio_s=1.0, d1=1.0)
    policy.buffered.add("cont")
    scheduler = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    scheduler._codec_deadline = policy
    scheduler.input_coordinator = None
    monkeypatch.setattr(
        scheduler_module.VLLMScheduler, "_free_request", lambda self, request, delay=False: (None, None)
    )
    scheduler._free_request(SimpleNamespace(request_id="cont"))
    assert "cont" not in policy.ledgers and "cont" not in policy.buffered


def _build_target(*, additional_config, async_scheduling=False, native=False):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(additional_config=additional_config),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        _native_data_plane=native,
        _batch_window=False,
        _first_chunk_express=False,
    )


@pytest.mark.parametrize(
    ("idle_wait", "async_scheduling", "native", "enabled"),
    [
        ("0.05", False, False, True),
        ("0", False, False, False),
        ("0.05", True, False, False),
        ("0.05", False, True, False),
    ],
)
def test_build_reads_additional_config_and_its_preconditions(monkeypatch, idle_wait, async_scheduling, native, enabled):
    monkeypatch.setenv("VLLM_OMNI_STAGE_IDLE_WAIT_S", idle_wait)
    target = _build_target(
        additional_config={"codec_deadline": {"enabled": True}}, async_scheduling=async_scheduling, native=native
    )
    assert isinstance(_build_codec_deadline(target), CodecDeadlinePolicy) is enabled
    assert _build_codec_deadline(_build_target(additional_config={})) is None


def test_a_stream_without_its_first_chunk_marks_the_onset_pending(monkeypatch) -> None:
    policy = CodecDeadlinePolicy(CodecDeadlineConfig(enabled=True))
    seen: list[bool] = []
    real_plan = policy.plan

    def plan(ready, now, *, onset_pending=False):
        seen.append(onset_pending)
        return real_plan(ready, now, onset_pending=onset_pending)

    monkeypatch.setattr(policy, "plan", plan)
    now = scheduler_module.time.monotonic()
    prewarmed = _chunk_request("prewarmed", prompt_token_ids=[], additional_information=_meta(0))
    _codec_deadline_plan(_deadline_scheduler(monkeypatch, [*_requests(), prewarmed], policy=policy), now)
    _codec_deadline_plan(_deadline_scheduler(monkeypatch, _requests(), policy=policy), now)
    assert seen == [True, False]


def test_update_from_output_feeds_the_ledger_and_the_step_time() -> None:
    """The real ``update_from_output``: dispatch snapshot -> ledger, and the step's row count -> T(1)."""
    import torch
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
    from vllm_omni.core.sched.codec_deadline import seed_step_s

    request = Request(
        request_id="first",
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=0.0,
        block_hasher=None,
    )
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 2
    request.num_in_flight_tokens = 1

    policy = CodecDeadlinePolicy(CodecDeadlineConfig(enabled=True))
    policy.on_scheduled("first", scheduler_module.ChunkMeta(0, 0, False, False, 25, 0), 1.0)
    policy.last_schedule_t = scheduler_module.time.monotonic() - 0.5

    sched = MagicMock()
    sched._codec_deadline = policy
    sched._audio_seconds = OmniGenerationScheduler._audio_seconds
    sched._first_chunk_express = False
    sched._express_min_slack_s = 0.0
    sched._batch_window = False
    sched.requests = {"first": request}
    sched.perf_metrics = None
    sched.chunk_transfer_adapter = SimpleNamespace(
        is_done_receiving_chunks=lambda _request_id: False,
        segment_finished_requests=set(),
    )
    sched.running = [request]
    sched.waiting = MagicMock()
    sched.skipped_waiting = MagicMock()
    sched.structured_output_manager.accept_tokens.return_value = True
    sched._pending_finish_reqs = []
    sched.recompute_kv_load_failures = False
    sched.connector = None
    sched.kv_cache_manager.take_events.return_value = None
    sched.kv_cache_manager.estimate_cached_tokens.return_value = 0
    sched.finished_req_ids_dict = {}
    sched.make_stats.return_value = None
    sched._handle_stopped_request.return_value = False
    sched._free_request.return_value = (None, None)

    scheduler_output = MagicMock(spec=SchedulerOutput)
    scheduler_output.num_scheduled_tokens = {"first": 1}
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.num_invalid_spec_tokens = 0
    model_runner_output = MagicMock(spec=ModelRunnerOutput)
    model_runner_output.sampled_token_ids = [[]]
    model_runner_output.logprobs = None
    model_runner_output.prompt_logprobs_dict = {}
    model_runner_output.pooler_output = None
    model_runner_output.num_nans_in_logits = None
    model_runner_output.kv_connector_output = None
    model_runner_output.cudagraph_stats = None
    model_runner_output.req_id_to_index = {"first": 0}
    model_runner_output.routed_experts = None
    model_runner_output.multimodal_outputs = [{"model_outputs": torch.zeros(2400), "sr": 24000}]

    OmniGenerationScheduler.update_from_output(sched, scheduler_output, model_runner_output)

    ledger = policy.ledgers["first"]
    assert ledger.audio_s == pytest.approx(0.1)
    assert ledger.play_end - ledger.anchor == pytest.approx(0.5 + 0.1)  # first segment: client prebuffer
    assert "first" not in policy.dispatched
    assert policy.step_time.mean[0] > seed_step_s(1)  # one-row step of ~0.5 s observed
