from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# isort: off
import vllm_omni  # noqa: F401
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
import vllm_omni.core.sched.omni_generation_scheduler as generation_scheduler_module
from vllm_omni.model_executor.models.indextts2.scheduler import (
    IndexTTS2GenerationScheduler,
)

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _EmptyRequestQueue:
    def __bool__(self) -> bool:
        return False

    def __len__(self) -> int:
        return 0

    def prepend_requests(self, _requests) -> None:
        return None


def _make_request(request_id: str, prompt_token_ids: list[int] | None = None) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids or [1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )


def _make_scheduler(running: list[Request]) -> IndexTTS2GenerationScheduler:
    scheduler = object.__new__(IndexTTS2GenerationScheduler)
    scheduler.max_num_scheduled_tokens = 16
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.allocate_slots.return_value = SimpleNamespace(
        get_block_ids=lambda: [],
    )
    scheduler.kv_cache_manager.get_num_common_prefix_blocks.return_value = []
    scheduler.running = running
    scheduler.requests = {request.request_id: request for request in running}
    scheduler.waiting = _EmptyRequestQueue()
    scheduler.policy = MagicMock()
    scheduler.chunk_transfer_adapter = None
    scheduler._retains_state_across_chunks = False
    scheduler._pending_finish_reqs = []
    scheduler.input_coordinator = None
    scheduler._consume_pending_connector_output = lambda *_args, **_kwargs: None
    scheduler._process_pending_input_timeouts = lambda: None
    scheduler.scheduler_config = SimpleNamespace(enable_chunked_prefill=True)
    scheduler.num_lookahead_tokens = 0
    scheduler.log_stats = False
    scheduler.use_v2_model_runner = False
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    scheduler._make_cached_request_data = lambda **_kwargs: SimpleNamespace(
        req_ids=[],
        resumed_req_ids=[],
        new_token_ids=[],
        all_token_ids=[],
        new_block_ids=[],
        num_computed_tokens=[],
        num_output_tokens=[],
    )
    scheduler.prev_step_scheduled_req_ids = set()
    scheduler.needs_kv_cache_zeroing = False
    scheduler.encoder_cache_manager = SimpleNamespace(get_freed_mm_hashes=lambda: [])
    scheduler.finished_req_ids = set()
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler._update_after_schedule = lambda _output: None
    scheduler.max_num_running_reqs = 32
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=1))
    scheduler._stepwise_generation = True
    return scheduler


def _make_update_scheduler(session: Request) -> MagicMock:
    scheduler = MagicMock()
    scheduler._stepwise_generation = True
    scheduler._should_finish_generation_request = (
        IndexTTS2GenerationScheduler._should_finish_generation_request.__get__(scheduler)
    )
    scheduler.requests = {session.request_id: session}
    scheduler.perf_metrics = None
    scheduler.chunk_transfer_adapter = None
    scheduler._handle_stopped_request.return_value = True
    scheduler._free_request.return_value = (None, None)
    scheduler.running = [session]
    scheduler.structured_output_manager.should_advance.return_value = False
    scheduler.recompute_kv_load_failures = False
    scheduler.connector = None
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.finished_req_ids_dict = {}
    scheduler.make_stats.return_value = None
    return scheduler


def _model_output(request_id: str, *, finished: bool) -> SimpleNamespace:
    return SimpleNamespace(
        sampled_token_ids=[[]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        multimodal_outputs=[None],
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        req_id_to_index={request_id: 0},
        routed_experts=None,
        generation_finished_req_ids={request_id} if finished else set(),
    )


def test_cached_stepwise_generation_does_not_resend_payload() -> None:
    payload = {"mel_codes": [1, 2, 3]}
    request = SimpleNamespace(additional_information=payload)
    scheduler = object.__new__(IndexTTS2GenerationScheduler)

    scheduler._stepwise_generation = True
    assert scheduler._cached_additional_information(request) is None

    scheduler._stepwise_generation = False
    assert scheduler._cached_additional_information(request) is payload


def test_omni_scheduler_output_preserves_stepwise_request_order() -> None:
    base = SimpleNamespace(**{name: None for name in SchedulerOutput.__dataclass_fields__})
    scheduler = object.__new__(OmniSchedulerMixin)
    scheduler.input_coordinator = None

    output = scheduler._wrap_omni_scheduler_output(
        base,
        stepwise_req_ids=["cached-b", "cached-a"],
    )

    assert output.stepwise_req_ids == ["cached-b", "cached-a"]


def test_zero_token_stepwise_schedule_does_not_allocate_kv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_request("req-zero-token")
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = len(request.prompt_token_ids)
    scheduler = _make_scheduler([request])
    scheduler.kv_cache_manager.allocate_slots.side_effect = AssertionError("cached stepwise work must not allocate KV")
    monkeypatch.setattr(
        generation_scheduler_module,
        "create_request_queue",
        lambda _policy: _EmptyRequestQueue(),
    )

    output = scheduler.schedule()

    scheduler.kv_cache_manager.allocate_slots.assert_not_called()
    assert output.total_num_scheduled_tokens == 0
    assert output.num_scheduled_tokens == {}
    assert output.stepwise_req_ids == [request.request_id]


def test_stepwise_work_survives_exhausted_token_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_request = _make_request("req-token-budget", list(range(16)))
    token_request.status = RequestStatus.RUNNING
    stepwise_request = _make_request("req-stepwise-after-budget")
    stepwise_request.status = RequestStatus.RUNNING
    stepwise_request.num_computed_tokens = len(stepwise_request.prompt_token_ids)
    scheduler = _make_scheduler([token_request, stepwise_request])
    monkeypatch.setattr(
        generation_scheduler_module,
        "create_request_queue",
        lambda _policy: _EmptyRequestQueue(),
    )

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {token_request.request_id: 16}
    assert output.stepwise_req_ids == [
        token_request.request_id,
        stepwise_request.request_id,
    ]


def test_stepwise_request_waits_for_model_completion() -> None:
    session = _make_request("req-stepwise-cfm")
    session.status = RequestStatus.RUNNING
    session.num_computed_tokens = len(session.prompt_token_ids)
    scheduler = _make_update_scheduler(session)
    scheduler_output = MagicMock(spec=SchedulerOutput)
    scheduler_output.num_scheduled_tokens = {session.request_id: 1}
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.num_invalid_spec_tokens = 0

    IndexTTS2GenerationScheduler.update_from_output(
        scheduler,
        scheduler_output,
        _model_output(session.request_id, finished=False),
    )

    assert session.status == RequestStatus.RUNNING
    scheduler._handle_stopped_request.assert_not_called()

    IndexTTS2GenerationScheduler.update_from_output(
        scheduler,
        scheduler_output,
        _model_output(session.request_id, finished=True),
    )
    assert session.status == RequestStatus.FINISHED_STOPPED


def test_zero_token_stepwise_completion_is_consumed() -> None:
    session = _make_request("req-zero-token-complete")
    session.status = RequestStatus.RUNNING
    session.num_computed_tokens = len(session.prompt_token_ids)
    scheduler = _make_update_scheduler(session)
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={},
        stepwise_req_ids=[session.request_id],
        scheduled_spec_decode_tokens={},
        num_invalid_spec_tokens=0,
    )

    IndexTTS2GenerationScheduler.update_from_output(
        scheduler,
        scheduler_output,
        _model_output(session.request_id, finished=True),
    )

    assert session.status == RequestStatus.FINISHED_STOPPED


def test_zero_token_stepwise_output_does_not_consume_stale_tokens() -> None:
    session = _make_request("req-zero-token-stale")
    session.status = RequestStatus.RUNNING
    session.num_computed_tokens = len(session.prompt_token_ids)
    session.num_stale_output_tokens = 1
    scheduler = _make_update_scheduler(session)
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={},
        stepwise_req_ids=[session.request_id],
        scheduled_spec_decode_tokens={},
        num_invalid_spec_tokens=0,
    )

    IndexTTS2GenerationScheduler.update_from_output(
        scheduler,
        scheduler_output,
        _model_output(session.request_id, finished=False),
    )

    assert session.num_stale_output_tokens == 1
    assert session.status == RequestStatus.RUNNING
