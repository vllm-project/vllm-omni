from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.request import RequestStatus

import vllm_omni.core.sched.omni_ar_scheduler as scheduler_mod
import vllm_omni.model_executor.models.voxcpm2.scheduler as voxcpm2_scheduler_mod
from vllm_omni.model_executor.models.voxcpm2.runtime_config import _VoxCPM2RuntimeConfig
from vllm_omni.model_executor.models.voxcpm2.scheduler import VoxCPM2OmniARAsyncScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _MockQueue:
    def __init__(self, items: list | None = None) -> None:
        self._items = list(items or [])

    def __bool__(self) -> bool:
        return bool(self._items)

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def add_request(self, request) -> None:
        self._items.append(request)

    def pop_request(self):
        return self._items.pop(0)

    def prepend_request(self, request) -> None:
        self._items.insert(0, request)

    def prepend_requests(self, requests) -> None:
        self._items = list(requests) + self._items


class _MockRequest:
    def __init__(
        self,
        request_id: str,
        *,
        status: RequestStatus = RequestStatus.RUNNING,
        num_prompt_tokens: int = 4,
        num_computed_tokens: int = 4,
        num_output_placeholders: int = 0,
    ) -> None:
        self.request_id = request_id
        self.status = status
        self.num_prompt_tokens = num_prompt_tokens
        self.num_computed_tokens = num_computed_tokens
        self.num_output_placeholders = num_output_placeholders
        self._all_token_ids = list(range(num_computed_tokens))

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)


@pytest.fixture(autouse=True)
def _mock_cuda_graph_platform(monkeypatch) -> None:
    monkeypatch.setattr(voxcpm2_scheduler_mod.current_omni_platform, "is_cuda", lambda: True)


def _make_scheduler(
    *,
    enable_unified_decode_graph: bool | None = True,
    deterministic_cfm_noise: bool = False,
    runtime_config_on_model_config: bool = False,
    prefill_interval: int = 6,
) -> VoxCPM2OmniARAsyncScheduler:
    sched = VoxCPM2OmniARAsyncScheduler.__new__(VoxCPM2OmniARAsyncScheduler)
    hf_config = SimpleNamespace()
    model_config = SimpleNamespace(hf_config=hf_config)
    if enable_unified_decode_graph is not None or deterministic_cfm_noise:
        runtime_config = SimpleNamespace(
            enable_unified_decode_graph=True if enable_unified_decode_graph is None else enable_unified_decode_graph,
            deterministic_cfm_noise=deterministic_cfm_noise,
            unified_decode_graph_prefill_interval=prefill_interval,
        )
        if runtime_config_on_model_config:
            model_config.voxcpm2_runtime_config = runtime_config
        else:
            hf_config.voxcpm2_runtime_config = runtime_config
    sched.vllm_config = SimpleNamespace(model_config=model_config)
    sched._decode_steps_since_prefill = 0
    sched._temporarily_unscheduled_req_ids = set()
    sched.max_num_running_reqs = 8
    sched.num_waiting_for_streaming_input = 0
    sched.requests = {}
    return sched


def test_voxcpm2_unified_decode_graph_defers_waiting_when_decode_ready() -> None:
    scheduler = _make_scheduler()
    scheduler.running = [_MockRequest("decode")]
    scheduler.waiting = _MockQueue([_MockRequest("prefill", status=RequestStatus.WAITING)])

    assert scheduler._should_defer_waiting_for_unified_decode_graph()


def test_voxcpm2_unified_decode_graph_does_not_defer_without_decode_ready() -> None:
    scheduler = _make_scheduler()
    scheduler.running = [_MockRequest("prefill-running", num_prompt_tokens=8, num_computed_tokens=4)]
    scheduler.waiting = _MockQueue([_MockRequest("waiting", status=RequestStatus.WAITING)])

    assert not scheduler._should_defer_waiting_for_unified_decode_graph()


def test_voxcpm2_unified_decode_graph_does_not_defer_when_disabled() -> None:
    scheduler = _make_scheduler(enable_unified_decode_graph=False)
    scheduler.running = [_MockRequest("decode")]
    scheduler.waiting = _MockQueue([_MockRequest("prefill", status=RequestStatus.WAITING)])

    assert not scheduler._should_defer_waiting_for_unified_decode_graph()


def test_voxcpm2_unified_decode_graph_uses_model_runtime_defaults() -> None:
    scheduler = _make_scheduler(enable_unified_decode_graph=None)
    scheduler.running = [_MockRequest("decode")]
    scheduler.waiting = _MockQueue([_MockRequest("prefill", status=RequestStatus.WAITING)])

    assert scheduler._should_defer_waiting_for_unified_decode_graph()


def test_voxcpm2_unified_decode_graph_reads_model_config_runtime_config() -> None:
    scheduler = _make_scheduler(enable_unified_decode_graph=True, runtime_config_on_model_config=True)
    scheduler.running = [_MockRequest("decode")]
    scheduler.waiting = _MockQueue([_MockRequest("prefill", status=RequestStatus.WAITING)])

    assert scheduler._should_defer_waiting_for_unified_decode_graph()


def test_voxcpm2_unified_decode_graph_does_not_defer_with_deterministic_noise() -> None:
    scheduler = _make_scheduler(deterministic_cfm_noise=True)
    scheduler.running = [_MockRequest("decode")]
    scheduler.waiting = _MockQueue([_MockRequest("prefill", status=RequestStatus.WAITING)])

    assert not scheduler._should_defer_waiting_for_unified_decode_graph()


def test_voxcpm2_unified_decode_graph_does_not_defer_without_cuda_graph(monkeypatch) -> None:
    monkeypatch.setattr(voxcpm2_scheduler_mod.current_omni_platform, "is_cuda", lambda: False)
    scheduler = _make_scheduler()
    scheduler.running = [_MockRequest("decode")]
    scheduler.waiting = _MockQueue([_MockRequest("prefill", status=RequestStatus.WAITING)])

    assert not scheduler._should_defer_waiting_for_unified_decode_graph()


def test_voxcpm2_periodically_schedules_prefill_only_admission() -> None:
    scheduler = _make_scheduler(prefill_interval=3)
    scheduler.running = [_MockRequest("decode")]
    scheduler.waiting = _MockQueue([_MockRequest("prefill", status=RequestStatus.WAITING)])

    assert not scheduler._should_run_prefill_only_admission()
    assert not scheduler._should_run_prefill_only_admission()
    assert scheduler._should_run_prefill_only_admission()
    assert scheduler._decode_steps_since_prefill == 0


def test_voxcpm2_periodic_prefill_admission_is_opt_in() -> None:
    scheduler = _make_scheduler(prefill_interval=0)
    scheduler.running = [_MockRequest("decode")]
    scheduler.waiting = _MockQueue([_MockRequest("prefill", status=RequestStatus.WAITING)])

    for _ in range(12):
        assert not scheduler._should_run_prefill_only_admission()
    assert scheduler._decode_steps_since_prefill == 0


def test_voxcpm2_prefill_interval_clamps_negative_values() -> None:
    assert _VoxCPM2RuntimeConfig._coerce_value("unified_decode_graph_prefill_interval", -1, 0) == 0


def test_voxcpm2_prefill_only_admission_waits_for_a_free_slot() -> None:
    scheduler = _make_scheduler(prefill_interval=2)
    scheduler.running = [_MockRequest(f"decode-{i}") for i in range(8)]
    scheduler.waiting = _MockQueue([_MockRequest("prefill", status=RequestStatus.WAITING)])

    assert not scheduler._should_run_prefill_only_admission()
    assert not scheduler._should_run_prefill_only_admission()
    assert scheduler._decode_steps_since_prefill == 1

    scheduler.running.pop()
    assert scheduler._should_run_prefill_only_admission()


def test_temporarily_unscheduled_request_gets_async_resume_tokens() -> None:
    scheduler = _make_scheduler()
    request = _MockRequest("decode", num_computed_tokens=6)
    scheduler.requests = {request.request_id: request}
    scheduler._temporarily_unscheduled_req_ids = {request.request_id}
    cached_reqs = SimpleNamespace(req_ids=[request.request_id], all_token_ids={})
    scheduler_output = SimpleNamespace(scheduled_cached_reqs=cached_reqs)

    scheduler._restore_temporarily_unscheduled_metadata(scheduler_output)

    assert cached_reqs.all_token_ids == {request.request_id: request._all_token_ids}
    assert not scheduler._temporarily_unscheduled_req_ids


def test_prefill_only_admission_pauses_decode_and_restores_queues(monkeypatch) -> None:
    scheduler = _make_scheduler(prefill_interval=1)
    decode = _MockRequest("decode")
    prefill = _MockRequest("prefill", status=RequestStatus.WAITING)
    scheduler.running = [decode]
    scheduler.waiting = _MockQueue([prefill])
    scheduler.policy = "fcfs"
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = None
    scheduler._consume_pending_connector_output = lambda model_mode: None

    monkeypatch.setattr(scheduler_mod, "create_request_queue", lambda _policy: _MockQueue())

    def fake_upstream_schedule(self, throttle_prefills: bool = False):
        assert not self.running
        assert self.waiting._items == [prefill]
        self.running.append(self.waiting.pop_request())
        raise RuntimeError("stop after prefill admission")

    monkeypatch.setattr(scheduler_mod.VLLMScheduler, "schedule", fake_upstream_schedule)

    with pytest.raises(RuntimeError, match="stop after prefill admission"):
        scheduler.schedule()

    assert scheduler.running == [decode, prefill]
    assert not scheduler.waiting
    assert scheduler._temporarily_unscheduled_req_ids == {decode.request_id}


def test_unified_decode_graph_deferral_restores_waiting_queue(monkeypatch) -> None:
    scheduler = _make_scheduler()
    scheduler.running = [_MockRequest("decode")]
    original_waiting_req = _MockRequest("waiting", status=RequestStatus.WAITING)
    deferred_by_upstream = _MockRequest("deferred-by-upstream", status=RequestStatus.WAITING)
    original_waiting = _MockQueue([original_waiting_req])
    scheduler.waiting = original_waiting
    scheduler.policy = "fcfs"
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = None
    scheduler._consume_pending_connector_output = lambda model_mode: None

    monkeypatch.setattr(scheduler_mod, "create_request_queue", lambda _policy: _MockQueue())

    def fake_upstream_schedule(self, throttle_prefills: bool = False):
        assert self.waiting is not original_waiting
        assert not self.waiting
        self.waiting.add_request(deferred_by_upstream)
        raise RuntimeError("stop before output wrapping")

    monkeypatch.setattr(scheduler_mod.VLLMScheduler, "schedule", fake_upstream_schedule)

    with pytest.raises(RuntimeError, match="stop before output wrapping"):
        scheduler.schedule()

    assert scheduler.waiting._items == [deferred_by_upstream, original_waiting_req]
