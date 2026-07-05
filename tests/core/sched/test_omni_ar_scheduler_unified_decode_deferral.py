from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.request import RequestStatus

import vllm_omni.core.sched.omni_ar_scheduler as scheduler_mod
import vllm_omni.model_executor.models.voxcpm2.scheduler as voxcpm2_scheduler_mod
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
) -> VoxCPM2OmniARAsyncScheduler:
    sched = VoxCPM2OmniARAsyncScheduler.__new__(VoxCPM2OmniARAsyncScheduler)
    hf_config = SimpleNamespace()
    model_config = SimpleNamespace(hf_config=hf_config)
    if enable_unified_decode_graph is not None or deterministic_cfm_noise:
        runtime_config = SimpleNamespace(
            enable_unified_decode_graph=True if enable_unified_decode_graph is None else enable_unified_decode_graph,
            deterministic_cfm_noise=deterministic_cfm_noise,
        )
        if runtime_config_on_model_config:
            model_config.voxcpm2_runtime_config = runtime_config
        else:
            hf_config.voxcpm2_runtime_config = runtime_config
    sched.vllm_config = SimpleNamespace(model_config=model_config)
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
