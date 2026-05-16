from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.request import RequestStatus

import vllm_omni.core.sched.omni_ar_scheduler as ar_sched_mod
import vllm_omni.core.sched.omni_generation_scheduler as gen_sched_mod
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeInputCoordinator:
    def __init__(self) -> None:
        self.freed: list[str] = []

    def free_finished_request(self, request_id: str) -> None:
        self.freed.append(request_id)


class DummyRequest:
    request_id = "req-free"
    client_index = 0
    additional_information = None

    def is_finished(self) -> bool:
        return True


@pytest.mark.parametrize(
    ("scheduler_cls", "scheduler_mod"),
    [
        (OmniARScheduler, ar_sched_mod),
        (OmniGenerationScheduler, gen_sched_mod),
    ],
)
def test_finish_requests_cleans_input_coordinator_for_finished_ids(
    monkeypatch: pytest.MonkeyPatch,
    scheduler_cls,
    scheduler_mod,
) -> None:
    coordinator = FakeInputCoordinator()
    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = coordinator

    def fake_finish_requests(self, request_ids, finished_status):
        assert request_ids == ["req-a", "req-b"]
        assert finished_status == RequestStatus.FINISHED_STOPPED
        return [("req-a", 0), ("req-b", 1)]

    monkeypatch.setattr(scheduler_mod.VLLMScheduler, "finish_requests", fake_finish_requests)

    result = scheduler_cls.finish_requests(
        scheduler,
        ["req-a", "req-b"],
        RequestStatus.FINISHED_STOPPED,
    )

    assert result == [("req-a", 0), ("req-b", 1)]
    assert coordinator.freed == ["req-a", "req-b"]


def test_ar_free_request_cleans_input_coordinator_on_normal_free() -> None:
    coordinator = FakeInputCoordinator()
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.input_coordinator = coordinator
    scheduler._omits_kv_transfer_cache = {"req-free": True}
    scheduler.encoder_cache_manager = SimpleNamespace(free=lambda request: None)
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler._new_prompt_len_snapshot = {"req-free": 3}
    scheduler._connector_finished = lambda request: (False, None)
    scheduler._should_transfer_kv_for_request = lambda request_id: False
    scheduler._free_blocks = lambda request: None

    result = OmniARScheduler._free_request(scheduler, DummyRequest())

    assert result is None
    assert coordinator.freed == ["req-free"]
    assert scheduler._omits_kv_transfer_cache == {}
    assert scheduler._new_prompt_len_snapshot == {}


def test_generation_free_request_cleans_input_coordinator(monkeypatch: pytest.MonkeyPatch) -> None:
    coordinator = FakeInputCoordinator()
    scheduler = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    scheduler.input_coordinator = coordinator

    def fake_free_request(self, request, delay_free_blocks=False):
        assert delay_free_blocks is True
