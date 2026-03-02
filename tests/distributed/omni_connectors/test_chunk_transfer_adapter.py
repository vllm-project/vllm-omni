# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections import deque
from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture
from vllm.v1.request import RequestStatus

from vllm_omni.distributed.omni_connectors.transfer_adapter.base import OmniTransferAdapterBase
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
    OmniChunkTransferAdapter,
)
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class DummyWaitingQueue(list):
    def prepend_requests(self, requests):
        self[:0] = list(requests)

    def add_request(self, request):
        self.append(request)


def _req(req_id: str, status: RequestStatus, external_req_id: str | None = None):
    return SimpleNamespace(
        request_id=req_id,
        external_req_id=external_req_id or req_id,
        status=status,
        prompt_token_ids=[],
        num_computed_tokens=0,
        additional_information=None,
        is_finished=lambda: status == RequestStatus.FINISHED_STOPPED,
    )


@pytest.fixture
def build_adapter(monkeypatch, mocker: MockerFixture):
    def _build(*, stage_id: int = 1, model_mode: str = "ar", max_num_seqs: int = 2):
        connector = mocker.MagicMock()
        connector.stage_id = stage_id
        connector.get.return_value = None
        connector.put.return_value = (True, 1, {})

        def _fake_base_init(self, config):
            self.config = config
            self._pending_load_reqs = deque()
            self._finished_load_reqs = set()
            self._pending_save_reqs = deque()
            self._finished_save_reqs = set()
            self.stop_event = threading.Event()

        monkeypatch.setattr(OmniTransferAdapterBase, "__init__", _fake_base_init)
        monkeypatch.setattr(
            OmniChunkTransferAdapter,
            "create_connector",
            classmethod(lambda cls, _model_config: connector),
        )

        model_config = SimpleNamespace(worker_type=model_mode)
        scheduler_config = SimpleNamespace(max_num_seqs=max_num_seqs)
        adapter = OmniChunkTransferAdapter(
            SimpleNamespace(model_config=model_config, scheduler_config=scheduler_config)
        )
        return adapter, connector

    return _build


@pytest.mark.parametrize(
    ("raw_cfg", "expected_name", "expected_extra"),
    [
        (None, "SharedMemoryConnector", {}),
        (SimpleNamespace(name="YuanrongConnector", extra={"k": "v"}), "YuanrongConnector", {"k": "v"}),
    ],
)
def test_create_connector_config_parsing(monkeypatch, raw_cfg, expected_name, expected_extra):
    captured = {}

    def _fake_create(spec):
        captured["spec"] = spec
        return "ok"

    monkeypatch.setattr(
        "vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter"
        ".OmniConnectorFactory.create_connector",
        _fake_create,
    )

    model_config = SimpleNamespace(stage_connector_config=raw_cfg) if raw_cfg is not None else SimpleNamespace()
    connector = OmniChunkTransferAdapter.create_connector(model_config)

    assert connector == "ok"
    assert isinstance(captured["spec"], ConnectorSpec)
    assert captured["spec"].name == expected_name
    assert captured["spec"].extra == expected_extra


def test_load_poll(build_adapter):
    adapter, connector = build_adapter(stage_id=2, model_mode="ar")
    request = _req("req-1", RequestStatus.WAITING, external_req_id="external-1")

    adapter.load_async(request)
    payload = {"code_predictor_codes": [[1]], "hidden_states": torch.tensor([[2.0]]), "finished": True}
    connector.get.return_value = (payload, 16)
    adapter._poll_single_request(request)

    assert request.additional_information == payload
    assert adapter.get_req_chunk["req-1"] == 1
    assert "req-1" in adapter._finished_load_reqs
    assert "req-1" in adapter.finished_requests
    assert "req-1" not in adapter._pending_load_reqs


def test_save_async(build_adapter):
    adapter, _ = build_adapter(stage_id=1)
    request = _req("req-1", RequestStatus.WAITING, external_req_id="external-1")

    adapter.custom_process_next_stage_input_func = lambda **kwargs: {"x": [1], "finished": False}
    adapter.save_async(pooling_output=None, request=request)
    adapter.custom_process_next_stage_input_func = lambda **kwargs: {}
    adapter.save_async(pooling_output=None, request=request)

    task = adapter._pending_save_reqs.popleft()
    assert task["is_finished"] is False


def test_update_request_payload(build_adapter):
    adapter, _ = build_adapter()

    adapter._update_request_payload("ext", {"h": torch.tensor([[1.0]]), "codes": [1], "finished": False})
    merged = adapter._update_request_payload("ext", {"h": torch.tensor([[2.0]]), "codes": [2], "finished": True})

    assert torch.equal(merged["h"], torch.tensor([[1.0], [2.0]]))
    assert merged["codes"] == [1, 2]
    assert merged["finished"] is True


def test_process_and_restore_queues(build_adapter):
    adapter, _ = build_adapter(stage_id=1, max_num_seqs=8)
    waiting_req = _req("w1", RequestStatus.WAITING)
    running_req = _req("r1", RequestStatus.RUNNING)
    waiting_queue = DummyWaitingQueue([waiting_req])
    running_queue = [running_req]

    adapter.process_pending_chunks(waiting_queue, running_queue)
    assert waiting_req.status == RequestStatus.WAITING_FOR_CHUNK
    assert running_req.status == RequestStatus.WAITING_FOR_CHUNK
    assert waiting_queue == []
    assert running_queue == []

    adapter.restore_queues(waiting_queue, running_queue)
    assert waiting_queue == [waiting_req]
    assert running_queue == [running_req]
    assert adapter.waiting_for_chunk_waiting_requests == deque()
    assert adapter.waiting_for_chunk_running_requests == deque()


def test_postprocess_scheduler_output(build_adapter):
    adapter, _ = build_adapter()
    adapter.requests_with_ready_chunks = {"new-ready", "cached-ready", "leftover"}

    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="new-ready")],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["cached-ready", "missing"]),
    )
    requests = {"cached-ready": SimpleNamespace(additional_information={"k": "v"})}

    adapter.postprocess_scheduler_output(scheduler_output, requests)

    cached_info = scheduler_output.scheduled_cached_reqs.additional_information
    assert cached_info["cached-ready"] == {"k": "v"}
    assert cached_info["missing"] is None
    assert adapter.requests_with_ready_chunks == {"leftover"}


# ---------------------------------------------------------------
# Cleanup tests
# ---------------------------------------------------------------


def _populate_adapter_state(adapter, req_id="req-1", ext_id="ext-1"):
    """Fill every per-request structure so cleanup can be verified."""
    adapter.finished_requests.add(req_id)
    adapter.get_req_chunk[req_id] = 3
    adapter.requests_with_ready_chunks.add(req_id)
    adapter.request_ids_mapping[req_id] = ext_id
    adapter._pending_load_reqs.append(SimpleNamespace(request_id=req_id))
    adapter._finished_load_reqs.add(req_id)

    adapter.put_req_chunk[ext_id] = 5
    adapter.request_payload[ext_id] = {"hidden": [1, 2]}
    adapter.code_prompt_token_ids[ext_id] = [[10, 20]]


def test_cleanup_clears_all_state(build_adapter):
    """After cleanup, no per-request key should remain in any dict/set."""
    adapter, _ = build_adapter(stage_id=1)
    req_id, ext_id = "req-1", "ext-1"
    _populate_adapter_state(adapter, req_id, ext_id)

    adapter.cleanup(req_id, ext_id)

    assert req_id not in adapter.finished_requests
    assert req_id not in adapter.get_req_chunk
    assert req_id not in adapter.requests_with_ready_chunks
    assert req_id not in adapter.request_ids_mapping
    assert all(getattr(r, "request_id", None) != req_id for r in adapter._pending_load_reqs)
    assert req_id not in adapter._finished_load_reqs

    assert ext_id not in adapter.put_req_chunk
    assert ext_id not in adapter.request_payload
    assert ext_id not in adapter.code_prompt_token_ids


def test_cleanup_infers_external_id(build_adapter):
    """When external_req_id is None, cleanup should look it up from the mapping."""
    adapter, _ = build_adapter(stage_id=1)
    req_id, ext_id = "req-2", "ext-2"
    _populate_adapter_state(adapter, req_id, ext_id)

    adapter.cleanup(req_id)

    assert ext_id not in adapter.put_req_chunk
    assert ext_id not in adapter.request_payload


def test_cleanup_idempotent(build_adapter):
    """Calling cleanup multiple times for the same (or nonexistent) request must not raise."""
    adapter, _ = build_adapter(stage_id=1)

    try:
        adapter.cleanup("nonexistent")
        adapter.cleanup("nonexistent")
    except Exception as e:
        pytest.fail(f"cleanup should be idempotent: {e}")

    req_id, ext_id = "req-3", "ext-3"
    _populate_adapter_state(adapter, req_id, ext_id)
    adapter.cleanup(req_id, ext_id)

    try:
        adapter.cleanup(req_id, ext_id)
    except Exception as e:
        pytest.fail(f"second cleanup should be idempotent: {e}")


def test_cleanup_request_id_reuse_not_polluted(build_adapter):
    """After cleanup, reusing the same request_id must not be treated as finished."""
    adapter, _ = build_adapter(stage_id=1)
    req_id, ext_id = "req-reuse", "ext-reuse"
    _populate_adapter_state(adapter, req_id, ext_id)

    adapter.cleanup(req_id, ext_id)

    assert req_id not in adapter.finished_requests
    assert req_id not in adapter.get_req_chunk


def test_cleanup_preserves_pending_save(build_adapter):
    """Cleanup must NOT remove _pending_save_reqs to avoid losing unsent chunks."""
    adapter, _ = build_adapter(stage_id=1)
    req_id, ext_id = "req-4", "ext-4"
    _populate_adapter_state(adapter, req_id, ext_id)

    pending_task = {"put_key": f"{ext_id}_1_0", "data": {"x": 1}}
    adapter._pending_save_reqs.append(pending_task)

    adapter.cleanup(req_id, ext_id)

    assert len(adapter._pending_save_reqs) == 1


def test_cleanup_only_affects_target_request(build_adapter):
    """Cleanup for one request must not affect another request's state."""
    adapter, _ = build_adapter(stage_id=1)
    _populate_adapter_state(adapter, "req-a", "ext-a")
    _populate_adapter_state(adapter, "req-b", "ext-b")

    adapter.cleanup("req-a", "ext-a")

    assert "req-b" in adapter.finished_requests
    assert "req-b" in adapter.get_req_chunk
    assert "ext-b" in adapter.put_req_chunk
    assert "ext-b" in adapter.request_payload
    assert "ext-b" in adapter.code_prompt_token_ids
    assert "req-b" in adapter.request_ids_mapping


def test_cleanup_after_poll_flow(build_adapter):
    """Simulate full load_async -> poll -> finished -> cleanup cycle."""
    adapter, connector = build_adapter(stage_id=2, model_mode="ar")
    request = _req("req-flow", RequestStatus.WAITING, external_req_id="ext-flow")

    adapter.load_async(request)

    adapter.request_ids_mapping["req-flow"] = "ext-flow"
    payload = {"hidden_states": torch.tensor([[1.0]]), "finished": True}
    connector.get.return_value = (payload, 8)
    adapter._poll_single_request(request)

    assert "req-flow" in adapter.finished_requests
    assert adapter.get_req_chunk["req-flow"] == 1
    assert "req-flow" in adapter.request_ids_mapping

    adapter.cleanup("req-flow", "ext-flow")

    assert "req-flow" not in adapter.finished_requests
    assert "req-flow" not in adapter.get_req_chunk
    assert "req-flow" not in adapter.request_ids_mapping
    assert "ext-flow" not in adapter.request_payload


# ---------------------------------------------------------------
# Scheduler trigger tests
# ---------------------------------------------------------------


class _HashableRequest(SimpleNamespace):
    """SimpleNamespace that can be added to a set (needed by scheduler internals)."""

    def __hash__(self):
        return hash(self.request_id)

    def __eq__(self, other):
        return getattr(other, "request_id", None) == self.request_id


def test_generation_scheduler_calls_cleanup_on_finished(monkeypatch, mocker: MockerFixture):
    """OmniGenerationScheduler must call adapter.cleanup when request finishes."""
    cleanup_calls = []

    adapter_mock = mocker.MagicMock()
    adapter_mock.finished_requests = {"req-s1"}
    adapter_mock.cleanup = lambda *a, **kw: cleanup_calls.append((a, kw))

    from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler

    scheduler = mocker.MagicMock()
    scheduler.chunk_transfer_adapter = adapter_mock
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.perf_metrics = None
    scheduler.log_stats = False
    scheduler.recompute_kv_load_failures = False
    scheduler.structured_output_manager = mocker.MagicMock()
    scheduler.structured_output_manager.should_advance.return_value = False
    scheduler.finished_req_ids_dict = {}
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.kv_event_publisher = mocker.MagicMock()

    request = _HashableRequest(
        request_id="req-s1",
        external_req_id="ext-s1",
        status=RequestStatus.RUNNING,
        is_finished=lambda: False,
        num_computed_tokens=10,
        num_prompt_tokens=10,
        prompt_token_ids=list(range(10)),
        num_output_placeholders=0,
        sampling_params=None,
        pooling_params=None,
        stop_reason=None,
        client_index=0,
        take_events=lambda: [],
        trace_headers=None,
        num_cached_tokens=0,
        num_external_computed_tokens=0,
        num_nans_in_logits=0,
        get_finished_reason=lambda: "stop",
    )
    scheduler.requests = {"req-s1": request}

    scheduler._handle_stopped_request = mocker.MagicMock(return_value=True)
    scheduler._free_request = mocker.MagicMock(return_value=None)
    scheduler._get_routed_experts = mocker.MagicMock(return_value=None)
    scheduler.running = [request]
    scheduler.waiting = mocker.MagicMock()
    scheduler.waiting.remove_requests = mocker.MagicMock()
    scheduler.make_stats = mocker.MagicMock(return_value=None)

    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req-s1": 10},
        scheduled_spec_decode_tokens={},
        num_invalid_spec_tokens=0,
    )
    model_runner_output = SimpleNamespace(
        sampled_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        req_id_to_index={"req-s1": 0},
    )

    OmniGenerationScheduler.update_from_output(scheduler, scheduler_output, model_runner_output)

    assert len(cleanup_calls) == 1
    args, _ = cleanup_calls[0]
    assert args[0] == "req-s1"
    assert args[1] == "ext-s1"
