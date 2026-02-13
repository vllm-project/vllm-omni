# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
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
    )


@pytest.fixture
def build_adapter(monkeypatch):
    def _build(*, stage_id: int = 1, model_mode: str = "ar", max_num_seqs: int = 2):
        connector = MagicMock()
        connector.stage_id = stage_id
        connector.get.return_value = None
        connector.put.return_value = (True, 1, {})

        def _fake_base_init(self, config):
            self.config = config
            self._pending_load_reqs = {}
            self._finished_load_reqs = set()
            self._pending_save_reqs = {}
            self._finished_save_reqs = set()
            self.stop_event = threading.Event()
            self.lock = threading.Lock()

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
    adapter._poll_single_request("req-1")

    connector.get.assert_called_once_with("1", "2", "external-1_1_0")
    assert request.additional_information == payload
    assert adapter.get_req_chunk["req-1"] == 1
    assert "req-1" in adapter._finished_load_reqs
    assert "req-1" in adapter.finished_requests
    assert "req-1" not in adapter._pending_load_reqs


def test_save_async(build_adapter):
    adapter, _ = build_adapter(stage_id=1)
    request = SimpleNamespace(external_req_id="external-1")

    adapter.custom_process_next_stage_input_func = lambda **kwargs: {"x": [1], "finished": False}
    adapter.save_async(pooling_output=None, request=request)
    adapter.custom_process_next_stage_input_func = lambda **kwargs: {}
    adapter.save_async(pooling_output=None, request=request)

    assert adapter.put_req_chunk["external-1"] == 1
    queued = adapter._pending_save_reqs["external-1"]
    assert len(queued) == 1
    assert queued[0]["put_key"] == "external-1_1_0"


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
