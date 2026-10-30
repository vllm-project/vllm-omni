# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configured MRv2 payload edges over production SHM, with manually stepped I/O."""

import uuid
from types import SimpleNamespace

import pytest
import torch

from tests.worker.test_omni_connector_mixin import MixinHost, _make_model_config, _make_request
from vllm_omni.config.stage_routing import StageRouting
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.engine.stage_init_utils import get_stage_connector_spec

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _host(mocker, stage_id, route, async_chunk):
    config = _make_model_config(async_chunk=async_chunk)
    config.stage_id = stage_id
    config.stage_connector_config = get_stage_connector_spec(None, stage_id, async_chunk, route)
    host = MixinHost()
    host.vllm_config = SimpleNamespace(model_config=config)
    # Initialize all production state without starting competing I/O threads.
    mocker.patch.object(host, "_create_connector", return_value=None)
    host.init_omni_connectors(config)
    host._omni_connector = SharedMemoryConnector({"stage_id": stage_id})
    return host


@pytest.mark.parametrize(("source", "target"), [(0, 2), (2, 1)])
@pytest.mark.parametrize("async_chunk", [False, True])
def test_mrv2_payload_uses_configured_shm_edge(mocker, source, target, async_chunk):
    route = StageRouting.from_transitions(3, ((0, 2), (2, 1)))
    sender = _host(mocker, source, route, async_chunk)
    receiver = _host(mocker, target, route, async_chunk)
    request = _make_request("r", f"configured-mrv2-{uuid.uuid4().hex}")
    receiver.register_chunk_recv(request)
    try:
        for index in range(3 if async_chunk else 1):
            payload = {"codes": {"audio": torch.tensor([index])}, "meta": {"finished": torch.tensor(False)}}
            if async_chunk:
                assert sender._enqueue_chunk_payload(request, payload)[0]
            else:
                assert sender.send_full_payload_outputs(None, {"r": (payload, request)}) == ["r"]
            send_key = request.external_req_id if async_chunk else request.request_id
            task = sender._pending_save_reqs[send_key].popleft()
            assert sender._send_single_request(task)
            assert receiver._poll_single_request("r")
            assert receiver._get_req_chunk["r"] == index + 1
            expected = list(range(index + 1)) if async_chunk else [index]
            assert receiver.get_local_stage_payload("r")["codes"]["audio"].tolist() == expected
            receiver.get_omni_connector_output()
    finally:
        sender.shutdown_omni_connectors()
        receiver.shutdown_omni_connectors()


def test_mrv2_terminal_has_no_outgoing_transport(mocker):
    host = _host(mocker, 2, StageRouting.from_transitions(3, ((0, 2),)), True)
    request = _make_request("r")
    try:
        assert host.send_full_payload_outputs(None, {"r": ({"value": 1}, request)}) == []
        assert host._enqueue_chunk_payload(request, {"value": 1}) == (True, None)
        assert not host._pending_save_reqs and not host._put_req_chunk
    finally:
        host.shutdown_omni_connectors()
