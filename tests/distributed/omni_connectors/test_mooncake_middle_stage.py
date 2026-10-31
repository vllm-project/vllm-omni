# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise middle-stage routing with real ZMQ and a CPU transfer-engine double."""

import ctypes
from contextlib import ExitStack

import pytest
import torch
from vllm.utils.network_utils import get_open_port

from vllm_omni.data_entry_keys import HiddenStatesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.distributed.omni_connectors.connectors import mooncake_transfer_engine_connector as mooncake
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec, OmniTransferConfig
from vllm_omni.distributed.omni_connectors.utils.initialization import (
    compute_connector_zmq_port,
    resolve_connector_spec,
)
from vllm_omni.engine.stage_init_utils import get_stage_connector_spec

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


def _transfer_config(first_port, second_port):
    return OmniTransferConfig(
        connectors={
            (str(stage), str(stage + 1)): ConnectorSpec(
                name="MooncakeTransferEngineConnector",
                extra={"host": "127.0.0.1", "zmq_port": port, "memory_pool_size": 65536, "protocol": "tcp"},
            )
            for stage, port in enumerate((first_port, second_port))
        }
    )


@pytest.mark.parametrize("async_chunk", [False, True])
@pytest.mark.parametrize("replica_id,local_rank", [(0, 0), (2, 3)])
def test_middle_stage_preserves_both_endpoints(async_chunk, replica_id, local_rank):
    config = _transfer_config(12000, 14000)
    config.connectors[("0", "1")].extra["host"] = "10.0.0.1"
    config.connectors[("1", "2")].extra["host"] = "10.0.0.2"
    spec = get_stage_connector_spec(config, 1, async_chunk)
    resolved = resolve_connector_spec(
        ConnectorSpec(**spec), stage_id=1, role=spec["extra"]["role"], replica_id=replica_id, local_rank=local_rank
    )
    assert resolved.extra["sender_host"] == "10.0.0.1"
    assert resolved.extra["sender_zmq_port"] == compute_connector_zmq_port(
        12000, purpose="request_forwarding", from_stage=0, replica_id=replica_id, local_rank=local_rank
    )
    assert resolved.extra["host"] == "10.0.0.2"
    assert resolved.extra["zmq_port"] == compute_connector_zmq_port(
        14000, purpose="request_forwarding", from_stage=1, replica_id=replica_id, local_rank=local_rank
    )


class _CpuTransferEngine:
    def initialize(self, *args):
        return 0

    def get_rpc_port(self):
        return 1

    def register_memory(self, address, size):
        return 0

    def unregister_memory(self, address):
        return 0

    def batch_transfer_sync_write(self, remote, sources, destinations, sizes):
        for source, destination, size in zip(sources, destinations, sizes, strict=True):
            ctypes.memmove(destination, source, size)
        return 0


def test_three_stage_payload_roundtrip(monkeypatch):
    # Keep constructor roles, sockets, serialization, pool ownership and put/get
    # real. Only native DMA and pinned allocation are replaced for a CPU test.
    monkeypatch.setattr(mooncake, "TransferEngine", _CpuTransferEngine)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    config = _transfer_config(get_open_port(), get_open_port() - 1)
    with ExitStack() as stack:
        connectors = []
        for stage in range(3):
            spec = get_stage_connector_spec(config, stage, True)
            resolved = resolve_connector_spec(ConnectorSpec(**spec), stage_id=stage, role=spec["extra"]["role"])
            connectors.append(stack.enter_context(mooncake.MooncakeTransferEngineConnector(resolved.extra)))
        first, middle, last = connectors
        assert first.can_put
        assert middle.can_put
        assert not last.can_put
        assert last._listener_thread is None
        assert last.put("2", "3", "receive-only", b"invalid") == (False, 0, None)

        for chunk in range(2):
            hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4) + chunk
            payload = OmniPayloadStruct(
                hidden_states=HiddenStatesStruct(output=hidden), meta=MetaStruct(finished=bool(chunk))
            )
            key = f"session_{chunk}"
            assert first.put("0", "1", key, payload)[0]
            received, _ = middle.get("0", "1", key)
            torch.testing.assert_close(received["hidden_states"]["output"], hidden)
            assert received["meta"]["finished"] == bool(chunk)
            assert middle.put("1", "2", key, received)[0]
            forwarded, _ = last.get("1", "2", key)
            torch.testing.assert_close(forwarded["hidden_states"]["output"], hidden)
            assert forwarded["meta"]["finished"] == bool(chunk)


def test_middle_stage_rejects_mixed_connector_types():
    config = _transfer_config(12000, 14000)
    config.connectors[("1", "2")] = ConnectorSpec(name="SharedMemoryConnector", extra={})
    with pytest.raises(ValueError, match="middle stage requires"):
        get_stage_connector_spec(config, 1, True)
