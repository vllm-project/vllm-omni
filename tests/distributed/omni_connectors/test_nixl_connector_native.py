# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import multiprocessing as mp
import socket
import time
from typing import Any

import pytest
import torch

from vllm_omni.data_entry_keys import HiddenStatesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.parallel]


def _native_nixl_available() -> bool:
    try:
        from vllm.distributed.nixl_utils import NixlWrapper
    except ImportError:
        return False
    return NixlWrapper is not None and torch.cuda.is_available() and torch.accelerator.device_count() >= 2


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _producer(port: int, ready: Any, consumed: Any, result: Any) -> None:
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector

    current_omni_platform.set_device(0)
    connector = NixlConnector(
        {
            "role": "sender",
            "host": "127.0.0.1",
            "zmq_port": port,
            "agent_name": "native-smoke-producer",
        }
    )
    try:
        payload = OmniPayloadStruct(
            hidden_states=HiddenStatesStruct(output=torch.arange(8, dtype=torch.float32, device="cuda:0")),
            meta=MetaStruct(token_role_ids=torch.tensor([1, 2, 3], dtype=torch.int64, device="cpu")),
            request_id="native-smoke",
        )
        success, size, metadata = connector.put("0", "1", "native-smoke", payload)
        result.put(("put", success, size, metadata["kind"] if metadata else None))
        ready.set()
        if not consumed.wait(timeout=30):
            raise TimeoutError("consumer did not complete the native NIXL transfer")
        deadline = time.monotonic() + 10
        while connector._pending and time.monotonic() < deadline:
            time.sleep(0.01)
        result.put(("cleanup", len(connector._pending), len(connector._registered_descs)))
    except Exception as error:
        result.put(("producer_error", repr(error)))
        raise
    finally:
        connector.close()


def _consumer(port: int, ready: Any, consumed: Any, result: Any) -> None:
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector

    if not ready.wait(timeout=30):
        raise TimeoutError("producer did not publish native NIXL metadata")
    current_omni_platform.set_device(1)
    connector = NixlConnector(
        {
            "role": "receiver",
            "sender_host": "127.0.0.1",
            "sender_zmq_port": port,
            "receive_device": "cuda",
            "agent_name": "native-smoke-consumer",
        }
    )
    try:
        received = None
        deadline = time.monotonic() + 30
        while received is None and time.monotonic() < deadline:
            received = connector.get(
                "0",
                "1",
                "native-smoke",
                {"source_host": "127.0.0.1", "source_port": port},
            )
        if received is None:
            raise TimeoutError("native NIXL transfer did not complete")
        payload, size = received
        hidden = payload["hidden_states"]["output"]
        token_roles = payload["meta"]["token_role_ids"]
        result.put(
            (
                "get",
                size,
                hidden.cpu().tolist(),
                str(hidden.dtype),
                list(hidden.shape),
                str(hidden.device),
                token_roles.cpu().tolist(),
                str(token_roles.dtype),
                list(token_roles.shape),
                str(token_roles.device),
            )
        )
        consumed.set()
    except Exception as error:
        result.put(("consumer_error", repr(error)))
        raise
    finally:
        connector.close()


@pytest.mark.skipif(not _native_nixl_available(), reason="requires NIXL and at least two CUDA devices")
def test_native_two_process_structured_mixed_device_transfer():
    context = mp.get_context("spawn")
    ready = context.Event()
    consumed = context.Event()
    result = context.Queue()
    port = _free_port()
    producer = context.Process(target=_producer, args=(port, ready, consumed, result))
    consumer = context.Process(target=_consumer, args=(port, ready, consumed, result))

    producer.start()
    consumer.start()
    producer.join(timeout=60)
    consumer.join(timeout=60)
    if producer.is_alive():
        producer.terminate()
    if consumer.is_alive():
        consumer.terminate()

    records = [result.get(timeout=5) for _ in range(3)]
    assert producer.exitcode == 0, records
    assert consumer.exitcode == 0, records
    assert records[0][0] == "put"
    assert records[0][1] is True
    assert records[0][2] >= 56
    assert records[0][3] == "structured"
    get_record = next(record for record in records if record[0] == "get")
    assert get_record[1] == records[0][2]
    assert get_record[2:6] == (
        list(range(8)),
        "torch.float32",
        [8],
        "cuda:1",
    )
    assert get_record[6:] == ([1, 2, 3], "torch.int64", [3], "cuda:1")
    assert next(record for record in records if record[0] == "cleanup") == ("cleanup", 0, 0)
