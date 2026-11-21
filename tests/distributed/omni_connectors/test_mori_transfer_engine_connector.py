# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU-only unit tests for ``MoriTransferEngineConnector``.

The Mori runtime (``mori.io``), the RDMA fabric and the ZMQ peers are all
faked here, so the whole file runs under ``pytest -m "core_model and cpu"``
without a Mori install, an RDMA NIC, a second process or an accelerator.

What is covered:

* the msgspec wire structs used by the pull/query handshake,
* configuration parsing and validation in ``__init__`` (backend selection,
  role handling, pool device constraints, RDMA device resolution),
* the ``put()`` / ``get()`` data paths for raw tensors, pool-backed
  ``ManagedBuffer`` zero-copy puts, serialised Python objects, pool
  exhaustion and duplicate keys,
* metadata bookkeeping (``health()``, metrics, ``cleanup()``, ``close()``),
* the sender-side listener handlers ``_handle_pull_request`` /
  ``_handle_query_request`` including the length-mismatch guard, and
  request-socket caching.

The RDMA transfers themselves (``IOEngine.batch_write``), the ZMQ transport
and the ``mori`` native library stay out of scope: they need real hardware and
are exercised by the RDMA-gated integration suites instead.
"""

from __future__ import annotations

import queue
import socket
import threading
import time
from typing import Any

import msgspec
import pytest
import torch
import zmq

from vllm_omni.distributed.omni_connectors.connectors import mori_transfer_engine_connector as mori_mod
from vllm_omni.distributed.omni_connectors.connectors.mori_transfer_engine_connector import (
    INFO_NOT_FOUND,
    QUERY_INFO,
    TRANS_DONE,
    TRANS_ERROR,
    MoriPullRequest,
    MoriTransferEngineConnector,
    QueryRequest,
    QueryResponse,
)
from vllm_omni.distributed.omni_connectors.utils.memory_pool import BufferAllocator, ManagedBuffer

pytestmark = [pytest.mark.parallel, pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Fake Mori runtime
# ---------------------------------------------------------------------------


class FakeMemoryDesc:
    """Minimal stand-in for ``mori.io.MemoryDesc`` (pack/unpack only)."""

    def __init__(self, key: str = "mem", base: int = 0, size: int = 0) -> None:
        self.key = key
        self.base = base
        self.size = size

    def pack(self) -> bytes:
        return msgspec.msgpack.encode({"key": self.key, "base": self.base, "size": self.size})

    @classmethod
    def unpack(cls, packed: bytes) -> FakeMemoryDesc:
        payload = msgspec.msgpack.decode(packed)
        return cls(payload["key"], payload["base"], payload["size"])


class FakeEngineDesc:
    """Minimal stand-in for ``mori.io.EngineDesc``."""

    def __init__(self, key: str = "peer-engine", host: str = "127.0.0.1", port: int = 12345) -> None:
        self.key = key
        self.host = host
        self.port = port

    def pack(self) -> bytes:
        return msgspec.msgpack.encode({"key": self.key, "host": self.host, "port": self.port})

    @classmethod
    def unpack(cls, packed: bytes) -> FakeEngineDesc:
        payload = msgspec.msgpack.decode(packed)
        return cls(payload["key"], payload["host"], payload["port"])


class FakeTransferStatus:
    """Stand-in for ``mori.cpp.TransferStatus``."""

    def __init__(self, failed: bool = False, message: str = "ok") -> None:
        self.failed = failed
        self.message = message
        self.waited = False

    def Wait(self) -> None:  # noqa: N802 - mirrors the Mori pybind name
        self.waited = True

    def Failed(self) -> bool:  # noqa: N802 - mirrors the Mori pybind name
        return self.failed

    def Message(self) -> str:  # noqa: N802 - mirrors the Mori pybind name
        return self.message


class FakeIOEngineConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeRdmaBackendConfig:
    def __init__(self, *args: Any) -> None:
        self.args = args


class FakeXgmiBackendConfig:
    def __init__(self, *args: Any) -> None:
        self.args = args
        self.num_streams = 0
        self.num_events = 0


class FakeBackendType:
    RDMA = "fake.backend.rdma"
    XGMI = "fake.backend.xgmi"


class FakePollCqMode:
    POLLING = "fake.polling"


class FakeIOEngine:
    """In-memory stand-in for ``mori.io.IOEngine``."""

    instances: list[FakeIOEngine] = []

    def __init__(self, key: str, config: FakeIOEngineConfig | None) -> None:
        self.key = key
        self.config = config
        self.backends: list[tuple[Any, Any]] = []
        self.registered_tensors: list[torch.Tensor] = []
        self.registered_remote: list[FakeEngineDesc] = []
        self.deregistered_memory: list[FakeMemoryDesc] = []
        self.deregistered_remote: list[FakeEngineDesc] = []
        self.batch_writes: list[tuple[Any, ...]] = []
        self.statuses: list[FakeTransferStatus] = []
        self._transfer_uid = 0
        self.register_engine_error: Exception | None = None
        self.deregister_error: Exception | None = None
        FakeIOEngine.instances.append(self)

    # -- engine lifecycle ---------------------------------------------------
    def create_backend(self, backend_type: Any, config: Any) -> None:
        self.backends.append((backend_type, config))

    def get_engine_desc(self) -> FakeEngineDesc:
        return FakeEngineDesc(key=self.key, host="127.0.0.1", port=4242)

    def register_torch_tensor(self, tensor: torch.Tensor) -> FakeMemoryDesc:
        self.registered_tensors.append(tensor)
        return FakeMemoryDesc(key=f"mem-{len(self.registered_tensors)}", size=tensor.nbytes)

    def deregister_memory(self, mem_desc: FakeMemoryDesc) -> None:
        if self.deregister_error is not None:
            raise self.deregister_error
        self.deregistered_memory.append(mem_desc)

    # -- remote engines -----------------------------------------------------
    def register_remote_engine(self, engine_desc: FakeEngineDesc) -> None:
        if self.register_engine_error is not None:
            raise self.register_engine_error
        self.registered_remote.append(engine_desc)

    def deregister_remote_engine(self, engine_desc: FakeEngineDesc) -> None:
        if self.deregister_error is not None:
            raise self.deregister_error
        self.deregistered_remote.append(engine_desc)

    # -- transfers ----------------------------------------------------------
    def allocate_transfer_uid(self) -> int:
        self._transfer_uid += 1
        return self._transfer_uid

    def batch_write(self, *args: Any) -> list[FakeTransferStatus]:
        self.batch_writes.append(args)
        return self.statuses or [FakeTransferStatus()]


class FakePoolView:
    """Slice of :class:`FakePoolTensor` that behaves like a flat uint8 tensor."""

    def __init__(self, size: int, device: str = "cpu") -> None:
        self.size = size
        self.device = torch.device(device)
        self.is_cuda = device != "cpu"
        self.bytes = bytearray(size)

    def copy_(self, src: Any) -> FakePoolView:
        if isinstance(src, torch.Tensor):
            data = src.cpu().numpy().tobytes()
        elif isinstance(src, FakePoolView):
            data = bytes(src.bytes)
        else:
            data = bytes(src)
        self.bytes = bytearray(data[: self.size])
        return self

    def fill_(self, value: int) -> FakePoolView:
        self.bytes = bytearray([value] * self.size)
        return self

    def to_bytes(self) -> bytes:
        return bytes(self.bytes)

    def numpy(self) -> Any:
        return torch.frombuffer(bytearray(self.bytes), dtype=torch.uint8).numpy()


class FakePoolTensor:
    """Stand-in for the pinned host pool tensor allocated in ``__init__``."""

    def __init__(self, size: int, device: str = "cpu") -> None:
        self.nbytes = size
        self.device = torch.device(device)
        self.is_cuda = device != "cpu"
        self.pin_memory_calls = 0
        self._flat = FakePoolView(size, device)

    def __getitem__(self, item: slice) -> FakePoolView:
        start = item.start or 0
        stop = item.stop if item.stop is not None else self.nbytes
        return FakePoolView(max(0, stop - start), self.device.type)

    def data_ptr(self) -> int:
        return 0x7F0000

    def pin_memory(self) -> FakePoolTensor:
        self.pin_memory_calls += 1
        return self


class FakeZmqSocket:
    """Scripted stand-in for a ``zmq.Socket``."""

    def __init__(self, responses: list[Any] | None = None, on_send: Any = None) -> None:
        self.responses = list(responses or [])
        self.on_send = on_send
        self.sent: list[bytes] = []
        self.connected: list[str] = []
        self.options: dict[int, int] = {}
        self.closed = False
        self.close_linger: int | None = None

    def connect(self, addr: str) -> None:
        self.connected.append(addr)

    def setsockopt(self, option: int, value: int) -> None:
        self.options[option] = value

    def send(self, payload: bytes, *args: Any) -> None:
        self.sent.append(payload)
        if self.on_send is not None:
            self.on_send(self, payload)

    def recv(self) -> bytes:
        if not self.responses:
            raise AssertionError("FakeZmqSocket.recv() called with no scripted response")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self, linger: int | None = None) -> None:
        self.closed = True
        self.close_linger = linger


class FakeZmqContext:
    """Stand-in for ``zmq.Context``."""

    def __init__(self) -> None:
        self.sockets: list[FakeZmqSocket] = []
        self.terminated = False
        self.scripted: list[list[Any]] = []
        self.on_send: Any = None

    def socket(self, kind: int) -> FakeZmqSocket:
        responses = self.scripted.pop(0) if self.scripted else []
        sock = FakeZmqSocket(responses=responses, on_send=self.on_send)
        self.sockets.append(sock)
        return sock

    def term(self) -> None:
        self.terminated = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_fake_engine() -> None:
    FakeIOEngine.instances.clear()


@pytest.fixture
def fake_mori(monkeypatch: pytest.MonkeyPatch) -> type[FakeIOEngine]:
    """Replace the optional ``mori`` bindings imported by the connector."""
    monkeypatch.setattr(mori_mod, "IOEngine", FakeIOEngine, raising=False)
    monkeypatch.setattr(mori_mod, "EngineDesc", FakeEngineDesc, raising=False)
    monkeypatch.setattr(mori_mod, "MemoryDesc", FakeMemoryDesc, raising=False)
    monkeypatch.setattr(mori_mod, "IOEngineConfig", FakeIOEngineConfig, raising=False)
    monkeypatch.setattr(mori_mod, "BackendType", FakeBackendType, raising=False)
    monkeypatch.setattr(mori_mod, "PollCqMode", FakePollCqMode, raising=False)
    monkeypatch.setattr(mori_mod, "RdmaBackendConfig", FakeRdmaBackendConfig, raising=False)
    monkeypatch.setattr(mori_mod, "XgmiBackendConfig", FakeXgmiBackendConfig, raising=False)
    monkeypatch.setattr(mori_mod, "TransferStatus", FakeTransferStatus, raising=False)
    return FakeIOEngine


@pytest.fixture(autouse=True)
def patch_wire_descs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose fake ``EngineDesc`` / ``MemoryDesc`` even when mori is absent.

    The module binds these names inside a ``try: import mori ... except
    ImportError`` block, so a machine without Mori has no ``EngineDesc`` /
    ``MemoryDesc`` attribute at all.
    """
    monkeypatch.setattr(mori_mod, "EngineDesc", FakeEngineDesc, raising=False)
    monkeypatch.setattr(mori_mod, "MemoryDesc", FakeMemoryDesc, raising=False)


@pytest.fixture
def fake_pool(monkeypatch: pytest.MonkeyPatch) -> FakePoolTensor:
    """Avoid allocating a real pinned pool inside ``__init__``."""
    pool = FakePoolTensor(4096)
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: pool)
    return pool


def _make_connector(
    *,
    role: str = "sender",
    pool_size: int = 8192,
    sender_host: str | None = "127.0.0.1",
    sender_zmq_port: int | None = 50052,
    can_put: bool | None = None,
    scripted: list[list[Any]] | None = None,
    on_send: Any = None,
) -> MoriTransferEngineConnector:
    """Build a connector without running the RDMA/zmq ``__init__`` bring-up."""
    connector = object.__new__(MoriTransferEngineConnector)
    connector._closed = False
    connector._bind_error = None
    connector._stop_event = threading.Event()
    connector._sender_executor = None
    connector._listener_thread = None
    connector._listener_ready = threading.Event()
    connector._local_buffers = {}
    connector._local_buffers_lock = threading.Lock()
    connector._req_local = threading.local()
    connector._worker_local = threading.local()
    connector._last_ttl_check = time.monotonic()
    connector._registered_engines = {}
    connector._registered_engines_lock = threading.Lock()
    connector._metrics = {"puts": 0, "gets": 0, "bytes_transferred": 0, "errors": 0, "timeouts": 0}

    connector.config = {}
    connector.stage_id = -1
    connector.host = "127.0.0.1"
    connector.zmq_port = 50051
    connector.backend_type = "rdma"
    connector.device_name = ""
    connector.pool_size = pool_size
    connector.pool_device = "cpu"
    connector.sender_host = sender_host
    connector.sender_zmq_port = sender_zmq_port
    connector.role = role
    connector.can_put = (role == "sender") if can_put is None else can_put
    connector.engine_key = "omni-test-engine"

    connector.engine = FakeIOEngine(connector.engine_key, None)
    connector.engine_desc = FakeEngineDesc(key=connector.engine_key)
    connector.engine_desc_packed = connector.engine_desc.pack()

    connector.pool = torch.zeros(pool_size, dtype=torch.uint8)
    connector.pool_mem_desc = FakeMemoryDesc(key="pool", size=pool_size)
    connector.pool_mem_desc_packed = connector.pool_mem_desc.pack()
    connector.base_ptr = connector.pool.data_ptr()
    connector.allocator = BufferAllocator(pool_size, alignment=4096)

    fake_ctx = FakeZmqContext()
    if scripted is not None:
        fake_ctx.scripted = [list(items) for items in scripted]
    if on_send is not None:
        fake_ctx.on_send = on_send
    connector.zmq_ctx = fake_ctx
    return connector


def _stage_put(connector: MoriTransferEngineConnector, data: Any, key: str = "req") -> tuple[bool, int, Any]:
    return connector.put("stage_0", "stage_1", key, data)


def _stored_buffer(connector: MoriTransferEngineConnector, key: str = "req@stage_0_stage_1") -> tuple:
    return connector._local_buffers[key]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_into_pool(connector: MoriTransferEngineConnector, offset: int, payload: bytes) -> None:
    view = connector.pool[offset : offset + len(payload)]
    view.copy_(torch.frombuffer(bytearray(payload), dtype=torch.uint8))


# ---------------------------------------------------------------------------
# Wire structs
# ---------------------------------------------------------------------------


class TestMoriProtocolMessages:
    def test_pull_request_roundtrip(self) -> None:
        request = MoriPullRequest(
            request_id="req@0_1",
            engine_desc_packed=b"\x01\x02",
            mem_desc_packed=b"\x03\x04",
            dst_offset=4096,
            length=512,
        )
        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(request), type=MoriPullRequest)

        assert decoded.request_id == "req@0_1"
        assert decoded.engine_desc_packed == b"\x01\x02"
        assert decoded.mem_desc_packed == b"\x03\x04"
        assert decoded.dst_offset == 4096
        assert decoded.length == 512

    def test_pull_request_is_msgspec_struct(self) -> None:
        assert issubclass(MoriPullRequest, msgspec.Struct)
        assert issubclass(QueryRequest, msgspec.Struct)
        assert issubclass(QueryResponse, msgspec.Struct)

    def test_query_request_roundtrip(self) -> None:
        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(QueryRequest(request_id="k@0_1")), type=QueryRequest)
        assert decoded.request_id == "k@0_1"

    def test_query_response_roundtrip(self) -> None:
        response = QueryResponse(request_id="k@0_1", data_size=8192, is_fast_path=True)
        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(response), type=QueryResponse)

        assert decoded.request_id == "k@0_1"
        assert decoded.data_size == 8192
        assert decoded.is_fast_path is True

    def test_control_frames_are_distinct(self) -> None:
        frames = {TRANS_DONE, TRANS_ERROR, QUERY_INFO, INFO_NOT_FOUND}
        assert len(frames) == 4
        for frame in frames:
            assert frame
        # QUERY_INFO is used as a prefix, so it must not be a prefix of another frame.
        others = [f for f in frames if f != QUERY_INFO]
        assert not any(other.startswith(QUERY_INFO) for other in others)

    @pytest.mark.parametrize(
        "payload, expected",
        [
            (QUERY_INFO + b"body", True),
            (QUERY_INFO, True),
            (TRANS_DONE, False),
            (msgspec.msgpack.encode(QueryRequest(request_id="x")), False),
        ],
    )
    def test_query_prefix_detection(self, payload: bytes, expected: bool) -> None:
        assert payload.startswith(QUERY_INFO) is expected


class TestModuleContract:
    def test_supports_raw_data(self) -> None:
        assert MoriTransferEngineConnector.supports_raw_data is True

    def test_supported_backends(self) -> None:
        assert mori_mod._SUPPORTED_BACKENDS == ("rdma", "xgmi")

    def test_buffer_ttl_is_positive(self) -> None:
        assert mori_mod._BUFFER_TTL_SECONDS > 0

    def test_constructor_requires_mori(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mori_mod, "IOEngine", None)
        with pytest.raises(ImportError, match="Mori is not available"):
            MoriTransferEngineConnector({"role": "receiver"})


# ---------------------------------------------------------------------------
# Configuration / __init__
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_rejects_invalid_backend_type(self, fake_mori: type[FakeIOEngine], fake_pool: FakePoolTensor) -> None:
        with pytest.raises(ValueError, match="Invalid backend_type"):
            MoriTransferEngineConnector({"role": "receiver", "backend_type": "nvlink"})

    def test_rejects_invalid_role(self, fake_mori: type[FakeIOEngine], fake_pool: FakePoolTensor) -> None:
        with pytest.raises(ValueError, match="Invalid role"):
            MoriTransferEngineConnector({"role": "producer"})

    def test_xgmi_requires_cuda_pool(self, fake_mori: type[FakeIOEngine], fake_pool: FakePoolTensor) -> None:
        with pytest.raises(ValueError, match="requires memory_pool_device='cuda'"):
            MoriTransferEngineConnector({"role": "receiver", "backend_type": "xgmi"})

    def test_xgmi_ignores_device_name(
        self,
        fake_mori: type[FakeIOEngine],
        fake_pool: FakePoolTensor,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        connector = MoriTransferEngineConnector(
            {
                "role": "receiver",
                "backend_type": "xgmi",
                "memory_pool_device": "cuda",
                "device_name": "mlx5_0",
            }
        )

        assert connector.device_name == ""
        assert "is ignored for backend_type='xgmi'" in caplog.text

    def test_rdma_backend_config_forwarded(self, fake_mori: type[FakeIOEngine], fake_pool: FakePoolTensor) -> None:
        connector = MoriTransferEngineConnector(
            {
                "role": "receiver",
                "qp_per_transfer": 2,
                "post_batch_size": 16,
                "num_worker_threads": 3,
            }
        )

        engine = connector.engine
        backend_type, config = engine.backends[0]
        assert backend_type == FakeBackendType.RDMA
        assert config.args == (2, 16, 3, FakePollCqMode.POLLING, False)

    def test_xgmi_backend_config_forwarded(self, fake_mori: type[FakeIOEngine], fake_pool: FakePoolTensor) -> None:
        connector = MoriTransferEngineConnector(
            {
                "role": "receiver",
                "backend_type": "xgmi",
                "memory_pool_device": "cuda",
                "xgmi_num_streams": 8,
                "xgmi_num_events": 4,
            }
        )

        backend_type, config = connector.engine.backends[0]
        assert backend_type == FakeBackendType.XGMI
        assert config.num_streams == 8
        assert config.num_events == 4

    def test_rdma_device_from_env(
        self,
        fake_mori: type[FakeIOEngine],
        fake_pool: FakePoolTensor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MORI_RDMA_DEVICES", "mlx5_1")

        connector = MoriTransferEngineConnector({"role": "receiver"})

        assert connector.device_name == "mlx5_1"

    def test_rdma_device_from_config_takes_precedence(
        self,
        fake_mori: type[FakeIOEngine],
        fake_pool: FakePoolTensor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MORI_RDMA_DEVICES", "mlx5_from_env")

        connector = MoriTransferEngineConnector({"role": "receiver", "device_name": "mlx5_cfg"})

        assert connector.device_name == "mlx5_cfg"

    def test_host_auto_uses_detected_ip(
        self,
        fake_mori: type[FakeIOEngine],
        fake_pool: FakePoolTensor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(MoriTransferEngineConnector, "_get_local_ip", staticmethod(lambda: "10.0.0.7"))

        connector = MoriTransferEngineConnector({"role": "receiver", "host": "auto"})

        assert connector.host == "10.0.0.7"

    def test_receiver_role_does_not_start_listener(
        self,
        fake_mori: type[FakeIOEngine],
        fake_pool: FakePoolTensor,
    ) -> None:
        connector = MoriTransferEngineConnector({"role": "receiver"})

        assert connector.can_put is False
        assert connector._listener_thread is None
        assert connector.get_connection_info()["can_put"] is False

    def test_receiver_with_sender_info(self, fake_mori: type[FakeIOEngine], fake_pool: FakePoolTensor) -> None:
        connector = MoriTransferEngineConnector(
            {"role": "receiver", "sender_host": "10.0.0.9", "sender_zmq_port": 50061}
        )

        assert connector.can_put is False
        assert connector.sender_host == "10.0.0.9"
        assert connector.sender_zmq_port == 50061

    def test_sender_role_starts_listener(
        self,
        fake_mori: type[FakeIOEngine],
        fake_pool: FakePoolTensor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        started = threading.Event()

        def _fake_listener(self: MoriTransferEngineConnector) -> None:
            started.set()
            self._listener_ready.set()

        monkeypatch.setattr(MoriTransferEngineConnector, "_zmq_listener_loop", _fake_listener)

        connector = MoriTransferEngineConnector({"role": "sender"})
        connector._listener_thread.join(timeout=5.0)

        assert started.is_set()
        assert connector.can_put is True
        assert connector._bind_error is None

    def test_sender_bind_error_is_fatal(
        self,
        fake_mori: type[FakeIOEngine],
        fake_pool: FakePoolTensor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bind_error = zmq.ZMQError(errno=98, msg="Address already in use")

        def _failing_listener(self: MoriTransferEngineConnector) -> None:
            self._bind_error = bind_error
            self._listener_ready.set()

        monkeypatch.setattr(MoriTransferEngineConnector, "_zmq_listener_loop", _failing_listener)

        with pytest.raises(RuntimeError, match="failed to bind ZMQ"):
            MoriTransferEngineConnector({"role": "sender"})

    def test_pool_registration_failure_propagates(
        self,
        fake_mori: type[FakeIOEngine],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("oom")))

        with pytest.raises(RuntimeError, match="oom"):
            MoriTransferEngineConnector({"role": "receiver"})

    def test_defaults_and_connection_info(self, fake_mori: type[FakeIOEngine], fake_pool: FakePoolTensor) -> None:
        connector = MoriTransferEngineConnector({"role": "receiver", "stage_id": 3, "zmq_port": 1234})

        assert connector.stage_id == 3
        assert connector.backend_type == "rdma"
        assert connector.pool_device == "cpu"
        assert connector.allocator.alignment == 4096
        assert connector._metrics == {"puts": 0, "gets": 0, "bytes_transferred": 0, "errors": 0, "timeouts": 0}
        info = connector.get_connection_info()
        assert info["host"] == "127.0.0.1"
        assert info["zmq_port"] == 1234
        assert info["engine_key"] == connector.engine_key
        assert connector.engine.registered_tensors == [connector.pool]

    def test_update_sender_info(self) -> None:
        connector = _make_connector(role="receiver", sender_host=None, sender_zmq_port=None)

        connector.update_sender_info("10.1.2.3", 6000)

        assert connector.sender_host == "10.1.2.3"
        assert connector.sender_zmq_port == 6000

    def test_get_local_ip_falls_back_to_loopback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(*args: Any, **kwargs: Any) -> None:
            raise OSError("no route")

        monkeypatch.setattr(socket, "socket", _raise)
        monkeypatch.setattr(socket, "gethostbyname", _raise)

        assert MoriTransferEngineConnector._get_local_ip() == "127.0.0.1"


# ---------------------------------------------------------------------------
# put()
# ---------------------------------------------------------------------------


class TestPut:
    def test_put_on_closed_connector_raises(self) -> None:
        connector = _make_connector()
        connector._closed = True

        with pytest.raises(RuntimeError, match="Cannot put"):
            _stage_put(connector, torch.ones(4, dtype=torch.uint8))

    def test_receiver_only_connector_rejects_put(self) -> None:
        connector = _make_connector(role="receiver")

        assert _stage_put(connector, torch.ones(4, dtype=torch.uint8)) == (False, 0, None)
        assert connector._local_buffers == {}
        assert connector._metrics["puts"] == 0

    def test_rejects_empty_bytes(self) -> None:
        connector = _make_connector()

        assert _stage_put(connector, b"") == (False, 0, None)
        assert connector._local_buffers == {}

    def test_rejects_empty_tensor(self) -> None:
        connector = _make_connector()

        assert _stage_put(connector, torch.empty(0, dtype=torch.uint8)) == (False, 0, None)
        assert connector._local_buffers == {}

    def test_tensor_fast_path(self) -> None:
        connector = _make_connector()
        tensor = torch.arange(16, dtype=torch.uint8)

        ok, size, metadata = _stage_put(connector, tensor)

        assert ok is True
        assert size == tensor.nbytes
        assert metadata == {
            "source_host": "127.0.0.1",
            "source_port": 50051,
            "data_size": tensor.nbytes,
            "is_fast_path": True,
        }
        offset, stored_size, holder, should_release, is_fast, _ts = _stored_buffer(connector)
        assert stored_size == tensor.nbytes
        assert should_release is True
        assert is_fast is True
        assert holder.to_bytes() == tensor.numpy().tobytes()
        assert offset % 4096 == 0
        assert connector._metrics["puts"] == 1
        assert connector._metrics["bytes_transferred"] == tensor.nbytes

    def test_non_contiguous_tensor_is_copied(self) -> None:
        connector = _make_connector()
        base = torch.arange(32, dtype=torch.uint8).reshape(4, 8)
        transposed = base.t()

        ok, size, _ = _stage_put(connector, transposed)

        assert ok is True
        assert size == transposed.nbytes
        _, _, holder, _, _, _ = _stored_buffer(connector)
        assert holder.to_bytes() == transposed.contiguous().numpy().tobytes()

    def test_object_payload_uses_serializer(self) -> None:
        connector = _make_connector()
        payload = {"tokens": [1, 2, 3], "nested": {"flag": True}}

        ok, size, metadata = _stage_put(connector, payload)

        assert ok is True
        assert metadata is not None
        assert metadata["is_fast_path"] is False
        assert size == len(mori_mod.OmniSerializer.serialize(payload))
        _, stored_size, holder, _, is_fast, _ = _stored_buffer(connector)
        assert is_fast is False
        assert stored_size == size
        assert mori_mod.OmniSerializer.deserialize(holder.to_bytes()) == payload

    def test_bytes_payload_roundtrip_size(self) -> None:
        connector = _make_connector()
        payload = b"mori-payload" * 8

        ok, size, metadata = _stage_put(connector, payload)

        assert ok is True
        assert size == len(payload)
        assert metadata is not None
        assert metadata["data_size"] == len(payload)
        _, _, holder, _, _, _ = _stored_buffer(connector)
        assert holder.to_bytes() == payload

    def test_managed_buffer_from_same_pool_is_zero_copy(self) -> None:
        connector = _make_connector()
        offset = connector.allocator.alloc(1024)
        buffer = ManagedBuffer(connector.allocator, offset, 1024, connector.pool)
        buffer.tensor.fill_(7)
        free_blocks_before = list(connector.allocator.free_blocks)

        ok, size, _ = _stage_put(connector, buffer)

        assert ok is True
        assert size == 1024
        stored_offset, stored_size, holder, should_release, _, _ = _stored_buffer(connector)
        assert stored_offset == offset
        assert stored_size == 1024
        assert holder is buffer
        assert should_release is False
        assert connector.allocator.free_blocks == free_blocks_before

    def test_managed_buffer_from_foreign_pool_is_copied(self) -> None:
        connector = _make_connector()
        foreign_allocator = BufferAllocator(2048, alignment=64)
        foreign_pool = torch.full((2048,), 3, dtype=torch.uint8)
        buffer = ManagedBuffer(foreign_allocator, 0, 128, foreign_pool)

        ok, size, _ = _stage_put(connector, buffer)

        assert ok is True
        assert size == 128
        _, _, holder, should_release, _, _ = _stored_buffer(connector)
        assert holder is not buffer
        assert should_release is True
        assert holder.to_bytes() == bytes([3]) * 128

    def test_empty_managed_buffer_is_rejected(self) -> None:
        connector = _make_connector()
        buffer = ManagedBuffer(connector.allocator, 0, 0, connector.pool)

        assert _stage_put(connector, buffer) == (False, 0, None)
        assert connector._local_buffers == {}

    def test_pool_exhaustion_returns_failure(self) -> None:
        connector = _make_connector(pool_size=4096)

        ok, size, metadata = _stage_put(connector, torch.ones(8192, dtype=torch.uint8))

        assert (ok, size, metadata) == (False, 0, None)
        assert connector._local_buffers == {}

    def test_duplicate_key_releases_stale_buffer(self) -> None:
        connector = _make_connector()
        first = torch.ones(64, dtype=torch.uint8)
        second = torch.zeros(64, dtype=torch.uint8)

        assert _stage_put(connector, first, key="dup")[0] is True
        first_entry = _stored_buffer(connector, "dup@stage_0_stage_1")
        assert _stage_put(connector, second, key="dup")[0] is True
        second_entry = _stored_buffer(connector, "dup@stage_0_stage_1")

        assert first_entry[2]._released is True
        assert second_entry[2].to_bytes() == bytes(64)

    def test_unexpected_error_is_counted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connector = _make_connector()

        def _boom(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("allocator bug")

        monkeypatch.setattr(connector.allocator, "alloc", _boom)

        assert _stage_put(connector, torch.ones(8, dtype=torch.uint8)) == (False, 0, None)
        assert connector._metrics["errors"] == 1


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_on_closed_connector_raises(self) -> None:
        connector = _make_connector(role="receiver")
        connector._closed = True

        with pytest.raises(RuntimeError, match="Cannot get"):
            connector.get("stage_0", "stage_1", "req", {"source_host": "127.0.0.1", "source_port": 1, "data_size": 4})

    def test_missing_metadata_requires_sender_info(self) -> None:
        connector = _make_connector(role="receiver", sender_host=None, sender_zmq_port=None)

        with pytest.raises(RuntimeError, match="requires sender info"):
            connector.get("stage_0", "stage_1", "req", None)

    def test_metadata_query_returns_none_when_sender_has_no_buffer(self) -> None:
        connector = _make_connector(role="receiver", scripted=[[INFO_NOT_FOUND]])

        assert connector.get("stage_0", "stage_1", "req", None) is None

    def test_invalid_metadata_host_returns_none(self) -> None:
        connector = _make_connector(role="receiver")

        assert (
            connector.get("stage_0", "stage_1", "req", {"source_host": "auto", "source_port": 1, "data_size": 8})
            is None
        )

    def test_zero_size_metadata_returns_none(self) -> None:
        connector = _make_connector(role="receiver")

        metadata = {"source_host": "127.0.0.1", "source_port": 5000, "data_size": 0, "is_fast_path": True}
        assert connector.get("stage_0", "stage_1", "req", metadata) is None

    def test_allocation_failure_returns_none(self) -> None:
        connector = _make_connector(role="receiver", pool_size=4096)
        metadata = {"source_host": "127.0.0.1", "source_port": 5000, "data_size": 8192, "is_fast_path": True}

        assert connector.get("stage_0", "stage_1", "req", metadata) is None

    def test_fast_path_returns_managed_buffer(self) -> None:
        payload = b"fast-path" * 4
        written: dict[str, int] = {}

        def _on_send(sock: FakeZmqSocket, raw: bytes) -> None:
            request = msgspec.msgpack.decode(raw, type=MoriPullRequest)
            written["offset"] = request.dst_offset
            _write_into_pool(connector, request.dst_offset, payload)

        connector = _make_connector(role="receiver", on_send=_on_send, scripted=[[TRANS_DONE]])
        metadata = {
            "source_host": "127.0.0.1",
            "source_port": 5000,
            "data_size": len(payload),
            "is_fast_path": True,
        }

        result = connector.get("stage_0", "stage_1", "req", metadata)

        assert result is not None
        received, size = result
        assert isinstance(received, ManagedBuffer)
        assert size == len(payload)
        assert received.to_bytes() == payload
        assert received.offset == written["offset"]
        assert connector._metrics["gets"] == 1
        assert connector._metrics["bytes_transferred"] == len(payload)

    def test_slow_path_deserializes_and_releases_buffer(self) -> None:
        payload = {"answer": 42, "tokens": [7, 8, 9]}
        serialized = mori_mod.OmniSerializer.serialize(payload)

        def _on_send(sock: FakeZmqSocket, raw: bytes) -> None:
            request = msgspec.msgpack.decode(raw, type=MoriPullRequest)
            _write_into_pool(connector, request.dst_offset, serialized)

        connector = _make_connector(role="receiver", on_send=_on_send, scripted=[[TRANS_DONE]])
        metadata = {
            "source_host": "127.0.0.1",
            "source_port": 5000,
            "data_size": len(serialized),
            "is_fast_path": False,
        }

        result = connector.get("stage_0", "stage_1", "req", metadata)

        assert result == (payload, len(serialized))
        # The staging buffer must be released once the object is deserialised.
        assert connector.allocator.free_blocks[0] == (0, connector.pool_size)

    def test_error_response_returns_none(self) -> None:
        connector = _make_connector(role="receiver", scripted=[[TRANS_ERROR]])
        metadata = {"source_host": "127.0.0.1", "source_port": 5000, "data_size": 16, "is_fast_path": True}

        assert connector.get("stage_0", "stage_1", "req", metadata) is None
        assert connector._metrics["errors"] == 1

    def test_transport_error_is_counted_and_socket_invalidated(self) -> None:
        connector = _make_connector(role="receiver", scripted=[[zmq.Again()]])
        metadata = {"source_host": "127.0.0.1", "source_port": 5000, "data_size": 16, "is_fast_path": True}

        assert connector.get("stage_0", "stage_1", "req", metadata) is None
        assert connector._metrics["timeouts"] == 1
        assert getattr(connector._req_local, "cache", {}) == {}


class TestMetadataQuery:
    def test_query_returns_metadata(self) -> None:
        response = QueryResponse(request_id="req@stage_0_stage_1", data_size=32, is_fast_path=False)
        connector = _make_connector(role="receiver", scripted=[[msgspec.msgpack.encode(response)]])

        metadata = connector._query_metadata_from_sender("req@stage_0_stage_1")

        assert metadata == {
            "source_host": "127.0.0.1",
            "source_port": 50052,
            "data_size": 32,
            "is_fast_path": False,
        }
        sock = connector.zmq_ctx.sockets[0]
        assert sock.connected == ["tcp://127.0.0.1:50052"]
        assert sock.sent[0].startswith(QUERY_INFO)

    def test_query_not_found_returns_none(self) -> None:
        connector = _make_connector(role="receiver", scripted=[[INFO_NOT_FOUND]])

        assert connector._query_metadata_from_sender("missing@stage_0_stage_1") is None

    def test_query_error_returns_none_and_invalidates_socket(self) -> None:
        connector = _make_connector(role="receiver", scripted=[[zmq.Again()]])

        assert connector._query_metadata_from_sender("req@stage_0_stage_1") is None
        assert getattr(connector._req_local, "cache", {}) == {}


# ---------------------------------------------------------------------------
# cleanup() / health() / close()
# ---------------------------------------------------------------------------


class TestCleanupHealthClose:
    def test_cleanup_requires_both_stages(self) -> None:
        connector = _make_connector()

        with pytest.raises(ValueError, match="requires both from_stage and to_stage"):
            connector.cleanup("req", from_stage="stage_0")

    def test_cleanup_composes_key_with_stages(self) -> None:
        connector = _make_connector()
        assert _stage_put(connector, torch.ones(32, dtype=torch.uint8))[0] is True

        connector.cleanup("req", "stage_0", "stage_1")

        assert connector._local_buffers == {}
        assert connector.allocator.free_blocks == [(0, connector.pool_size)]

    def test_cleanup_without_stages_uses_raw_key(self) -> None:
        connector = _make_connector()
        assert _stage_put(connector, torch.ones(32, dtype=torch.uint8))[0] is True

        connector.cleanup("req@stage_0_stage_1")

        assert connector._local_buffers == {}

    def test_cleanup_unknown_key_is_noop(self) -> None:
        connector = _make_connector()

        connector.cleanup("nope@stage_0_stage_1")

        assert connector._local_buffers == {}
        assert connector.allocator.free_blocks == [(0, connector.pool_size)]

    def test_cleanup_keeps_borrowed_managed_buffer(self) -> None:
        connector = _make_connector()
        offset = connector.allocator.alloc(64)
        buffer = ManagedBuffer(connector.allocator, offset, 64, connector.pool)
        assert _stage_put(connector, buffer)[0] is True

        connector.cleanup("req", "stage_0", "stage_1")

        assert connector._local_buffers == {}
        assert buffer._released is False

    def test_health_reports_metrics(self) -> None:
        connector = _make_connector()
        _stage_put(connector, torch.ones(16, dtype=torch.uint8))

        health = connector.health()

        assert health["status"] == "healthy"
        assert health["host"] == "127.0.0.1"
        assert health["pool_device"] == "cpu"
        assert health["pool_size"] == 8192
        assert health["engine_key"] == "omni-test-engine"
        assert health["puts"] == 1
        assert health["bytes_transferred"] == 16

    def test_health_after_close_is_unhealthy(self) -> None:
        connector = _make_connector()
        connector.close()

        health = connector.health()

        assert health["status"] == "unhealthy"
        assert health["error"] == "Connector is closed"

    def test_close_is_idempotent(self) -> None:
        connector = _make_connector()

        connector.close()
        connector.close()

        assert connector._closed is True

    def test_close_releases_buffers_and_engine_resources(self) -> None:
        connector = _make_connector()
        engine = connector.engine
        assert _stage_put(connector, torch.ones(64, dtype=torch.uint8))[0] is True
        remote_desc = FakeEngineDesc(key="peer")
        connector._registered_engines["peer"] = remote_desc

        connector.close()

        assert connector._local_buffers == {}
        assert [desc.key for desc in engine.deregistered_memory] == ["pool"]
        assert engine.deregistered_remote == [remote_desc]
        assert connector.engine is None
        assert connector.pool is None

    def test_close_terminates_zmq_context_and_sockets(self) -> None:
        connector = _make_connector(role="receiver", scripted=[[]])
        connector._query_metadata_from_sender("req@stage_0_stage_1")
        sock = connector.zmq_ctx.sockets[0]

        connector.close()

        assert sock.closed is True
        assert sock.close_linger == 0
        assert connector.zmq_ctx.terminated is True

    def test_close_tolerates_deregister_failures(self) -> None:
        connector = _make_connector()
        connector.engine.deregister_error = RuntimeError("engine gone")

        connector.close()

        assert connector._closed is True
        assert connector.engine is None

    def test_close_tolerates_socket_and_context_failures(self) -> None:
        connector = _make_connector(role="receiver", scripted=[[]])
        connector._query_metadata_from_sender("req@stage_0_stage_1")
        sock = connector.zmq_ctx.sockets[0]

        def _close_boom(linger: int | None = None) -> None:
            raise RuntimeError("socket already gone")

        def _term_boom() -> None:
            raise RuntimeError("context already terminated")

        sock.close = _close_boom  # type: ignore[method-assign]
        connector.zmq_ctx.term = _term_boom  # type: ignore[method-assign]

        connector.close()

        assert connector._closed is True

    def test_close_tolerates_remote_engine_deregister_failure(self) -> None:
        connector = _make_connector()
        connector._registered_engines["peer"] = FakeEngineDesc(key="peer")
        connector.engine.deregister_error = RuntimeError("fabric gone")

        connector.close()

        assert connector._registered_engines == {}
        assert connector.engine is None

    def test_close_stops_listener_thread(self) -> None:
        connector = _make_connector()
        stop_seen = threading.Event()

        def _fake_listener() -> None:
            connector._stop_event.wait(5.0)
            stop_seen.set()

        connector._listener_thread = threading.Thread(target=_fake_listener, daemon=True)
        connector._listener_thread.start()

        connector.close()

        assert stop_seen.wait(5.0) is True


# ---------------------------------------------------------------------------
# Sender-side listener handlers
# ---------------------------------------------------------------------------


class TestRemoteRegistration:
    def test_registers_remote_engine_once(self) -> None:
        connector = _make_connector()
        desc = FakeEngineDesc(key="peer")

        first = connector._ensure_remote_registered(desc.pack())
        second = connector._ensure_remote_registered(desc.pack())

        assert first == "peer"
        assert second == "peer"
        assert [d.key for d in connector.engine.registered_remote] == ["peer"]

    def test_registration_error_propagates(self) -> None:
        connector = _make_connector()
        connector.engine.register_engine_error = RuntimeError("register failed")

        with pytest.raises(RuntimeError, match="register failed"):
            connector._ensure_remote_registered(FakeEngineDesc(key="peer").pack())


class TestListenerHandlers:
    @staticmethod
    def _collect_notify(connector: MoriTransferEngineConnector) -> list[str]:
        notified: list[str] = []

        def _notify(addr: str) -> None:
            notified.append(addr)

        connector._notify_listener = _notify  # type: ignore[assignment]
        return notified

    def test_query_handler_reports_ready_buffer(self) -> None:
        connector = _make_connector()
        connector._local_buffers["req@0_1"] = (0, 128, None, False, True, time.monotonic())
        connector._notify_listener = lambda addr: None
        response_queue: queue.Queue = queue.Queue()

        connector._handle_query_request(
            response_queue, "inproc://n", b"id", msgspec.msgpack.encode(QueryRequest("req@0_1"))
        )

        identity, payload = response_queue.get_nowait()
        assert identity == b"id"
        decoded = msgspec.msgpack.decode(payload, type=QueryResponse)
        assert decoded.data_size == 128
        assert decoded.is_fast_path is True

    def test_query_handler_reports_missing_buffer(self) -> None:
        connector = _make_connector()
        connector._notify_listener = lambda addr: None
        response_queue: queue.Queue = queue.Queue()

        connector._handle_query_request(
            response_queue, "inproc://n", b"id", msgspec.msgpack.encode(QueryRequest("gone"))
        )

        assert response_queue.get_nowait() == (b"id", INFO_NOT_FOUND)

    def test_query_handler_rejects_malformed_payload(self) -> None:
        connector = _make_connector()
        connector._notify_listener = lambda addr: None
        response_queue: queue.Queue = queue.Queue()

        connector._handle_query_request(response_queue, "inproc://n", b"id", b"not-msgpack")

        assert response_queue.get_nowait() == (b"id", INFO_NOT_FOUND)

    def test_pull_request_unknown_key_returns_error(self) -> None:
        connector = _make_connector()
        notified = self._collect_notify(connector)
        response_queue: queue.Queue = queue.Queue()
        pull = MoriPullRequest(
            request_id="missing@0_1",
            engine_desc_packed=connector.engine_desc_packed,
            mem_desc_packed=connector.pool_mem_desc_packed,
            dst_offset=0,
            length=16,
        )

        connector._handle_pull_request(response_queue, "inproc://n", b"id", msgspec.msgpack.encode(pull))

        assert response_queue.get_nowait() == (b"id", TRANS_ERROR)
        assert notified == ["inproc://n"]

    def test_pull_request_length_mismatch_is_refused(self) -> None:
        connector = _make_connector()
        notified = self._collect_notify(connector)
        response_queue: queue.Queue = queue.Queue()
        assert _stage_put(connector, torch.ones(64, dtype=torch.uint8))[0] is True
        pull = MoriPullRequest(
            request_id="req@stage_0_stage_1",
            engine_desc_packed=connector.engine_desc_packed,
            mem_desc_packed=connector.pool_mem_desc_packed,
            dst_offset=0,
            length=32,
        )

        connector._handle_pull_request(response_queue, "inproc://n", b"id", msgspec.msgpack.encode(pull))

        assert response_queue.get_nowait() == (b"id", TRANS_ERROR)
        assert connector.engine.batch_writes == []
        assert "req@stage_0_stage_1" in connector._local_buffers
        assert notified == ["inproc://n"]

    def test_pull_request_success_writes_and_cleans_up(self) -> None:
        connector = _make_connector()
        notified = self._collect_notify(connector)
        response_queue: queue.Queue = queue.Queue()
        assert _stage_put(connector, torch.ones(64, dtype=torch.uint8))[0] is True
        source_offset, source_size, _, _, _, _ = _stored_buffer(connector)
        pull = MoriPullRequest(
            request_id="req@stage_0_stage_1",
            engine_desc_packed=connector.engine_desc_packed,
            mem_desc_packed=connector.pool_mem_desc_packed,
            dst_offset=12288,
            length=source_size,
        )

        connector._handle_pull_request(response_queue, "inproc://n", b"id", msgspec.msgpack.encode(pull))

        assert response_queue.get_nowait() == (b"id", TRANS_DONE)
        assert len(connector.engine.batch_writes) == 1
        local_mems, local_offsets, remote_mems, remote_offsets, sizes, uids = connector.engine.batch_writes[0]
        assert local_mems == [connector.pool_mem_desc]
        assert local_offsets == [[source_offset]]
        assert isinstance(remote_mems[0], FakeMemoryDesc)
        assert remote_offsets == [[12288]]
        assert sizes == [[source_size]]
        assert len(uids) == 1
        assert "req@stage_0_stage_1" not in connector._local_buffers
        assert connector.engine.registered_remote[0].key == "omni-test-engine"
        assert notified == ["inproc://n"]

    def test_pull_request_failed_transfer_retains_buffer(self) -> None:
        connector = _make_connector()
        self._collect_notify(connector)
        connector.engine.statuses = [FakeTransferStatus(failed=True, message="cq error")]
        response_queue: queue.Queue = queue.Queue()
        assert _stage_put(connector, torch.ones(64, dtype=torch.uint8))[0] is True
        _, source_size, _, _, _, _ = _stored_buffer(connector)
        pull = MoriPullRequest(
            request_id="req@stage_0_stage_1",
            engine_desc_packed=connector.engine_desc_packed,
            mem_desc_packed=connector.pool_mem_desc_packed,
            dst_offset=0,
            length=source_size,
        )

        connector._handle_pull_request(response_queue, "inproc://n", b"id", msgspec.msgpack.encode(pull))

        assert response_queue.get_nowait() == (b"id", TRANS_ERROR)
        assert "req@stage_0_stage_1" in connector._local_buffers

    def test_pull_request_routes_query_prefix(self) -> None:
        connector = _make_connector()
        notified = self._collect_notify(connector)
        response_queue: queue.Queue = queue.Queue()
        connector._local_buffers["req@0_1"] = (0, 32, None, False, False, time.monotonic())
        payload = QUERY_INFO + msgspec.msgpack.encode(QueryRequest("req@0_1"))

        connector._handle_pull_request(response_queue, "inproc://n", b"id", payload)

        identity, body = response_queue.get_nowait()
        assert (identity, msgspec.msgpack.decode(body, type=QueryResponse).data_size) == (b"id", 32)
        assert notified == ["inproc://n"]

    def test_pull_request_handles_exceptions(self) -> None:
        connector = _make_connector()
        notified = self._collect_notify(connector)
        response_queue: queue.Queue = queue.Queue()

        connector._handle_pull_request(response_queue, "inproc://n", b"id", b"garbage")

        assert response_queue.get_nowait() == (b"id", TRANS_ERROR)
        assert notified == ["inproc://n"]


class TestStaleBufferCleanup:
    def test_expired_buffers_are_reclaimed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connector = _make_connector()
        assert _stage_put(connector, torch.ones(64, dtype=torch.uint8))[0] is True
        stale_ts = time.monotonic() - mori_mod._BUFFER_TTL_SECONDS - 1
        connector._local_buffers["req@stage_0_stage_1"] = (
            *connector._local_buffers["req@stage_0_stage_1"][:5],
            stale_ts,
        )

        connector._cleanup_stale_buffers()

        assert connector._local_buffers == {}
        assert connector.allocator.free_blocks == [(0, connector.pool_size)]

    def test_fresh_buffers_are_kept(self) -> None:
        connector = _make_connector()
        assert _stage_put(connector, torch.ones(64, dtype=torch.uint8))[0] is True

        connector._cleanup_stale_buffers()

        assert "req@stage_0_stage_1" in connector._local_buffers


class TestRequestSocketCache:
    def test_socket_is_cached_per_address(self) -> None:
        connector = _make_connector(role="receiver")

        first = connector._get_req_socket("tcp://127.0.0.1:5000", timeout_ms=250)
        second = connector._get_req_socket("tcp://127.0.0.1:5000", timeout_ms=1000)

        assert first is second
        assert first.options[zmq.SNDTIMEO] == 1000
        assert first.options[zmq.RCVTIMEO] == 1000
        assert len(connector.zmq_ctx.sockets) == 1

    def test_invalidate_closes_and_drops_socket(self) -> None:
        connector = _make_connector(role="receiver")
        sock = connector._get_req_socket("tcp://127.0.0.1:5000", timeout_ms=250)

        connector._invalidate_req_socket("tcp://127.0.0.1:5000")

        assert sock.closed is True
        assert connector._req_local.cache == {}

    def test_invalidate_unknown_address_is_noop(self) -> None:
        connector = _make_connector(role="receiver")

        connector._invalidate_req_socket("tcp://127.0.0.1:5000")

        assert getattr(connector._req_local, "cache", None) is None


class TestNotifyListener:
    def test_notify_socket_is_reused_per_address(self) -> None:
        connector = _make_connector()

        connector._notify_listener("inproc://a")
        first = connector._worker_local.notify_socket
        connector._notify_listener("inproc://a")

        assert connector._worker_local.notify_socket is first
        assert len(first.connected) == 1

    def test_notify_reconnects_when_address_changes(self) -> None:
        connector = _make_connector()

        connector._notify_listener("inproc://a")
        first = connector._worker_local.notify_socket
        connector._notify_listener("inproc://b")

        assert first.closed is True
        assert connector._worker_local.notify_socket is not first
        assert connector._worker_local.notify_addr == "inproc://b"

    def test_notify_swallows_send_failures(self) -> None:
        connector = _make_connector()
        connector._notify_listener("inproc://a")
        sock = connector._worker_local.notify_socket

        def _boom(payload: bytes) -> None:
            raise zmq.ZMQError(errno=11, msg="again")

        sock.send = _boom  # type: ignore[method-assign]

        connector._notify_listener("inproc://a")

        assert connector._worker_local.notify_socket is None
        assert connector._worker_local.notify_addr is None


# ---------------------------------------------------------------------------
# Real ZMQ loopback smoke (no Mori, no RDMA)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSenderLoopback:
    def test_sender_answers_query_requests(self, fake_mori: type[FakeIOEngine], fake_pool: FakePoolTensor) -> None:
        port = _free_port()
        connector = MoriTransferEngineConnector({"role": "sender", "host": "127.0.0.1", "zmq_port": port})
        try:
            assert connector._listener_thread is not None
            assert connector._listener_thread.is_alive()

            ctx = zmq.Context()
            req = ctx.socket(zmq.REQ)
            req.setsockopt(zmq.RCVTIMEO, 5000)
            req.connect(f"tcp://127.0.0.1:{port}")
            try:
                req.send(QUERY_INFO + msgspec.msgpack.encode(QueryRequest("absent@0_1")))
                assert req.recv() == INFO_NOT_FOUND
            finally:
                req.close(linger=0)
                ctx.term()
        finally:
            connector.close()

    def test_sender_reports_put_buffer_via_query(
        self, fake_mori: type[FakeIOEngine], fake_pool: FakePoolTensor
    ) -> None:
        port = _free_port()
        connector = MoriTransferEngineConnector({"role": "sender", "host": "127.0.0.1", "zmq_port": port})
        try:
            ok, size, _ = connector.put("stage_0", "stage_1", "kbuf", torch.ones(16, dtype=torch.uint8))
            assert ok is True

            ctx = zmq.Context()
            req = ctx.socket(zmq.REQ)
            req.setsockopt(zmq.RCVTIMEO, 5000)
            req.connect(f"tcp://127.0.0.1:{port}")
            try:
                req.send(QUERY_INFO + msgspec.msgpack.encode(QueryRequest("kbuf@stage_0_stage_1")))
                response = msgspec.msgpack.decode(req.recv(), type=QueryResponse)
            finally:
                req.close(linger=0)
                ctx.term()

            assert response.request_id == "kbuf@stage_0_stage_1"
            assert response.data_size == size
            assert response.is_fast_path is True
        finally:
            connector.close()
