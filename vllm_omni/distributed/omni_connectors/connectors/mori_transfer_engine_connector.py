# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""OmniConnector backed by the Mori RDMA transfer engine.

Implements the ``OmniConnector`` interface using Mori's ``IOEngine`` /
``MemoryDesc`` API to perform zero-copy RDMA transfers between
disaggregated prefill and decode workers.

Notable design points:
  * Remote peers must be registered via ``register_remote_engine()``
    with the peer's ``EngineDesc`` before any data transfer can proceed.
  * Memory regions are described by ``MemoryDesc`` objects (serialisable
    via pack/unpack) rather than raw virtual addresses.
  * Transfers are dispatched through ``IOEngine.batch_write()`` and
    tracked asynchronously via ``TransferStatus`` objects.

ZMQ handshake protocol:
  * **Pull request** – receiver sends ``MoriPullRequest`` (msgspec-encoded)
    containing its ``EngineDesc`` and pool ``MemoryDesc`` so the sender can
    register the remote engine and RDMA-write directly into the receiver's
    pool at the specified offset.
  * **Query request** – receiver sends ``QUERY_INFO`` prefix + ``QueryRequest``
    to check whether data is ready on the sender side (metadata-less get path).
"""

import os
import queue
import socket
import threading
import time as _time_mod
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import msgspec
import torch
import zmq

from ..utils.logging import get_connector_logger
from ..utils.serialization import OmniSerializer
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

try:
    from mori.cpp import TransferStatus  # noqa: F401
    from mori.io import (
        BackendType,
        EngineDesc,
        IOEngine,
        IOEngineConfig,
        MemoryDesc,
        PollCqMode,
        RdmaBackendConfig,
        XgmiBackendConfig,
    )
except ImportError:
    IOEngine = None

# Supported backend types for Mori. Kept as string constants so configuration
# (YAML / CLI / dict) stays transport-agnostic. ``rdma`` uses NIC-based RDMA
# (RoCE / IB, GDR-capable); ``xgmi`` uses AMD Infinity Fabric GPU-to-GPU
# direct links and therefore requires a CUDA pool.
_SUPPORTED_BACKENDS = ("rdma", "xgmi")

_BUFFER_TTL_SECONDS = 300

TRANS_DONE = b"trans_done"
TRANS_ERROR = b"trans_error"
QUERY_INFO = b"query_info"
INFO_NOT_FOUND = b"info_not_found"


# ---------------------------------------------------------------------------
# ZMQ message types
# ---------------------------------------------------------------------------


class MoriPullRequest(msgspec.Struct):
    """Receiver → sender: request an RDMA write into the receiver's pool."""

    request_id: str
    engine_desc_packed: bytes
    mem_desc_packed: bytes
    dst_offset: int
    length: int


class QueryRequest(msgspec.Struct):
    """Receiver → sender: query whether data for *request_id* is ready."""

    request_id: str


class QueryResponse(msgspec.Struct):
    """Sender → receiver: metadata about a ready buffer."""

    request_id: str
    data_size: int
    is_fast_path: bool


# ---------------------------------------------------------------------------
# Pool memory management (shared with MooncakeTransferEngineConnector)
# ---------------------------------------------------------------------------


class BufferAllocator:
    """Simple first-fit allocator over a contiguous memory pool."""

    def __init__(self, total_size: int, alignment: int = 4096):
        self.total_size = total_size
        self.alignment = alignment
        self.lock = threading.Lock()
        self.free_blocks: list[tuple[int, int]] = [(0, total_size)]

    def alloc(self, size: int) -> int:
        aligned = (size + self.alignment - 1) // self.alignment * self.alignment
        with self.lock:
            for i, (start, bsz) in enumerate(self.free_blocks):
                if bsz >= aligned:
                    remainder = bsz - aligned
                    if remainder > 0:
                        self.free_blocks[i] = (start + aligned, remainder)
                    else:
                        self.free_blocks.pop(i)
                    return start
        raise MemoryError(f"Out of memory in buffer pool. Requested {size} bytes (aligned {aligned}).")

    def free(self, offset: int, size: int) -> None:
        aligned = (size + self.alignment - 1) // self.alignment * self.alignment
        with self.lock:
            for start, length in self.free_blocks:
                if offset == start and aligned == length:
                    return
                if offset >= start and offset + aligned <= start + length:
                    return
                if not (offset + aligned <= start or start + length <= offset):
                    raise RuntimeError(
                        f"Memory corruption: freeing {offset}-{offset + aligned} "
                        f"overlaps with free block {start}-{start + length}"
                    )
            self.free_blocks.append((offset, aligned))
            self.free_blocks.sort()

            i = 0
            while i < len(self.free_blocks) - 1:
                cs, csz = self.free_blocks[i]
                ns, nsz = self.free_blocks[i + 1]
                if cs + csz == ns:
                    self.free_blocks[i] = (cs, csz + nsz)
                    self.free_blocks.pop(i + 1)
                else:
                    i += 1


class ManagedBuffer:
    """Zero-copy view into the global memory pool.

    Callers **must** call :meth:`release` (or use the context manager) after
    they are done reading/writing the buffer so the region is returned to the
    :class:`BufferAllocator`.
    """

    def __init__(
        self,
        allocator: BufferAllocator,
        offset: int,
        size: int,
        pool_tensor: torch.Tensor,
    ):
        self.allocator = allocator
        self.offset = offset
        self.size = size
        self.pool_tensor = pool_tensor
        self._released = False

    def release(self) -> None:
        if not self._released:
            self.allocator.free(self.offset, self.size)
            self._released = True

    def __del__(self) -> None:
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @property
    def tensor(self) -> torch.Tensor:
        return self.pool_tensor[self.offset : self.offset + self.size]

    def as_tensor(self, dtype: torch.dtype, shape: tuple) -> torch.Tensor:
        itemsize = torch.tensor([], dtype=dtype).element_size()
        expected = itemsize
        for d in shape:
            expected *= d
        if expected != self.size:
            raise ValueError(f"Shape {shape} dtype {dtype} needs {expected} bytes, buffer has {self.size}")
        if self.offset % itemsize != 0:
            raise RuntimeError(f"Buffer offset {self.offset} not aligned for {dtype}")
        return self.tensor.view(dtype).reshape(shape)

    def to_bytes(self) -> bytes:
        t = self.tensor
        if t.is_cuda:
            t = t.cpu()
        return t.numpy().tobytes()


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class MoriTransferEngineConnector(OmniConnectorBase):
    """OmniConnector backed by the Mori RDMA IOEngine.

    Topology: 1 sender ↔ 1 receiver per key (same as
    ``MooncakeTransferEngineConnector``).
    """

    supports_raw_data: bool = True

    # ------------------------------------------------------------------ init
    def __init__(self, config: dict[str, Any]):
        if IOEngine is None:
            raise ImportError("Mori is not available. Install via: pip install mori")

        self._closed = False
        self._bind_error: Exception | None = None

        # Teardown-safe defaults
        self._stop_event = threading.Event()
        self._sender_executor: ThreadPoolExecutor | None = None
        self._listener_thread: threading.Thread | None = None
        self._listener_ready = threading.Event()
        self._local_buffers: dict[str, Any] = {}
        self._local_buffers_lock = threading.Lock()
        self._req_local = threading.local()
        self._worker_local = threading.local()
        self._last_ttl_check: float = _time_mod.monotonic()

        # Track remote engines already registered with IOEngine
        self._registered_engines: set[str] = set()
        self._registered_engines_lock = threading.Lock()

        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "errors": 0,
            "timeouts": 0,
        }

        self.config = config

        # Stage id is read by OmniChunkTransferAdapter (process_pending_chunks /
        # _send_single_request / _poll_single_request / ...) to decide whether
        # this connector sits at the sender or receiver end of a stage pair.
        # Kept parallel to ``SharedMemoryConnector`` and other OmniConnector
        # implementations. Populated by ``build_stage_connectors`` via the
        # ``stage_id`` key that ``get_connectors_config_for_stage`` injects.
        self.stage_id = config.get("stage_id", -1)

        # ---- Host / ZMQ ----
        host_cfg = config.get("host", "127.0.0.1")
        if host_cfg.lower() == "auto":
            self.host = self._get_local_ip()
            logger.info(f"Auto-detected local IP: {self.host}")
        else:
            self.host = host_cfg
        self.zmq_port = config.get("zmq_port", 50051)

        # ---- Backend selection ----
        # Defaults to "rdma" so existing deployments (pre-XGMI support) keep
        # working unchanged. ``xgmi`` selects AMD Infinity Fabric GPU-to-GPU
        # direct links; see ``mori.io.XgmiBackendConfig``.
        backend_type_cfg = str(config.get("backend_type", "rdma")).lower()
        if backend_type_cfg not in _SUPPORTED_BACKENDS:
            raise ValueError(f"Invalid backend_type={backend_type_cfg!r}. Supported: {list(_SUPPORTED_BACKENDS)}.")
        self.backend_type = backend_type_cfg

        # ---- RDMA device (RDMA backend only) ----
        self.device_name = ""
        if self.backend_type == "rdma":
            self.device_name = config.get("device_name", "")
            if not self.device_name:
                env_dev = os.environ.get("MORI_RDMA_DEVICES", "")
                if env_dev:
                    self.device_name = env_dev
                    logger.info(f"Using MORI_RDMA_DEVICES from env: {self.device_name}")
        elif config.get("device_name"):
            logger.warning(
                "device_name=%r is ignored for backend_type='xgmi' (XGMI does not use an RDMA NIC).",
                config.get("device_name"),
            )

        # ---- Pool config ----
        self.pool_size = config.get("memory_pool_size", 1024**3)
        self.pool_device = config.get("memory_pool_device", "cpu")
        if self.backend_type == "xgmi" and self.pool_device == "cpu":
            raise ValueError(
                "backend_type='xgmi' requires memory_pool_device='cuda': "
                "XGMI is a GPU-to-GPU fabric and cannot address CPU memory."
            )

        # ---- Sender info (receiver / dual uses this when metadata=None) ----
        self.sender_host = config.get("sender_host", None)
        self.sender_zmq_port = config.get("sender_zmq_port", None)

        # ---- Role ----
        # ``sender``   : binds the ZMQ listener, accepts ``put()``.
        # ``receiver`` : no listener, only ``get()`` by querying an
        #                upstream sender whose endpoint lives in
        #                ``sender_host`` / ``sender_zmq_port``.
        # ``dual``     : single instance that simultaneously binds a
        #                listener (serves inbound pull requests from a
        #                downstream receiver) AND maintains an upstream
        #                endpoint (so its own ``get()`` can pull from an
        #                upstream sender).  Used by middle stages on the
        #                chunk_transfer_adapter path -- the adapter keeps
        #                its historical "one connector per stage" model
        #                (like ``SharedMemoryConnector``), and a dual
        #                instance is how a role-bound RDMA connector
        #                exposes put+get from the same object.  The role
        #                is chosen by the framework (see
        #                ``get_connectors_config_for_stage``) based on
        #                whether the stage has incoming / outgoing edges;
        #                ``MoriTransferEngineConnector`` itself does not
        #                reverse-infer its deployment mode.
        role = str(config.get("role", "sender")).lower()
        if role not in {"sender", "receiver", "dual"}:
            raise ValueError(
                f"Invalid role={role!r} for MoriTransferEngineConnector. "
                "Expected 'sender', 'receiver', or 'dual'."
            )
        self.role = role
        self.can_put = role in ("sender", "dual")

        # ---- Mori IOEngine ----
        if self.device_name:
            os.environ["MORI_RDMA_DEVICES"] = self.device_name

        engine_config = IOEngineConfig(host=self.host, port=0)
        self.engine_key = f"omni-{role}-{uuid.uuid4().hex[:8]}-pid{os.getpid()}-{self.host}"
        self.engine = IOEngine(self.engine_key, engine_config)

        # ---- Backend creation (per backend_type) ----
        # ``IOEngine.batch_write`` is backend-agnostic, so only construction
        # differs; the data-plane / ZMQ handshake paths below are identical.
        if self.backend_type == "xgmi":
            xgmi_cfg = XgmiBackendConfig()
            xgmi_cfg.num_streams = config.get("xgmi_num_streams", 64)
            xgmi_cfg.num_events = config.get("xgmi_num_events", 64)
            self.engine.create_backend(BackendType.XGMI, xgmi_cfg)
            logger.info(f"Mori backend: XGMI (num_streams={xgmi_cfg.num_streams}, num_events={xgmi_cfg.num_events})")
        else:  # rdma
            qp_per_transfer = config.get("qp_per_transfer", 1)
            post_batch_size = config.get("post_batch_size", -1)
            num_workers = config.get("num_worker_threads", 1)
            rdma_cfg = RdmaBackendConfig(
                qp_per_transfer,
                post_batch_size,
                num_workers,
                PollCqMode.POLLING,
                False,
            )
            self.engine.create_backend(BackendType.RDMA, rdma_cfg)
            logger.info(
                f"Mori backend: RDMA (qp_per_transfer={qp_per_transfer}, "
                f"num_workers={num_workers}, device={self.device_name or 'auto'})"
            )

        self.engine_desc: EngineDesc = self.engine.get_engine_desc()
        self.engine_desc_packed: bytes = self.engine_desc.pack()
        logger.info(f"Mori IOEngine ready: key={self.engine_key} at {self.engine_desc.host}:{self.engine_desc.port}")

        # ---- Pool allocation & Mori memory registration ----
        logger.info(
            f"Allocating {self.backend_type.upper()} pool: {self.pool_size / 1024**2:.2f} MB on {self.pool_device}"
        )
        try:
            if self.pool_device == "cpu":
                self.pool = torch.empty(self.pool_size, dtype=torch.uint8).pin_memory()
            else:
                self.pool = torch.empty(
                    self.pool_size,
                    dtype=torch.uint8,
                    device=self.pool_device,
                )
            self.pool_mem_desc: MemoryDesc = self.engine.register_torch_tensor(self.pool)
            self.pool_mem_desc_packed: bytes = self.pool_mem_desc.pack()
            self.base_ptr = self.pool.data_ptr()
        except Exception as e:
            logger.error(f"Failed to allocate/register pool: {e}")
            raise

        self.allocator = BufferAllocator(self.pool_size, alignment=4096)

        # ---- ZMQ / threading ----
        self.zmq_ctx = zmq.Context()
        self._sender_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mori-sender")

        logger.info(
            f"MoriTransferEngineConnector config:\n"
            f"  Local: host={self.host}, zmq_port={self.zmq_port}\n"
            f"  Remote: sender_host={self.sender_host}, "
            f"sender_zmq_port={self.sender_zmq_port}\n"
            f"  Role: can_put={self.can_put}, "
            f"configured_role={config.get('role', 'sender')}"
        )

        if self.can_put:
            self._last_ttl_check = _time_mod.monotonic()
            self._listener_thread = threading.Thread(target=self._zmq_listener_loop, daemon=True)
            self._listener_thread.start()
            self._listener_ready.wait(timeout=1.0)
            if self._bind_error is not None:
                raise RuntimeError(
                    f"MoriTransferEngineConnector failed to bind ZMQ on {self.host}:{self.zmq_port}: {self._bind_error}"
                ) from self._bind_error
            if self.role == "dual":
                logger.info(
                    f"MoriTransferEngineConnector DUAL ready "
                    f"(ZMQ listening on {self.host}:{self.zmq_port}, "
                    f"upstream sender at {self.sender_host}:{self.sender_zmq_port})"
                )
            else:
                logger.info(f"MoriTransferEngineConnector SENDER ready (ZMQ on {self.host}:{self.zmq_port})")
        else:
            if not self.sender_host or self.sender_host.lower() == "auto":
                logger.info("MoriTransferEngineConnector RECEIVER: awaiting sender info via update_sender_info().")
            else:
                logger.info(
                    f"MoriTransferEngineConnector RECEIVER ready (sender at {self.sender_host}:{self.sender_zmq_port})"
                )

    # -------------------------------------------------------- public helpers
    def get_connection_info(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "zmq_port": self.zmq_port,
            "engine_key": self.engine_key,
            "can_put": self.can_put,
        }

    def update_sender_info(self, sender_host: str, sender_zmq_port: int) -> None:
        """Inject the sender's ZMQ endpoint into the receiver connector."""
        self.sender_host = sender_host
        self.sender_zmq_port = sender_zmq_port
        logger.info(f"Sender info updated: host={sender_host!r}, zmq_port={sender_zmq_port}")

    # -------------------------------------------------- internal helpers
    @staticmethod
    def _get_local_ip() -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            try:
                return socket.gethostbyname(socket.gethostname())
            except Exception:
                return "127.0.0.1"

    def _get_req_socket(self, zmq_addr: str, timeout_ms: int) -> zmq.Socket:
        cache: dict[str, zmq.Socket] | None = getattr(self._req_local, "cache", None)
        if cache is None:
            cache = {}
            self._req_local.cache = cache

        sock = cache.get(zmq_addr)
        if sock is None:
            sock = self.zmq_ctx.socket(zmq.REQ)
            sock.connect(zmq_addr)
            cache[zmq_addr] = sock
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        return sock

    def _invalidate_req_socket(self, zmq_addr: str) -> None:
        cache: dict[str, zmq.Socket] | None = getattr(self._req_local, "cache", None)
        if cache is None:
            return
        sock = cache.pop(zmq_addr, None)
        if sock is not None:
            try:
                sock.close(linger=0)
            except Exception:
                pass

    def _ensure_remote_registered(self, engine_desc_packed: bytes) -> str:
        """Register a remote engine with the local IOEngine (idempotent)."""
        desc = EngineDesc.unpack(engine_desc_packed)
        key = desc.key
        with self._registered_engines_lock:
            if key not in self._registered_engines:
                self.engine.register_remote_engine(desc)
                self._registered_engines.add(key)
                logger.debug(f"Registered remote Mori engine: {key}")
        return key

    # ----------------------------------------------------------------- put()
    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        if self._closed:
            raise RuntimeError("Cannot put: MoriTransferEngineConnector is closed")
        if not self.can_put:
            logger.warning(f"Rejecting put for {put_key}: connector is receiver-only")
            return False, 0, None

        put_key = self._make_key(put_key, from_stage, to_stage)
        try:
            src_offset = 0
            size = 0
            holder = None
            is_fast_path = True
            should_release = False

            # Reject empty payloads early
            if isinstance(data, bytes) and len(data) == 0:
                return False, 0, None
            if isinstance(data, torch.Tensor) and data.nbytes == 0:
                return False, 0, None

            # Serialize non-raw types
            if not isinstance(data, (ManagedBuffer, torch.Tensor, bytes)):
                data = OmniSerializer.serialize(data)
                is_fast_path = False

            if isinstance(data, ManagedBuffer):
                if data.pool_tensor.data_ptr() != self.pool.data_ptr():
                    data = data.tensor.contiguous()
                else:
                    src_offset = data.offset
                    size = data.size
                    holder = data
                    should_release = False

            if isinstance(data, (torch.Tensor, bytes)):
                if isinstance(data, torch.Tensor):
                    size = data.nbytes
                    tensor_data = data
                else:
                    size = len(data)
                    tensor_data = torch.frombuffer(data, dtype=torch.uint8)

                try:
                    offset = self.allocator.alloc(size)
                    holder = ManagedBuffer(self.allocator, offset, size, self.pool)
                    should_release = True
                except MemoryError:
                    logger.error(f"Pool exhausted for {size} bytes")
                    return False, 0, None

                try:
                    dst_t = holder.tensor
                    if isinstance(data, torch.Tensor):
                        if not data.is_contiguous():
                            data = data.contiguous()
                        src_view = data.view(torch.uint8).flatten()
                        if src_view.device != dst_t.device:
                            dst_t.copy_(src_view, non_blocking=True)
                            if src_view.is_cuda:
                                with torch.cuda.device(src_view.device):
                                    torch.cuda.current_stream().synchronize()
                            elif dst_t.is_cuda:
                                with torch.cuda.device(dst_t.device):
                                    torch.cuda.current_stream().synchronize()
                        else:
                            dst_t.copy_(src_view)
                            if dst_t.is_cuda:
                                with torch.cuda.device(dst_t.device):
                                    torch.cuda.current_stream().synchronize()
                    else:
                        dst_t.copy_(tensor_data)
                        if dst_t.is_cuda:
                            with torch.cuda.device(dst_t.device):
                                torch.cuda.current_stream().synchronize()
                except Exception as e:
                    holder.release()
                    logger.error(f"Failed to copy data to pool: {e}")
                    return False, 0, None

                src_offset = offset

            if size <= 0:
                if should_release and isinstance(holder, ManagedBuffer):
                    holder.release()
                return False, 0, None

            with self._local_buffers_lock:
                old = self._local_buffers.pop(put_key, None)
                if old:
                    _, _, oh, osr, _, _ = old
                    if osr and isinstance(oh, ManagedBuffer):
                        oh.release()
                        logger.warning(f"Released stale buffer for duplicate key: {put_key}")
                # (offset, size, holder, should_release, is_fast_path, ts)
                self._local_buffers[put_key] = (
                    src_offset,
                    size,
                    holder,
                    should_release,
                    is_fast_path,
                    _time_mod.monotonic(),
                )

            metadata = {
                "source_host": self.host,
                "source_port": self.zmq_port,
                "data_size": size,
                "is_fast_path": is_fast_path,
            }
            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size
            return True, size, metadata

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(f"Put failed for {put_key}: {e}", exc_info=True)
            return False, 0, None

    # ------------------------------------------------ query (no-metadata get)
    def _query_metadata_from_sender(self, get_key: str) -> dict[str, Any] | None:
        zmq_addr = f"tcp://{self.sender_host}:{self.sender_zmq_port}"
        req_socket = self._get_req_socket(zmq_addr, timeout_ms=5000)
        try:
            q = QueryRequest(request_id=get_key)
            req_socket.send(QUERY_INFO + msgspec.msgpack.encode(q))
            resp = req_socket.recv()
            if resp == INFO_NOT_FOUND:
                return None
            qr = msgspec.msgpack.decode(resp, type=QueryResponse)
            return {
                "source_host": self.sender_host,
                "source_port": self.sender_zmq_port,
                "data_size": qr.data_size,
                "is_fast_path": qr.is_fast_path,
            }
        except Exception as e:
            self._invalidate_req_socket(zmq_addr)
            logger.debug(f"Metadata query failed for {get_key}: {e}")
            return None

    # ----------------------------------------------------------------- get()
    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        if self._closed:
            raise RuntimeError("Cannot get: MoriTransferEngineConnector is closed")

        get_key = self._make_key(get_key, from_stage, to_stage)
        _t0 = _time_mod.perf_counter()

        # Resolve metadata
        if not metadata:
            if not self.sender_host or not self.sender_zmq_port or str(self.sender_host).lower() == "auto":
                raise RuntimeError("get(metadata=None) requires sender info. Call update_sender_info() first.")
            metadata = self._query_metadata_from_sender(get_key)
            if not metadata:
                return None

        _t1 = _time_mod.perf_counter()
        _query_ms = (_t1 - _t0) * 1000

        src_host = metadata.get("source_host")
        src_port = metadata.get("source_port")
        data_size = metadata.get("data_size", 0)
        is_fast_path = metadata.get("is_fast_path", False)

        if not src_host or not src_port or str(src_host).lower() == "auto":
            logger.error(f"Invalid metadata for {get_key}")
            return None
        if data_size == 0:
            logger.warning(f"Skipping get for {get_key}: data_size is 0")
            return None

        # Allocate destination buffer
        try:
            offset = self.allocator.alloc(data_size)
            recv_buf = ManagedBuffer(self.allocator, offset, data_size, self.pool)
        except MemoryError:
            logger.error(f"Failed to allocate {data_size} bytes for get")
            return None

        _t2 = _time_mod.perf_counter()
        _alloc_ms = (_t2 - _t1) * 1000

        # Build pull request with Mori-specific EngineDesc + MemoryDesc
        pull_req = MoriPullRequest(
            request_id=get_key,
            engine_desc_packed=self.engine_desc_packed,
            mem_desc_packed=self.pool_mem_desc_packed,
            dst_offset=offset,
            length=data_size,
        )

        _base_timeout_ms = 30000
        _size_timeout_ms = max(0, (data_size // (100 * 1024 * 1024))) * 5000
        _total_timeout_ms = _base_timeout_ms + _size_timeout_ms
        zmq_addr = f"tcp://{src_host}:{src_port}"
        req_socket = self._get_req_socket(zmq_addr, timeout_ms=_total_timeout_ms)

        try:
            req_socket.send(msgspec.msgpack.encode(pull_req))
            resp = req_socket.recv()

            _t3 = _time_mod.perf_counter()
            _rdma_ms = (_t3 - _t2) * 1000

            if resp == TRANS_DONE:
                if self.pool.is_cuda:
                    with torch.cuda.device(self.pool.device):
                        torch.cuda.current_stream().synchronize()

                _t4 = _time_mod.perf_counter()
                _sync_ms = (_t4 - _t3) * 1000

                if is_fast_path:
                    _total_ms = (_time_mod.perf_counter() - _t0) * 1000
                    _mbps = (data_size / 1024 / 1024) / (_total_ms / 1000) if _total_ms > 0 else 0
                    logger.info(
                        f"[MORI GET] {get_key}: query={_query_ms:.1f}ms, "
                        f"alloc={_alloc_ms:.1f}ms, rdma={_rdma_ms:.1f}ms, "
                        f"sync={_sync_ms:.1f}ms, total={_total_ms:.1f}ms, "
                        f"{_mbps:.1f} MB/s (fast_path)"
                    )
                    self._metrics["gets"] += 1
                    self._metrics["bytes_transferred"] += data_size
                    return recv_buf, data_size
                else:
                    try:
                        raw = recv_buf.to_bytes()
                        val = OmniSerializer.deserialize(raw)
                        _total_ms = (_time_mod.perf_counter() - _t0) * 1000
                        _mbps = (data_size / 1024 / 1024) / (_total_ms / 1000) if _total_ms > 0 else 0
                        logger.info(f"[MORI GET] {get_key}: total={_total_ms:.1f}ms, {_mbps:.1f} MB/s (deserialized)")
                        self._metrics["gets"] += 1
                        self._metrics["bytes_transferred"] += data_size
                        return val, data_size
                    finally:
                        recv_buf.release()
            else:
                self._metrics["errors"] += 1
                logger.error(f"MORI get failed: received {resp} instead of TRANS_DONE")
                recv_buf.release()
                return None
        except Exception as e:
            self._invalidate_req_socket(zmq_addr)
            self._metrics["timeouts"] += 1
            logger.error(f"MORI get error: {e}", exc_info=True)
            recv_buf.release()
            return None

    # ------------------------------------------------------------- cleanup()
    def cleanup(
        self,
        request_id: str,
        from_stage: str | None = None,
        to_stage: str | None = None,
    ) -> None:
        if (from_stage is None) != (to_stage is None):
            raise ValueError("cleanup() requires both from_stage and to_stage, or neither.")
        if from_stage is not None and to_stage is not None:
            request_id = self._make_key(request_id, from_stage, to_stage)
        with self._local_buffers_lock:
            item = self._local_buffers.pop(request_id, None)
            if item:
                _, _, holder, sr, _, _ = item
                if sr and isinstance(holder, ManagedBuffer):
                    holder.release()

    # -------------------------------------------------------------- health()
    def health(self) -> dict[str, Any]:
        if self._closed:
            return {"status": "unhealthy", "error": "Connector is closed"}
        return {
            "status": "healthy",
            "host": self.host,
            "pool_device": self.pool_device,
            "pool_size": self.pool_size,
            "engine_key": self.engine_key,
            **self._metrics,
        }

    # --------------------------------------------------------------- close()
    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        logger.info("Closing MoriTransferEngineConnector...")

        self._stop_event.set()

        if self._listener_thread is not None and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)
            if self._listener_thread.is_alive():
                logger.warning("Listener thread did not stop gracefully")

        if self._sender_executor is not None:
            self._sender_executor.shutdown(wait=True, cancel_futures=False)

        with self._local_buffers_lock:
            for _k, item in list(self._local_buffers.items()):
                _, _, holder, sr, _, _ = item
                if sr and isinstance(holder, ManagedBuffer):
                    holder.release()
            self._local_buffers.clear()

        cache: dict[str, zmq.Socket] | None = getattr(self._req_local, "cache", None)
        if cache:
            for _addr, sock in cache.items():
                try:
                    sock.close(linger=0)
                except Exception:
                    pass
            cache.clear()

        try:
            if hasattr(self, "zmq_ctx"):
                self.zmq_ctx.term()
        except Exception as e:
            logger.warning(f"Failed to terminate ZMQ context: {e}")

        self.pool = None  # type: ignore[assignment]
        logger.info("MoriTransferEngineConnector closed.")

    # ============================================================== LISTENER
    def _cleanup_stale_buffers(self) -> None:
        now = _time_mod.monotonic()
        with self._local_buffers_lock:
            stale = [k for k, v in self._local_buffers.items() if now - v[5] > _BUFFER_TTL_SECONDS]
            for k in stale:
                item = self._local_buffers.pop(k)
                _, _, holder, sr, _, _ = item
                if sr and isinstance(holder, ManagedBuffer):
                    holder.release()
                logger.warning(f"TTL expired ({_BUFFER_TTL_SECONDS}s): reclaimed buffer for {k}")

    def _zmq_listener_loop(self) -> None:
        sock = self.zmq_ctx.socket(zmq.ROUTER)
        bound = False

        # Try the configured port first
        try:
            sock.bind(f"tcp://{self.host}:{self.zmq_port}")
            bound = True
        except zmq.ZMQError as exc:
            if exc.errno in (zmq.EADDRINUSE, 98):
                logger.warning(f"ZMQ port {self.zmq_port} already in use, falling back to OS-assigned port")
            else:
                logger.error(f"ZMQ bind failed on {self.host}:{self.zmq_port}: {exc}")
                self.can_put = False
                self._bind_error = exc
                self._listener_ready.set()
                return

        # Fallback: let the OS pick a free port
        if not bound:
            try:
                sock.bind(f"tcp://{self.host}:*")
                bound = True
            except zmq.ZMQError as exc:
                logger.error(f"ZMQ bind failed on {self.host}:*: {exc}")
                self.can_put = False
                self._bind_error = exc
                self._listener_ready.set()
                return

        # Read back the actual port (works for both configured and
        # OS-assigned ports) and update self.zmq_port so that put()
        # metadata and get_connection_info() return the correct value.
        endpoint = sock.getsockopt(zmq.LAST_ENDPOINT).decode()
        actual_port = int(endpoint.rsplit(":", 1)[1])
        if actual_port != self.zmq_port:
            logger.info(f"ZMQ listener bound on {self.host}:{actual_port} (configured: {self.zmq_port})")
        self.zmq_port = actual_port

        self._listener_ready.set()

        notify_addr = f"inproc://mori-notify-{id(self)}"
        notify_recv = self.zmq_ctx.socket(zmq.PULL)
        notify_recv.bind(notify_addr)

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        poller.register(notify_recv, zmq.POLLIN)

        response_queue: queue.Queue = queue.Queue()

        try:
            while not self._stop_event.is_set():
                try:
                    events = dict(poller.poll(1000))

                    if notify_recv in events:
                        while True:
                            try:
                                notify_recv.recv(zmq.NOBLOCK)
                            except zmq.Again:
                                break

                    while True:
                        try:
                            identity, response = response_queue.get_nowait()
                            sock.send_multipart([identity, b"", response])
                        except queue.Empty:
                            break

                    now = _time_mod.monotonic()
                    if now - self._last_ttl_check >= 10.0:
                        self._last_ttl_check = now
                        self._cleanup_stale_buffers()

                    if sock in events:
                        frames = sock.recv_multipart()
                        if len(frames) >= 2:
                            self._sender_executor.submit(
                                self._handle_pull_request,
                                response_queue,
                                notify_addr,
                                frames[0],
                                frames[-1],
                            )
                except zmq.ContextTerminated:
                    break
                except Exception:
                    logger.debug("Listener loop error", exc_info=True)
        finally:
            try:
                notify_recv.close(linger=0)
                sock.close(linger=0)
            except Exception:
                pass

    def _handle_pull_request(
        self,
        response_queue: queue.Queue,
        notify_addr: str,
        identity: bytes,
        payload: bytes,
    ) -> None:
        try:
            if payload.startswith(QUERY_INFO):
                self._handle_query_request(
                    response_queue,
                    notify_addr,
                    identity,
                    payload[len(QUERY_INFO) :],
                )
                return

            pull = msgspec.msgpack.decode(payload, type=MoriPullRequest)

            with self._local_buffers_lock:
                item = self._local_buffers.get(pull.request_id)

            if not item:
                response_queue.put((identity, TRANS_ERROR))
                self._notify_listener(notify_addr)
                return

            src_offset, src_size, _, _, _, _ = item

            # Register the receiver's IOEngine (idempotent)
            self._ensure_remote_registered(pull.engine_desc_packed)

            # Reconstruct the receiver's pool MemoryDesc
            remote_mem = MemoryDesc.unpack(pull.mem_desc_packed)

            # RDMA write from local pool → remote pool
            transfer_uid = self.engine.allocate_transfer_uid()
            statuses = self.engine.batch_write(
                [self.pool_mem_desc],
                [[src_offset]],
                [remote_mem],
                [[pull.dst_offset]],
                [[pull.length]],
                [transfer_uid],
            )

            success = True
            for st in statuses:
                st.Wait()
                if st.Failed():
                    logger.error(f"RDMA write failed for {pull.request_id}: {st.Message()}")
                    success = False

            if success:
                self.cleanup(pull.request_id)
                response_queue.put((identity, TRANS_DONE))
            else:
                logger.warning(f"RDMA write failed for {pull.request_id}. Buffer retained for retry.")
                response_queue.put((identity, TRANS_ERROR))

        except Exception as e:
            logger.error(f"Pull request handler error: {e}", exc_info=True)
            response_queue.put((identity, TRANS_ERROR))

        self._notify_listener(notify_addr)

    def _handle_query_request(
        self,
        response_queue: queue.Queue,
        notify_addr: str,
        identity: bytes,
        payload: bytes,
    ) -> None:
        try:
            q = msgspec.msgpack.decode(payload, type=QueryRequest)
            with self._local_buffers_lock:
                item = self._local_buffers.get(q.request_id)
            if not item:
                response_queue.put((identity, INFO_NOT_FOUND))
            else:
                _, data_size, _, _, is_fast, _ = item
                resp = QueryResponse(
                    request_id=q.request_id,
                    data_size=data_size,
                    is_fast_path=is_fast,
                )
                response_queue.put((identity, msgspec.msgpack.encode(resp)))
        except Exception as e:
            logger.error(f"Query handler error: {e}")
            response_queue.put((identity, INFO_NOT_FOUND))

        self._notify_listener(notify_addr)

    def _notify_listener(self, notify_addr: str) -> None:
        try:
            local = self._worker_local
            sock = getattr(local, "notify_socket", None)
            cached_addr = getattr(local, "notify_addr", None)
            if sock is None or cached_addr != notify_addr:
                if sock is not None:
                    sock.close(linger=0)
                sock = self.zmq_ctx.socket(zmq.PUSH)
                sock.connect(notify_addr)
                local.notify_socket = sock
                local.notify_addr = notify_addr
            sock.send(b"", zmq.NOBLOCK)
        except Exception:
            local.notify_socket = None
            local.notify_addr = None
