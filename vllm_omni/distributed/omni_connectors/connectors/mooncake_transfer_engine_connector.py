# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import queue
import socket
import threading
import time as _time_mod
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import msgspec
import torch
import zmq

from ..utils.logging import get_connector_logger
from ..utils.memory_pool import BufferAllocator, ManagedBuffer
from ..utils.serialization import OmniSerializer
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

try:
    from mooncake.engine import TransferEngine
except ImportError:
    TransferEngine = None

# Stale buffer TTL: buffers older than this are automatically reclaimed
# to prevent memory leaks when receiver crashes or gives up.
_BUFFER_TTL_SECONDS = 300  # 5 minutes
_BASE_TRANSFER_TIMEOUT_MS = 30_000
_TRANSFER_TIMEOUT_STEP_BYTES = 100 * 1024 * 1024
_TRANSFER_TIMEOUT_PER_STEP_MS = 5_000
_CAPACITY_WARNING_INTERVAL_SECONDS = 1.0

# ZMQ Message constants
TRANS_DONE = b"trans_done"
# Legacy errors carry no lifetime proof. Never release a published destination
# on TRANS_ERROR, including when an older peer sends it.
TRANS_ERROR = b"trans_error"
TRANS_NOT_SUBMITTED = b"trans_not_submitted"
TRANS_UNKNOWN = b"trans_unknown"
_PROTOCOL_VERSION = 2

# Unknown native operations must retain the engine and registered memory pool.
_RETAINED_CONNECTORS: set["MooncakeTransferEngineConnector"] = set()


class _ReceivePendingError(TimeoutError):
    """Destination ownership has moved to the connector's pending set."""


QUERY_INFO = b"query_info"
INFO_NOT_FOUND = b"info_not_found"


@dataclass
class QueryRequest:
    """Request to query metadata for a specific key."""

    request_id: str


@dataclass
class QueryResponse:
    """Response containing metadata for a request."""

    request_id: str
    data_size: int
    is_fast_path: bool
    protocol_version: int = 0
    native_tcp_proto_version: int = 0


@dataclass
class MooncakeAgentMetadata:
    """
    Metadata exchanged via ZMQ Handshake.
    """

    remote_hostname: str
    remote_port: int  # RDMA Port
    request_id: str
    dst_addrs: list[int]
    lengths: list[int]
    protocol_version: int = 0
    native_tcp_proto_version: int = 0


class MooncakeTransferEngineConnector(OmniConnectorBase):
    """
    OmniConnector implementation using Mooncake Transfer Engine with a managed memory pool.
    Supports both CPU (Pinned) and GPU memory pools, and both RDMA and TCP protocols.
    Topology limitations (current implementation):
        Current design focuses on peer-to-peer communication between stages.
        - **1 sender → 1 receiver per key**: After a successful RDMA write the
          sender immediately cleans up the buffer (``cleanup()``), so only the
          first receiver to pull a given key will succeed.  Broadcast / multicast
          (1 sender → N receivers sharing the same data) is not yet supported.
        - **1 receiver → N senders**: Supported via partial metadata.  The
          manager constructs metadata with the target sender's
          ``source_host`` / ``source_port`` (computed from ``from_rank``)
          and passes it to ``get(metadata=...)``.  The connector detects
          that ``data_size`` is missing, queries the specified sender at
          the given address to fill it in, then performs the RDMA pull.
          This enables heterogeneous TP (sender TP > receiver TP) where a
          single receiver must pull KV shards from multiple sender ranks.

    Future work:
        - Support 1 sender → N receivers (e.g. reference-counted buffers, or
          explicit ``retain()`` / ``release()`` semantics so the buffer survives
          multiple pulls).
    """

    # RDMA connector copies raw bytes/tensor directly to the memory pool
    # without going through OmniSerializer, so the sender can use
    # to_gpu_tensor() / to_bytes() fast-paths.
    supports_raw_data: bool = True

    def __init__(self, config: dict[str, Any]):
        if TransferEngine is None:
            raise ImportError("Mooncake not available")

        self._state_lock = threading.RLock()
        self._closed = False
        self._closing = False
        self._close_scheduled = False
        self._active_puts = 0
        self._active_handlers = 0
        self._receiving_keys: set[str] = set()
        self._pending_receives: dict[str, tuple[ManagedBuffer, zmq.Socket | None]] = {}
        self._bind_error: Exception | None = None  # fatal ZMQ bind error from listener thread

        # --- Early init of all teardown-related fields ---
        # If __init__ fails later (engine init, pool alloc, etc.), __del__ → close()
        # must not crash on missing attributes.  Safe defaults here ensure close()
        # can always run without AttributeError.
        self._stop_event = threading.Event()
        self._sender_executor: ThreadPoolExecutor | None = None
        self._listener_thread: threading.Thread | None = None
        self._listener_ready = threading.Event()
        self._local_buffers: dict[str, Any] = {}
        self._local_buffers_lock = self._state_lock
        self._put_lock = threading.Lock()
        self._req_local = threading.local()
        self._worker_local = threading.local()
        self._last_ttl_check: float = _time_mod.monotonic()
        self._sender_endpoints: dict[int, tuple[str, int]] = {}
        self._in_flight_buffers: set[str] = set()
        self._last_capacity_warning = 0.0

        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "errors": 0,
            "timeouts": 0,
            "receive_capacity_rejections": 0,
        }

        self.config = config
        self.stage_id = config.get("stage_id", -1)
        self.max_inflight_receives = int(config.get("max_inflight_receives", 64))
        self.max_inflight_sends = int(config.get("max_inflight_sends", 64))
        if self.max_inflight_receives < 1 or self.max_inflight_sends < 1:
            raise ValueError("Mooncake in-flight limits must be positive")
        self._sender_slots = threading.BoundedSemaphore(self.max_inflight_sends)
        host_config = config.get("host")
        host_value = "auto" if host_config is None else str(host_config)
        # Default sender/receiver bootstrap to a routable local IP so the
        # advertised endpoint matches the interface Mooncake binds.
        if host_value.lower() == "auto" or host_value in {"", "*", "0.0.0.0", "::"}:
            self.host = self._get_local_ip()
            logger.info(f"Auto-detected local IP for RDMA: {self.host}")
        else:
            self.host = host_value
        self.zmq_port = config.get("zmq_port", 50051)
        self.protocol = config.get("protocol", "rdma")
        if os.environ.get("MC_TCP_PROTO") == "1":
            raise ValueError("Mooncake protocol v2 requires acknowledged TCP completion; unset MC_TCP_PROTO=1")
        try:
            self.native_tcp_proto_version = int(config.get("native_tcp_proto_version", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("native_tcp_proto_version must be an integer") from exc
        if self.protocol.lower() == "tcp" and self.native_tcp_proto_version < 2:
            raise ValueError(
                "Mooncake TCP requires native_tcp_proto_version=2; "
                "the installed SDK capability must be verified explicitly"
            )

        # --- RDMA Device Configuration ---
        # Specify device names to filter (comma-separated), or empty for all devices.
        # Example: "mlx5_0,mlx5_1" to use only specific NICs.
        # This is important for environments with mixed InfiniBand/RoCE NICs.
        self.device_name = config.get("device_name", "")
        if not self.device_name:
            env_device = os.environ.get("RDMA_DEVICE_NAME", "")
            if env_device:
                self.device_name = env_device
                logger.info(f"Using RDMA_DEVICE_NAME from env: {self.device_name}")

        # --- Memory Pool Configuration ---
        self.pool_size = config.get("memory_pool_size", 1024**3)  # Default 1GB
        self.pool_device = config.get("memory_pool_device", "cpu")
        self.buffer_ttl_seconds = float(config.get("buffer_ttl_seconds", _BUFFER_TTL_SECONDS))

        # --- Sender Configuration (for receiver to query without metadata) ---
        # When receiver doesn't have metadata, it uses these to connect to sender
        self.sender_host = config.get("sender_host", None)
        self.sender_zmq_port = config.get("sender_zmq_port", None)

        # --- Role ---
        # "sender": bind ZMQ listener, accept put() calls.
        # "receiver": skip ZMQ bind, only accept get() calls.
        # The orchestration layer (get_connectors_config_for_stage /
        # kv_transfer_manager) is responsible for injecting the correct role.
        role = str(config.get("role", "sender")).lower()
        if role not in {"sender", "receiver"}:
            raise ValueError(
                f"Invalid role={role!r} for MooncakeTransferEngineConnector. Expected 'sender' or 'receiver'."
            )
        self.can_put = role == "sender"

        self.engine_id = str(uuid.uuid4())

        self._init_engine()
        self._init_pool()
        self._init_listener(config)

    def _requires_tcp_completion(self) -> bool:
        """Return whether the effective native path may use TCP."""
        return self.protocol.lower() == "tcp" or (
            self.protocol.lower() == "rdma" and "MC_FORCE_TCP" in os.environ
        )

    def _validate_rdma_transport(self) -> None:
        """Reject RDMA configurations that would silently fall back to TCP."""
        if "MC_FORCE_TCP" in os.environ:
            raise ValueError(
                "Mooncake protocol 'rdma' cannot be used with MC_FORCE_TCP; "
                "unset MC_FORCE_TCP or configure protocol='tcp' with "
                "native_tcp_proto_version=2"
            )
        # MC_FORCE_HCA makes the native engine fail closed if no HCA is
        # available, so no topology probe is needed in that mode.
        if "MC_FORCE_HCA" in os.environ:
            return
        get_topology = getattr(self.engine, "get_local_topology", None)
        if get_topology is None:
            raise RuntimeError(
                "Mooncake RDMA requires get_local_topology() or MC_FORCE_HCA=1 "
                "to prevent an implicit TCP fallback"
            )
        try:
            topology = json.loads(get_topology(self.device_name or None))
        except Exception as exc:
            raise RuntimeError(
                "Unable to verify an HCA for Mooncake protocol 'rdma'; "
                "set MC_FORCE_HCA=1 to make native initialization fail closed"
            ) from exc

        hcas: set[str] = set()

        def collect(value: Any) -> None:
            if isinstance(value, str):
                hcas.add(value)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        if isinstance(topology, dict):
            for value in topology.values():
                collect(value)
        requested = {
            item.strip() for item in str(self.device_name).split(",") if item.strip()
        }
        if not hcas or (requested and not requested.issubset(hcas)):
            selected = ", ".join(sorted(requested)) or "any HCA"
            raise RuntimeError(
                f"Mooncake protocol 'rdma' requires an available HCA ({selected}); "
                "refusing an implicit TCP fallback"
            )

    def _init_engine(self) -> None:
        """Initialize the Mooncake engine and discover the local RPC port."""
        self.engine = TransferEngine()
        if self.protocol.lower() == "rdma":
            self._validate_rdma_transport()
        # Note: For P2P handshake mode, local_hostname should be just the IP address.
        # Mooncake will auto-assign an RPC port, retrievable via get_rpc_port().
        ret = self.engine.initialize(self.host, "P2PHANDSHAKE", self.protocol, self.device_name)
        if ret != 0:
            raise RuntimeError(f"Mooncake Engine initialization failed with code {ret}")

        self.rpc_port = self.engine.get_rpc_port()
        logger.info(f"MooncakeTransferEngineConnector initialized at {self.host}:{self.rpc_port}")

    def _init_pool(self) -> None:
        """Allocate and register the memory pool used by TransferEngine."""
        logger.info(f"Allocating RDMA Memory Pool: {self.pool_size / 1024**2:.2f} MB on {self.pool_device}")
        try:
            if self.pool_device == "cpu":
                self.pool = torch.empty(self.pool_size, dtype=torch.uint8).pin_memory()
                self.base_ptr = self.pool.data_ptr()
            else:
                self.pool = torch.empty(self.pool_size, dtype=torch.uint8, device=self.pool_device)
                self.base_ptr = self.pool.data_ptr()

            # Register the entire pool
            ret = self.engine.register_memory(self.base_ptr, self.pool_size)
            if ret != 0:
                raise RuntimeError("Failed to register memory pool with Mooncake Engine")

        except Exception as e:
            logger.error(f"Failed to allocate/register memory pool: {e}")
            raise

        self.allocator = BufferAllocator(self.pool_size, alignment=4096)  # 4KB alignment for safety

    def _init_listener(self, config: dict[str, Any]) -> None:
        """Initialize control-plane sockets and start the sender listener."""
        self.zmq_ctx = zmq.Context()
        self._sender_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mooncake-sender")

        # Log complete connector configuration for debugging
        logger.info(
            f"MooncakeTransferEngineConnector config summary:\n"
            f"  Local: host={self.host}, zmq_port={self.zmq_port}, rpc_port={self.rpc_port}\n"
            f"  Remote: sender_host={self.sender_host}, sender_zmq_port={self.sender_zmq_port}\n"
            f"  Role: can_put={self.can_put}, configured_role={config.get('role', 'sender')}"
        )

        # Only sender needs ZMQ listener to handle pull requests
        if self.can_put:
            self._last_ttl_check = _time_mod.monotonic()  # reset after slow init
            self._listener_thread = threading.Thread(target=self._zmq_listener_loop, daemon=True)
            self._listener_thread.start()
            # Wait briefly for listener to bind (or fail)
            self._listener_ready.wait(timeout=1.0)
            # Propagate any bind error to the caller — sender must bind successfully.
            if self._bind_error is not None:
                raise RuntimeError(
                    f"MooncakeTransferEngineConnector failed to bind ZMQ on "
                    f"{self.host}:{self.zmq_port}: {self._bind_error}"
                ) from self._bind_error
            logger.info(
                f"MooncakeTransferEngineConnector started as SENDER (ZMQ listener on {self.host}:{self.zmq_port})"
            )
        else:
            # Receiver mode — sender address is provided per-request via
            # metadata from put() through the queue, not pre-configured.
            if not self.sender_host or self.sender_host.lower() == "auto":
                logger.info(
                    "MooncakeTransferEngineConnector receiver: sender_host='auto', "
                    "awaiting sender info via task from orchestrator."
                )
            else:
                logger.info(
                    f"MooncakeTransferEngineConnector started as RECEIVER "
                    f"(will query sender at {self.sender_host}:{self.sender_zmq_port})"
                )

    def get_connection_info(self) -> dict[str, Any]:
        """Get connection info for this connector (useful for sender to share with receivers)."""
        return {
            "host": self.host,
            "zmq_port": self.zmq_port,
            "rpc_port": self.rpc_port,
            "can_put": self.can_put,
        }

    def update_sender_info(
        self,
        sender_host: str,
        sender_zmq_port: int,
        sender_rank: int | None = None,
    ) -> None:
        """Inject a sender's ZMQ endpoint into the receiver connector.

        When ``sender_rank`` is ``None`` (default), sets the single default
        sender used by ``get()`` when no rank is specified — this preserves
        backward-compatible 1:1 semantics.

        When ``sender_rank`` is an integer, the endpoint is stored in a
        per-rank registry for internal use (e.g. by
        ``_query_metadata_from_sender(sender_rank=R)``).
        """
        if sender_rank is not None:
            self._sender_endpoints[sender_rank] = (sender_host, sender_zmq_port)
            logger.info(
                "Sender info updated for rank %s: host=%r, zmq_port=%s",
                sender_rank,
                sender_host,
                sender_zmq_port,
            )
        else:
            self.sender_host = sender_host
            self.sender_zmq_port = sender_zmq_port
            logger.info(
                "Sender info updated (default): host=%r, zmq_port=%s",
                sender_host,
                sender_zmq_port,
            )

    def _get_local_ip(self) -> str:
        """
        Auto-detect the local IP address that can be used for RDMA communication.
        This tries to get the IP that would be used to communicate externally,
        not the loopback address.
        """
        try:
            # Create a socket to determine the local IP used for external communication
            # We don't actually connect, just use the socket to get routing info
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                # Use Google's DNS as a target (doesn't actually connect)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
            return local_ip
        except Exception as e:
            logger.warning(f"Failed to auto-detect local IP: {e}, falling back to hostname lookup")
            try:
                # Fallback: use hostname resolution
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
                return local_ip
            except Exception as e2:
                logger.error(f"Failed to get local IP via hostname: {e2}, using 127.0.0.1")
                return "127.0.0.1"

    def _deadline_expired(self) -> bool:
        deadline = getattr(self._req_local, "deadline", None)
        return deadline is not None and _time_mod.monotonic() >= deadline

    def _deadline_timeout_ms(self, timeout_ms: int) -> int:
        """Limit a blocking ZMQ operation to the active receive deadline."""
        deadline = getattr(self._req_local, "deadline", None)
        if deadline is None:
            return timeout_ms
        remaining_ms = int((deadline - _time_mod.monotonic()) * 1000)
        if remaining_ms <= 0:
            raise TimeoutError("Mooncake receive deadline expired")
        return min(timeout_ms, max(1, remaining_ms))

    def _get_req_socket(self, zmq_addr: str, timeout_ms: int) -> zmq.Socket:
        """Get or create a thread-local cached ZMQ REQ socket.

        Each calling thread maintains its own ``{zmq_addr: socket}`` map
        stored in ``self._req_local``, so concurrent ``get()`` /
        ``_query_metadata_from_sender()`` calls from different threads
        never share a REQ socket — which would violate ZMQ's strict
        send→recv ordering constraint.

        If a call fails, use ``_invalidate_req_socket()`` to discard the
        broken socket; the next call will transparently create a fresh one.
        """
        cache: dict[str, zmq.Socket] = getattr(self._req_local, "cache", None)  # type: ignore[assignment]
        if cache is None:
            cache = {}
            self._req_local.cache = cache

        sock = cache.get(zmq_addr)
        if sock is None:
            sock = self.zmq_ctx.socket(zmq.REQ)
            sock.connect(zmq_addr)
            cache[zmq_addr] = sock
        # Set timeouts per-call (safe to change between send/recv cycles).
        # SNDTIMEO guards against send() blocking when the peer is unreachable
        # or its receive buffer is full.
        timeout_ms = self._deadline_timeout_ms(timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        return sock

    def _invalidate_req_socket(self, zmq_addr: str) -> None:
        """Close and remove a thread-local cached REQ socket (e.g., after recv timeout)."""
        cache: dict[str, zmq.Socket] = getattr(self._req_local, "cache", None)  # type: ignore[assignment]
        if cache is None:
            return
        sock = cache.pop(zmq_addr, None)
        if sock is not None:
            try:
                sock.close(linger=0)
            except Exception:
                pass  # Best-effort; zmq_ctx.term() will reclaim if needed

    def put(self, from_stage: str, to_stage: str, put_key: str, data: Any) -> tuple[bool, int, dict[str, Any] | None]:
        """
        Producer Side.
        Exposes data for RDMA transfer.

        Behavior by data type:
        - ManagedBuffer: Copy to an independently owned source allocation
        - torch.Tensor / bytes: Copy to pool, is_fast_path=True
        - Other (dict, etc.): Serialize to bytes, copy to pool, is_fast_path=False
          (receiver will deserialize automatically)
        """
        if self._closed:
            raise RuntimeError("Cannot put data: MooncakeTransferEngineConnector is closed")

        if not self.can_put:
            logger.warning(
                f"Rejecting put for {put_key}: connector is in receiver mode "
                f"(sender fallback occurred, ZMQ port likely occupied by another instance)"
            )
            return False, 0, None

        put_key = self._make_key(put_key, from_stage, to_stage)

        # Serialize concurrent put() calls on the same connector to prevent
        # races in the alloc -> copy -> store-metadata flow at the Mooncake
        # C++ engine level.
        with self._put_lock:
            with self._state_lock:
                if self._closed or self._closing or put_key in self._in_flight_buffers:
                    return False, 0, None
                self._active_puts += 1
            try:
                # An externally owned ManagedBuffer can be released by its
                # caller. Copy it so this connector controls native source lifetime.
                if isinstance(data, ManagedBuffer):
                    data = data.tensor
                return self._put_impl(put_key, data)
            finally:
                with self._state_lock:
                    self._active_puts -= 1
                    if self._closing:
                        self._schedule_close()

    def _put_impl(self, put_key: str, data: Any) -> tuple[bool, int, dict[str, Any] | None]:
        """Internal put implementation, called under _put_lock."""
        try:
            src_addr = 0
            size = 0
            holder = None
            is_fast_path = True
            should_release_holder = False

            # Reject empty data early — alloc(0) has undefined behaviour and
            # would leave a zombie entry in _local_buffers that the receiver
            # side can never consume (get() returns None for data_size==0).
            if isinstance(data, bytes) and len(data) == 0:
                logger.warning(f"Rejecting put for {put_key}: empty bytes payload")
                return False, 0, None
            if isinstance(data, torch.Tensor) and data.nbytes == 0:
                logger.warning(f"Rejecting put for {put_key}: zero-size tensor")
                return False, 0, None

            # Pre-process: serialize non-raw types before entering the copy path.
            # This avoids the previous recursive put() pattern and keeps
            # _local_buffers writes atomic (single write, no override needed).
            if not isinstance(data, (torch.Tensor, bytes)):
                data = OmniSerializer.serialize(data)
                is_fast_path = False  # Receiver must deserialize

            if isinstance(data, (torch.Tensor, bytes)):
                # Copy Path
                # 1. Determine size
                if isinstance(data, torch.Tensor):
                    size = data.nbytes
                    tensor_data = data
                else:
                    size = len(data)
                    # Convert bytes to tensor for copy
                    tensor_data = torch.frombuffer(data, dtype=torch.uint8)

                # 2. Alloc from pool
                try:
                    offset = self.allocator.alloc(size)
                    holder = ManagedBuffer(self.allocator, offset, size, self.pool)
                    should_release_holder = True  # We created it, we release it
                except MemoryError:
                    logger.error(f"Pool exhausted, cannot put data size {size}")
                    return False, 0, None

                # 3. Copy data to pool
                # Handle device mismatch for copy
                try:
                    dst_tensor = holder.tensor
                    if isinstance(data, torch.Tensor):
                        if not data.is_contiguous():
                            data = data.contiguous()

                        # View as flat uint8
                        src_view = data.view(torch.uint8).flatten()
                        if src_view.device != dst_tensor.device:
                            dst_tensor.copy_(src_view, non_blocking=True)
                            # Ensure copy is complete before exposing buffer to RDMA
                            # Must sync on SOURCE device for D2H copies, or DST for H2D
                            if src_view.is_cuda:
                                with torch.cuda.device(src_view.device):
                                    torch.cuda.current_stream().synchronize()
                            elif dst_tensor.is_cuda:
                                with torch.cuda.device(dst_tensor.device):
                                    torch.cuda.current_stream().synchronize()
                        else:
                            dst_tensor.copy_(src_view)
                            if dst_tensor.is_cuda:
                                with torch.cuda.device(dst_tensor.device):
                                    torch.cuda.current_stream().synchronize()
                    else:
                        # bytes -> tensor copy
                        # torch.frombuffer creates CPU tensor. If pool is GPU, copy_ handles H2D.
                        dst_tensor.copy_(tensor_data)
                        if dst_tensor.is_cuda:
                            with torch.cuda.device(dst_tensor.device):
                                torch.cuda.current_stream().synchronize()
                except Exception as e:
                    # Copy failed, release the allocated buffer to prevent leak
                    holder.release()
                    logger.error(f"Failed to copy data to pool: {e}")
                    return False, 0, None

                src_addr = self.base_ptr + offset

            # Final guard: reject zero-size payloads regardless of how they
            # got here (ManagedBuffer(size=0), empty serialization output, etc.).
            # This is the unified catch-all that complements the early checks
            # for bytes/Tensor above.
            if size <= 0:
                if should_release_holder and isinstance(holder, ManagedBuffer):
                    holder.release()
                logger.warning(f"Rejecting put for {put_key}: final payload size is {size}")
                return False, 0, None

            with self._local_buffers_lock:
                if put_key in self._in_flight_buffers:
                    if should_release_holder and isinstance(holder, ManagedBuffer):
                        holder.release()
                    return False, 0, None
                # Release old buffer if put_key already exists (prevents pool leak)
                old_item = self._local_buffers.pop(put_key, None)
                if old_item:
                    _, _, old_holder, old_should_release, _, _ = old_item
                    if old_should_release and isinstance(old_holder, ManagedBuffer):
                        old_holder.release()
                        logger.warning(f"Released old buffer for duplicate put_key: {put_key}")

                # Store: (src_addrs, lengths, holder, should_release, is_fast_path, created_at)
                self._local_buffers[put_key] = (
                    [src_addr],
                    [size],
                    holder,
                    should_release_holder,
                    is_fast_path,
                    _time_mod.monotonic(),
                )

            # Metadata for Consumer
            metadata = {
                "source_host": self.host,
                "protocol_version": _PROTOCOL_VERSION,
                "native_tcp_proto_version": self.native_tcp_proto_version,
                "source_port": self.zmq_port,
                "data_size": size,
                "is_fast_path": is_fast_path,  # Hint: True = return ManagedBuffer, False = deserialize
            }

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size

            return True, size, metadata

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(f"RDMA Put failed for {put_key}: {e}", exc_info=True)
            return False, 0, None

    def _resolve_sender_endpoint(self, sender_rank: int | None = None) -> tuple[str, int] | None:
        """Return ``(host, zmq_port)`` for *sender_rank*.

        Resolution order:
        1. Per-rank registry (``_sender_endpoints[sender_rank]``)
        2. Default sender (``sender_host`` / ``sender_zmq_port``)
        3. ``None`` if nothing is configured.
        """
        if sender_rank is not None and sender_rank in self._sender_endpoints:
            return self._sender_endpoints[sender_rank]
        host = getattr(self, "sender_host", None)
        port = getattr(self, "sender_zmq_port", None)
        if host and port and str(host).lower() != "auto":
            return (host, int(port))
        return None

    def _query_metadata_at(self, get_key: str, host: str, port: int) -> dict[str, Any] | None:
        """Query metadata from a sender endpoint via ZMQ.

        Returns ``{source_host, source_port, data_size, is_fast_path}``
        or ``None`` when the key is not found / the query fails.
        """
        if self._deadline_expired():
            return None
        zmq_addr = f"tcp://{host}:{port}"
        try:
            req_socket = self._get_req_socket(zmq_addr, timeout_ms=5000)
            req_socket.send(QUERY_INFO + msgspec.msgpack.encode(QueryRequest(request_id=get_key)))
            req_socket.setsockopt(zmq.RCVTIMEO, self._deadline_timeout_ms(5000))
            resp = req_socket.recv()
            if resp == INFO_NOT_FOUND:
                return None
            query_resp = msgspec.msgpack.decode(resp, type=QueryResponse)
            return {
                "source_host": host,
                "protocol_version": query_resp.protocol_version,
                "native_tcp_proto_version": query_resp.native_tcp_proto_version,
                "source_port": port,
                "data_size": query_resp.data_size,
                "is_fast_path": query_resp.is_fast_path,
            }
        except Exception as e:
            self._invalidate_req_socket(zmq_addr)
            logger.debug("Failed to query metadata at %s for %s: %s", zmq_addr, get_key, e)
            return None

    def _query_metadata_from_sender(self, get_key: str, sender_rank: int | None = None) -> dict[str, Any] | None:
        """Query metadata from sender via ZMQ (fallback when ``metadata=None``).

        ``get()`` supports three metadata resolution paths::

            get(metadata=?)
            ├── Path 1: metadata has data_size (adapter path)
            │     → use metadata directly → RDMA pull
            ├── Path 2: metadata has source_host/port but no data_size
            │     → _query_metadata_at(host, port) → get data_size → RDMA pull
            └── Path 3: metadata=None (KV-transfer polling path)
                  → _query_metadata_from_sender(get_key)   ← this method
                  │
                  ├── sender endpoint resolved (via update_sender_info)
                  │     → ZMQ query → get data_size/is_fast_path
                  │     → construct metadata → RDMA pull
                  └── sender endpoint unresolved
                        → return None → caller retries or times out

        When *sender_rank* is provided, the query is routed to that
        rank's endpoint (registered via ``update_sender_info(rank=...)``).
        Otherwise the default sender is used.
        """
        endpoint = self._resolve_sender_endpoint(sender_rank)
        if endpoint is None:
            return None
        return self._query_metadata_at(get_key, *endpoint)

    def get_with_deadline(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
        *,
        deadline: float,
    ) -> tuple[Any, int] | None:
        """Bound network waits; retain abandoned destinations until the write ACK.

        A timeout cannot cancel the remote write. Its destination stays allocated
        and registered until a terminal response arrives, including during close.

        ``None`` is a retryable result for an unavailable payload, an expired
        deadline, or a receiver at ``max_inflight_receives``. The latter is
        intentional backpressure: no destination buffer is allocated until an
        in-flight receive completes. Capacity rejections are counted and logged
        at a bounded rate so they remain distinguishable operationally without
        changing the connector return type.
        """
        previous_deadline = getattr(self._req_local, "deadline", None)
        self._req_local.deadline = min(deadline, previous_deadline) if previous_deadline is not None else deadline
        try:
            if self._deadline_expired():
                return None
            return self.get(from_stage, to_stage, get_key, metadata)
        finally:
            if previous_deadline is None:
                try:
                    del self._req_local.deadline
                except AttributeError:
                    pass
            else:
                self._req_local.deadline = previous_deadline

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        key = self._make_key(get_key, from_stage, to_stage)
        capacity_rejected = False
        capacity_warning = False
        inflight_receives = 0
        with self._state_lock:
            if self._closed or self._closing:
                raise RuntimeError("Cannot get data: MooncakeTransferEngineConnector is closing or closed")
            if key in self._pending_receives or key in self._receiving_keys:
                return None
            if len(self._receiving_keys) + len(self._pending_receives) >= self.max_inflight_receives:
                self._metrics["receive_capacity_rejections"] += 1
                capacity_rejected = True
                inflight_receives = len(self._receiving_keys) + len(self._pending_receives)
                now = _time_mod.monotonic()
                if now - self._last_capacity_warning >= _CAPACITY_WARNING_INTERVAL_SECONDS:
                    self._last_capacity_warning = now
                    capacity_warning = True
            else:
                self._receiving_keys.add(key)
        if capacity_rejected and capacity_warning:
            logger.warning(
                "Mooncake receive capacity exhausted; retrying until the deadline "
                "(inflight=%d, limit=%d)",
                inflight_receives,
                self.max_inflight_receives,
            )
        if capacity_rejected:
            return None
        try:
            return self._get_impl(from_stage, to_stage, get_key, metadata)
        finally:
            with self._state_lock:
                self._receiving_keys.discard(key)
                if self._closing:
                    self._schedule_close()

    def _get_impl(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        """Consumer Side.  Allocates from local pool and pulls data via RDMA.

        Metadata resolution:

        1. ``metadata`` provided **with** ``data_size`` → use directly (RDMA pull).
        2. ``metadata`` provided with ``source_host``/``source_port`` but
           **without** ``data_size`` → query that specific sender for
           ``data_size`` / ``is_fast_path``, then RDMA pull.  This is the
           heterogeneous-TP path where the manager knows the target sender
           endpoint but not the payload size.
        3. ``metadata=None`` → query the default sender (set via
           ``update_sender_info()``) for the full metadata.

        Returns:
            ``(data, size)`` on success, ``None`` on failure.

            - **is_fast_path=True** (tensor *or* bytes payload):
                Returns ``(ManagedBuffer, size)``.
                **CALLER MUST call ``ManagedBuffer.release()`` after consuming.**
            - **is_fast_path=False** (serialized Python object):
                Returns ``(DeserializedObject, size)``.
                Buffer is auto-released internally after deserialization.
        """
        get_key = self._make_key(get_key, from_stage, to_stage)
        _t0 = _time_mod.perf_counter()
        metadata = self._resolve_metadata(get_key, metadata)
        if not metadata or self._deadline_expired():
            return None
        if metadata.get("protocol_version") != _PROTOCOL_VERSION:
            raise RuntimeError("Mooncake deadline receive requires protocol v2 on both peers")
        if self._requires_tcp_completion() and (
            metadata.get("native_tcp_proto_version") != 2 or self.native_tcp_proto_version != 2
        ):
            raise RuntimeError("Mooncake TCP acknowledged completion is not available on both peers")

        _t1 = _time_mod.perf_counter()
        query_ms = (_t1 - _t0) * 1000

        src_host = metadata.get("source_host")
        src_port = metadata.get("source_port")
        data_size = int(metadata.get("data_size", 0) or 0)
        is_fast_path = bool(metadata.get("is_fast_path", False))
        if not src_host or not src_port or str(src_host).lower() == "auto":
            logger.error(
                f"Invalid metadata for {get_key}: source_host={src_host!r}, "
                f"source_port={src_port!r}. Cannot establish RDMA connection."
            )
            return None

        if data_size == 0:
            logger.warning(f"Skipping get for {get_key}: data_size is 0 (metadata={metadata})")
            return None

        try:
            recv_buffer, dst_ptr = self._alloc_recv_buffer(data_size, is_fast_path)
        except MemoryError:
            return None

        if self._deadline_expired():
            recv_buffer.release()
            return None

        _t2 = _time_mod.perf_counter()
        alloc_ms = (_t2 - _t1) * 1000
        try:
            resp = self._request_transfer(
                get_key=get_key,
                source_host=str(src_host),
                source_port=int(src_port),
                dst_ptr=dst_ptr,
                data_size=data_size,
                recv_buffer=recv_buffer,
            )
        except _ReceivePendingError:
            self._metrics["timeouts"] += 1
            return None
        except (TimeoutError, zmq.Again) as e:
            self._metrics["timeouts"] += 1
            logger.debug("RDMA Get reached its receive deadline: %s", e)
            recv_buffer.release()
            return None
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(f"RDMA transfer request failed: {e}", exc_info=True)
            recv_buffer.release()
            return None

        _t3 = _time_mod.perf_counter()
        rdma_ms = (_t3 - _t2) * 1000
        if self._deadline_expired():
            self._metrics["timeouts"] += 1
            recv_buffer.release()
            return None
        if resp != TRANS_DONE:
            self._metrics["errors"] += 1
            logger.error("RDMA Get failed: received %r instead of TRANS_DONE", resp)
            recv_buffer.release()
            return None

        try:
            sync_ms = self._sync_receive_buffer_if_needed()
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(f"Failed to synchronize Mooncake receive buffer: {e}", exc_info=True)
            recv_buffer.release()
            return None

        if self._deadline_expired():
            self._metrics["timeouts"] += 1
            recv_buffer.release()
            return None

        try:
            return self._decode_result(
                get_key=get_key,
                recv_buffer=recv_buffer,
                data_size=data_size,
                is_fast_path=is_fast_path,
                query_ms=query_ms,
                alloc_ms=alloc_ms,
                rdma_ms=rdma_ms,
                sync_ms=sync_ms,
                start_time=_t0,
            )
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(f"RDMA result decode failed for {get_key}: {e}", exc_info=True)
            return None

    def _resolve_metadata(self, get_key: str, metadata: dict[str, Any] | None) -> dict[str, Any] | None:
        """Resolve full transfer metadata for a receive request."""
        if not metadata:
            # Path 3: no metadata at all — query default sender
            if not self.sender_host or not self.sender_zmq_port or str(self.sender_host).lower() == "auto":
                raise RuntimeError(
                    f"get(metadata=None) requires sender info to be resolved, "
                    f"but sender_host={self.sender_host!r}, sender_zmq_port={self.sender_zmq_port!r}. "
                    f"Call update_sender_info(host, port) before using get() without metadata."
                )
            return self._query_metadata_from_sender(get_key)

        if "data_size" in metadata:
            # Path 1: metadata is complete
            return metadata

        # Path 2: partial metadata (host/port only) — query that sender
        partial_host = metadata.get("source_host")
        partial_port = metadata.get("source_port")
        if not partial_host or not partial_port:
            logger.warning(
                "get(%s): partial metadata missing source_host/source_port, cannot resolve data_size. metadata=%s",
                get_key,
                metadata,
            )
            return None
        return self._query_metadata_at(get_key, str(partial_host), int(partial_port))

    def _alloc_recv_buffer(self, data_size: int, is_fast_path: bool) -> tuple[ManagedBuffer, int]:
        """Allocate the receive-side buffer and return ``(buffer, dst_ptr)``."""
        # For non-fast-path serialized objects, allocate a few extra bytes as
        # padding. Mooncake's TCP/memcpy transport may prepend a protocol byte.
        padding = 4 if not is_fast_path else 0
        alloc_size = data_size + padding
        try:
            offset = self.allocator.alloc(alloc_size)
        except MemoryError:
            logger.error(f"Failed to allocate {alloc_size} bytes in receive pool")
            raise
        recv_buffer = ManagedBuffer(self.allocator, offset, alloc_size, self.pool)
        return recv_buffer, self.base_ptr + offset

    def _request_transfer(
        self,
        get_key: str,
        source_host: str,
        source_port: int,
        dst_ptr: int,
        data_size: int,
        recv_buffer: ManagedBuffer,
    ) -> bytes:
        """Request a remote WRITE; a sent request owns its destination until ACK."""
        agent_meta = MooncakeAgentMetadata(
            remote_hostname=self.host,
            remote_port=self.rpc_port,
            request_id=get_key,
            dst_addrs=[dst_ptr],
            lengths=[data_size],
            protocol_version=_PROTOCOL_VERSION,
            native_tcp_proto_version=self.native_tcp_proto_version,
        )
        size_timeout_ms = (data_size // _TRANSFER_TIMEOUT_STEP_BYTES) * _TRANSFER_TIMEOUT_PER_STEP_MS
        timeout_ms = _BASE_TRANSFER_TIMEOUT_MS + size_timeout_ms
        # A transfer uses a dedicated socket so an abandoned reply cannot be
        # consumed by a subsequent request on this thread's metadata socket.
        req_socket = self.zmq_ctx.socket(zmq.REQ)
        sent = False
        deferred = False
        received = False
        try:
            req_socket.setsockopt(zmq.LINGER, 0)
            req_socket.setsockopt(zmq.SNDTIMEO, self._deadline_timeout_ms(timeout_ms))
            req_socket.connect(f"tcp://{source_host}:{source_port}")
            req_socket.send(msgspec.msgpack.encode(agent_meta))
            sent = True
            req_socket.setsockopt(zmq.RCVTIMEO, self._deadline_timeout_ms(timeout_ms))
            response = req_socket.recv()
            received = True
            if response not in (TRANS_DONE, TRANS_NOT_SUBMITTED):
                raise RuntimeError(f"Mooncake write completion is unknown: {response!r}")
            return response
        except Exception as exc:
            if sent:
                # No exception, timeout, or local clock proves the remote write
                # stopped. Transfer exclusive socket/buffer ownership under the
                # lock; the caller must not release this allocation.
                deferred = True
                self._defer_receive(get_key, recv_buffer, req_socket, wait_for_ack=not received)
                raise _ReceivePendingError(get_key) from exc
            raise
        finally:
            if not deferred:
                req_socket.close(linger=0)

    def _defer_receive(self, get_key: str, buffer: ManagedBuffer, sock: zmq.Socket, *, wait_for_ack: bool) -> None:
        with self._state_lock:
            # A received ambiguous reply is final on this REQ socket. The SDK
            # may still own the memory, but there is no further ACK to wait for.
            self._pending_receives[get_key] = (buffer, sock if wait_for_ack else None)
            _RETAINED_CONNECTORS.add(self)
            if not wait_for_ack:
                sock.close(linger=0)
                logger.error("Mooncake completion unknown for %s; retaining destination until process exit", get_key)
                return
            waiter = threading.Thread(
                target=self._finish_receive, args=(get_key,), daemon=True, name="mooncake-completion"
            )
            try:
                waiter.start()
            except Exception:
                sock.close(linger=0)
                self._pending_receives[get_key] = (buffer, None)
                logger.exception("Cannot start Mooncake completion waiter for %s; retaining destination", get_key)

    def _finish_receive(self, get_key: str) -> None:
        with self._state_lock:
            buffer, sock = self._pending_receives[get_key]
        assert sock is not None
        completed = False
        try:
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            while True:
                try:
                    response = sock.recv()
                    break
                except zmq.Again:
                    continue
            if response not in (TRANS_DONE, TRANS_NOT_SUBMITTED):
                raise RuntimeError(f"Mooncake write completion is unknown: {response!r}")
            if response == TRANS_DONE:
                self._sync_receive_buffer_if_needed()
            completed = True
        except Exception:
            logger.exception("Mooncake completion for %s is unknown; retaining destination until process exit", get_key)
        finally:
            sock.close(linger=0)
        with self._state_lock:
            if completed:
                buffer.release()
                self._pending_receives.pop(get_key)
            else:
                self._pending_receives[get_key] = (buffer, None)
            self._refresh_retention()
            if self._closing:
                self._schedule_close()

    def _refresh_retention(self) -> None:
        """Called under _state_lock; keep all possibly native-owned memory alive."""
        if self._pending_receives or self._in_flight_buffers:
            _RETAINED_CONNECTORS.add(self)
        elif not self._closing:
            _RETAINED_CONNECTORS.discard(self)

    def _schedule_close(self) -> None:
        """Finalize from a separate thread, never join the current sender worker."""
        if self._closing and not self._close_scheduled:
            self._close_scheduled = True
            try:
                threading.Thread(target=self.close, name="mooncake-close", daemon=True).start()
            except Exception:
                self._close_scheduled = False
                logger.exception("Cannot schedule Mooncake close; retaining registered memory")

    def _sync_receive_buffer_if_needed(self) -> float:
        """Synchronize GPU-visible receive memory until Mooncake exposes a finer fence."""
        t0 = _time_mod.perf_counter()
        if self.pool.is_cuda:
            with torch.cuda.device(self.pool.device):
                torch.cuda.current_stream().synchronize()
        return (_time_mod.perf_counter() - t0) * 1000

    def _decode_result(
        self,
        get_key: str,
        recv_buffer: ManagedBuffer,
        data_size: int,
        is_fast_path: bool,
        query_ms: float,
        alloc_ms: float,
        rdma_ms: float,
        sync_ms: float,
        start_time: float,
    ) -> tuple[Any, int]:
        """Decode a completed transfer result."""
        if is_fast_path:
            try:
                total_ms = (_time_mod.perf_counter() - start_time) * 1000
                mbps = (data_size / 1024 / 1024) / (total_ms / 1000) if total_ms > 0 else 0
                logger.info(
                    f"[RDMA GET] {get_key}: query={query_ms:.1f}ms, alloc={alloc_ms:.1f}ms, "
                    f"rdma={rdma_ms:.1f}ms, sync={sync_ms:.1f}ms, "
                    f"total={total_ms:.1f}ms, {mbps:.1f} MB/s (fast_path, zero-copy)"
                )
                self._metrics["gets"] += 1
                self._metrics["bytes_transferred"] += data_size
                return recv_buffer, data_size
            except Exception:
                recv_buffer.release()
                raise

        try:
            copy_start = _time_mod.perf_counter()
            raw_bytes = recv_buffer.to_bytes()
            copy_ms = (_time_mod.perf_counter() - copy_start) * 1000

            deser_start = _time_mod.perf_counter()
            payload = raw_bytes[:data_size]
            try:
                value = OmniSerializer.deserialize(payload)
            except msgspec.DecodeError:
                payload = raw_bytes[1 : data_size + 1]
                value = OmniSerializer.deserialize(payload)
            deser_ms = (_time_mod.perf_counter() - deser_start) * 1000

            total_ms = (_time_mod.perf_counter() - start_time) * 1000
            mbps = (data_size / 1024 / 1024) / (total_ms / 1000) if total_ms > 0 else 0
            logger.info(
                f"[RDMA GET] {get_key}: query={query_ms:.1f}ms, alloc={alloc_ms:.1f}ms, "
                f"rdma={rdma_ms:.1f}ms, sync={sync_ms:.1f}ms, copy={copy_ms:.1f}ms, "
                f"deser={deser_ms:.1f}ms, total={total_ms:.1f}ms, {mbps:.1f} MB/s"
            )
            self._metrics["gets"] += 1
            self._metrics["bytes_transferred"] += data_size
            return value, data_size
        finally:
            recv_buffer.release()

    def cleanup(self, request_id: str, from_stage: str | None = None, to_stage: str | None = None) -> None:
        """Release the producer-side buffer associated with the request.

        Args:
            request_id: The key used in ``put()``.
            from_stage: Optional source stage.  When both *from_stage* and
                *to_stage* are provided the method applies ``_make_key()``
                internally, so callers can pass the same raw key they used
                in ``put()`` / ``get()`` without knowing the internal format.
                When omitted the *request_id* is used as-is (suitable for
                internal callers that already hold the transformed key).
            to_stage: Optional destination stage (see *from_stage*).
        """
        if (from_stage is None) != (to_stage is None):
            raise ValueError(
                f"cleanup() requires both from_stage and to_stage, or neither. "
                f"Got from_stage={from_stage!r}, to_stage={to_stage!r}"
            )
        if from_stage is not None and to_stage is not None:
            request_id = self._make_key(request_id, from_stage, to_stage)
        with self._local_buffers_lock:
            if request_id in self._in_flight_buffers:
                return
            item = self._local_buffers.pop(request_id, None)
            if item:
                self._in_flight_buffers.discard(request_id)
                # item is (src_addrs, lengths, holder, should_release, is_fast_path, created_at)
                _, _, holder, should_release, _, _ = item
                if should_release and isinstance(holder, ManagedBuffer):
                    # We own this buffer (created internally), so we must release it.
                    holder.release()
                # If holder was externally owned (should_release=False), we do nothing.
                # If holder was something else (e.g. Tensor), GC handles it.

    def health(self) -> dict[str, Any]:
        if self._closed:
            return {"status": "unhealthy", "error": "Connector is closed"}

        return {
            "status": "healthy",
            "host": self.host,
            "metadata_server": None,
            "master": None,
            "protocol": self.protocol,
            "pool_device": self.pool_device,
            "pool_size": self.pool_size,
            **self._metrics,
        }

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        with self._state_lock:
            self._close_scheduled = False
            if self._closed:
                return
            self._closing = True
            if (
                self._receiving_keys
                or self._pending_receives
                or self._in_flight_buffers
                or self._active_handlers
                or self._active_puts
            ):
                _RETAINED_CONNECTORS.add(self)
                logger.warning("Mooncake close deferred while operations may still own registered memory")
                return
            self._closed = True
            _RETAINED_CONNECTORS.discard(self)
        self._close_resources()

    def _close_resources(self) -> None:
        """
        Gracefully shutdown the connector and release all resources.
        This method should be called when the connector is no longer needed.
        Idempotent: safe to call multiple times.
        """
        # Idempotent guard — use getattr for safety: if __init__ raised before
        # setting _closed, __del__ → close() would hit AttributeError.
        logger.info("Closing MooncakeTransferEngineConnector...")

        # Stop accepting new transfers, then finish existing writes while the
        # listener can still send their ACKs. Otherwise a receiver that timed
        # out would retain its destination forever despite a completed write.
        self.can_put = False
        if self._sender_executor is not None:
            self._sender_executor.shutdown(wait=True, cancel_futures=False)
        self._stop_event.set()
        if self._listener_thread is not None and self._listener_thread.is_alive():
            # Each listener socket wait is bounded. Join before destroying its
            # context: destroy() is unsafe while another thread owns a socket.
            self._listener_thread.join()

        # 4. Release all pending buffers
        with self._local_buffers_lock:
            for req_id, item in list(self._local_buffers.items()):
                _, _, holder, should_release, _, _ = item
                if should_release and isinstance(holder, ManagedBuffer):
                    holder.release()
            self._local_buffers.clear()
            self._in_flight_buffers.clear()

        # 5. Close thread-local cached REQ sockets (for the calling thread).
        #    Sockets cached in *other* threads are not directly accessible here
        #    because they live in per-thread ``threading.local()`` storage.
        #    They will be forcefully closed when ``zmq_ctx.destroy()`` is called
        #    in step 7 below, which is the intended cleanup path.
        cache: dict[str, zmq.Socket] = getattr(self._req_local, "cache", None)  # type: ignore[assignment]
        if cache:
            for addr, sock in cache.items():
                try:
                    sock.close(linger=0)
                except Exception:
                    pass  # Best-effort; zmq_ctx.destroy() below will reclaim
            cache.clear()

        # 6. Unregister memory from engine (if supported)
        try:
            if hasattr(self, "engine") and hasattr(self.engine, "unregister_memory"):
                # Mooncake API only takes address, not size
                self.engine.unregister_memory(self.base_ptr)
        except Exception as e:
            logger.warning(f"Failed to unregister memory: {e}")

        # 7. Close ZMQ contexts (also reclaims sockets cached in other threads)
        try:
            if hasattr(self, "zmq_ctx"):
                # No get() or completion waiter still owns a socket here.
                # term() alone would wait forever on other threads' idle caches.
                self.zmq_ctx.destroy(linger=0)
        except Exception as e:
            logger.warning(f"Failed to terminate ZMQ context: {e}")

        # 8. Release pool tensor reference (let GC handle actual deallocation)
        # Note: We set to None instead of del to avoid AttributeError on repeated access
        self.pool = None

        logger.info("MooncakeTransferEngineConnector closed.")

    # -------------------------------------------------------
    # Listener Logic
    # -------------------------------------------------------

    def _cleanup_stale_buffers(self) -> None:
        """Reclaim stale buffers not currently used by a transfer.

        Prevents permanent memory leaks when a receiver crashes or times out
        without ever pulling the data.
        """
        if self.buffer_ttl_seconds <= 0:
            return
        now = _time_mod.monotonic()
        with self._local_buffers_lock:
            stale_keys = [
                k
                for k, v in self._local_buffers.items()
                if k not in self._in_flight_buffers and now - v[5] > self.buffer_ttl_seconds
            ]
            for k in stale_keys:
                item = self._local_buffers.pop(k)
                _, _, holder, should_release, _, _ = item
                if should_release and isinstance(holder, ManagedBuffer):
                    holder.release()
                logger.warning(f"TTL expired ({self.buffer_ttl_seconds}s): cleaned up stale buffer for {k}")

    def _zmq_listener_loop(self):
        socket = self.zmq_ctx.socket(zmq.ROUTER)
        # A non-reading peer must not block draining other completion replies.
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        try:
            socket.bind(f"tcp://{self.host}:{self.zmq_port}")
        except zmq.ZMQError as exc:
            # Any bind failure (EADDRINUSE, EADDRNOTAVAIL, EACCES, etc.)
            # is fatal for a sender — fail fast so __init__ propagates the error.
            # There is no silent receiver fallback; roles are explicitly assigned.
            logger.error(f"ZMQ bind failed on {self.host}:{self.zmq_port}: {exc} (errno={exc.errno})")
            self.can_put = False
            self._bind_error = exc
            self._listener_ready.set()
            return  # let __init__ propagate the error

        # Successfully bound - signal ready
        self._listener_ready.set()

        # Create inproc socket pair for worker thread notifications
        # This allows workers to wake up the listener immediately when done
        notify_addr = f"inproc://notify-{id(self)}"
        notify_recv = self.zmq_ctx.socket(zmq.PULL)
        notify_recv.bind(notify_addr)

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        poller.register(notify_recv, zmq.POLLIN)

        # Response queue for thread-safe socket operations
        response_queue: queue.Queue = queue.Queue()

        try:
            while not self._stop_event.is_set() or not response_queue.empty():
                try:
                    # Poll for incoming requests or notifications (with timeout)
                    events = dict(poller.poll(1000))

                    # Process notifications (drain all)
                    if notify_recv in events:
                        while True:
                            try:
                                notify_recv.recv(zmq.NOBLOCK)
                            except zmq.Again:
                                break

                    # Process any pending responses (non-blocking)
                    while True:
                        try:
                            identity, response = response_queue.get_nowait()
                            socket.send_multipart([identity, b"", response])
                        except queue.Empty:
                            break

                    # Periodic TTL cleanup (~every 10s)
                    now = _time_mod.monotonic()
                    if now - self._last_ttl_check >= 10.0:
                        self._last_ttl_check = now
                        self._cleanup_stale_buffers()

                    # Process incoming requests
                    if socket in events:
                        frames = socket.recv_multipart()
                        if len(frames) >= 2:
                            reply = None
                            with self._state_lock:
                                if self._closing or not self._sender_slots.acquire(blocking=False):
                                    reply = INFO_NOT_FOUND if frames[-1].startswith(QUERY_INFO) else TRANS_NOT_SUBMITTED
                                else:
                                    self._active_handlers += 1
                                    try:
                                        assert self._sender_executor is not None
                                        self._sender_executor.submit(
                                            self._handle_pull_request,
                                            response_queue,
                                            notify_addr,
                                            frames[0],
                                            frames[-1],
                                        )
                                    except Exception:
                                        self._active_handlers -= 1
                                        self._sender_slots.release()
                                        reply = (
                                            INFO_NOT_FOUND if frames[-1].startswith(QUERY_INFO) else TRANS_NOT_SUBMITTED
                                        )
                                        logger.exception("Cannot submit Mooncake request handler")
                            if reply is not None:
                                socket.send_multipart([frames[0], b"", reply])
                except zmq.ContextTerminated:
                    break
                except Exception:
                    logger.debug("Listener loop error", exc_info=True)
        finally:
            try:
                notify_recv.close(linger=0)
                socket.close(linger=0)
            except Exception:
                pass  # Best-effort cleanup during listener shutdown

    def _handle_pull_request(self, response_queue: queue.Queue, notify_addr: str, identity, payload):
        """Own a bounded sender slot, including after an ambiguous SDK failure."""
        uncertain = False
        key: str | None = None
        response = TRANS_NOT_SUBMITTED
        try:
            if payload.startswith(QUERY_INFO):
                self._handle_query_request(response_queue, notify_addr, identity, payload[len(QUERY_INFO) :])
                return
            meta = msgspec.msgpack.decode(payload, type=MooncakeAgentMetadata)
            # Legacy consumers release memory on any error. Reject before
            # invoking the SDK; mixed-version transfers are not supported.
            if meta.protocol_version != _PROTOCOL_VERSION:
                response = TRANS_ERROR
                return
            if self._requires_tcp_completion() and (
                meta.native_tcp_proto_version != 2 or self.native_tcp_proto_version != 2
            ):
                response = TRANS_NOT_SUBMITTED
                return
            with self._state_lock:
                item = self._local_buffers.get(meta.request_id)
                if self._closing or item is None or meta.request_id in self._in_flight_buffers:
                    return
                src_addrs, src_lengths, _, _, _, _ = item
                if (
                    len(meta.dst_addrs) != len(src_addrs)
                    or meta.lengths != src_lengths
                    or not all(address > 0 for address in meta.dst_addrs)
                ):
                    return
                key = meta.request_id
                self._in_flight_buffers.add(key)
                _RETAINED_CONNECTORS.add(self)
                # A native error/exception does not prove the write stopped.
                # Pin both ends before entering the synchronous SDK call.
                uncertain = True
            remote = f"{meta.remote_hostname}:{meta.remote_port}"
            response = TRANS_UNKNOWN
            # Run on the existing sender worker. Receiver socket deadlines do
            # not cancel this call or establish that a failed write has drained.
            status = self.engine.batch_transfer_sync_write(remote, src_addrs, meta.dst_addrs, src_lengths)
            if status != 0:
                logger.error("Mooncake completion unknown for %s (status=%s); retaining source", key, status)
                return
            response = TRANS_DONE
            with self._state_lock:
                self._in_flight_buffers.remove(key)
                uncertain = False
                self.cleanup(key)
                self._refresh_retention()
        except Exception:
            if response != TRANS_DONE:
                response = TRANS_UNKNOWN if uncertain else TRANS_NOT_SUBMITTED
            logger.exception("Mooncake sender failed; destination outcome=%r", response)
        finally:
            if not payload.startswith(QUERY_INFO):
                response_queue.put((identity, response))
                self._notify_listener(notify_addr)
            with self._state_lock:
                self._active_handlers -= 1
                if not uncertain:
                    self._sender_slots.release()
                if self._closing:
                    self._schedule_close()

    def _handle_query_request(self, response_queue: queue.Queue, notify_addr: str, identity, payload):
        """Handle metadata query request."""
        try:
            query = msgspec.msgpack.decode(payload, type=QueryRequest)

            with self._local_buffers_lock:
                item = self._local_buffers.get(query.request_id)

            if not item:
                response_queue.put((identity, INFO_NOT_FOUND))
            else:
                src_addrs, src_lengths, _, _, is_fast_path, _ = item

                resp = QueryResponse(
                    request_id=query.request_id,
                    data_size=src_lengths[0] if src_lengths else 0,
                    is_fast_path=is_fast_path,
                    protocol_version=_PROTOCOL_VERSION,
                    native_tcp_proto_version=self.native_tcp_proto_version,
                )
                response_queue.put((identity, msgspec.msgpack.encode(resp)))

        except Exception as e:
            logger.error(f"Query request failed: {e}")
            response_queue.put((identity, INFO_NOT_FOUND))

        self._notify_listener(notify_addr)

    def _notify_listener(self, notify_addr: str):
        """Send notification to wake up listener thread.

        Uses a per-worker-thread cached PUSH socket to avoid the overhead
        of creating/connecting/closing a socket on every call.
        """
        try:
            local = self._worker_local
            sock = getattr(local, "notify_socket", None)
            cached_addr = getattr(local, "notify_addr", None)
            if sock is None or cached_addr != notify_addr:
                # First call on this thread, or address changed – create socket
                if sock is not None:
                    sock.close(linger=0)
                sock = self.zmq_ctx.socket(zmq.PUSH)
                sock.connect(notify_addr)
                local.notify_socket = sock
                local.notify_addr = notify_addr
            sock.send(b"", zmq.NOBLOCK)
        except Exception:
            # Socket may be broken; clear cache so next call recreates it
            local.notify_socket = None
            local.notify_addr = None
