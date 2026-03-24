# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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

# ZMQ Message constants
TRANS_DONE = b"trans_done"
TRANS_ERROR = b"trans_error"
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


class BufferAllocator:
    """
    Manages the allocation of memory segments within the registered pool.
    Thread-safe implementation using a simple free list.
    """

    def __init__(self, total_size: int, alignment: int = 4096):
        self.total_size = total_size
        self.alignment = alignment
        self.lock = threading.Lock()
        # Free list: [(start, size), ...] sorted by start
        self.free_blocks = [(0, total_size)]

    def alloc(self, size: int) -> int:
        """
        Allocates a block of 'size' bytes.
        Returns the starting offset.
        """
        # Align size upwards
        aligned_size = (size + self.alignment - 1) // self.alignment * self.alignment

        with self.lock:
            for i, (start, block_size) in enumerate(self.free_blocks):
                if block_size >= aligned_size:
                    # Found a block
                    new_start = start + aligned_size
                    new_size = block_size - aligned_size

                    if new_size > 0:
                        self.free_blocks[i] = (new_start, new_size)
                    else:
                        self.free_blocks.pop(i)
                    return start

        raise MemoryError(f"Out of memory in buffer pool. Requested {size} bytes (aligned {aligned_size}).")

    def free(self, offset: int, size: int):
        """
        Frees a previously allocated block.
        """
        aligned_size = (size + self.alignment - 1) // self.alignment * self.alignment

        with self.lock:
            # Check for double-free and corruption
            for start, length in self.free_blocks:
                # Case 1: Exact match = double free, safe to ignore
                if offset == start and aligned_size == length:
                    logger.warning(f"Double free detected at offset {offset}, size {aligned_size}. Ignoring.")
                    return
                # Case 2: Block is fully contained within an existing free block = also double free
                # This happens when the block was freed and then merged with adjacent blocks
                if offset >= start and offset + aligned_size <= start + length:
                    logger.warning(
                        f"Double free detected: block {offset}-{offset + aligned_size} "
                        f"is already within free block {start}-{start + length}. Ignoring."
                    )
                    return
                # Case 3: Partial overlap (but not fully contained) = memory corruption
                if not (offset + aligned_size <= start or start + length <= offset):
                    raise RuntimeError(
                        f"Memory corruption detected: freeing {offset}-{offset + aligned_size} "
                        f"partially overlaps with free block {start}-{start + length}"
                    )

            self.free_blocks.append((offset, aligned_size))
            self.free_blocks.sort()  # Sort by offset

            # Merge adjacent blocks
            i = 0
            while i < len(self.free_blocks) - 1:
                curr_start, curr_size = self.free_blocks[i]
                next_start, next_size = self.free_blocks[i + 1]

                if curr_start + curr_size == next_start:
                    self.free_blocks[i] = (curr_start, curr_size + next_size)
                    self.free_blocks.pop(i + 1)
                else:
                    i += 1


class ManagedBuffer:
    """
    A temporary view into the global memory pool.
    Must be kept alive while the data view is being used.
    """

    def __init__(self, allocator: BufferAllocator, offset: int, size: int, pool_tensor: torch.Tensor):
        self.allocator = allocator
        self.offset = offset
        self.size = size
        self.pool_tensor = pool_tensor
        self._released = False

    def release(self):
        """Explicitly release the buffer back to the pool."""
        if not self._released:
            self.allocator.free(self.offset, self.size)
            self._released = True

    def __del__(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @property
    def tensor(self) -> torch.Tensor:
        """
        Returns a 1D uint8 zero-copy view of the buffer.
        """
        return self.pool_tensor[self.offset : self.offset + self.size]

    def as_tensor(self, dtype: torch.dtype, shape: tuple) -> torch.Tensor:
        """
        Returns a typed, shaped zero-copy view.
        Validates size, shape, and alignment.
        """
        itemsize = torch.tensor([], dtype=dtype).element_size()

        # Calculate expected size
        expected_bytes = itemsize
        for dim in shape:
            if dim < 0:
                raise ValueError("Dynamic dimension (-1) is not supported in as_tensor")
            expected_bytes *= dim

        if expected_bytes != self.size:
            raise ValueError(
                f"Shape {shape} with dtype {dtype} requires {expected_bytes} bytes, but buffer size is {self.size}"
            )

        # Check alignment (offset must be divisible by itemsize)
        if self.offset % itemsize != 0:
            raise RuntimeError(f"Buffer offset {self.offset} is not aligned for dtype {dtype} (itemsize {itemsize})")

        raw_view = self.tensor
        # view() requires contiguous memory, slice of contiguous tensor is contiguous
        typed_view = raw_view.view(dtype)
        return typed_view.reshape(shape)

    def to_bytes(self) -> bytes:
        """
        Returns a copy of the data as python bytes.
        Performs D2H copy if pool is on GPU.
        """
        t = self.tensor
        if t.is_cuda:
            t = t.cpu()
        return t.numpy().tobytes()


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
        - **1 receiver → 1 sender**: ``update_sender_info()`` stores a single
          ``(sender_host, sender_zmq_port)`` pair, so a receiver can only query
          metadata from one sender at a time.

    Future work:
        - Support 1 sender → N receivers (e.g. reference-counted buffers, or
          explicit ``retain()`` / ``release()`` semantics so the buffer survives
          multiple pulls).
        - Support 1 receiver → N senders (e.g. a sender registry mapping
          ``get_key`` prefixes to different sender endpoints).
    """

    # RDMA connector copies raw bytes/tensor directly to the memory pool
    # without going through OmniSerializer, so the sender can use
    # to_gpu_tensor() / to_bytes() fast-paths.
    supports_raw_data: bool = True

    def __init__(self, config: dict[str, Any]):
        if TransferEngine is None:
            raise ImportError("Mooncake not available")

        self._closed = False
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
        self._local_buffers_lock = threading.Lock()
        self._req_local = threading.local()
        self._worker_local = threading.local()
        self._last_ttl_check: float = _time_mod.monotonic()

        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "errors": 0,
            "timeouts": 0,
        }

        self.config = config
        host_config = config.get("host", "127.0.0.1")
        # Support "auto" to auto-detect local IP address
        if host_config.lower() == "auto":
            self.host = self._get_local_ip()
            logger.info(f"Auto-detected local IP for RDMA: {self.host}")
        else:
            self.host = host_config
        self.zmq_port = config.get("zmq_port", 50051)
        self.protocol = config.get("protocol", "rdma")

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

        # --- Mooncake Engine Init ---
        self.engine = TransferEngine()
        # Note: For P2P handshake mode, local_hostname should be just the IP address.
        # Mooncake will auto-assign an RPC port, retrievable via get_rpc_port().
        ret = self.engine.initialize(self.host, "P2PHANDSHAKE", self.protocol, self.device_name)
        if ret != 0:
            raise RuntimeError(f"Mooncake Engine initialization failed with code {ret}")

        self.rpc_port = self.engine.get_rpc_port()
        logger.info(f"MooncakeTransferEngineConnector initialized at {self.host}:{self.rpc_port}")

        # --- Pool Allocation & Registration ---
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

        # --- State Management & Background Threads ---
        # Most fields already initialized at the top of __init__ for teardown
        # safety.  Only create the real ZMQ context and thread pool here.
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

    def update_sender_info(self, sender_host: str, sender_zmq_port: int) -> None:
        """
        Inject the sender's ZMQ endpoint into the receiver connector.
        Used for NO METADATA GET calls.(E.g: KV-cache transfer path)
        Must be called before using get() without metadata!
        Otherwise, get() will raise an error.
        """
        self.sender_host = sender_host
        self.sender_zmq_port = sender_zmq_port
        logger.info(f"Sender info updated: host={sender_host!r}, zmq_port={sender_zmq_port}")

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
        - ManagedBuffer (from this pool): Zero-copy, is_fast_path=True
        - ManagedBuffer (different pool): Fallback to tensor copy path
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
            if not isinstance(data, (ManagedBuffer, torch.Tensor, bytes)):
                data = OmniSerializer.serialize(data)
                is_fast_path = False  # Receiver must deserialize

            if isinstance(data, ManagedBuffer):
                # Zero-Copy Path
                # Validate that the buffer belongs to this connector's pool.
                if data.pool_tensor.data_ptr() != self.pool.data_ptr():
                    # Fallback to copy path: extract tensor and let the
                    # torch.Tensor branch below handle the copy.
                    logger.warning("ManagedBuffer from different pool detected. Falling back to copy path.")
                    data = data.tensor.contiguous()
                    # Fall through to torch.Tensor branch
                else:
                    src_addr = self.base_ptr + data.offset
                    size = data.size
                    holder = data  # Keep the buffer alive
                    should_release_holder = False  # Caller owns it

            # Use 'if' (not 'elif') so the ManagedBuffer fallback above can
            # convert to Tensor and flow into this branch seamlessly.
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

    def _query_metadata_from_sender(self, get_key: str) -> dict[str, Any] | None:
        """Query metadata from sender via ZMQ (fallback when ``metadata=None``).

        ``get()`` supports two metadata resolution paths::

            get(metadata=?)
            ├── metadata provided (adapter path)
            │     → use metadata directly (source_host/port/data_size)
            │     → RDMA pull
            └── metadata=None (KV-transfer polling path)
                  → _query_metadata_from_sender(get_key)   ← this method
                  │
                  ├── sender_host resolved (via update_sender_info)
                  │     → ZMQ query → get data_size/is_fast_path
                  │     → construct metadata → RDMA pull
                  └── sender_host unresolved ("auto" / None)
                        → return None → caller retries or times out

        For the second path, the caller must call
        :meth:`update_sender_info` before ``get()`` to resolve the sender's ZMQ endpoint.
        Support the two paths in case that the orchestrator pushes the request info
        to different stages at the same time knowing metadata or not.
        """
        zmq_addr = f"tcp://{self.sender_host}:{self.sender_zmq_port}"
        req_socket = self._get_req_socket(zmq_addr, timeout_ms=5000)

        try:
            # Send query request
            query = QueryRequest(request_id=get_key)
            req_socket.send(QUERY_INFO + msgspec.msgpack.encode(query))
            resp = req_socket.recv()

            if resp == INFO_NOT_FOUND:
                return None

            # Parse response
            query_resp = msgspec.msgpack.decode(resp, type=QueryResponse)
            return {
                # source_host/source_port are used for verification
                "source_host": self.sender_host,
                "source_port": self.sender_zmq_port,
                "data_size": query_resp.data_size,
                "is_fast_path": query_resp.is_fast_path,
            }
        except Exception as e:
            # Socket may be stuck in bad state after timeout; discard it
            self._invalidate_req_socket(zmq_addr)
            logger.debug(f"Failed to query metadata for {get_key}: {e}")
            return None

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        """
        Consumer Side.
        Allocates from local pool and pulls data via RDMA.

        If metadata is not provided, will attempt to query it from sender
        using configured sender_host/sender_zmq_port.

        Returns:
            ``(data, size)`` on success, ``None`` on failure.

            - **is_fast_path=True** (tensor *or* bytes payload):
                Returns ``(ManagedBuffer, size)``.
                **CALLER MUST call ``ManagedBuffer.release()`` after consuming.**
                Note: even if the producer ``put()`` raw ``bytes``, the consumer
                receives a ``ManagedBuffer`` — use ``buf.to_bytes()`` to obtain
                a ``bytes`` copy, or ``buf.tensor`` for zero-copy access.
            - **is_fast_path=False** (serialized Python object):
                Returns ``(DeserializedObject, size)``.
                Buffer is auto-released internally after deserialization.
        """
        if self._closed:
            raise RuntimeError("Cannot get data: MooncakeTransferEngineConnector is closed")

        get_key = self._make_key(get_key, from_stage, to_stage)

        _t0 = _time_mod.perf_counter()

        # If no metadata provided, try to query from sender
        if not metadata:
            # Must insert sender info before using get() without metadata.
            if not self.sender_host or not self.sender_zmq_port or str(self.sender_host).lower() == "auto":
                raise RuntimeError(
                    f"get(metadata=None) requires sender info to be resolved, "
                    f"but sender_host={self.sender_host!r}, sender_zmq_port={self.sender_zmq_port!r}. "
                    f"Call update_sender_info(host, port) before using get() without metadata."
                )
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
            logger.error(
                f"Invalid metadata for {get_key}: source_host={src_host!r}, "
                f"source_port={src_port!r}. Cannot establish RDMA connection."
            )
            return None

        if data_size == 0:
            logger.warning(f"Skipping get for {get_key}: data_size is 0 (metadata={metadata})")
            return None

        # 1. Allocate Destination Buffer from Pool
        try:
            offset = self.allocator.alloc(data_size)
            recv_buffer = ManagedBuffer(self.allocator, offset, data_size, self.pool)
            dst_ptr = self.base_ptr + offset
        except MemoryError:
            logger.error(f"Failed to allocate {data_size} bytes in receive pool")
            return None

        _t2 = _time_mod.perf_counter()
        _alloc_ms = (_t2 - _t1) * 1000

        # 2. Prepare Handshake
        agent_meta = MooncakeAgentMetadata(
            remote_hostname=self.host,
            remote_port=self.rpc_port,
            request_id=get_key,
            dst_addrs=[dst_ptr],
            lengths=[data_size],
        )

        # 3. ZMQ Transaction (uses cached socket to avoid TCP reconnection)
        # Timeout scales with data size: base 30s + 5s per 100MB.
        # Large RDMA transfers (esp. loopback or cross-NIC) can take tens of
        # seconds; if we time out too early the receiver closes and deregisters
        # its memory, causing the sender's in-flight write to fail with ret=-1.
        _base_timeout_ms = 30000
        _size_timeout_ms = max(0, (data_size // (100 * 1024 * 1024))) * 5000
        _total_timeout_ms = _base_timeout_ms + _size_timeout_ms
        zmq_addr = f"tcp://{src_host}:{src_port}"
        req_socket = self._get_req_socket(zmq_addr, timeout_ms=_total_timeout_ms)

        try:
            req_socket.send(msgspec.msgpack.encode(agent_meta))
            resp = req_socket.recv()

            _t3 = _time_mod.perf_counter()
            _rdma_ms = (_t3 - _t2) * 1000

            if resp == TRANS_DONE:
                # Success
                # Ensure data is visible on GPU
                # Note: RDMA write visibility on GPU usually requires some form of fence/sync.
                # torch.cuda.synchronize() is a heavy hammer but safe.
                # Ideally Mooncake engine provides a way to poll for completion that guarantees visibility.
                # TODO(wzliu): Replace synchronize with cuda event in the future for better performance.
                if self.pool.is_cuda:
                    with torch.cuda.device(self.pool.device):
                        torch.cuda.current_stream().synchronize()

                _t4 = _time_mod.perf_counter()
                _sync_ms = (_t4 - _t3) * 1000

                if is_fast_path:
                    # Return ManagedBuffer directly for ALL fast_path data
                    # (both bytes and tensor payloads).  Caller is responsible
                    # for consuming the data and calling release() afterwards.
                    # This avoids the expensive to_bytes() copy (~54ms for 115MB).
                    #
                    # Usage patterns for callers:
                    #   - Zero-copy read: mv = memoryview(buf.tensor.numpy())
                    #   - Copy to GPU:    buf.tensor.to(device); buf.release()
                    #   - Fallback:       data = buf.to_bytes(); buf.release()
                    _t5 = _time_mod.perf_counter()
                    _total_ms = (_t5 - _t0) * 1000
                    _mbps = (data_size / 1024 / 1024) / (_total_ms / 1000) if _total_ms > 0 else 0
                    logger.info(
                        f"[RDMA GET] {get_key}: query={_query_ms:.1f}ms, alloc={_alloc_ms:.1f}ms, "
                        f"rdma={_rdma_ms:.1f}ms, sync={_sync_ms:.1f}ms, "
                        f"total={_total_ms:.1f}ms, {_mbps:.1f} MB/s (fast_path, zero-copy)"
                    )
                    self._metrics["gets"] += 1
                    self._metrics["bytes_transferred"] += data_size
                    return recv_buffer, data_size
                else:
                    # If it was a serialized object or generic bytes, we assume standard Omni behavior:
                    # Deserialize and return object. This requires a copy (to_bytes).
                    # We MUST release the buffer after deserialization.
                    try:
                        _t_copy_start = _time_mod.perf_counter()
                        raw_bytes = recv_buffer.to_bytes()
                        _t_copy_end = _time_mod.perf_counter()
                        _copy_ms = (_t_copy_end - _t_copy_start) * 1000

                        _t_deser_start = _time_mod.perf_counter()
                        val = OmniSerializer.deserialize(raw_bytes)
                        _t_deser_end = _time_mod.perf_counter()
                        _deser_ms = (_t_deser_end - _t_deser_start) * 1000

                        _total_ms = (_t_deser_end - _t0) * 1000
                        _mbps = (data_size / 1024 / 1024) / (_total_ms / 1000) if _total_ms > 0 else 0
                        logger.info(
                            f"[RDMA GET] {get_key}: query={_query_ms:.1f}ms, alloc={_alloc_ms:.1f}ms, "
                            f"rdma={_rdma_ms:.1f}ms, sync={_sync_ms:.1f}ms, copy={_copy_ms:.1f}ms, "
                            f"deser={_deser_ms:.1f}ms, total={_total_ms:.1f}ms, {_mbps:.1f} MB/s"
                        )
                        self._metrics["gets"] += 1
                        self._metrics["bytes_transferred"] += data_size
                        return val, data_size
                    finally:
                        recv_buffer.release()
            else:
                self._metrics["errors"] += 1
                logger.error(f"RDMA Get failed: received {resp} instead of TRANS_DONE")
                recv_buffer.release()
                return None
        except Exception as e:
            # Socket may be stuck after timeout; discard it
            self._invalidate_req_socket(zmq_addr)
            self._metrics["timeouts"] += 1
            logger.error(f"RDMA Get error: {e}", exc_info=True)
            recv_buffer.release()
            return None

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
            item = self._local_buffers.pop(request_id, None)
            if item:
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
        """
        Gracefully shutdown the connector and release all resources.
        This method should be called when the connector is no longer needed.
        Idempotent: safe to call multiple times.
        """
        # Idempotent guard — use getattr for safety: if __init__ raised before
        # setting _closed, __del__ → close() would hit AttributeError.
        if getattr(self, "_closed", True):
            return
        self._closed = True

        logger.info("Closing MooncakeTransferEngineConnector...")

        # 1. Signal listener thread to stop
        self._stop_event.set()

        # 2. Wait for listener thread to finish (only if sender mode)
        if self._listener_thread is not None and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)
            if self._listener_thread.is_alive():
                logger.warning("Listener thread did not stop gracefully")

        # 3. Shutdown sender executor (may be None if __init__ failed early)
        if self._sender_executor is not None:
            self._sender_executor.shutdown(wait=True, cancel_futures=False)

        # 4. Release all pending buffers
        with self._local_buffers_lock:
            for req_id, item in list(self._local_buffers.items()):
                _, _, holder, should_release, _, _ = item
                if should_release and isinstance(holder, ManagedBuffer):
                    holder.release()
            self._local_buffers.clear()

        # 5. Close thread-local cached REQ sockets (for the calling thread).
        #    Sockets cached in *other* threads are not directly accessible here
        #    because they live in per-thread ``threading.local()`` storage.
        #    They will be forcefully closed when ``zmq_ctx.term()`` is called
        #    in step 7 below, which is the intended cleanup path.
        cache: dict[str, zmq.Socket] = getattr(self._req_local, "cache", None)  # type: ignore[assignment]
        if cache:
            for addr, sock in cache.items():
                try:
                    sock.close(linger=0)
                except Exception:
                    pass  # Best-effort; zmq_ctx.term() below will reclaim
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
                self.zmq_ctx.term()
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
        """Reclaim buffers older than ``_BUFFER_TTL_SECONDS``.
        Prevents permanent memory leaks when a receiver crashes or times out
        without ever pulling the data.
        TODO(wzliu): In extreme rare case, long transfer time, there might exist
        TTL cleanup vs in-flight transfer conflict, which will be handled in the next PR.
        """
        now = _time_mod.monotonic()
        with self._local_buffers_lock:
            stale_keys = [k for k, v in self._local_buffers.items() if now - v[5] > _BUFFER_TTL_SECONDS]
            for k in stale_keys:
                item = self._local_buffers.pop(k)
                _, _, holder, should_release, _, _ = item
                if should_release and isinstance(holder, ManagedBuffer):
                    holder.release()
                logger.warning(f"TTL expired ({_BUFFER_TTL_SECONDS}s): cleaned up stale buffer for {k}")

    def _zmq_listener_loop(self):
        socket = self.zmq_ctx.socket(zmq.ROUTER)
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
            while not self._stop_event.is_set():
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
                            # Submit to thread pool
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
                socket.close(linger=0)
            except Exception:
                pass  # Best-effort cleanup during listener shutdown

    def _handle_pull_request(self, response_queue: queue.Queue, notify_addr: str, identity, payload):
        """
        Handle pull request or query request in worker thread.
        Results are put into response_queue and listener is notified via inproc.
        """
        try:
            # Check if this is a query request
            if payload.startswith(QUERY_INFO):
                self._handle_query_request(response_queue, notify_addr, identity, payload[len(QUERY_INFO) :])
                return

            # Normal RDMA transfer request
            meta = msgspec.msgpack.decode(payload, type=MooncakeAgentMetadata)

            with self._local_buffers_lock:
                item = self._local_buffers.get(meta.request_id)

            if not item:
                response_queue.put((identity, TRANS_ERROR))
                self._notify_listener(notify_addr)
                return

            src_addrs, src_lengths, _, _, _, _ = item
            remote_session = f"{meta.remote_hostname}:{meta.remote_port}"

            # RDMA Write
            ret = self.engine.batch_transfer_sync_write(remote_session, src_addrs, meta.dst_addrs, src_lengths)

            if ret == 0:
                self.cleanup(meta.request_id)
                response_queue.put((identity, TRANS_DONE))
            else:
                # Keep buffer in _local_buffers so receiver can retry on transient failures.
                # Buffer will be cleaned up when: (a) a retry succeeds, (b) close() is called,
                # or (c) a future TTL mechanism reclaims stale entries.
                logger.warning(
                    f"RDMA write failed for {meta.request_id} to {remote_session} "
                    f"(ret={ret}). Buffer retained for retry."
                )
                response_queue.put((identity, TRANS_ERROR))

        except Exception as e:
            logger.error(f"Push failed: {e}")
            response_queue.put((identity, TRANS_ERROR))

        # Notify listener thread that response is ready
        self._notify_listener(notify_addr)

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
