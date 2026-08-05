# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import socket
import threading
import time as _time_mod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import msgspec
import torch
import zmq

from vllm_omni.distributed.omni_connectors.connectors.base import OmniConnectorBase
from vllm_omni.distributed.omni_connectors.utils.env import (
    AUTO_DEVICE_VALUES,
    AUTO_HOST_VALUES,
    expand_env_int,
    expand_env_value,
)
from vllm_omni.distributed.omni_connectors.utils.local_rank import get_connector_local_rank
from vllm_omni.distributed.omni_connectors.utils.logging import get_connector_logger
from vllm_omni.distributed.omni_connectors.utils.memory_pool import BufferAllocator, ManagedBuffer
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

logger = get_connector_logger(__name__)

try:
    from yr.datasystem import TransferEngine
except ImportError:
    TransferEngine = None

_BUFFER_TTL_SECONDS = 300

QUERY_INFO = b"query_info"
CLEANUP_KEY = b"cleanup_key"
ACK_OK = b"ack_ok"
ACK_ERROR = b"ack_error"
INFO_NOT_FOUND = b"info_not_found"


@dataclass
class QueryRequest:
    """Request to query metadata for a specific key."""

    request_id: str


@dataclass
class CleanupRequest:
    """Response containing metadata for a request."""

    request_id: str


@dataclass
class QueryResponse:
    request_id: str
    rpc_port: int
    data_size: int
    is_fast_path: bool
    src_addrs: list[int]
    lengths: list[int]


def _resolve_device_name(configured_device: Any, protocol: str) -> str:
    device_name = str(expand_env_value(configured_device or "auto")).strip()
    if device_name.lower() not in AUTO_DEVICE_VALUES:
        return device_name
    if protocol.lower() in {"ascend", "hccl", "npu"}:
        return f"npu:{get_connector_local_rank()}"
    return ""


def _resolve_pool_device(configured_device: Any) -> str:
    pool_device = str(expand_env_value(configured_device or "npu")).strip().lower()
    if pool_device in AUTO_DEVICE_VALUES:
        return "npu"
    if not (pool_device == "npu" or pool_device.startswith("npu:")):
        raise ValueError(
            f"YuanrongTransferEngineConnector requires an NPU memory pool. Got memory_pool_device={pool_device!r}."
        )
    return pool_device


class YuanrongTransferEngineConnector(OmniConnectorBase):
    """Pull-based connector backed by Yuanrong transfer_engine."""

    supports_raw_data: bool = True

    def __init__(self, config: dict[str, Any]):
        if TransferEngine is None:
            raise ImportError("Yuanrong is not available.")

        self._closed = False
        self._bind_error: Exception | None = None
        self._stop_event = threading.Event()
        self._listener_thread: threading.Thread | None = None
        self._listener_ready = threading.Event()
        self._control_executor: ThreadPoolExecutor | None = None
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

        self.config = {key: expand_env_value(value) for key, value in config.items()}
        config = self.config

        raw_host_config = config.get("host")
        host_config = "auto" if raw_host_config is None else str(raw_host_config).strip()
        self.host = self._get_local_ip() if host_config.lower() in AUTO_HOST_VALUES else host_config
        self.zmq_port = self._resolve_port(config.get("zmq_port"), self.host, "zmq_port")
        self.rpc_port = self._resolve_port(config.get("rpc_port"), self.host, "rpc_port")
        self.protocol = str(config.get("protocol", "ascend"))
        self.device_name = _resolve_device_name(config.get("device_name", "auto"), self.protocol)
        self.pool_size = int(config.get("memory_pool_size", 1024**3))
        self.pool_device = _resolve_pool_device(config.get("memory_pool_device", "npu"))
        self.sender_host = config.get("sender_host")
        sender_zmq_port = config.get("sender_zmq_port")
        self.sender_zmq_port = self._resolve_optional_port(sender_zmq_port, "sender_zmq_port")

        role = str(config.get("role", "sender")).lower()
        if role not in {"sender", "receiver"}:
            raise ValueError(
                f"Invalid role={role!r} for YuanrongTransferEngineConnector. Expected 'sender' or 'receiver'."
            )
        self.can_put = role == "sender"

        self.engine = TransferEngine()
        local_endpoint = f"{self.host}:{self.rpc_port}"
        self._ensure_result_ok(
            self.engine.initialize(local_endpoint, self.protocol, self.device_name),
            f"initialize transfer engine on {local_endpoint}",
        )
        resolved_rpc_port = int(self.engine.get_rpc_port())
        if resolved_rpc_port > 0:
            self.rpc_port = resolved_rpc_port

        logger.info(
            "YuanrongTransferEngineConnector initialized at %s:%s (protocol=%s, device=%s)",
            self.host,
            self.rpc_port,
            self.protocol,
            self.device_name,
        )

        try:
            self.pool = torch.empty(self.pool_size, dtype=torch.uint8, device=self.pool_device)
            self.base_ptr = int(self.pool.data_ptr())
            self._ensure_result_ok(
                self.engine.register_memory(self.base_ptr, self.pool_size),
                f"register memory pool ({self.pool_size} bytes)",
            )
        except Exception as exc:
            logger.error("Failed to allocate/register transfer_engine pool: %s", exc)
            raise

        self.allocator = BufferAllocator(self.pool_size, alignment=4096)
        self.zmq_ctx = zmq.Context()
        self._control_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="yuanrong-transfer-control")

        if self.can_put:
            self._listener_thread = threading.Thread(target=self._zmq_listener_loop, daemon=True)
            self._listener_thread.start()
            self._listener_ready.wait(timeout=1.0)
            if self._bind_error is not None:
                raise RuntimeError(
                    f"YuanrongTransferEngineConnector failed to bind ZMQ on "
                    f"{self.host}:{self.zmq_port}: {self._bind_error}"
                ) from self._bind_error
        else:
            logger.info(
                "YuanrongTransferEngineConnector started in receiver mode (sender_host=%r, sender_zmq_port=%r)",
                self.sender_host,
                self.sender_zmq_port,
            )

    @staticmethod
    def _ensure_result_ok(result: Any, action: str) -> None:
        if result.is_error():
            raise RuntimeError(f"{action} failed: {result.to_string()}")

    @staticmethod
    def _resolve_optional_port(configured_port: Any, field_name: str) -> int | None:
        configured_port = expand_env_value(configured_port, strict=True, field_name=field_name)
        if configured_port is None or str(configured_port).strip().lower() in {"", "auto"}:
            return None
        return expand_env_int(configured_port, field_name)

    @classmethod
    def _resolve_port(cls, configured_port: Any, host: str, field_name: str) -> int:
        resolved_port = cls._resolve_optional_port(configured_port, field_name)
        if resolved_port is not None:
            return resolved_port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def _get_local_ip() -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                return sock.getsockname()[0]
        except Exception:
            try:
                return socket.gethostbyname(socket.gethostname())
            except Exception:
                return "127.0.0.1"

    @staticmethod
    def _sync_tensor_device(tensor: torch.Tensor | None) -> None:
        if tensor is None:
            return
        device = getattr(tensor, "device", None)
        if device is None or getattr(device, "type", "cpu") == "cpu":
            return

        backend = getattr(torch, device.type, None)
        if backend is None:
            return

        current_stream = getattr(backend, "current_stream", None)
        if callable(current_stream):
            try:
                current_stream(device).synchronize()
                return
            except TypeError:
                try:
                    current_stream().synchronize()
                    return
                except Exception:
                    pass
            except Exception:
                pass

        synchronize = getattr(backend, "synchronize", None)
        if callable(synchronize):
            try:
                synchronize(device)
            except TypeError:
                synchronize()

    def get_connection_info(self) -> dict[str, Any]:
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
    ) -> None:
        self.sender_host = sender_host
        self.sender_zmq_port = int(sender_zmq_port)

    def _get_req_socket(self, zmq_addr: str, timeout_ms: int) -> zmq.Socket:
        cache: dict[str, zmq.Socket] = getattr(self._req_local, "cache", None)  # type: ignore[assignment]
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
        cache: dict[str, zmq.Socket] = getattr(self._req_local, "cache", None)  # type: ignore[assignment]
        if cache is None:
            return
        sock = cache.pop(zmq_addr, None)
        if sock is not None:
            try:
                sock.close(linger=0)
            except Exception:
                pass

    def put(self, from_stage: str, to_stage: str, put_key: str, data: Any) -> tuple[bool, int, dict[str, Any] | None]:
        if self._closed:
            raise RuntimeError("Cannot put data: YuanrongTransferEngineConnector is closed")

        if not self.can_put:
            self._metrics["errors"] += 1
            logger.warning("Rejecting put for %s: connector is in receiver mode", put_key)
            return False, 0, None

        put_key = self._make_key(put_key, from_stage, to_stage)

        try:
            src_addr = 0
            size = 0
            holder = None
            is_fast_path = True
            should_release_holder = False

            if isinstance(data, bytes) and len(data) == 0:
                self._metrics["errors"] += 1
                logger.warning("Rejecting put for %s: empty bytes payload", put_key)
                return False, 0, None
            if isinstance(data, torch.Tensor) and data.nbytes == 0:
                self._metrics["errors"] += 1
                logger.warning("Rejecting put for %s: zero-size tensor", put_key)
                return False, 0, None

            if not isinstance(data, (ManagedBuffer, torch.Tensor, bytes)):
                data = OmniSerializer.serialize(data)
                is_fast_path = False

            if isinstance(data, ManagedBuffer):
                if data.pool_tensor.data_ptr() != self.pool.data_ptr():
                    logger.warning("ManagedBuffer from different pool detected. Falling back to copy path.")
                    data = data.tensor.contiguous()
                else:
                    src_addr = self.base_ptr + data.offset
                    size = data.size
                    holder = data
                    should_release_holder = False

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
                    should_release_holder = True
                except MemoryError:
                    self._metrics["errors"] += 1
                    logger.error("Pool exhausted, cannot put data size %s", size)
                    return False, 0, None

                try:
                    dst_tensor = holder.tensor
                    if isinstance(data, torch.Tensor):
                        if not data.is_contiguous():
                            data = data.contiguous()
                        src_view = data.view(torch.uint8).flatten()
                        dst_tensor.copy_(src_view, non_blocking=src_view.device != dst_tensor.device)
                        self._sync_tensor_device(src_view)
                        self._sync_tensor_device(dst_tensor)
                    else:
                        dst_tensor.copy_(tensor_data)
                        self._sync_tensor_device(dst_tensor)
                except Exception as exc:
                    holder.release()
                    self._metrics["errors"] += 1
                    logger.error("Failed to copy data to pool: %s", exc)
                    return False, 0, None

                src_addr = self.base_ptr + offset

            if size <= 0:
                if should_release_holder and isinstance(holder, ManagedBuffer):
                    holder.release()
                self._metrics["errors"] += 1
                logger.warning("Rejecting put for %s: final payload size is %s", put_key, size)
                return False, 0, None

            with self._local_buffers_lock:
                old_item = self._local_buffers.pop(put_key, None)
                if old_item:
                    _, _, old_holder, old_should_release, _, _ = old_item
                    if old_should_release and isinstance(old_holder, ManagedBuffer):
                        old_holder.release()
                self._local_buffers[put_key] = (
                    [src_addr],
                    [size],
                    holder,
                    should_release_holder,
                    is_fast_path,
                    _time_mod.monotonic(),
                )

            metadata = {
                "source_host": self.host,
                "source_port": self.zmq_port,
                "source_rpc_port": self.rpc_port,
                "source_addrs": [src_addr],
                "lengths": [size],
                "data_size": size,
                "is_fast_path": is_fast_path,
            }

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size
            return True, size, metadata
        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error("transfer_engine put failed for %s: %s", put_key, exc, exc_info=True)
            return False, 0, None

    def _query_metadata_at(self, get_key: str, host: str, port: int) -> dict[str, Any] | None:
        zmq_addr = f"tcp://{host}:{port}"
        req_socket = self._get_req_socket(zmq_addr, timeout_ms=5000)

        try:
            query = QueryRequest(request_id=get_key)
            req_socket.send(QUERY_INFO + msgspec.msgpack.encode(query))
            resp = req_socket.recv()
            if resp == INFO_NOT_FOUND:
                return None

            query_resp = msgspec.msgpack.decode(resp, type=QueryResponse)
            return {
                "source_host": host,
                "source_port": port,
                "source_rpc_port": query_resp.rpc_port,
                "source_addrs": query_resp.src_addrs,
                "lengths": query_resp.lengths,
                "data_size": query_resp.data_size,
                "is_fast_path": query_resp.is_fast_path,
            }
        except Exception as exc:
            self._invalidate_req_socket(zmq_addr)
            logger.debug("Failed to query metadata at %s for %s: %s", zmq_addr, get_key, exc)
            return None

    @classmethod
    def _buffer_to_bytes(cls, buffer: ManagedBuffer) -> bytes:
        tensor = buffer.tensor
        cls._sync_tensor_device(tensor)
        if getattr(tensor.device, "type", "cpu") != "cpu":
            tensor = tensor.cpu()
        return tensor.numpy().tobytes()

    def _request_sender_cleanup(self, request_id: str, source_host: str | None, source_port: int | None) -> bool:
        if not source_host or not source_port:
            return False

        zmq_addr = f"tcp://{source_host}:{source_port}"
        req_socket = self._get_req_socket(zmq_addr, timeout_ms=3000)

        try:
            payload = CleanupRequest(request_id=request_id)
            req_socket.send(CLEANUP_KEY + msgspec.msgpack.encode(payload))
            return req_socket.recv() == ACK_OK
        except Exception as exc:
            self._invalidate_req_socket(zmq_addr)
            logger.warning("Failed to cleanup sender buffer for %s: %s", request_id, exc)
            return False

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        if self._closed:
            raise RuntimeError("Cannot get data: YuanrongTransferEngineConnector is closed")

        get_key = self._make_key(get_key, from_stage, to_stage)
        t0 = _time_mod.perf_counter()

        if not metadata:
            if not self.sender_host or self.sender_zmq_port is None or str(self.sender_host).lower() == "auto":
                self._metrics["errors"] += 1
                raise RuntimeError(
                    "get(metadata=None) requires sender_host and sender_zmq_port to be resolved. "
                    "Call update_sender_info(host, port) before polling."
                )
            metadata = self._query_metadata_at(get_key, str(self.sender_host), int(self.sender_zmq_port))
            if not metadata:
                return None
        elif not (metadata.get("source_host") and metadata.get("source_rpc_port") and metadata.get("source_addrs")):
            partial_host = metadata.get("source_host")
            partial_port = metadata.get("source_port")
            if not partial_host or not partial_port:
                self._metrics["errors"] += 1
                logger.warning(
                    "get(%s): partial metadata missing source_host/source_port. metadata=%s",
                    get_key,
                    metadata,
                )
                return None
            queried = self._query_metadata_at(get_key, str(partial_host), int(partial_port))
            if not queried:
                return None
            metadata = queried

        t1 = _time_mod.perf_counter()
        query_ms = (t1 - t0) * 1000

        source_host = metadata.get("source_host")
        source_port = metadata.get("source_port")
        source_rpc_port = metadata.get("source_rpc_port")
        source_addrs = list(metadata.get("source_addrs") or [])
        lengths = list(metadata.get("lengths") or [])
        data_size = int(metadata.get("data_size", 0))
        is_fast_path = bool(metadata.get("is_fast_path", False))

        if not source_host or not source_rpc_port or str(source_host).lower() == "auto":
            self._metrics["errors"] += 1
            logger.error(
                "Invalid metadata for %s: source_host=%r, source_rpc_port=%r",
                get_key,
                source_host,
                source_rpc_port,
            )
            return None
        if not source_addrs:
            self._metrics["errors"] += 1
            logger.error("Invalid metadata for %s: missing source_addrs", get_key)
            return None
        if not lengths:
            lengths = [data_size]
        if len(source_addrs) != len(lengths):
            self._metrics["errors"] += 1
            logger.error(
                "Invalid metadata for %s: source_addrs/lengths mismatch (%s vs %s)",
                get_key,
                len(source_addrs),
                len(lengths),
            )
            return None

        total_size = sum(int(length) for length in lengths)
        if data_size <= 0:
            data_size = total_size
        if data_size <= 0:
            self._metrics["errors"] += 1
            logger.warning("Skipping get for %s: data_size is 0", get_key)
            return None

        try:
            offset = self.allocator.alloc(data_size)
            recv_buffer = ManagedBuffer(self.allocator, offset, data_size, self.pool)
            base_dst_ptr = self.base_ptr + offset
        except MemoryError:
            self._metrics["errors"] += 1
            logger.error("Failed to allocate %s bytes in receive pool", data_size)
            return None

        dst_addrs: list[int] = []
        cursor = 0
        for length in lengths:
            dst_addrs.append(base_dst_ptr + cursor)
            cursor += int(length)
        if cursor != data_size:
            recv_buffer.release()
            self._metrics["errors"] += 1
            logger.error("Length sum mismatch for %s: lengths_sum=%s, data_size=%s", get_key, cursor, data_size)
            return None

        t2 = _time_mod.perf_counter()
        alloc_ms = (t2 - t1) * 1000
        target_hostname = f"{source_host}:{int(source_rpc_port)}"

        try:
            result = self.engine.batch_transfer_sync_read(target_hostname, dst_addrs, source_addrs, lengths)
            self._ensure_result_ok(result, f"read from {target_hostname}")
            self._sync_tensor_device(recv_buffer.tensor)

            t3 = _time_mod.perf_counter()
            read_ms = (t3 - t2) * 1000

            cleanup_ok = self._request_sender_cleanup(get_key, source_host, int(source_port) if source_port else None)
            if not cleanup_ok:
                logger.warning("Sender-side cleanup not confirmed for %s", get_key)

            if is_fast_path:
                total_ms = (_time_mod.perf_counter() - t0) * 1000
                mbps = (data_size / 1024 / 1024) / (total_ms / 1000) if total_ms > 0 else 0
                logger.info(
                    "[YR GET] %s: query=%.1fms, alloc=%.1fms, read=%.1fms, total=%.1fms, %.1f MB/s "
                    "(fast_path, zero-copy)",
                    get_key,
                    query_ms,
                    alloc_ms,
                    read_ms,
                    total_ms,
                    mbps,
                )
                self._metrics["gets"] += 1
                self._metrics["bytes_transferred"] += data_size
                return recv_buffer, data_size

            try:
                copy_start = _time_mod.perf_counter()
                raw_bytes = self._buffer_to_bytes(recv_buffer)
                copy_end = _time_mod.perf_counter()
                value = OmniSerializer.deserialize(raw_bytes)
                end = _time_mod.perf_counter()
                total_ms = (end - t0) * 1000
                mbps = (data_size / 1024 / 1024) / (total_ms / 1000) if total_ms > 0 else 0
                logger.info(
                    "[YR GET] %s: query=%.1fms, alloc=%.1fms, read=%.1fms, copy=%.1fms, total=%.1fms, %.1f MB/s",
                    get_key,
                    query_ms,
                    alloc_ms,
                    read_ms,
                    (copy_end - copy_start) * 1000,
                    total_ms,
                    mbps,
                )
                self._metrics["gets"] += 1
                self._metrics["bytes_transferred"] += data_size
                return value, data_size
            finally:
                recv_buffer.release()
        except Exception as exc:
            self._metrics["errors"] += 1
            self._metrics["timeouts"] += 1
            logger.error("transfer_engine get failed for %s: %s", get_key, exc, exc_info=True)
            recv_buffer.release()
            return None

    def cleanup(self, request_id: str, from_stage: str | None = None, to_stage: str | None = None) -> None:
        if (from_stage is None) != (to_stage is None):
            raise ValueError(
                "cleanup() requires both from_stage and to_stage, or neither. "
                f"Got from_stage={from_stage!r}, to_stage={to_stage!r}"
            )
        if from_stage is not None and to_stage is not None:
            request_id = self._make_key(request_id, from_stage, to_stage)

        with self._local_buffers_lock:
            item = self._local_buffers.pop(request_id, None)
            if not item:
                return
            _, _, holder, should_release, _, _ = item
            if should_release and isinstance(holder, ManagedBuffer):
                holder.release()

    def health(self) -> dict[str, Any]:
        if self._closed:
            return {"status": "unhealthy", "error": "Connector is closed"}

        return {
            "status": "healthy",
            "host": self.host,
            "zmq_port": self.zmq_port,
            "rpc_port": self.rpc_port,
            "protocol": self.protocol,
            "pool_device": self.pool_device,
            "pool_size": self.pool_size,
            **self._metrics,
        }

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True

        self._stop_event.set()

        if self._listener_thread is not None and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)

        if self._control_executor is not None:
            self._control_executor.shutdown(wait=True, cancel_futures=False)

        with self._local_buffers_lock:
            for _, item in list(self._local_buffers.items()):
                _, _, holder, should_release, _, _ = item
                if should_release and isinstance(holder, ManagedBuffer):
                    holder.release()
            self._local_buffers.clear()

        cache: dict[str, zmq.Socket] = getattr(self._req_local, "cache", None)  # type: ignore[assignment]
        if cache:
            for sock in cache.values():
                try:
                    sock.close(linger=0)
                except Exception:
                    pass
            cache.clear()

        try:
            if hasattr(self, "engine") and hasattr(self.engine, "unregister_memory"):
                result = self.engine.unregister_memory(self.base_ptr)
                if result.is_error():
                    logger.warning("Failed to unregister memory: %s", result.to_string())
        except Exception as exc:
            logger.warning("Failed to unregister memory: %s", exc)

        try:
            if hasattr(self, "engine") and hasattr(self.engine, "finalize"):
                result = self.engine.finalize()
                if result.is_error():
                    logger.warning("Failed to finalize transfer engine: %s", result.to_string())
        except Exception as exc:
            logger.warning("Failed to finalize transfer engine: %s", exc)

        try:
            if hasattr(self, "zmq_ctx"):
                self.zmq_ctx.term()
        except Exception as exc:
            logger.warning("Failed to terminate ZMQ context: %s", exc)

        self.pool = None

    def _cleanup_stale_buffers(self) -> None:
        now = _time_mod.monotonic()
        with self._local_buffers_lock:
            stale_keys = [k for k, v in self._local_buffers.items() if now - v[5] > _BUFFER_TTL_SECONDS]
            for key in stale_keys:
                item = self._local_buffers.pop(key)
                _, _, holder, should_release, _, _ = item
                if should_release and isinstance(holder, ManagedBuffer):
                    holder.release()
                logger.warning("TTL expired (%ss): cleaned up stale buffer for %s", _BUFFER_TTL_SECONDS, key)

    def _zmq_listener_loop(self) -> None:
        socket_obj = self.zmq_ctx.socket(zmq.ROUTER)
        try:
            socket_obj.bind(f"tcp://{self.host}:{self.zmq_port}")
        except zmq.ZMQError as exc:
            logger.error("ZMQ bind failed on %s:%s: %s", self.host, self.zmq_port, exc)
            self.can_put = False
            self._bind_error = exc
            self._listener_ready.set()
            return

        self._listener_ready.set()

        notify_addr = f"inproc://notify-{id(self)}"
        notify_recv = self.zmq_ctx.socket(zmq.PULL)
        notify_recv.bind(notify_addr)

        poller = zmq.Poller()
        poller.register(socket_obj, zmq.POLLIN)
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
                            socket_obj.send_multipart([identity, b"", response])
                        except queue.Empty:
                            break

                    now = _time_mod.monotonic()
                    if now - self._last_ttl_check >= 10.0:
                        self._last_ttl_check = now
                        self._cleanup_stale_buffers()

                    if socket_obj in events:
                        frames = socket_obj.recv_multipart()
                        if len(frames) >= 2:
                            self._control_executor.submit(
                                self._handle_control_request,
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
                socket_obj.close(linger=0)
            except Exception:
                pass

    def _handle_control_request(self, response_queue: queue.Queue, notify_addr: str, identity, payload: bytes) -> None:
        try:
            if payload.startswith(QUERY_INFO):
                self._handle_query_request(response_queue, notify_addr, identity, payload[len(QUERY_INFO) :])
                return
            if payload.startswith(CLEANUP_KEY):
                self._handle_cleanup_request(response_queue, notify_addr, identity, payload[len(CLEANUP_KEY) :])
                return

            response_queue.put((identity, ACK_ERROR))
        except Exception as exc:
            logger.error("Control request failed: %s", exc)
            response_queue.put((identity, ACK_ERROR))

        self._notify_listener(notify_addr)

    def _handle_query_request(self, response_queue: queue.Queue, notify_addr: str, identity, payload: bytes) -> None:
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
                    rpc_port=self.rpc_port,
                    data_size=sum(int(length) for length in src_lengths),
                    is_fast_path=is_fast_path,
                    src_addrs=src_addrs,
                    lengths=src_lengths,
                )
                response_queue.put((identity, msgspec.msgpack.encode(resp)))
        except Exception as exc:
            logger.error("Query request failed: %s", exc)
            response_queue.put((identity, INFO_NOT_FOUND))

        self._notify_listener(notify_addr)

    def _handle_cleanup_request(self, response_queue: queue.Queue, notify_addr: str, identity, payload: bytes) -> None:
        try:
            request = msgspec.msgpack.decode(payload, type=CleanupRequest)
            self.cleanup(request.request_id)
            response_queue.put((identity, ACK_OK))
        except Exception as exc:
            logger.error("Cleanup request failed: %s", exc)
            response_queue.put((identity, ACK_ERROR))

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
