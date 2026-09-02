# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import msgspec
import torch
import zmq

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

_SCHEMA_VERSION = 1
_KIND_TENSORS = "tensors"
_KIND_STRUCTURED = "structured"
_KIND_OBJECT = "object"
_INIT_AGENT = "NIXL_INIT_AGENT"
_TENSOR_MARKER = "__nixl_tensor_index__"
_TUPLE_MARKER = "__nixl_tuple__"

# Handshake control-plane messages. A ``put`` payload is only readable once the
# consumer knows the producer's NIXL agent metadata and memory descriptors, so a
# consumer that was not handed that metadata out-of-band asks for it here.
_GET_META_MSG = b"nixl_get_meta"
_XFER_DONE_MSG = b"nixl_xfer_done"
_META_NOT_FOUND = b"nixl_meta_not_found"
_ACK = b"nixl_ack"


@dataclass
class _DeferredTransfer:
    tensors: list[torch.Tensor]
    registrations: list[Any] = field(default_factory=list)
    dlists: list[Any] = field(default_factory=list)
    handles: list[Any] = field(default_factory=list)
    remote_agent: Any = None


@dataclass
class _PendingPayload:
    tensors: list[torch.Tensor]
    registrations: list[Any]
    deadline: float


class NixlConnector(OmniConnectorBase):
    """OmniConnector backed by vLLM's native NIXL wrapper.

    This connector intentionally depends on vLLM's optional NIXL integration
    (``vllm.distributed.nixl_utils``). It transfers raw tensor payloads
    directly through NIXL READ operations. Non-tensor Python
    payloads are serialized with OmniSerializer, packed into a uint8 CPU tensor,
    and moved through the same NIXL path.

    A NIXL READ needs the producer's agent metadata and memory descriptors,
    which ``put`` returns to its caller. Callers that can forward that
    dictionary to the consumer (for example the diffusion stage handle) keep
    working unchanged. Callers that cannot -- notably the generic stage payload
    path, which drops the ``put`` metadata and invokes ``get`` with
    ``metadata=None`` -- rely on the ZMQ handshake enabled by setting
    ``zmq_port``: the producer serves its metadata from a ROUTER socket and the
    consumer fetches it by key.
    """

    supports_raw_data: bool = True

    def __init__(self, config: dict[str, Any]):
        self.config = dict(config or {})
        self.stage_id = int(self.config.get("stage_id", 0))
        self._closed = True
        self._registered_descs: list[Any] = []
        self._pending: dict[str, _PendingPayload] = {}
        self._published: dict[str, dict[str, Any]] = {}
        self._state_lock = threading.RLock()
        self._remote_agents: list[str] = []
        self._metrics: dict[str, int] = {
            "puts": 0,
            "gets": 0,
            "errors": 0,
            "bytes_transferred": 0,
        }

        from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config

        if NixlWrapper is None:
            raise RuntimeError("NIXL is not available. Install the optional nixl/rixl package to use NixlConnector.")

        backends = self.config.get("backends", ["UCX"])
        self._backends = list(backends) if isinstance(backends, (list, tuple)) else [str(backends)]
        if nixl_agent_config is None:
            agent_config = None
        else:
            non_ucx_backends = [backend for backend in self._backends if backend != "UCX"]
            if non_ucx_backends:
                agent_config = nixl_agent_config(backends=self._backends, capture_telemetry=False)
            else:
                num_threads = int(self.config.get("num_threads", 4))
                agent_config = nixl_agent_config(num_threads=num_threads, capture_telemetry=False)

        self._agent = NixlWrapper(str(self.config.get("agent_name", uuid.uuid4())), agent_config)
        self._receive_device = self._parse_device(self.config.get("receive_device"))
        self._default_memory_type = self.config.get("memory_type")
        # The lease bounds how long a ``put`` payload stays registered while it
        # waits to be read. A consumer stage may legitimately queue for a long
        # time (e.g. video diffusion with concurrency above the replica count),
        # so the default must be far larger than any plausible queueing delay.
        # The environment variables intentionally take precedence over the
        # deployment YAML, which ships a fixed 300s lease that is too short for
        # long-queueing pipelines and cannot be overridden per deployment.
        self._lease_seconds = float(
            os.environ.get("VLLM_OMNI_NIXL_LEASE_S") or self.config.get("lease_seconds", 3600.0)
        )
        self._transfer_timeout_s = float(
            os.environ.get("VLLM_OMNI_NIXL_XFER_TIMEOUT_S") or self.config.get("transfer_timeout_s", 300.0)
        )
        self._poll_interval_s = float(self.config.get("poll_interval_s", 0.001))
        self._closed = False
        self._init_handshake()
        self._lease_wakeup = threading.Event()
        self._lease_thread: threading.Thread | None = None
        if self._role != "receiver":
            self._lease_thread = threading.Thread(
                target=self._lease_reaper_loop,
                name="nixl-lease-reaper",
                daemon=True,
            )
            self._lease_thread.start()
        self._transfer_wakeup = threading.Event()
        self._deferred_transfers: list[_DeferredTransfer] = []
        self._transfer_thread: threading.Thread | None = None
        if self._role != "sender":
            self._transfer_thread = threading.Thread(
                target=self._transfer_reaper_loop,
                name="nixl-transfer-reaper",
                daemon=True,
            )
            self._transfer_thread.start()

    def _init_handshake(self) -> None:
        """Set up the ZMQ control plane used when ``get`` receives no metadata.

        The handshake stays off unless ``zmq_port`` is configured, so
        deployments that forward the ``put`` metadata themselves keep their
        current behaviour and open no extra sockets.
        """
        role = self.config.get("role")
        self._role = str(role).lower() if role else None
        self._zmq_port = self.config.get("zmq_port")
        self._sender_host = self.config.get("sender_host")
        self._sender_zmq_port = self.config.get("sender_zmq_port")
        self._handshake_timeout_ms = int(self.config.get("handshake_timeout_ms", 5000))
        self._handshake_max_wait_s = float(self.config.get("handshake_max_wait_s", 60.0))
        self._handshake_retry_s = float(self.config.get("handshake_retry_s", 0.05))
        self._zmq_ctx: zmq.Context | None = None
        self._req_local = threading.local()
        self._listener_thread: threading.Thread | None = None
        self._listener_ready = threading.Event()
        self._stop_event = threading.Event()
        self._bind_error: BaseException | None = None
        self.host: str | None = None
        self._serving_handshake = False

        self._handshake_enabled = self._zmq_port is not None or self._sender_host is not None
        if not self._handshake_enabled:
            return

        self._zmq_ctx = zmq.Context()
        # A receiver only dials out, so it never needs a port of its own.
        if self._zmq_port is None or self._role == "receiver":
            return

        host_value = str(self.config.get("host", "auto"))
        self.host = (
            self._get_local_ip()
            if host_value.lower() == "auto" or host_value in {"", "*", "0.0.0.0", "::"}
            else host_value
        )
        self._zmq_port = int(self._zmq_port)
        self._listener_thread = threading.Thread(
            target=self._handshake_listener_loop, name="nixl-handshake", daemon=True
        )
        self._listener_thread.start()
        self._listener_ready.wait(timeout=5.0)
        if self._bind_error is not None:
            raise RuntimeError(
                f"NixlConnector failed to bind handshake socket on {self.host}:{self._zmq_port}"
            ) from self._bind_error
        self._serving_handshake = True
        logger.info("NixlConnector handshake listener bound on %s:%s", self.host, self._zmq_port)

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        if self._closed:
            raise RuntimeError("Cannot put data: NixlConnector is closed")

        try:
            self._cleanup_expired_pending()
            kind = _KIND_TENSORS
            if self._is_tensor_payload(data):
                tensors, tensor_specs = self._normalize_tensor_payload(data)
            elif self._contains_tensor(data):
                skeleton, payload_tensors = self._extract_tensor_leaves(data)
                header = self.serialize_obj(skeleton)
                header_tensor = torch.frombuffer(header, dtype=torch.uint8).clone().contiguous()
                tensors, tensor_specs = self._normalize_tensor_payload([header_tensor, *payload_tensors])
                kind = _KIND_STRUCTURED
            else:
                payload = self.serialize_obj(data)
                tensor = torch.frombuffer(payload, dtype=torch.uint8).clone().contiguous()
                tensors, tensor_specs = self._normalize_tensor_payload(tensor)
                kind = _KIND_OBJECT

            grouped_tensors: dict[str, list[tuple[int, torch.Tensor]]] = {}
            for tensor_index, tensor in enumerate(tensors):
                memory_type = self._resolve_memory_type(tensor)
                grouped_tensors.setdefault(memory_type, []).append((tensor_index, tensor))

            descriptor_groups = []
            registered_descs = []
            try:
                for memory_type, indexed_tensors in grouped_tensors.items():
                    tensor_indices = [tensor_index for tensor_index, _ in indexed_tensors]
                    regions = [self._tensor_region(tensor) for _, tensor in indexed_tensors]
                    reg_descs = self._agent.get_reg_descs(regions, memory_type)
                    self._agent.register_memory(reg_descs, backends=self._backends)
                    registered_descs.append(reg_descs)
                    descriptor_groups.append(
                        {
                            "memory_type": memory_type,
                            "tensor_indices": tensor_indices,
                            "regions": regions,
                        }
                    )
            except Exception:
                for reg_descs in registered_descs:
                    self._safe_call(self._agent.deregister_memory, reg_descs)
                raise
            with self._state_lock:
                self._registered_descs.extend(registered_descs)

            size = sum(spec["size"] for spec in tensor_specs)
            metadata = {
                "schema_version": _SCHEMA_VERSION,
                "kind": kind,
                "agent_metadata": self._agent.get_agent_metadata(),
                "descriptor_groups": descriptor_groups,
                "tensor_specs": tensor_specs,
                "size": size,
            }
            if self._serving_handshake:
                metadata["sender_host"] = self.host
                metadata["sender_zmq_port"] = self._zmq_port
            previous = self._take_pending(put_key)
            if previous is not None:
                self._release_pending(previous)
            with self._state_lock:
                self._pending[put_key] = _PendingPayload(
                    tensors=tensors,
                    registrations=registered_descs,
                    deadline=time.monotonic() + self._lease_seconds,
                )
                if self._serving_handshake:
                    self._published[put_key] = metadata
            self._lease_wakeup.set()
            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size
            logger.debug("NixlConnector put %s->%s key=%s size=%d", from_stage, to_stage, put_key, size)
            return True, size, metadata
        except Exception:
            self._metrics["errors"] += 1
            logger.error("NixlConnector put failed for %s", put_key, exc_info=True)
            return False, 0, None

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        if self._closed:
            raise RuntimeError("Cannot get data: NixlConnector is closed")

        remote_agent = None
        local_reg_descs_list = []
        dlist_handles = []
        xfer_handles = []
        try:
            metadata = self._resolve_metadata(get_key, metadata)
            if not isinstance(metadata, dict) or metadata.get("schema_version") != _SCHEMA_VERSION:
                logger.error("NixlConnector get has invalid metadata for %s", get_key)
                return None

            tensor_specs = metadata.get("tensor_specs")
            if not isinstance(tensor_specs, list):
                raise RuntimeError(f"Invalid NIXL metadata for {get_key}: missing tensor_specs")
            descriptor_groups = self._validated_descriptor_groups(metadata, len(tensor_specs))

            local_tensors = [self._allocate_tensor_from_spec(spec, metadata.get("kind")) for spec in tensor_specs]
            remote_agent = self._agent.add_remote_agent(metadata["agent_metadata"])
            self._remote_agents.append(remote_agent)
            for descriptor_group in descriptor_groups:
                remote_memory_type = descriptor_group["memory_type"]
                indexed_regions = list(
                    zip(descriptor_group["tensor_indices"], descriptor_group["regions"], strict=True)
                )
                local_groups: dict[str, list[tuple[int, Any]]] = {}
                for tensor_index, remote_region in indexed_regions:
                    local_memory_type = self._resolve_memory_type(local_tensors[tensor_index])
                    local_groups.setdefault(local_memory_type, []).append((tensor_index, remote_region))

                for local_memory_type, entries in local_groups.items():
                    group_tensors = [local_tensors[tensor_index] for tensor_index, _ in entries]
                    local_regions = [self._tensor_region(tensor) for tensor in group_tensors]
                    local_reg_descs = self._agent.get_reg_descs(local_regions, local_memory_type)
                    self._agent.register_memory(local_reg_descs, backends=self._backends)
                    local_reg_descs_list.append(local_reg_descs)

                    remote_regions = [tuple(region)[:3] for _, region in entries]
                    remote_descs = self._agent.get_xfer_descs(remote_regions, remote_memory_type)
                    local_descs = self._agent.get_xfer_descs(
                        [tuple(region)[:3] for region in local_regions], local_memory_type
                    )
                    remote_dlist = self._agent.prep_xfer_dlist(remote_agent, remote_descs)
                    local_dlist = self._agent.prep_xfer_dlist(_INIT_AGENT, local_descs)
                    dlist_handles.extend([local_dlist, remote_dlist])

                    desc_ids = list(range(len(entries)))
                    xfer_handle = self._agent.make_prepped_xfer(
                        "READ",
                        local_dlist,
                        desc_ids,
                        remote_dlist,
                        desc_ids,
                    )
                    xfer_handles.append(xfer_handle)
                    self._agent.transfer(xfer_handle)

            for xfer_handle in xfer_handles:
                try:
                    self._wait_for_transfer(xfer_handle, get_key)
                except TimeoutError:
                    self._defer_transfer(
                        _DeferredTransfer(
                            tensors=local_tensors,
                            registrations=local_reg_descs_list,
                            dlists=dlist_handles,
                            handles=xfer_handles,
                            remote_agent=remote_agent,
                        )
                    )
                    local_tensors = []
                    local_reg_descs_list = []
                    dlist_handles = []
                    xfer_handles = []
                    remote_agent = None
                    raise
            for xfer_handle in xfer_handles:
                self._agent.release_xfer_handle(xfer_handle)
            xfer_handles.clear()

            size = int(metadata.get("size", sum(spec.get("size", 0) for spec in tensor_specs)))
            if metadata.get("kind") == _KIND_OBJECT:
                raw = local_tensors[0].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                payload = self.deserialize_obj(raw)
            elif metadata.get("kind") == _KIND_STRUCTURED:
                raw = local_tensors[0].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                skeleton = self.deserialize_obj(raw)
                payload = self._restore_tensor_leaves(skeleton, local_tensors[1:])
            else:
                payload = local_tensors[0] if len(local_tensors) == 1 else local_tensors
            self._notify_transfer_done(get_key, metadata)
            self._metrics["gets"] += 1
            self._metrics["bytes_transferred"] += size
            logger.debug("NixlConnector get %s->%s key=%s size=%d", from_stage, to_stage, get_key, size)
            return payload, size
        except Exception:
            self._metrics["errors"] += 1
            logger.error("NixlConnector get failed for %s", get_key, exc_info=True)
            return None
        finally:
            for xfer_handle in xfer_handles:
                self._safe_call(self._agent.release_xfer_handle, xfer_handle)
            for dlist_handle in dlist_handles:
                self._safe_call(self._agent.release_dlist_handle, dlist_handle)
            if remote_agent is not None:
                self._safe_call(self._agent.remove_remote_agent, remote_agent)
                if remote_agent in self._remote_agents:
                    self._remote_agents.remove(remote_agent)
            for local_reg_descs in local_reg_descs_list:
                self._safe_call(self._agent.deregister_memory, local_reg_descs)

    def cleanup(self, request_id: str) -> None:
        pending = self._take_pending(request_id)
        if pending is None:
            return
        self._release_pending(pending)

    def _release_pending(self, pending: _PendingPayload) -> None:
        for reg_descs in pending.registrations:
            self._safe_call(self._agent.deregister_memory, reg_descs)
            with self._state_lock:
                if reg_descs in self._registered_descs:
                    self._registered_descs.remove(reg_descs)

    def _take_pending(self, request_id: str, expected: _PendingPayload | None = None) -> _PendingPayload | None:
        with self._state_lock:
            pending = self._pending.get(request_id)
            if pending is None or (expected is not None and pending is not expected):
                return None
            self._published.pop(request_id, None)
            return self._pending.pop(request_id)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        self._lease_wakeup.set()
        self._transfer_wakeup.set()
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=5.0)
            self._listener_thread = None
        if self._lease_thread is not None:
            self._lease_thread.join(timeout=5.0)
            self._lease_thread = None
        if self._transfer_thread is not None:
            self._transfer_thread.join(timeout=5.0)
            self._transfer_thread = None
        while self._deferred_transfers:
            self._reap_deferred_transfers()
            if self._deferred_transfers:
                time.sleep(max(self._poll_interval_s, 0.01))
        if self._zmq_ctx is not None:
            # destroy() rather than term(): REQ sockets live in thread-local
            # caches this thread cannot reach, and term() blocks until every
            # socket in the context is closed.
            self._safe_call(self._zmq_ctx.destroy, 0)
            self._zmq_ctx = None
        for request_id in list(self._pending):
            self.cleanup(request_id)
        deferred_agents = {
            transfer.remote_agent for transfer in self._deferred_transfers if transfer.remote_agent is not None
        }
        for agent_name in [agent for agent in self._remote_agents if agent not in deferred_agents]:
            self._safe_call(self._agent.remove_remote_agent, agent_name)
            self._remote_agents.remove(agent_name)
        for reg_descs in list(self._registered_descs):
            self._safe_call(self._agent.deregister_memory, reg_descs)
        self._registered_descs.clear()

    def health(self) -> dict[str, Any]:
        return {
            "status": "unhealthy" if self._closed else "healthy",
            "pending_requests": len(self._pending),
            **self._metrics,
        }

    def get_connection_info(self) -> dict[str, Any]:
        """Endpoint a consumer needs to reach this producer's handshake socket."""
        return {"host": self.host, "zmq_port": self._zmq_port}

    def update_sender_info(self, sender_host: str, sender_zmq_port: int) -> None:
        """Register the producer handshake endpoint on the consumer side.

        Used when the endpoint is only known after both stages have started and
        therefore cannot be baked into the connector config.
        """
        self._sender_host = sender_host
        self._sender_zmq_port = int(sender_zmq_port)
        if self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
            self._handshake_enabled = True

    @staticmethod
    def _metadata_endpoint(metadata: dict[str, Any] | None) -> tuple[str, int] | None:
        if not isinstance(metadata, dict):
            return None
        host = metadata.get("sender_host")
        port = metadata.get("sender_zmq_port")
        if not host or not port:
            host = metadata.get("source_host")
            port = metadata.get("source_port")
        if not host or not port:
            return None
        return str(host), int(port)

    def _resolve_metadata(self, get_key: str, metadata: dict[str, Any] | None) -> dict[str, Any] | None:
        """Return complete NIXL transfer metadata for ``get_key``.

        Callers that forwarded the ``put`` metadata are served directly; the
        remaining cases fall back to the handshake, either at the endpoint
        named in a partial metadata dict or at the configured default producer.
        """
        if isinstance(metadata, dict) and metadata.get("schema_version") == _SCHEMA_VERSION:
            return metadata

        endpoint = self._metadata_endpoint(metadata)
        if endpoint is None and self._sender_host and self._sender_zmq_port:
            endpoint = (str(self._sender_host), int(self._sender_zmq_port))
        if endpoint is None:
            logger.error(
                "NixlConnector get(%s) received no usable metadata and no producer handshake "
                "endpoint is configured. Set sender_host/sender_zmq_port on the consumer, "
                "pass source_host/source_port for the producer rank, or forward the metadata "
                "returned by put().",
                get_key,
            )
            return None
        if self._zmq_ctx is None:
            logger.error(
                "NixlConnector get(%s) needs the handshake but no ZMQ context was created; "
                "set zmq_port or sender_host in the connector config.",
                get_key,
            )
            return None

        return self._query_metadata_at(get_key, *endpoint)

    def _query_metadata_at(self, get_key: str, host: str, port: int) -> dict[str, Any] | None:
        """Fetch transfer metadata for ``get_key`` from a producer's ROUTER socket.

        A consumer can outrun its producer, so a missing key is retried until
        ``handshake_max_wait_s`` instead of failing the request immediately.
        """
        zmq_addr = f"tcp://{host}:{port}"
        deadline = time.monotonic() + self._handshake_max_wait_s
        request = _GET_META_MSG + msgspec.msgpack.encode({"key": get_key})
        while not self._stop_event.is_set():
            sock = self._get_req_socket(zmq_addr)
            try:
                sock.send(request)
                reply = sock.recv()
            except Exception:
                self._invalidate_req_socket(zmq_addr)
                logger.debug("NixlConnector handshake query to %s failed for %s", zmq_addr, get_key, exc_info=True)
                reply = _META_NOT_FOUND
            if reply != _META_NOT_FOUND:
                return msgspec.msgpack.decode(reply)
            if time.monotonic() >= deadline:
                logger.error(
                    "NixlConnector handshake timed out after %.0fs waiting for key %s at %s",
                    self._handshake_max_wait_s,
                    get_key,
                    zmq_addr,
                )
                return None
            time.sleep(self._handshake_retry_s)
        return None

    def _notify_transfer_done(self, get_key: str, metadata: dict[str, Any]) -> None:
        """Tell the producer its buffer is drained so it can deregister now.

        Without this the producer would hold the registration until the lease
        expires, which for long-lived stages means unbounded growth.
        """
        endpoint = self._metadata_endpoint(metadata)
        if endpoint is None or self._zmq_ctx is None:
            return
        host, port = endpoint
        zmq_addr = f"tcp://{host}:{port}"
        sock = self._get_req_socket(zmq_addr)
        try:
            sock.send(_XFER_DONE_MSG + msgspec.msgpack.encode({"key": get_key}))
            sock.recv()
        except Exception:
            self._invalidate_req_socket(zmq_addr)
            logger.debug("NixlConnector failed to notify completion for %s", get_key, exc_info=True)

    def _handshake_listener_loop(self) -> None:
        router = self._zmq_ctx.socket(zmq.ROUTER)
        try:
            router.bind(f"tcp://{self.host}:{self._zmq_port}")
        except zmq.ZMQError as exc:
            logger.error("NixlConnector handshake bind failed on %s:%s: %s", self.host, self._zmq_port, exc)
            self._bind_error = exc
            self._listener_ready.set()
            router.close(linger=0)
            return
        self._listener_ready.set()

        poller = zmq.Poller()
        poller.register(router, zmq.POLLIN)
        try:
            while not self._stop_event.is_set():
                try:
                    if not dict(poller.poll(500)):
                        self._cleanup_expired_pending()
                        continue
                    identity, _, payload = router.recv_multipart()
                    router.send_multipart([identity, b"", self._handle_handshake_message(payload)])
                except zmq.ContextTerminated:
                    break
                except Exception:
                    logger.debug("NixlConnector handshake listener error", exc_info=True)
        finally:
            self._safe_call(router.close, 0)

    def _handle_handshake_message(self, payload: bytes) -> bytes:
        if payload.startswith(_GET_META_MSG):
            key = msgspec.msgpack.decode(payload[len(_GET_META_MSG) :]).get("key")
            with self._state_lock:
                metadata = self._published.get(key)
            return _META_NOT_FOUND if metadata is None else msgspec.msgpack.encode(metadata)
        if payload.startswith(_XFER_DONE_MSG):
            key = msgspec.msgpack.decode(payload[len(_XFER_DONE_MSG) :]).get("key")
            self.cleanup(key)
            return _ACK
        logger.warning("NixlConnector handshake received an unknown message")
        return _META_NOT_FOUND

    def _get_req_socket(self, zmq_addr: str) -> zmq.Socket:
        """Return a thread-local REQ socket so concurrent calls never interleave."""
        cache: dict[str, zmq.Socket] | None = getattr(self._req_local, "cache", None)
        if cache is None:
            cache = {}
            self._req_local.cache = cache
        sock = cache.get(zmq_addr)
        if sock is None:
            sock = self._zmq_ctx.socket(zmq.REQ)
            sock.connect(zmq_addr)
            cache[zmq_addr] = sock
        sock.setsockopt(zmq.SNDTIMEO, self._handshake_timeout_ms)
        sock.setsockopt(zmq.RCVTIMEO, self._handshake_timeout_ms)
        return sock

    def _invalidate_req_socket(self, zmq_addr: str) -> None:
        cache: dict[str, zmq.Socket] | None = getattr(self._req_local, "cache", None)
        if cache is None:
            return
        sock = cache.pop(zmq_addr, None)
        if sock is not None:
            self._safe_call(sock.close, 0)

    @staticmethod
    def _get_local_ip() -> str:
        """Resolve the externally routable local address for the handshake bind."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("8.8.8.8", 80))
                return probe.getsockname()[0]
        except Exception:
            try:
                return socket.gethostbyname(socket.gethostname())
            except Exception:
                logger.warning("NixlConnector could not resolve a local IP; binding loopback")
                return "127.0.0.1"

    def _wait_for_transfer(self, handle: int, request_id: str) -> None:
        deadline = time.monotonic() + self._transfer_timeout_s
        while True:
            state = self._agent.check_xfer_state(handle)
            if state == "DONE":
                return
            if state != "PROC":
                raise RuntimeError(f"NIXL transfer for {request_id} failed with state={state}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"NIXL transfer for {request_id} timed out")
            time.sleep(self._poll_interval_s)

    def _cleanup_expired_pending(self) -> None:
        now = time.monotonic()
        with self._state_lock:
            expired = [
                (key, pending, sum(t.numel() * t.element_size() for t in pending.tensors))
                for key, pending in self._pending.items()
                if now >= pending.deadline
            ]
        for request_id, pending, size in expired:
            logger.warning(
                "NixlConnector lease expired for request %s after %.0fs; its %d bytes are "
                "being reclaimed while a consumer may still read them, which yields "
                "corrupt data. Raise VLLM_OMNI_NIXL_LEASE_S above the maximum "
                "producer-to-consumer queueing delay.",
                request_id,
                self._lease_seconds,
                size,
            )
            claimed = self._take_pending(request_id, expected=pending)
            if claimed is not None:
                self._release_pending(claimed)

    def _lease_reaper_loop(self) -> None:
        while not self._stop_event.is_set():
            now = time.monotonic()
            with self._state_lock:
                next_deadline = min((pending.deadline for pending in self._pending.values()), default=None)
            if next_deadline is None:
                timeout = None
            else:
                timeout = max(0.0, next_deadline - now)
            self._lease_wakeup.wait(timeout=timeout)
            self._lease_wakeup.clear()
            if not self._stop_event.is_set():
                self._cleanup_expired_pending()

    def _defer_transfer(self, transfer: _DeferredTransfer) -> None:
        with self._state_lock:
            self._deferred_transfers.append(transfer)
        self._transfer_wakeup.set()

    def _transfer_reaper_loop(self) -> None:
        while not self._stop_event.is_set():
            self._transfer_wakeup.wait(timeout=max(self._poll_interval_s, 0.01))
            self._transfer_wakeup.clear()
            self._reap_deferred_transfers()

    def _reap_deferred_transfers(self) -> None:
        with self._state_lock:
            transfers = list(self._deferred_transfers)
        for transfer in transfers:
            try:
                states = [self._agent.check_xfer_state(handle) for handle in transfer.handles]
            except Exception:
                logger.debug("Failed to poll deferred NIXL transfer", exc_info=True)
                continue
            if any(state == "PROC" for state in states):
                continue
            self._release_deferred_transfer(transfer)
            if (
                not transfer.handles
                and not transfer.dlists
                and not transfer.registrations
                and transfer.remote_agent is None
            ):
                transfer.tensors.clear()
                with self._state_lock:
                    if transfer in self._deferred_transfers:
                        self._deferred_transfers.remove(transfer)

    def _release_deferred_transfer(self, transfer: _DeferredTransfer) -> None:
        transfer.handles = self._release_owned_resources(self._agent.release_xfer_handle, transfer.handles)
        transfer.dlists = self._release_owned_resources(self._agent.release_dlist_handle, transfer.dlists)
        if transfer.remote_agent is not None:
            try:
                self._agent.remove_remote_agent(transfer.remote_agent)
            except Exception:
                logger.debug("Failed to remove deferred NIXL remote agent", exc_info=True)
            else:
                if transfer.remote_agent in self._remote_agents:
                    self._remote_agents.remove(transfer.remote_agent)
                transfer.remote_agent = None
        transfer.registrations = self._release_owned_resources(self._agent.deregister_memory, transfer.registrations)

    @staticmethod
    def _release_owned_resources(release: Any, resources: list[Any]) -> list[Any]:
        remaining = []
        for resource in resources:
            try:
                release(resource)
            except Exception:
                logger.debug("Failed to release deferred NIXL resource", exc_info=True)
                remaining.append(resource)
        return remaining

    def _resolve_memory_type(self, tensor: torch.Tensor) -> str:
        if self._default_memory_type is not None:
            return str(self._default_memory_type)
        if tensor.device.type == "cpu":
            return "DRAM"
        return "VRAM"

    @staticmethod
    def _validated_descriptor_groups(metadata: dict[str, Any], tensor_count: int) -> list[dict[str, Any]]:
        groups = metadata.get("descriptor_groups")
        if not isinstance(groups, list) or not groups:
            raise RuntimeError("Invalid NIXL metadata: missing descriptor_groups")

        seen_indices = []
        for group in groups:
            if not isinstance(group, dict):
                raise RuntimeError("Invalid NIXL metadata: descriptor group must be a mapping")
            indices = group.get("tensor_indices")
            regions = group.get("regions")
            if not isinstance(indices, list) or not isinstance(regions, list) or len(indices) != len(regions):
                raise RuntimeError("Invalid NIXL metadata: tensor_indices and regions must have equal lengths")
            if not group.get("memory_type"):
                raise RuntimeError("Invalid NIXL metadata: descriptor group is missing memory_type")
            seen_indices.extend(indices)

        if sorted(seen_indices) != list(range(tensor_count)):
            raise RuntimeError("Invalid NIXL metadata: tensor indices must form an exact partition")
        return groups

    @staticmethod
    def _tensor_region(tensor: torch.Tensor) -> tuple[int, int, int, str]:
        device_id = max(tensor.get_device(), 0) if tensor.device.type != "cpu" else 0
        return (tensor.data_ptr(), tensor.numel() * tensor.element_size(), device_id, "")

    @staticmethod
    def _is_tensor_payload(payload: Any) -> bool:
        return isinstance(payload, torch.Tensor) or (
            isinstance(payload, (list, tuple))
            and bool(payload)
            and all(isinstance(item, torch.Tensor) for item in payload)
        )

    @classmethod
    def _contains_tensor(cls, payload: Any) -> bool:
        if isinstance(payload, torch.Tensor):
            return True
        if isinstance(payload, dict):
            return any(cls._contains_tensor(value) for value in payload.values())
        if isinstance(payload, (list, tuple)):
            return any(cls._contains_tensor(value) for value in payload)
        return False

    @classmethod
    def _extract_tensor_leaves(cls, payload: Any) -> tuple[Any, list[torch.Tensor]]:
        tensors: list[torch.Tensor] = []

        def visit(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                index = len(tensors)
                tensors.append(value)
                return {_TENSOR_MARKER: index}
            if isinstance(value, dict):
                return {key: visit(item) for key, item in value.items()}
            if isinstance(value, list):
                return [visit(item) for item in value]
            if isinstance(value, tuple):
                return {_TUPLE_MARKER: [visit(item) for item in value]}
            return value

        return visit(payload), tensors

    @classmethod
    def _restore_tensor_leaves(cls, skeleton: Any, tensors: list[torch.Tensor]) -> Any:
        if isinstance(skeleton, dict):
            if set(skeleton) == {_TENSOR_MARKER}:
                return tensors[int(skeleton[_TENSOR_MARKER])]
            if set(skeleton) == {_TUPLE_MARKER}:
                return tuple(cls._restore_tensor_leaves(item, tensors) for item in skeleton[_TUPLE_MARKER])
            return {key: cls._restore_tensor_leaves(value, tensors) for key, value in skeleton.items()}
        if isinstance(skeleton, list):
            return [cls._restore_tensor_leaves(item, tensors) for item in skeleton]
        return skeleton

    @staticmethod
    def _normalize_tensor_payload(payload: Any) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
        tensors = [payload] if isinstance(payload, torch.Tensor) else list(payload)
        normalized: list[torch.Tensor] = []
        specs: list[dict[str, Any]] = []
        for tensor in tensors:
            contiguous = tensor.detach().contiguous()
            normalized.append(contiguous)
            specs.append(
                {
                    "shape": list(contiguous.shape),
                    "dtype": str(contiguous.dtype),
                    "device": str(contiguous.device),
                    "size": contiguous.numel() * contiguous.element_size(),
                }
            )
        return normalized, specs

    def _allocate_tensor_from_spec(self, spec: dict[str, Any], kind: str | None) -> torch.Tensor:
        shape = spec.get("shape")
        if not isinstance(shape, list):
            raise RuntimeError(f"Invalid NIXL tensor shape: {shape!r}")
        dtype_name = str(spec.get("dtype", "")).removeprefix("torch.")
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise RuntimeError(f"Unsupported NIXL tensor dtype: {spec.get('dtype')!r}")
        device = torch.device("cpu") if kind == _KIND_OBJECT else self._resolve_receive_device(spec.get("device"))
        return torch.empty(tuple(int(dim) for dim in shape), dtype=dtype, device=device)

    def _resolve_receive_device(self, spec_device: Any) -> torch.device:
        if self._receive_device is not None:
            return self._receive_device
        device = self._parse_device(spec_device)
        if device is None or device.type == "cpu":
            return torch.device("cpu")
        # The producer's device index is meaningless in this process, so keep
        # only its type and land the buffer on this stage's own card.
        backend = getattr(torch, device.type, None)
        index = backend.current_device() if hasattr(backend, "current_device") else 0
        return torch.device(device.type, index)

    @staticmethod
    def _parse_device(device_like: Any) -> torch.device | None:
        if device_like is None:
            return None
        try:
            return torch.device(device_like)
        except Exception as exc:
            raise RuntimeError(f"Invalid NIXL receive device: {device_like!r}") from exc

    @staticmethod
    def _safe_call(func: Any, *args: Any) -> None:
        try:
            func(*args)
        except Exception:
            logger.debug("Ignoring NIXL cleanup failure", exc_info=True)
