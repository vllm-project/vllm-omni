# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import math
import os
import socket
import sys
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import msgspec
import torch
import zmq

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase
from .paged_transfer import (
    KVPagePool,
    PageGeometry,
    PageOffer,
    PageReadyEvent,
    Region,
    ReservedKVPages,
    coalesce_page_regions,
)

if TYPE_CHECKING:
    from nixl._api import nixl_prepped_dlist_handle, nixl_xfer_handle

logger = get_connector_logger(__name__)

_SCHEMA_VERSION = 1
_KIND_TENSORS = "tensors"
_KIND_STRUCTURED = "structured"
_KIND_OBJECT = "object"
_INIT_AGENT = "NIXL_INIT_AGENT"
_TENSOR_MARKER = "__nixl_tensor_index__"
_TUPLE_MARKER = "__nixl_tuple__"

# Handshake control-plane messages. Consumers claim connector-managed payloads
# before READ, even with out-of-band metadata, and acknowledge completion here.
_GET_META_MSG = b"nixl_get_meta"
_XFER_DONE_MSG = b"nixl_xfer_done"
_PAGE_GEOMETRY_MISMATCH = b"nixl_page_geometry_mismatch"
_META_NOT_FOUND = b"nixl_meta_not_found"
_ACK = b"nixl_ack"

# NIXL has no remote READ cancellation primitive. A closed producer with an
# abandoned claim must keep both its agent and allocations alive until process
# exit; dropping the last Python reference is not a safe cancellation policy.
_RETAINED_PRODUCERS: set[Any] = set()


@dataclass
class _DeferredTransfer:
    tensors: list[torch.Tensor]
    registrations: list[Any] = field(default_factory=list)
    dlists: list[Any] = field(default_factory=list)
    handles: list[Any] = field(default_factory=list)
    remote_agent: Any = None
    source_key: str | None = None
    source_metadata: dict[str, Any] | None = None


@dataclass
class _PendingPayload:
    tensors: list[torch.Tensor]
    registrations: list[Any]
    deadline: float
    generation: str = field(default_factory=lambda: uuid.uuid4().hex)
    claims: set[str] = field(default_factory=set)
    page_pool: KVPagePool | None = None
    expected_readers: int = 0
    page_claim_ids: frozenset[str] | None = None


@dataclass
class _PageClaim:
    offer: PageOffer
    readers: frozenset[str]
    endpoint: tuple[str, int]


@dataclass
class _PageRead:
    key: str
    offer: PageOffer
    target: ReservedKVPages
    handle: nixl_xfer_handle | None = None
    dlists: list[nixl_prepped_dlist_handle] = field(default_factory=list)
    remote_agent: str | None = None
    dma_done: bool = False


class NixlConnector(OmniConnectorBase):
    """OmniConnector backed by vLLM's native NIXL wrapper.

    This connector intentionally depends on vLLM's optional NIXL integration
    (``vllm.distributed.nixl_utils``). It transfers raw tensor payloads
    directly through NIXL READ operations. Non-tensor Python
    payloads are serialized with OmniSerializer, packed into a uint8 CPU tensor,
    and moved through the same NIXL path.

    A NIXL READ needs the producer's agent metadata and memory descriptors,
    which ``put`` returns to its caller. Consumers use the ZMQ control plane to
    claim connector-managed payloads before READ and acknowledge completion,
    including when metadata is forwarded directly. Producers without a fixed
    ``zmq_port`` use an ephemeral listener advertised in that metadata. Callers
    passing ``metadata=None`` need a configured or request-specific producer
    endpoint to fetch and claim metadata by key.
    """

    supports_raw_data: bool = True
    page_offer_type: type[PageOffer] = PageOffer

    def __init__(self, config: dict[str, Any]):
        self.config = dict(config or {})
        self.stage_id = int(self.config.get("stage_id", 0))
        self._closed = True
        self._closing = False
        self._registered_descs: list[Any] = []
        self._pending: dict[str, _PendingPayload] = {}
        self._published: dict[str, dict[str, Any]] = {}
        self._state_lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._close_thread: threading.Thread | None = None
        self._remote_agents: list[str] = []
        self._remote_agent_names: dict[bytes, str] = {}
        self._remote_agent_users: dict[str, int] = {}
        self._page_pools: set[KVPagePool] = set()
        self._page_remote_agents: dict[tuple[KVPagePool, bytes], str] = {}
        self._page_reads: dict[str, _PageRead] = {}
        self._page_claims: dict[tuple[str, str, str], _PageClaim] = {}
        self._metrics: dict[str, int] = {
            "puts": 0,
            "gets": 0,
            "errors": 0,
            "bytes_transferred": 0,
            "page_exports": 0,
            "page_reads": 0,
            "page_descriptors_submitted": 0,
            "page_claim_queries": 0,
            "page_claim_cache_hits": 0,
            "page_pool_registrations": 0,
            "page_bytes_published": 0,
            "page_bytes_completed": 0,
            "page_regions_completed": 0,
        }

        from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config

        if NixlWrapper is None:
            raise RuntimeError("NIXL is not available. Install the optional nixl/rixl package to use NixlConnector.")

        backends = self.config.get("backends", ["UCX"])
        self._backends = list(backends) if isinstance(backends, (list, tuple)) else [str(backends)]
        if nixl_agent_config is None:
            agent_config = None
        else:
            from nixl import nixl_thread_sync_t

            config_module = sys.modules.get(nixl_agent_config.__module__)
            sync_type = getattr(config_module, "nixl_thread_sync_t", nixl_thread_sync_t)
            sync_mode = sync_type.NIXL_THREAD_SYNC_STRICT
            non_ucx_backends = [backend for backend in self._backends if backend != "UCX"]
            if non_ucx_backends:
                agent_config = nixl_agent_config(
                    backends=self._backends,
                    capture_telemetry=False,
                    sync_mode=sync_mode,
                )
            else:
                num_threads = int(self.config.get("num_threads", 4))
                agent_config = nixl_agent_config(
                    num_threads=num_threads,
                    capture_telemetry=False,
                    sync_mode=sync_mode,
                )

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
        if self._serving_handshake:
            self._lease_thread = threading.Thread(
                target=self._lease_reaper_loop,
                name="nixl-lease-reaper",
                daemon=True,
            )
            self._lease_thread.start()
        self._transfer_wakeup = threading.Event()
        self._deferred_transfers: list[_DeferredTransfer] = []
        self._abandoned_queries: list[tuple[str, str, str | None, str]] = []
        self._transfer_thread: threading.Thread | None = None
        if self._role != "sender":
            self._transfer_thread = threading.Thread(
                target=self._transfer_reaper_loop,
                name="nixl-transfer-reaper",
                daemon=True,
            )
            self._transfer_thread.start()

    def _init_handshake(self) -> None:
        """Set up the ZMQ metadata and payload-ownership control plane.

        Producers listen on a configured or ephemeral port, including for
        directly forwarded metadata. Receive-only connectors dial the producer;
        intermediate stages can also listen for their outgoing payloads.
        """
        role = self.config.get("role")
        self._role = str(role).lower() if role else None
        # Even out-of-band metadata needs an atomic claim before READ. An
        # ephemeral listener supplies that ownership channel when no fixed
        # data-plane metadata port was configured.
        self._zmq_port = self.config.get("zmq_port", 0 if self._role != "receiver" else None)
        self._sender_host = self.config.get("sender_host")
        self._sender_zmq_port = self.config.get("sender_zmq_port")
        self._handshake_timeout_ms = int(self.config.get("handshake_timeout_ms", 5000))
        self._metadata_query_timeout_ms = int(
            self.config.get("metadata_query_timeout_ms", min(self._handshake_timeout_ms, 10))
        )
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
        if self._zmq_port is None:
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
        if self._closed or getattr(self, "_closing", False):
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
                # Preserve spec/skeleton slots, but never register empty DMA regions.
                if tensor.numel() == 0:
                    continue
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
            with self._state_lock:
                previous = self._pending.get(put_key)
                if previous is not None and previous.claims:
                    for descs in registered_descs:
                        self._safe_call(self._agent.deregister_memory, descs)
                        self._registered_descs.remove(descs)
                    raise RuntimeError(f"Cannot replace claimed NIXL payload {put_key}")
                previous = self._take_pending(put_key)
                if previous is not None:
                    self._release_pending(previous)
                pending = _PendingPayload(
                    tensors=tensors,
                    registrations=registered_descs,
                    deadline=time.monotonic() + self._lease_seconds,
                )
                metadata["generation"] = pending.generation
                self._pending[put_key] = pending
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

    def get_with_deadline(self, from_stage, to_stage, get_key, metadata=None, *, deadline):
        """Bound control-plane and DMA waits without freeing active READ buffers."""
        self._req_local.deadline = deadline
        try:
            if time.monotonic() >= deadline:
                return None
            return self.get(from_stage, to_stage, get_key, metadata)
        finally:
            self._req_local.deadline = None

    def register_page_pool(self, pool: KVPagePool) -> None:
        """Register native allocations once, without taking allocator ownership."""
        with self._state_lock:
            if self._closed or self._closing:
                raise RuntimeError("Cannot register pages: NixlConnector is closed")
            if pool in self._page_pools:
                return
            if pool.registrations:
                raise ValueError("KV pool is already registered with another connector")
            memory_type = self._resolve_memory_type(next(iter(pool.caches.values())))
            regions = [(address, size, device, "") for address, size, device in pool.registration_regions()]
            descs = self._agent.get_reg_descs(regions, memory_type)
            self._agent.register_memory(descs, backends=self._backends)
            pool.registrations.append(descs)
            self._page_pools.add(pool)
            self._metrics["page_pool_registrations"] += 1

    def export_pages(
        self,
        key: str,
        pool: KVPagePool,
        block_ids: Sequence[int],
        num_tokens: int,
        *,
        expected_readers: int,
        ready: PageReadyEvent,
        generation: str | None = None,
    ) -> PageOffer:
        """Publish a fenced native block lease on the existing control plane."""
        regions = pool.regions(block_ids)
        if not 0 < num_tokens <= len(block_ids) * pool.block_size or expected_readers < 1:
            raise ValueError("A page export requires valid tokens and at least one reader")
        ready.synchronize()
        with self._state_lock:
            if self._closed or self._closing or pool not in self._page_pools or not self._serving_handshake:
                raise RuntimeError("Page exports require an open producer and a registered pool")
            if key in self._pending:
                raise ValueError(f"NIXL page export key {key!r} is still owned")
            generation = generation or uuid.uuid4().hex
            if generation in pool.exports:
                raise ValueError("Source KV allocation generation is already exported")
            offer = PageOffer(
                _SCHEMA_VERSION,
                "pages",
                generation,
                pool.epoch,
                pool.geometry,
                num_tokens,
                len(block_ids),
                expected_readers,
                regions,
                self._agent.get_agent_metadata(),
                str(self.host),
                int(self._zmq_port),
            )
            pool.exports.add(generation)
            self._pending[key] = _PendingPayload(
                list(pool.caches.values()),
                [],
                math.inf,
                generation=generation,
                page_pool=pool,
                expected_readers=expected_readers,
            )
            self._published[key] = msgspec.structs.asdict(offer)
            self._metrics["page_exports"] += 1
            self._metrics["page_bytes_published"] += sum(region[1] for region in regions)
        return offer

    def claim_pages(
        self,
        key: str,
        host: str,
        port: int,
        *,
        generation: str,
        claim_id: str,
        page_claim_ids: tuple[str, ...],
        geometry: PageGeometry,
    ) -> PageOffer | None:
        """Claim every CFG reader atomically before the first page READ."""
        readers = frozenset(page_claim_ids)
        with self._state_lock:
            submitted = {(read.key, read.offer.generation, read.offer.claim_id) for read in self._page_reads.values()}
            if (key, generation, claim_id) in submitted:
                return None
            cached = self._page_claims.get((key, generation, claim_id))
            if (
                cached is not None
                and cached.readers == readers
                and cached.endpoint == (host, port)
                and cached.offer.geometry == geometry
            ):
                self._metrics["page_claim_cache_hits"] += 1
                return cached.offer
        if self._zmq_ctx is None:
            with self._state_lock:
                if self._zmq_ctx is None:
                    self._zmq_ctx = zmq.Context()
        self._metrics["page_claim_queries"] += 1
        metadata = self._query_metadata_at(
            key,
            host,
            port,
            generation=generation,
            claim_id=claim_id,
            page_claim_ids=page_claim_ids,
            geometry=geometry,
        )
        if metadata is None:
            return None
        offer = msgspec.convert(metadata, type=self.page_offer_type)
        with self._state_lock:
            live_readers = msgspec.convert(metadata.get("page_claim_ids", (offer.claim_id,)), type=tuple[str, ...])
            submitted = {(read.key, read.offer.generation, read.offer.claim_id) for read in self._page_reads.values()}
            for reader in live_readers:
                if (key, offer.generation, reader) in submitted:
                    continue
                claimed = msgspec.structs.replace(offer, claim_id=reader)
                self._page_claims[(key, offer.generation, reader)] = _PageClaim(claimed, readers, (host, port))
        return offer

    def cancel_page_claim(self, key: str, offer: PageOffer) -> bool:
        """ACK an unsubmitted claim; active or ambiguous DMA cannot be cancelled."""
        claim = (key, offer.generation, offer.claim_id)
        with self._state_lock:
            if any(read.key == key and read.offer == offer for read in self._page_reads.values()):
                raise RuntimeError("Cannot cancel a submitted NIXL page READ")
            cached = self._page_claims.get(claim)
            if cached is None:
                return True
            if cached.offer != offer:
                raise ValueError("Cannot cancel a different NIXL page claim")
            if not self._notify_transfer_done(key, msgspec.structs.asdict(offer)):
                return False
            del self._page_claims[claim]
        return True

    def _reserve_page_read(self, key: str, offer: PageOffer, target: ReservedKVPages) -> tuple[str, tuple[Region, ...]]:
        """Consume a claim and pin its native destination before submitting DMA."""
        pool = target.pool
        regions = pool.validate_offer(offer, target.block_ids)
        if not offer.claim_id or target.allocation_generation < 0:
            raise ValueError("Page READ requires a source claim and a destination allocation generation")
        with self._state_lock:
            if self._closed or self._closing or pool not in self._page_pools:
                raise RuntimeError("Page READ requires an open receiver and a registered pool")
            if target.request_id in pool.reservations or any(
                set(target.block_ids).intersection(blocks) for _, blocks in pool.reservations.values()
            ):
                raise ValueError("Destination KV pages are still owned by a previous transfer or computation")
            claim = (key, offer.generation, offer.claim_id)
            cached = self._page_claims.get(claim)
            if cached is None or cached.offer != offer:
                raise ValueError("Stale or already submitted NIXL page claim")
            del self._page_claims[claim]
            read_id = uuid.uuid4().hex
            read = _PageRead(key, offer, target)
            # Ownership precedes any call that can submit DMA. Failed or unknown
            # submissions retain these native allocations until proven drained.
            pool.reservations[target.request_id] = (target.allocation_generation, target.block_ids)
            pool.reads.add(read_id)
            self._page_reads[read_id] = read
        return read_id, regions

    def read_into(self, key: str, offer: PageOffer, target: ReservedKVPages) -> str:
        """READ directly into a Scheduler reservation; no full-size receive tensor."""
        with self._state_lock:
            read_id, regions = self._reserve_page_read(key, offer, target)
            read = self._page_reads[read_id]
            pool = target.pool
            peer = (pool, offer.agent_metadata)
            if peer not in self._page_remote_agents:
                self._page_remote_agents[peer] = self._acquire_remote_agent(offer.agent_metadata)
            read.remote_agent = self._acquire_remote_agent(offer.agent_metadata)
            memory_type = self._resolve_memory_type(next(iter(pool.caches.values())))
            local_regions, remote_regions = coalesce_page_regions(
                regions, offer.regions, len(regions) // len(pool.geometry.layers)
            )
            local_descs = self._agent.get_xfer_descs(local_regions, memory_type)
            remote_descs = self._agent.get_xfer_descs(remote_regions, memory_type)
            read.dlists.append(self._agent.prep_xfer_dlist(_INIT_AGENT, local_descs))
            read.dlists.append(self._agent.prep_xfer_dlist(read.remote_agent, remote_descs))
            read.handle = self._agent.make_prepped_xfer(
                "READ",
                read.dlists[0],
                list(range(len(local_regions))),
                read.dlists[1],
                list(range(len(remote_regions))),
            )
            self._agent.transfer(read.handle)
            self._metrics["page_reads"] += 1
            self._metrics["page_descriptors_submitted"] += len(local_regions)
        return read_id

    def poll_page_read(self, read_id: str) -> bool:
        """Only DONE permits ACK; timeout and unknown state keep both pools pinned."""
        with self._state_lock:
            read = self._page_reads[read_id]
            if not read.dma_done:
                if read.handle is None:
                    raise RuntimeError("NIXL page submission did not return a transfer handle")
                state = self._agent.check_xfer_state(read.handle)
                if state == "PROC":
                    return False
                if state != "DONE":
                    raise RuntimeError(f"NIXL page READ state={state}; native pages remain pinned")
                self._agent.release_xfer_handle(read.handle)
                for dlist in read.dlists:
                    self._agent.release_dlist_handle(dlist)
                assert read.remote_agent is not None
                self._release_remote_agent(read.remote_agent)
                read.dma_done = True
            if not self._notify_transfer_done(read.key, msgspec.structs.asdict(read.offer)):
                return False
            read.target.pool.reads.remove(read_id)
            del self._page_reads[read_id]
            self._metrics["bytes_transferred"] += sum(region[1] for region in read.offer.regions)
            self._metrics["page_bytes_completed"] += sum(region[1] for region in read.offer.regions)
            self._metrics["page_regions_completed"] += len(read.offer.regions)
        return True

    def retire_pages(self, target: ReservedKVPages) -> None:
        """Release borrowed ownership after the final computation on these pages."""
        with self._state_lock:
            if target.pool.reservations.get(target.request_id) != (target.allocation_generation, target.block_ids):
                raise ValueError("Cannot retire a stale destination KV allocation")
            if any(read.target == target for read in self._page_reads.values()):
                raise RuntimeError("Cannot retire native KV pages with unfinished DMA or ACK")
            del target.pool.reservations[target.request_id]

    def unregister_page_pool(self, pool: KVPagePool) -> None:
        with self._state_lock:
            if pool.exports or pool.reads or pool.reservations:
                raise RuntimeError("Cannot deregister a KV pool owned by transfers or computation")
            for descs in pool.registrations:
                self._agent.deregister_memory(descs)
            for peer, name in list(self._page_remote_agents.items()):
                if peer[0] is pool:
                    self._release_remote_agent(name)
                    del self._page_remote_agents[peer]
            pool.registrations.clear()
            self._page_pools.discard(pool)

    def _acquire_remote_agent(self, metadata: bytes) -> str:
        with self._state_lock:
            name = self._remote_agent_names.get(metadata)
            if name is None:
                name = self._agent.add_remote_agent(metadata)
                self._remote_agent_names[metadata] = name
                if name not in self._remote_agents:
                    self._remote_agents.append(name)
            self._remote_agent_users[name] = self._remote_agent_users.get(name, 0) + 1
            return name

    def _release_remote_agent(self, name: str) -> None:
        with self._state_lock:
            users = self._remote_agent_users[name]
            if users > 1:
                self._remote_agent_users[name] = users - 1
                return
            self._agent.remove_remote_agent(name)
            del self._remote_agent_users[name]
            self._remote_agent_names = {
                metadata: agent for metadata, agent in self._remote_agent_names.items() if agent != name
            }
            self._remote_agents.remove(name)

    def abandon_get(self, get_key: str) -> None:
        claims = getattr(self._req_local, "claims", {})
        with self._state_lock:
            for query_key, claim_id in list(claims.items()):
                if query_key[1] == get_key:
                    self._abandoned_queries.append((*query_key, claim_id))
                    del claims[query_key]
        self._transfer_wakeup.set()

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        if self._closed or getattr(self, "_closing", False):
            raise RuntimeError("Cannot get data: NixlConnector is closed")

        remote_agent = None
        local_reg_descs_list = []
        dlist_handles = []
        xfer_handles = []
        source_metadata = None
        try:
            metadata = self._resolve_metadata(get_key, metadata)
            if not isinstance(metadata, dict) or metadata.get("schema_version") != _SCHEMA_VERSION:
                logger.error("NixlConnector get has invalid metadata for %s", get_key)
                return None

            source_metadata = metadata
            if metadata.get("kind") == "pages":
                raise ValueError("KV page offers require read_into() and scheduler-reserved destination pages")
            deadline = getattr(self._req_local, "deadline", None)
            if deadline is not None and time.monotonic() >= deadline:
                return None

            tensor_specs = metadata.get("tensor_specs")
            if not isinstance(tensor_specs, list):
                raise RuntimeError(f"Invalid NIXL metadata for {get_key}: missing tensor_specs")
            descriptor_groups = self._validated_descriptor_groups(metadata, tensor_specs)

            local_tensors = [self._allocate_tensor_from_spec(spec, metadata.get("kind")) for spec in tensor_specs]
            if descriptor_groups:
                remote_agent = self._acquire_remote_agent(metadata["agent_metadata"])
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
                self._wait_for_transfer(xfer_handle, get_key)
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
            self._metrics["gets"] += 1
            self._metrics["bytes_transferred"] += size
            logger.debug("NixlConnector get %s->%s key=%s size=%d", from_stage, to_stage, get_key, size)
            return payload, size
        except Exception:
            if xfer_handles and self._transfers_may_be_active(xfer_handles):
                self._defer_transfer(
                    _DeferredTransfer(
                        tensors=local_tensors,
                        registrations=local_reg_descs_list,
                        dlists=dlist_handles,
                        handles=xfer_handles,
                        remote_agent=remote_agent,
                        source_key=get_key,
                        source_metadata=source_metadata,
                    )
                )
                local_tensors = []
                local_reg_descs_list = []
                dlist_handles = []
                xfer_handles = []
                remote_agent = None
                source_metadata = None
            self._metrics["errors"] += 1
            logger.error("NixlConnector get failed for %s", get_key, exc_info=True)
            return None
        finally:
            for xfer_handle in xfer_handles:
                self._safe_call(self._agent.release_xfer_handle, xfer_handle)
            for dlist_handle in dlist_handles:
                self._safe_call(self._agent.release_dlist_handle, dlist_handle)
            if remote_agent is not None:
                self._safe_call(self._release_remote_agent, remote_agent)
            for local_reg_descs in local_reg_descs_list:
                self._safe_call(self._agent.deregister_memory, local_reg_descs)
            if source_metadata is not None:
                if not self._notify_transfer_done(get_key, source_metadata):
                    self._defer_transfer(
                        _DeferredTransfer(
                            tensors=[],
                            source_key=get_key,
                            source_metadata=source_metadata,
                        )
                    )

    def cleanup(self, request_id: str) -> None:
        pending = self._take_pending(request_id)
        if pending is None:
            return
        self._release_pending(pending)

    def _release_pending(self, pending: _PendingPayload) -> None:
        if pending.page_pool is not None:
            pending.page_pool.exports.discard(pending.generation)
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
            if pending.claims or (pending.page_pool is not None and pending.page_claim_ids is None):
                return None
            self._published.pop(request_id, None)
            return self._pending.pop(request_id)

    def close(self) -> None:
        with self._close_lock:
            self._close_once()
            if not self._closed and self._close_thread is None:
                self._close_thread = threading.Thread(
                    target=self._finish_close_in_background, name="nixl-close", daemon=True
                )
                self._close_thread.start()

    def _finish_close_in_background(self) -> None:
        retry = threading.Event()
        while not retry.wait(max(self._poll_interval_s, 0.1)):
            with self._close_lock:
                self._close_once()
                if self._closed:
                    return

    def _close_once(self) -> None:
        if self._closed:
            return
        if self._page_reads or any(pool.exports or pool.reservations for pool in self._page_pools):
            _RETAINED_PRODUCERS.add(self)
            self._closing = True
            return
        for pool in list(self._page_pools):
            self.unregister_page_pool(pool)
        self._closed = True
        # Keep completion ACKs available until the automatic closer drains ownership.
        for request_id in list(self._pending):
            self.cleanup(request_id)
        with self._state_lock:
            if any(pending.claims for pending in self._pending.values()):
                _RETAINED_PRODUCERS.add(self)
                self._closed = False
                self._closing = True
                if self._close_thread is None:
                    logger.warning("NIXL close deferred: remote READ claims still own source allocations")
                return
        _RETAINED_PRODUCERS.discard(self)
        self._stop_event.set()
        self._lease_wakeup.set()
        self._transfer_wakeup.set()
        for thread_name in ("_listener_thread", "_lease_thread", "_transfer_thread"):
            thread = getattr(self, thread_name)
            if thread is None:
                continue
            if thread is not threading.current_thread():
                thread.join(timeout=5.0)
            if thread.is_alive():
                _RETAINED_PRODUCERS.add(self)
                self._closed = False
                self._closing = True
                return
            setattr(self, thread_name, None)
        self._reap_deferred_transfers()
        if self._deferred_transfers or self._abandoned_queries:
            _RETAINED_PRODUCERS.add(self)
            self._closed = False
            self._closing = True
            if self._close_thread is None:
                logger.warning("NIXL close deferred: local DMA resources retained for automatic cleanup")
            return
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
            "status": "unhealthy" if self._closed or getattr(self, "_closing", False) else "healthy",
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

        Connector-managed metadata, including directly forwarded ``put`` results,
        requires a handshake claim at the supplied or configured producer endpoint.
        Only legacy externally owned metadata without a generation bypasses it.
        """
        direct = isinstance(metadata, dict) and metadata.get("schema_version") == _SCHEMA_VERSION
        if direct and "generation" not in metadata:
            # Legacy externally owned metadata has no connector-managed lease.
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
            with self._state_lock:
                if self._zmq_ctx is None:
                    self._zmq_ctx = zmq.Context()

        if direct:
            return self._query_metadata_at(get_key, *endpoint, generation=metadata["generation"])
        return self._query_metadata_at(get_key, *endpoint)

    def _query_metadata_at(
        self,
        get_key: str,
        host: str,
        port: int,
        *,
        generation: str | None = None,
        claim_id: str | None = None,
        page_claim_ids: tuple[str, ...] | None = None,
        geometry: PageGeometry | None = None,
    ) -> dict[str, Any] | None:
        """Fetch transfer metadata for ``get_key`` from a producer's ROUTER socket.

        Each call performs one bounded query so a missing key cannot starve
        other requests in the shared receive loop. The caller retries later.
        Retries after a lost reply reuse ownership until metadata is received.
        """
        zmq_addr = f"tcp://{host}:{port}"
        claims = getattr(self._req_local, "claims", None)
        if claims is None:
            claims = {}
            self._req_local.claims = claims
        query_key = (zmq_addr, get_key, generation)
        if page_claim_ids is None:
            claim_id = claims.setdefault(query_key, claim_id or uuid.uuid4().hex)
        request = _GET_META_MSG + msgspec.msgpack.encode(
            {
                "key": get_key,
                "generation": generation,
                "claim_id": claim_id,
                "page_claim_ids": page_claim_ids,
                "page_geometry": geometry,
            }
        )
        try:
            sock = self._get_req_socket(zmq_addr, self._metadata_query_timeout_ms)
            sock.send(request)
            reply = sock.recv()
        except Exception:
            self._invalidate_req_socket(zmq_addr)
            logger.debug("NixlConnector handshake query to %s failed for %s", zmq_addr, get_key, exc_info=True)
            return None
        if reply == _META_NOT_FOUND:
            claims.pop(query_key, None)
            return None
        if reply.startswith(_PAGE_GEOMETRY_MISMATCH):
            source = msgspec.msgpack.decode(reply[len(_PAGE_GEOMETRY_MISMATCH) :], type=PageGeometry)
            raise ValueError(f"Incompatible NIXL KV page geometry: source={source}, destination={geometry}")
        metadata = msgspec.msgpack.decode(reply)
        claims.pop(query_key, None)
        return metadata

    def _notify_transfer_done(self, get_key: str, metadata: dict[str, Any]) -> bool:
        """Tell the producer its buffer is drained so it can deregister now.

        Without this the producer would hold the registration until the lease
        expires, which for long-lived stages means unbounded growth.
        """
        endpoint = self._metadata_endpoint(metadata)
        if endpoint is None or self._zmq_ctx is None:
            return True
        host, port = endpoint
        zmq_addr = f"tcp://{host}:{port}"
        try:
            sock = self._get_req_socket(zmq_addr)
            sock.send(
                _XFER_DONE_MSG
                + msgspec.msgpack.encode(
                    {"key": get_key, "generation": metadata.get("generation"), "claim_id": metadata.get("claim_id")}
                )
            )
            sock.recv()
            return True
        except Exception:
            self._invalidate_req_socket(zmq_addr)
            logger.debug("NixlConnector failed to notify completion for %s", get_key, exc_info=True)
            return False

    def _handshake_listener_loop(self) -> None:
        router = self._zmq_ctx.socket(zmq.ROUTER)
        try:
            if self._zmq_port == 0:
                self._zmq_port = router.bind_to_random_port(f"tcp://{self.host}")
            else:
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
            request = msgspec.msgpack.decode(payload[len(_GET_META_MSG) :])
            key = request.get("key")
            claim = request.get("claim_id")
            with self._state_lock:
                pending = self._pending.get(key)
                metadata = self._published.get(key)
                if (
                    pending is None
                    or metadata is None
                    or not claim
                    or request.get("generation") not in (None, pending.generation)
                    or (not pending.claims and time.monotonic() >= pending.deadline)
                    or ((self._closed or self._closing) and claim not in pending.claims and pending.page_pool is None)
                ):
                    return _META_NOT_FOUND
                # Publish ownership before descriptors can leave this lock.
                # Failed/lost replies retain the claim conservatively.
                if pending.page_pool is not None:
                    readers = request.get("page_claim_ids")
                    if (
                        not isinstance(readers, (tuple, list))
                        or len(set(readers)) != pending.expected_readers
                        or claim not in readers
                        or (pending.page_claim_ids is not None and frozenset(readers) != pending.page_claim_ids)
                    ):
                        return _META_NOT_FOUND
                    geometry = msgspec.convert(request["page_geometry"], type=PageGeometry)
                    if geometry != pending.page_pool.geometry:
                        return _PAGE_GEOMETRY_MISMATCH + msgspec.msgpack.encode(pending.page_pool.geometry)
                    if pending.page_claim_ids is None:
                        pending.page_claim_ids = frozenset(readers)
                        pending.claims.update(readers)
                    elif claim not in pending.claims:
                        return _META_NOT_FOUND
                reply_metadata = {**metadata, "claim_id": claim}
                if pending.page_pool is not None:
                    reply_metadata["page_claim_ids"] = tuple(pending.claims)
                reply = msgspec.msgpack.encode(reply_metadata)
                pending.claims.add(claim)
                return reply
        if payload.startswith(_XFER_DONE_MSG):
            request = msgspec.msgpack.decode(payload[len(_XFER_DONE_MSG) :])
            key = request.get("key")
            with self._state_lock:
                pending = self._pending.get(key)
                if (
                    pending is not None
                    and request.get("generation") == pending.generation
                    and request.get("claim_id") in pending.claims
                ):
                    pending.claims.remove(request["claim_id"])
                    self.cleanup(key)
            return _ACK
        logger.warning("NixlConnector handshake received an unknown message")
        return _META_NOT_FOUND

    def _get_req_socket(self, zmq_addr: str, timeout_ms: int | None = None) -> zmq.Socket:
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
        timeout_ms = self._handshake_timeout_ms if timeout_ms is None else timeout_ms
        deadline = getattr(self._req_local, "deadline", None)
        if deadline is not None:
            remaining_ms = int((deadline - time.monotonic()) * 1000 / 2)
            if remaining_ms <= 0:
                raise TimeoutError("NIXL receive deadline expired")
            timeout_ms = min(timeout_ms, remaining_ms)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
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
        receive_deadline = getattr(self._req_local, "deadline", None)
        if receive_deadline is not None:
            deadline = min(deadline, receive_deadline)
        while True:
            state = self._agent.check_xfer_state(handle)
            if state == "DONE":
                return
            if state != "PROC":
                raise RuntimeError(f"NIXL transfer for {request_id} failed with state={state}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"NIXL transfer for {request_id} timed out")
            time.sleep(min(self._poll_interval_s, max(0.0, deadline - time.monotonic())))

    def _transfers_may_be_active(self, handles: list[Any]) -> bool:
        try:
            return any(self._agent.check_xfer_state(handle) not in {"DONE", "ERR"} for handle in handles)
        except Exception:
            return True

    def _cleanup_expired_pending(self) -> None:
        now = time.monotonic()
        with self._state_lock:
            expired = [
                (key, pending, sum(t.numel() * t.element_size() for t in pending.tensors))
                for key, pending in self._pending.items()
                if not pending.claims and now >= pending.deadline
            ]
        for request_id, pending, size in expired:
            logger.warning(
                "NixlConnector lease expired for request %s after %.0fs; its %d bytes are "
                "unclaimed and can safely be reclaimed.",
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
                next_deadline = min(
                    (
                        pending.deadline
                        for pending in self._pending.values()
                        if not pending.claims and pending.page_pool is None
                    ),
                    default=None,
                )
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
            queries = list(self._abandoned_queries)
        for query in queries:
            address, key, generation, claim_id = query
            endpoint = address.removeprefix("tcp://").rsplit(":", 1)
            metadata = self._query_metadata_at(
                key, endpoint[0], int(endpoint[1]), generation=generation, claim_id=claim_id
            )
            query_key = (address, key, generation)
            unresolved = query_key in getattr(self._req_local, "claims", {})
            if metadata is None and unresolved:
                continue
            if metadata is not None and not self._notify_transfer_done(key, metadata):
                self._defer_transfer(_DeferredTransfer(tensors=[], source_key=key, source_metadata=metadata))
            with self._state_lock:
                self._abandoned_queries.remove(query)
        with self._state_lock:
            transfers = list(self._deferred_transfers)
        for transfer in transfers:
            try:
                states = [self._agent.check_xfer_state(handle) for handle in transfer.handles]
            except Exception:
                logger.debug("Failed to poll deferred NIXL transfer", exc_info=True)
                continue
            if any(state not in {"DONE", "ERR"} for state in states):
                continue
            self._release_deferred_transfer(transfer)
            if (
                not transfer.handles
                and not transfer.dlists
                and not transfer.registrations
                and transfer.remote_agent is None
            ):
                transfer.tensors.clear()
                if transfer.source_metadata is not None:
                    if not self._notify_transfer_done(transfer.source_key, transfer.source_metadata):
                        continue
                    transfer.source_metadata = None
                with self._state_lock:
                    if transfer in self._deferred_transfers:
                        self._deferred_transfers.remove(transfer)

    def _release_deferred_transfer(self, transfer: _DeferredTransfer) -> None:
        transfer.handles = self._release_owned_resources(self._agent.release_xfer_handle, transfer.handles)
        transfer.dlists = self._release_owned_resources(self._agent.release_dlist_handle, transfer.dlists)
        if transfer.remote_agent is not None:
            try:
                self._release_remote_agent(transfer.remote_agent)
            except Exception:
                logger.debug("Failed to remove deferred NIXL remote agent", exc_info=True)
            else:
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
    def _validated_descriptor_groups(
        metadata: dict[str, Any], tensor_specs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # Derive emptiness from shape/dtype rather than the advertised size.
        sizes = []
        for spec in tensor_specs:
            if not isinstance(spec, dict):
                raise RuntimeError("Invalid NIXL metadata: tensor spec must be a mapping")
            shape = spec.get("shape")
            if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
                raise RuntimeError("Invalid NIXL metadata: invalid tensor shape")
            dtype = getattr(torch, str(spec.get("dtype", "")).removeprefix("torch."), None)
            if not isinstance(dtype, torch.dtype):
                raise RuntimeError("Invalid NIXL metadata: invalid tensor dtype")
            size = math.prod(shape) * dtype.itemsize
            if type(spec.get("size")) is not int or spec["size"] != size:
                raise RuntimeError("Invalid NIXL metadata: tensor size does not match shape/dtype")
            sizes.append(size)
        groups = metadata.get("descriptor_groups")
        if not isinstance(groups, list):
            raise RuntimeError("Invalid NIXL metadata: missing descriptor_groups")

        seen_indices = []
        for group in groups:
            if not isinstance(group, dict):
                raise RuntimeError("Invalid NIXL metadata: descriptor group must be a mapping")
            indices = group.get("tensor_indices")
            regions = group.get("regions")
            if not isinstance(indices, list) or not isinstance(regions, list) or len(indices) != len(regions):
                raise RuntimeError("Invalid NIXL metadata: tensor_indices and regions must have equal lengths")
            if not indices:
                raise RuntimeError("Invalid NIXL metadata: empty descriptor group")
            if not group.get("memory_type"):
                raise RuntimeError("Invalid NIXL metadata: descriptor group is missing memory_type")
            if any(type(index) is not int or not 0 <= index < len(sizes) for index in indices):
                raise RuntimeError("Invalid NIXL metadata: tensor indices must form an exact partition")
            for index, region in zip(indices, regions, strict=True):
                if (
                    not isinstance(region, (list, tuple))
                    or len(region) not in (3, 4)
                    or type(region[0]) is not int
                    or region[0] <= 0
                    or type(region[1]) is not int
                    or region[1] <= 0
                    or region[1] != sizes[index]
                    or type(region[2]) is not int
                    or region[2] < 0
                ):
                    raise RuntimeError("Invalid NIXL metadata: invalid tensor region or byte size")
            seen_indices.extend(indices)

        if sorted(seen_indices) != [index for index, size in enumerate(sizes) if size > 0]:
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
        if isinstance(payload, msgspec.Struct):
            payload = msgspec.structs.asdict(payload)
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
            if isinstance(value, msgspec.Struct):
                value = msgspec.structs.asdict(value)
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
