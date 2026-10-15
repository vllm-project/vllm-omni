# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native Scheduler/Worker hooks borrowing the Omni NIXL page capability."""

from __future__ import annotations

import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import msgspec
import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.kv_cache_interface import KVCacheConfig, is_full_attention_spec
from vllm.v1.request import Request, RequestStatus

from .nixl_connector import NixlConnector
from .paged_transfer import KVPagePool, ReservedKVPages

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput

    from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVRequest
    from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput


class PageTicket(msgspec.Struct, frozen=True):
    transfer_id: str
    source_generation: str
    source_host: str
    source_zmq_port: int
    num_transfer_tokens: int
    expected_readers: int
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class SourcePages:
    request_id: str
    ticket: PageTicket
    block_ids: tuple[int, ...]


@dataclass(frozen=True)
class TargetPages:
    request_id: str
    ticket: PageTicket
    block_ids: tuple[int, ...]
    allocation_generation: int
    claim_id: str
    peer_claims: tuple[str, ...] = ()


@dataclass
class PageConnectorMetadata(KVConnectorMetadata):
    sources: tuple[SourcePages, ...]
    targets: tuple[TargetPages, ...]


class _PageFence:
    def __init__(self, events: tuple[torch.cuda.Event, ...]):
        self.events = events

    def synchronize(self) -> None:
        for event in self.events:
            event.synchronize()


class OmniNixlKVConnector(KVConnectorBase_V1, SupportsHMA):
    """Opt-in TP=1 native KV handoff using one Omni NixlConnector per Worker."""

    transport_type: type[NixlConnector] = NixlConnector

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, role, kv_cache_config)
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("Omni NIXL page transfer currently requires TP=1 and PP=1")
        if len(kv_cache_config.kv_cache_groups) != 1 or not is_full_attention_spec(
            kv_cache_config.kv_cache_groups[0].kv_cache_spec
        ):
            raise ValueError("Omni NIXL page transfer requires one uniform full-attention cache group")
        self.block_size = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        self.extra = self._kv_transfer_config.kv_connector_extra_config
        self.namespace = self.extra["page_namespace"]
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError("page_namespace must identify the model revision and KV configuration")
        self._sources: dict[str, SourcePages] = {}
        self._pending_sources: set[str] = set()
        self._targets: dict[str, TargetPages] = {}
        self._exports: dict[str, SourcePages] = {}
        self._waiting: dict[str, TargetPages] = {}
        self._reads: dict[str, str] = {}
        self._reservations: dict[str, ReservedKVPages] = {}
        self._finished_sources: set[str] = set()
        self._published_sources: set[str] = set()
        self._retiring: set[str] = set()
        self._events: dict[str, torch.cuda.Event] = {}
        self._pool: KVPagePool | None = None
        self._transport: NixlConnector | None = None
        if role is KVConnectorRole.WORKER:
            producer = self._kv_transfer_config.is_kv_producer
            self._transport = self.transport_type(
                {
                    "role": "sender" if producer else "receiver",
                    "host": self._kv_transfer_config.kv_ip,
                    "zmq_port": int(self.extra["page_port"]) if producer else None,
                    "backends": self.extra.get("backends", ["UCX"]),
                    "metadata_query_timeout_ms": 10,
                    "handshake_timeout_ms": 100,
                }
            )

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str:
        return "LBNHC"

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        assert self._transport is not None
        prefix = self.extra.get("layer_name_prefix", "")
        if any(not name.startswith(prefix) for name in kv_caches):
            raise ValueError("Native KV layer names do not match layer_name_prefix")
        caches = {name.removeprefix(prefix): tensor for name, tensor in sorted(kv_caches.items())}
        if any(tensor.device.type != "cuda" for tensor in caches.values()):
            raise ValueError("Native Omni NIXL page transfer requires CUDA KV pools")
        layout = self._vllm_config.cache_config.kv_cache_layout
        if layout != "LBNHC":
            raise ValueError("Omni NIXL page transfer requires resolved physical layout LBNHC")
        self._pool = KVPagePool(caches, self.namespace, layout)
        if self._pool.block_size != self.block_size:
            raise ValueError("Omni NIXL pages require matching manager and kernel block sizes")
        self._transport.register_page_pool(self._pool)

    def get_num_new_matched_tokens(
        self,
        request: Request | DiffusionKVRequest,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        params = request.kv_transfer_params
        if not params or not params.get("do_remote_prefill"):
            return 0, False
        ticket = msgspec.convert(params, type=PageTicket)
        if request.prompt_token_ids is None:
            raise ValueError("Native page receive requires the model adapter's reusable prompt_token_ids")
        reusable = min(ticket.num_transfer_tokens, len(request.prompt_token_ids))
        # Equal lengths do not prove that the AR and DiT templates agree.
        reusable = next(
            (
                i
                for i, (source, target) in enumerate(zip(ticket.token_ids, request.prompt_token_ids))
                if source != target
            ),
            reusable,
        )
        tokens = reusable - num_computed_tokens
        return max(0, tokens), tokens > 0

    def update_state_after_alloc(
        self,
        request: Request | DiffusionKVRequest,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVRequest

        params = request.kv_transfer_params
        if not params or not params.get("do_remote_prefill"):
            return
        if not isinstance(request, DiffusionKVRequest):
            raise ValueError("Native page consumers require Scheduler-owned DiffusionKVRequest reservations")
        ticket = msgspec.convert(params, type=PageTicket)
        block_ids = tuple(block.block_id for block in blocks.blocks[0])
        if len(block_ids) != math.ceil(ticket.num_transfer_tokens / self.block_size):
            raise ValueError("Destination reservation must cover every physical source page")
        self._targets[request.request_id] = TargetPages(
            request.request_id,
            ticket,
            block_ids,
            request.allocation_generation,
            uuid.uuid4().hex,
        )

    def request_finished(
        self,
        request: Request | DiffusionKVRequest,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, object] | None]:
        params = request.kv_transfer_params
        if not self._kv_transfer_config.is_kv_producer or not params or not params.get("do_remote_decode"):
            if request.status == RequestStatus.FINISHED_ABORTED:
                self._targets.pop(request.request_id, None)
            return False, None
        if request.status == RequestStatus.FINISHED_ABORTED or request.num_computed_tokens <= 0:
            return False, None
        if not isinstance(request, Request):
            raise ValueError("Native page producers require an AR Request with token IDs")
        ticket = PageTicket(
            params["transfer_id"],
            uuid.uuid4().hex,
            self._kv_transfer_config.kv_ip,
            int(self.extra["page_port"]),
            request.num_computed_tokens,
            int(self.extra.get("expected_readers", 1)),
            tuple(request.all_token_ids[: request.num_computed_tokens]),
        )
        blocks = tuple(block_ids[: math.ceil(ticket.num_transfer_tokens / self.block_size)])
        self._sources[request.request_id] = SourcePages(request.request_id, ticket, blocks)
        self._pending_sources.add(request.request_id)
        return True, {**msgspec.structs.asdict(ticket), "do_remote_prefill": True, "do_remote_decode": False}

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, object] | None]:
        # The constructor accepts one uniform group, including when native HMA
        # is enabled. All block ownership still belongs to its native manager.
        assert len(block_ids) == 1
        return self.request_finished(request, block_ids[0])

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput | DiffusionSchedulerOutput,
    ) -> PageConnectorMetadata:
        groups: dict[tuple[str, str], list[TargetPages]] = defaultdict(list)
        for target in self._targets.values():
            groups[(target.ticket.transfer_id, target.ticket.source_generation)].append(target)
        targets: list[TargetPages] = []
        for group in groups.values():
            if len(group) != group[0].ticket.expected_readers:
                raise ValueError("All CFG reservations must be published in one page transfer metadata batch")
            claims = tuple(target.claim_id for target in group)
            targets.extend(replace(target, peer_claims=claims) for target in group)
        metadata = PageConnectorMetadata(tuple(self._sources.values()), tuple(targets))
        self._sources.clear()
        self._targets.clear()
        return metadata

    def has_pending_push_work(self) -> bool:
        # Keep the native engine stepping to publish fenced source leases and
        # report remote READ completion even when AR has no tokens left to run.
        return bool(self._pending_sources)

    def has_pending_block_frees(self) -> bool:
        return bool(self._pending_sources)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        self._pending_sources.difference_update(connector_output.finished_sending or ())

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: object) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, PageConnectorMetadata):
            raise TypeError("Omni NIXL Worker received incompatible connector metadata")
        self._exports.update((source.request_id, source) for source in metadata.sources)
        self._waiting.update((target.request_id, target) for target in metadata.targets)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: object,
    ) -> None:
        event = torch.cuda.Event()
        event.record()
        self._events[layer_name.removeprefix(self.extra.get("layer_name_prefix", ""))] = event

    def wait_for_save(self) -> None:
        assert self._transport is not None and self._pool is not None
        for source in self._exports.values():
            ticket = source.ticket
            if source.request_id in self._published_sources:
                continue
            fence = _PageFence(tuple(self._events[name] for name in self._pool.caches))
            self._transport.export_pages(
                ticket.transfer_id,
                self._pool,
                source.block_ids,
                ticket.num_transfer_tokens,
                expected_readers=ticket.expected_readers,
                ready=fence,
                generation=ticket.source_generation,
            )
            self._published_sources.add(source.request_id)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self._transport is not None and self._pool is not None
        # vLLM's no_forward path skips wait_for_save. Finished AR sources arrive
        # in that path, so publish their already-recorded fences here as well.
        self.wait_for_save()
        self._finished_sources.update(finished_req_ids.intersection(self._exports))
        self._retiring.update(finished_req_ids.intersection(set(self._reservations) | set(self._waiting)))
        sent, received = set(), set()
        for request_id, source in list(self._exports.items()):
            if (
                request_id in self._published_sources
                and source.ticket.source_generation not in self._pool.exports
                and request_id in self._finished_sources
            ):
                sent.add(request_id)
                del self._exports[request_id]
                self._finished_sources.remove(request_id)
                self._published_sources.remove(request_id)
        for request_id, target in list(self._waiting.items()):
            ticket = target.ticket
            offer = self._transport.claim_pages(
                ticket.transfer_id,
                ticket.source_host,
                ticket.source_zmq_port,
                generation=ticket.source_generation,
                claim_id=target.claim_id,
                page_claim_ids=target.peer_claims,
                geometry=self._pool.geometry,
            )
            if offer is None:
                continue
            if request_id in self._retiring:
                if self._transport.cancel_page_claim(ticket.transfer_id, offer):
                    del self._waiting[request_id]
                    self._retiring.remove(request_id)
                    received.add(request_id)
                continue
            reservation = ReservedKVPages(request_id, target.allocation_generation, self._pool, target.block_ids)
            read_id = self._transport.read_into(ticket.transfer_id, offer, reservation)
            self._reads[request_id] = read_id
            self._reservations[request_id] = reservation
            del self._waiting[request_id]
        for request_id, read_id in list(self._reads.items()):
            if self._transport.poll_page_read(read_id):
                received.add(request_id)
                del self._reads[request_id]
        for request_id in list(self._retiring - self._reads.keys() - self._waiting.keys()):
            self._transport.retire_pages(self._reservations[request_id])
            del self._reservations[request_id]
            self._retiring.remove(request_id)
        return sent, received

    def shutdown(self) -> None:
        if self._transport is not None:
            self._transport.close()

    def page_metrics(self) -> dict[str, int]:
        assert self._transport is not None and self._pool is not None
        return {
            **self._transport._metrics,
            "registered_pools": len(self._transport._page_pools),
            "source_leases": len(self._pool.exports),
            "destination_reservations": len(self._pool.reservations),
            "active_reads": len(self._pool.reads),
        }
