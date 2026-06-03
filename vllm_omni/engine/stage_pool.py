"""Unified stage-local runtime abstraction for vLLM-Omni."""

from __future__ import annotations

import asyncio
import time as _time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.metrics.stats import IterationStats

from vllm_omni.distributed.omni_coordinator import (
    LoadBalancer,
    OmniCoordClientForHub,
    ReplicaInfo,
    ReplicaStatus,
)
from vllm_omni.distributed.omni_coordinator.load_balancer import Task
from vllm_omni.engine.stage_client import (
    StagePoolClient,
    StagePoolDiffusionClient,
    StagePoolLLMClient,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.metrics.stats import StageRequestStats as StageRequestMetrics
from vllm_omni.metrics.stats import StageStats
from vllm_omni.metrics.utils import count_tokens_from_outputs

if TYPE_CHECKING:
    from vllm_omni.engine.orchestrator import OrchestratorRequestState

logger = init_logger(__name__)


@dataclass
class _ReplicaMetrics:
    """Per-replica metrics accumulators owned by a stage pool."""

    batch_seq: int = 0
    agg_total_tokens: int = 0
    agg_total_gen_time_ms: float = 0.0


class StagePool:
    """Replicas of one logical stage + per-stage routing (LB + affinity).

    The pool owns the head-side stage clients for one logical stage. It also
    absorbs the per-stage dispatch responsibility (load balancing, affinity
    tracking, bounded-wait pick) that used to live in a separate
    ``StageDispatcher`` class — see the design doc for the rationale.

    In distributed mode (when an :class:`OmniCoordClientForHub` and a
    :class:`LoadBalancer` are injected via :meth:`attach_hub` /
    :meth:`attach_load_balancer`), :meth:`pick` consults the hub's cached
    replica list and routes via the load balancer, sticking subsequent calls
    for the same ``request_id`` to the same replica.

    In non-distributed mode (no hub attached), :meth:`pick` falls back to the
    legacy ``select_replica_id`` round-robin path so the multi-stage
    in-process invocation is unchanged.

    Dynamic replica membership: when a remote replica is added or removed
    (driven by :class:`Orchestrator` via :meth:`add_client` /
    :meth:`remove_client`), the pool keeps stable integer ``replica_id``s by
    storing clients in a list whose entries can be ``None`` after a removal.
    Iteration callers should use :meth:`live_replica_ids` rather than
    ``range(pool.num_replicas)`` to skip the gaps.
    """

    DISPATCH_WAIT_TIMEOUT_S: float = 10.0
    DISPATCH_RETRY_INTERVAL_S: float = 0.1

    def __init__(
        self,
        stage_id: int,
        clients: StagePoolClient | list[StagePoolClient],
        *,
        output_processor: Any = None,
        stage_vllm_config: Any = None,
    ) -> None:
        if isinstance(clients, list):
            normalized_clients: list[StagePoolClient] = list(clients)
        else:
            normalized_clients = [clients]

        # Allow empty pools when running in distributed head mode for a
        # non-self stage; clients will arrive via add_client(...).
        self.stage_id = stage_id
        # Slots can become None after a dynamic remove_client (distributed mode);
        # iterate via live_replica_ids() to skip holes.
        self.clients: list[StagePoolClient | None] = list(normalized_clients)
        self._output_processor = output_processor
        self._stage_vllm_config = stage_vllm_config
        self._next_replica_id = 0
        self._request_bindings: dict[str, int] = {}
        self._replica_metrics: list[_ReplicaMetrics] = [_ReplicaMetrics() for _ in self.clients]

        # Distributed-mode state. Populated by add_client / remove_client.
        self._addr_to_replica_id: dict[str, int] = {}
        for replica_id, client in enumerate(self.clients):
            if client is not None:
                addr = self._client_input_addr(client)
                if addr is not None:
                    self._addr_to_replica_id[addr] = replica_id

        # Distributed-mode dispatch hooks (injected by Orchestrator on bring-up).
        self._hub: OmniCoordClientForHub | None = None
        self._lb: LoadBalancer | None = None
        # ``request_id`` → ``input_addr`` affinity (distributed mode only).
        # Kept separate from the legacy ``_request_bindings`` so the two
        # binding shapes do not collide.
        self._affinity: dict[str, str] = {}

    # ---- Stage-level properties ----

    @property
    def num_replicas(self) -> int:
        """Total slot count, including ``None`` holes from removed replicas.

        Use :meth:`live_replica_ids` to iterate only live entries.
        """
        return len(self.clients)

    @property
    def live_num_replicas(self) -> int:
        """Number of currently live (non-None) replicas in this pool."""
        return sum(1 for c in self.clients if c is not None)

    def live_replica_ids(self) -> list[int]:
        """Return the indices of currently live replicas in this pool."""
        return [i for i, c in enumerate(self.clients) if c is not None]

    @property
    def stage_type(self) -> str | None:
        client = self.stage_client
        return None if client is None else client.stage_type

    @property
    def final_output(self) -> bool:
        client = self.stage_client
        return False if client is None else bool(client.final_output)

    @property
    def stage_client(self) -> StagePoolClient | None:
        for client in self.clients:
            if client is not None:
                return client
        return None

    @property
    def llm_stage_client(self) -> StagePoolLLMClient:
        return cast(StagePoolLLMClient, self.stage_client)

    @property
    def stage_vllm_config(self) -> Any:
        return self._stage_vllm_config

    @property
    def output_processor(self) -> Any:
        return self._output_processor

    @property
    def is_distributed(self) -> bool:
        """True iff a hub has been attached (i.e. running in head-distributed mode)."""
        return self._hub is not None

    # ---- Distributed-mode dispatch hooks ----

    def attach_hub(self, hub: OmniCoordClientForHub | None) -> None:
        """Inject the shared :class:`OmniCoordClientForHub`.

        Called once by :class:`Orchestrator` after the hub is constructed.
        ``hub=None`` keeps the pool in legacy mode (no behavior change).
        """
        self._hub = hub

    def attach_load_balancer(self, lb: LoadBalancer | None) -> None:
        """Inject the per-pool :class:`LoadBalancer` for distributed-mode pick."""
        self._lb = lb

    # ---- Dynamic membership (distributed mode) ----

    @staticmethod
    def _client_input_addr(client: Any) -> str | None:
        """Return the input ZMQ address advertised by ``client`` if any.

        LLM clients expose ``client_addresses["input_address"]``; diffusion
        clients expose ``request_address``. Both are stable strings used by
        :class:`OmniCoordinator` to key replicas.
        """
        request_address = getattr(client, "request_address", None)
        if isinstance(request_address, str) and request_address:
            return request_address
        addrs = getattr(client, "client_addresses", None)
        if isinstance(addrs, dict):
            addr = addrs.get("input_address")
            if isinstance(addr, str) and addr:
                return addr
        return None

    def add_client(self, input_addr: str, client: Any) -> int:
        """Register a head-side client for ``input_addr``.

        Returns the assigned ``replica_id`` (index into :attr:`clients`).
        If the address is already known, replaces the existing client and
        returns its existing id (this should not happen in practice — the
        master server assigns unique slots — but the contract is idempotent
        to keep the dispatch layer robust).
        """
        if not input_addr:
            raise ValueError("input_addr must be a non-empty string")

        existing = self._addr_to_replica_id.get(input_addr)
        if existing is not None:
            self.clients[existing] = client
            return existing

        replica_id = len(self.clients)
        self.clients.append(client)
        self._addr_to_replica_id[input_addr] = replica_id
        self._replica_metrics.append(_ReplicaMetrics())
        return replica_id

    def remove_client(self, input_addr: str) -> Any | None:
        """Remove the client at ``input_addr``. Returns the removed client or ``None``.

        Slot is marked ``None`` to preserve indices for outstanding bindings.
        """
        replica_id = self._addr_to_replica_id.pop(input_addr, None)
        if replica_id is None:
            return None
        client = self.clients[replica_id]
        self.clients[replica_id] = None
        return client

    def get_client_by_addr(self, input_addr: str) -> Any | None:
        """Return the live client for ``input_addr`` if present."""
        replica_id = self._addr_to_replica_id.get(input_addr)
        if replica_id is None:
            return None
        return self.clients[replica_id]

    def get_replica_id_by_addr(self, input_addr: str) -> int | None:
        """Return the stable replica_id for ``input_addr`` if registered."""
        return self._addr_to_replica_id.get(input_addr)

    # ---- Per-request distributed dispatch ----

    async def pick(
        self,
        request_id: str,
        task: Task | None = None,
        *,
        affinity_request_id: str | None = None,
    ) -> int:
        """Return a replica id for ``request_id``.

        In distributed mode: consults the hub for UP replicas, runs the load
        balancer, and records affinity so future picks for the same
        ``request_id`` return the same replica. Bounded wait up to
        ``DISPATCH_WAIT_TIMEOUT_S`` when no UP replica is currently usable.

        In non-distributed (legacy) mode: delegates to
        :meth:`select_replica_id`.
        """
        if self._hub is None or self._lb is None:
            return self.select_replica_id(request_id, affinity_request_id=affinity_request_id)

        # 1. Sticky: previously bound and still serviceable?
        bound_addr = self._affinity.get(request_id)
        if bound_addr is not None:
            replica_id = self._serviceable_replica_id_for_addr(bound_addr)
            if replica_id is not None:
                return replica_id
            # Bound replica is gone or DOWN — fall through to re-select.
            self._affinity.pop(request_id, None)

        # 2. Inherited affinity (CFG companion sharing a parent request_id).
        if affinity_request_id is not None:
            parent_addr = self._affinity.get(affinity_request_id)
            if parent_addr is not None:
                replica_id = self._serviceable_replica_id_for_addr(parent_addr)
                if replica_id is not None:
                    self._affinity[request_id] = parent_addr
                    return replica_id

        # 3. Fresh pick: poll hub + LB with bounded wait.
        task = task or Task(request_id=request_id)
        deadline = _time.monotonic() + self.DISPATCH_WAIT_TIMEOUT_S
        while True:
            candidates = self._collect_serviceable_replicas()
            if candidates:
                # LB chose an index *into our candidates list*.
                lb_idx = self._lb.select(task, [rep for rep, _ in candidates])
                replica_info, replica_id = candidates[lb_idx]
                self._affinity[request_id] = replica_info.input_addr
                return replica_id

            now = _time.monotonic()
            if now >= deadline:
                raise RuntimeError(f"no UP replica for stage {self.stage_id} after {self.DISPATCH_WAIT_TIMEOUT_S:.1f}s")
            await asyncio.sleep(min(self.DISPATCH_RETRY_INTERVAL_S, deadline - now))

    def preselect_replica_id(
        self,
        request_id: str,
        task: Task | None = None,
        *,
        affinity_request_id: str | None = None,
    ) -> int | None:
        """Synchronously pick and bind a replica before request preprocessing.

        The main-thread input preprocessing path cannot await :meth:`pick`, but
        multimodal cache UUID scoping needs to know the same replica that
        :meth:`submit_initial` will later use. In distributed mode this checks
        the hub's cached replica snapshot once and records the selected input
        address in ``_affinity`` so the async submit path reuses the route. If
        no replica is currently serviceable, return ``None`` and let the async
        submit-time router wait without blocking the caller.
        """
        if self._hub is None or self._lb is None:
            return self.select_replica_id(request_id, affinity_request_id=affinity_request_id)

        bound_addr = self._affinity.get(request_id)
        if bound_addr is not None:
            replica_id = self._serviceable_replica_id_for_addr(bound_addr)
            if replica_id is not None:
                return replica_id
            self._affinity.pop(request_id, None)

        if affinity_request_id is not None:
            parent_addr = self._affinity.get(affinity_request_id)
            if parent_addr is not None:
                replica_id = self._serviceable_replica_id_for_addr(parent_addr)
                if replica_id is not None:
                    self._affinity[request_id] = parent_addr
                    return replica_id

        task = task or Task(request_id=request_id)
        candidates = self._collect_serviceable_replicas()
        if not candidates:
            return None

        lb_idx = self._lb.select(task, [rep for rep, _ in candidates])
        replica_info, replica_id = candidates[lb_idx]
        self._affinity[request_id] = replica_info.input_addr
        return replica_id

    def _collect_serviceable_replicas(self) -> list[tuple[ReplicaInfo, int]]:
        """Return list of ``(ReplicaInfo, replica_id)`` for UP, attached replicas."""
        if self._hub is None:
            return []
        snap = self._hub.get_replicas_for_stage(self.stage_id)
        out: list[tuple[ReplicaInfo, int]] = []
        for rep in snap.replicas:
            if rep.status != ReplicaStatus.UP:
                continue
            replica_id = self._addr_to_replica_id.get(rep.input_addr)
            if replica_id is None:
                continue  # Hub knows about it but head-side client not attached yet.
            if self.clients[replica_id] is None:
                continue
            out.append((rep, replica_id))
        return out

    def _serviceable_replica_id_for_addr(self, input_addr: str) -> int | None:
        """Return ``replica_id`` for ``input_addr`` iff currently UP + attached."""
        if self._hub is None:
            return None
        replica_id = self._addr_to_replica_id.get(input_addr)
        if replica_id is None or self.clients[replica_id] is None:
            return None
        snap = self._hub.get_replicas_for_stage(self.stage_id)
        for rep in snap.replicas:
            if rep.input_addr == input_addr and rep.status == ReplicaStatus.UP:
                return replica_id
        return None

    def bind(self, request_id: str, input_addr: str) -> None:
        """Explicitly record affinity (distributed mode)."""
        self._affinity[request_id] = input_addr

    def release(self, request_id: str) -> None:
        """Drop affinity (distributed mode) and legacy binding for ``request_id``."""
        self._affinity.pop(request_id, None)
        self.release_binding(request_id)

    def invalidate_addr(self, input_addr: str) -> list[str]:
        """Drop affinity rows pointing at ``input_addr``; return affected request ids."""
        affected: list[str] = [rid for rid, addr in self._affinity.items() if addr == input_addr]
        for rid in affected:
            self._affinity.pop(rid, None)
        return affected

    # ---- Legacy (non-distributed) route binding ----

    def get_bound_replica_id(self, request_id: str) -> int | None:
        """Return the currently bound replica id for *request_id* if present.

        In distributed mode the binding may have been recorded via
        :meth:`pick`; we honor it transparently here.
        """
        legacy = self._request_bindings.get(request_id)
        if legacy is not None:
            return legacy
        addr = self._affinity.get(request_id)
        if addr is None:
            return None
        return self._addr_to_replica_id.get(addr)

    def get_bound_client(self, request_id: str) -> StagePoolClient | None:
        """Return the currently bound client for *request_id* if present."""
        replica_id = self.get_bound_replica_id(request_id)
        if replica_id is None:
            return None
        return self.clients[replica_id]

    def get_bound_llm_client(self, request_id: str) -> StagePoolLLMClient | None:
        """Return the currently bound LLM client for *request_id* if present."""
        client = self.get_bound_client(request_id)
        if client is None:
            return None
        return cast(StagePoolLLMClient, client)

    def release_binding(self, request_id: str) -> None:
        """Drop the route binding for *request_id* in this stage."""
        self._request_bindings.pop(request_id, None)
        self._affinity.pop(request_id, None)

    def release_bindings(self, request_ids: list[str]) -> None:
        """Drop route bindings for the given request ids in this stage."""
        for request_id in request_ids:
            self.release_binding(request_id)

    def select_replica_id(
        self,
        request_id: str,
        *,
        affinity_request_id: str | None = None,
    ) -> int:
        """Pick a replica id for *request_id* and cache the choice (legacy path)."""
        cached = self.get_bound_replica_id(request_id)
        if cached is not None and self.clients[cached] is not None:
            return cached

        chosen: int | None = None
        if affinity_request_id is not None:
            parent = self.get_bound_replica_id(affinity_request_id)
            if parent is not None and self.clients[parent] is not None:
                chosen = parent

        if chosen is None:
            live = self.live_replica_ids()
            if not live:
                raise RuntimeError(f"stage {self.stage_id} has no live replicas")
            if len(live) == 1:
                chosen = live[0]
            else:
                # Round-robin over live replicas only.
                start = self._next_replica_id % len(live)
                chosen = live[start]
                self._next_replica_id = (self._next_replica_id + 1) % len(live)

        self._request_bindings[request_id] = chosen
        return chosen

    def _llm_client(self, replica_id: int) -> StagePoolLLMClient:
        client = self.clients[replica_id]
        if client is None:
            raise RuntimeError(f"stage {self.stage_id} replica {replica_id} is not attached")
        return cast(StagePoolLLMClient, client)

    def _diffusion_client(self, replica_id: int) -> StagePoolDiffusionClient:
        client = self.clients[replica_id]
        if client is None:
            raise RuntimeError(f"stage {self.stage_id} replica {replica_id} is not attached")
        return cast(StagePoolDiffusionClient, client)

    # ---- Metrics ----

    def build_stage_metrics(
        self,
        request_outputs: list[Any],
        *,
        submit_ts: float,
        replica_id: int,
    ) -> StageRequestMetrics:
        """Build stage metrics for outputs produced on one replica."""
        now = _time.time()
        stage_gen_time_ms = (now - submit_ts) * 1000.0

        num_tokens_out = count_tokens_from_outputs(request_outputs)
        num_tokens_in = 0
        if self.stage_id == 0:
            for ro in request_outputs:
                ptids = getattr(ro, "prompt_token_ids", None)
                if ptids is not None:
                    num_tokens_in += len(ptids)

        metrics = self._replica_metrics[replica_id]
        metrics.batch_seq += 1
        batch_id = metrics.batch_seq
        metrics.agg_total_tokens += num_tokens_out
        metrics.agg_total_gen_time_ms += stage_gen_time_ms

        return StageRequestMetrics(
            num_tokens_in=num_tokens_in,
            num_tokens_out=num_tokens_out,
            stage_gen_time_ms=stage_gen_time_ms,
            batch_id=batch_id,
            batch_size=1,
            replica_id=replica_id,
            rx_decode_time_ms=0.0,
            rx_transfer_bytes=0,
            rx_in_flight_time_ms=0.0,
            stage_stats=StageStats(
                total_token=metrics.agg_total_tokens,
                total_gen_time_ms=metrics.agg_total_gen_time_ms,
            ),
        )

    # ---- Stage-local admission ----

    async def submit_initial(
        self,
        request_id: str,
        req_state: OrchestratorRequestState,
        request: Any,
        *,
        prompt_text: Any = None,
        affinity_request_id: str | None = None,
        submit_kwargs: dict[str, Any] | None = None,
        params_override: Any = None,
    ) -> int:
        """Submit a stage-entry request into this pool."""
        params = params_override if params_override is not None else req_state.sampling_params_list[self.stage_id]
        # Convert plain vllm SamplingParams for single-stage diffusion models
        # that receive sampling params from the user/caller directly.
        if self.stage_type == "diffusion" and not isinstance(params, OmniDiffusionSamplingParams):
            params = OmniDiffusionSamplingParams()
        submit_kwargs = dict(submit_kwargs or {})
        if self.stage_type == "diffusion":
            replica_id = await self._pick_or_select(
                request_id,
                affinity_request_id=affinity_request_id,
            )
            client = self._diffusion_client(replica_id)
            if isinstance(request, list):
                await client.add_batch_request_async(request_id, request, params, **submit_kwargs)
            else:
                await client.add_request_async(request_id, request, params, **submit_kwargs)
            return replica_id

        replica_id = await self._pick_or_select(
            request_id,
            affinity_request_id=affinity_request_id,
        )
        client = self.clients[replica_id]
        if client is None:
            raise RuntimeError(f"stage {self.stage_id} replica {replica_id} is not attached")
        try:
            self.output_processor.add_request(
                request=request,
                prompt=prompt_text,
                parent_req=None,
                request_index=0,
                queue=None,
            )
        except Exception:
            self.release_binding(request_id)
            raise

        try:
            await self._llm_client(replica_id).add_request_async(request, **submit_kwargs)
        except Exception:
            self.release_binding(request_id)
            rollback = getattr(self.output_processor, "remove_request", None)
            if callable(rollback):
                try:
                    rollback(request_id)
                except Exception as rollback_error:
                    logger.warning(
                        "[StagePool] Failed to rollback output processor state for req=%s stage-%s: %s",
                        request_id,
                        self.stage_id,
                        rollback_error,
                    )
            raise
        return replica_id

    async def submit_update(
        self,
        request_id: str,
        req_state: OrchestratorRequestState,
        request: Any,
        *,
        prompt_text: Any = None,
    ) -> int:
        """Submit a streaming update to an already admitted request."""
        params = req_state.sampling_params_list[self.stage_id]
        if self.stage_type == "diffusion" and not isinstance(params, OmniDiffusionSamplingParams):
            params = OmniDiffusionSamplingParams()
        replica_id = self.get_bound_replica_id(request_id)
        if replica_id is None or self.clients[replica_id] is None:
            replica_id = await self._pick_or_select(request_id)

        client = self.clients[replica_id]
        if client is None:
            raise RuntimeError(f"stage {self.stage_id} replica {replica_id} is not attached")

        if self.stage_type == "diffusion":
            await self._diffusion_client(replica_id).add_request_async(request_id, request, params)
        else:
            # Refresh the shared output-processor state before yielding to the
            # stage client so streaming segments are merged against the latest
            # prompt/token metadata.
            self.output_processor.add_request(
                request=request,
                prompt=prompt_text,
                parent_req=None,
                request_index=0,
                queue=None,
            )
            await self._llm_client(replica_id).add_request_async(request)
        return replica_id

    async def _pick_or_select(
        self,
        request_id: str,
        *,
        affinity_request_id: str | None = None,
    ) -> int:
        """Bridge to ``pick`` in distributed mode or ``select_replica_id`` legacy."""
        if self.is_distributed:
            return await self.pick(request_id, affinity_request_id=affinity_request_id)
        return self.select_replica_id(request_id, affinity_request_id=affinity_request_id)

    # ---- Stage-local polling ----

    async def _poll_stage_raw(self, client: StagePoolLLMClient) -> EngineCoreOutputs | None:
        """Pull raw EngineCoreOutputs from a stage replica without processing."""
        outputs = await client.get_output_async()
        if not outputs.outputs:
            return None
        return outputs

    async def process_llm_raw_outputs(
        self,
        replica_id: int,
        raw_outputs: EngineCoreOutputs,
        iteration_stats: IterationStats | None = None,
    ) -> list[Any]:
        """Run the shared LLM output processor on one raw poll result."""
        raw_client = self.clients[replica_id]
        if raw_client is None:
            return []
        client = cast(StagePoolLLMClient, raw_client)
        processor = self.output_processor
        processed = processor.process_outputs(
            raw_outputs.outputs,
            raw_outputs.timestamp,
            iteration_stats,
        )

        if processed.reqs_to_abort:
            await client.abort_requests_async(processed.reqs_to_abort)

        if raw_outputs.scheduler_stats is not None:
            processor.update_scheduler_stats(raw_outputs.scheduler_stats)

        return processed.request_outputs

    async def poll_llm_raw_output(
        self,
        replica_id: int,
        *,
        timeout_s: float = 0.001,
    ) -> EngineCoreOutputs | None:
        """Poll raw EngineCore outputs from one LLM replica once."""
        raw_client = self.clients[replica_id]
        if raw_client is None:
            return None
        client = cast(StagePoolLLMClient, raw_client)
        try:
            return await asyncio.wait_for(
                self._poll_stage_raw(client),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            return None
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "[StagePool] _poll_stage_raw failed for stage-%s replica-%s",
                self.stage_id,
                replica_id,
            )
            raise

    def poll_diffusion_output(self, replica_id: int) -> Any | None:
        """Drain one ready diffusion output from the given replica if present."""
        raw_client = self.clients[replica_id]
        if raw_client is None:
            return None
        return cast(StagePoolDiffusionClient, raw_client).get_diffusion_output_nowait()

    # ---- Stage-local control plane ----

    async def abort_requests(self, request_ids: list[str]) -> None:
        """Abort the given requests in this stage pool.

        Request-bound abort routing stays inside the pool because route affinity
        (``request_id -> replica_id``) is pool-owned.
        """
        if not request_ids:
            return

        request_ids_by_replica: dict[int, list[str]] = {}
        for request_id in request_ids:
            replica_id = self.get_bound_replica_id(request_id)
            if replica_id is None or self.clients[replica_id] is None:
                logger.debug("[StagePool] abort: no live binding for req=%s in stage-%s", request_id, self.stage_id)
                continue
            request_ids_by_replica.setdefault(replica_id, []).append(request_id)

        for replica_id, replica_request_ids in request_ids_by_replica.items():
            client = self.clients[replica_id]
            if client is None:
                continue
            await client.abort_requests_async(replica_request_ids)

        # Clean up OutputProcessor state (e.g. mm_accumulated tensors) that
        # would otherwise leak — aborted requests never produce a final
        # EngineCoreOutput, so process_outputs() never fires its cleanup path.
        all_aborted = [rid for ids in request_ids_by_replica.values() for rid in ids]
        if all_aborted and self._output_processor is not None:
            self._output_processor.abort_requests(all_aborted, internal=True)

    async def collective_rpc(
        self,
        replica_id: int,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any] | Any:
        """Dispatch a stage-scoped control-plane RPC to one physical route."""
        kwargs = dict(kwargs or {})
        client = self.clients[replica_id]
        if client is None:
            return {
                "supported": False,
                "error": f"stage {self.stage_id} replica {replica_id} is not attached",
            }
        try:
            return await client.collective_rpc_async(
                method=method,
                timeout=timeout,
                args=args,
                kwargs=kwargs,
            )
        except Exception as exc:
            logger.exception(
                "[StagePool] collective_rpc failed: stage=%s replica=%s method=%s",
                self.stage_id,
                replica_id,
                method,
            )
            return {
                "supported": False,
                "error": str(exc),
            }

    def shutdown_replica(self, replica_id: int) -> None:
        """Shutdown one backend handle in this stage pool."""
        if replica_id >= len(self.clients):
            return
        client = self.clients[replica_id]
        if client is None:
            return
        try:
            client.shutdown()
            logger.info(
                "[StagePool] Stage %d replica %d shut down",
                self.stage_id,
                replica_id,
            )
        except Exception as e:
            logger.warning(
                "[StagePool] Failed to shutdown stage %d replica %d: %s",
                self.stage_id,
                replica_id,
                e,
            )
