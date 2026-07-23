"""Unified stage-local runtime abstraction for vLLM-Omni."""

from __future__ import annotations

import asyncio
import time as _time
from collections.abc import Mapping
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
from vllm_omni.metrics import definitions as defs
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
        self._unavailable_replicas: set[int] = set()
        self._replica_metrics: list[_ReplicaMetrics] = [_ReplicaMetrics() for _ in self.clients]
        self._output_timestamps_by_request: dict[str, list[float]] = {}
        self._non_empty_first_output_timestamps_by_request: dict[str, float] = {}
        self._audio_frames_by_request: dict[str, int] = {}
        self._audio_sample_rate_by_request: dict[str, int] = {}

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

    def add_client(self, input_addr: str, client: Any, *, replica_id: int | None = None) -> int:
        """Register a head-side client for ``input_addr``.

        Returns the assigned ``replica_id`` (index into :attr:`clients`).
        A coordinator-provided ``replica_id`` restores that exact slot after
        unregister/re-register. Without one, a known address keeps its slot
        and a new address takes the first vacant slot.
        """
        if not input_addr:
            raise ValueError("input_addr must be a non-empty string")
        if replica_id is not None and replica_id < 0:
            raise ValueError("replica_id must be non-negative")

        existing = self._addr_to_replica_id.get(input_addr)
        if replica_id is not None and existing is not None and existing != replica_id:
            raise ValueError(
                f"input address {input_addr!r} is already assigned to replica {existing}, not {replica_id}"
            )
        if replica_id is None:
            replica_id = existing
        if replica_id is None:
            replica_id = next(
                (index for index, existing_client in enumerate(self.clients) if existing_client is None),
                len(self.clients),
            )

        while len(self.clients) <= replica_id:
            self.clients.append(None)
            self._replica_metrics.append(_ReplicaMetrics())
        for known_addr, known_replica_id in list(self._addr_to_replica_id.items()):
            if known_replica_id == replica_id and known_addr != input_addr:
                self._addr_to_replica_id.pop(known_addr, None)
        self.clients[replica_id] = client
        self._addr_to_replica_id[input_addr] = replica_id
        self._unavailable_replicas.discard(replica_id)
        return replica_id

    def remove_client(self, input_addr: str) -> Any | None:
        """Remove the client at ``input_addr``. Returns the removed client or ``None``.

        Slot is marked ``None`` to preserve indices for outstanding bindings.
        """
        replica_id = self._addr_to_replica_id.get(input_addr)
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

    def replica_monitor_sample(self, replica_id: int) -> tuple[int, int]:
        """Return (outputs_queue_size, inflight) for orchestrator load diagnostics."""
        if replica_id < 0 or replica_id >= len(self.clients):
            return (0, 0)
        client = self.clients[replica_id]
        if client is None:
            return (0, 0)

        outputs_qsize = 0
        outputs_queue = getattr(client, "outputs_queue", None)
        if outputs_queue is not None:
            try:
                outputs_qsize = max(int(outputs_queue.qsize()), 0)
            except (AttributeError, NotImplementedError, RuntimeError, ValueError):
                pass

        if self._hub is not None:
            input_addr = self._client_input_addr(client)
            if input_addr is not None:
                for replica in self._hub.get_replicas_for_stage(self.stage_id).replicas:
                    if replica.input_addr == input_addr:
                        return (outputs_qsize, max(int(replica.queue_length), 0))
            # Distributed dispatch (pick): request_id -> input_addr in _affinity.
            inflight = sum(
                1 for input_addr in self._affinity.values() if self._addr_to_replica_id.get(input_addr) == replica_id
            )
        else:
            # Legacy local dispatch (select_replica_id): request_id -> replica_id.
            inflight = sum(1 for bound_replica_id in self._request_bindings.values() if bound_replica_id == replica_id)
        return (outputs_qsize, inflight)

    def release_binding(self, request_id: str) -> None:
        """Drop the route binding for *request_id* in this stage."""
        self._request_bindings.pop(request_id, None)
        self._affinity.pop(request_id, None)
        self._output_timestamps_by_request.pop(str(request_id), None)
        self._non_empty_first_output_timestamps_by_request.pop(str(request_id), None)
        self._audio_frames_by_request.pop(str(request_id), None)
        self._audio_sample_rate_by_request.pop(str(request_id), None)

    def release_bindings(self, request_ids: list[str]) -> None:
        """Drop route bindings for the given request ids in this stage."""
        for request_id in request_ids:
            self.release_binding(request_id)

    def release_replica_bindings(self, replica_id: int) -> list[str]:
        """Drop all route/session bindings owned by one physical replica."""
        released_request_ids = [
            request_id
            for request_id, bound_replica_id in list(self._request_bindings.items())
            if bound_replica_id == replica_id
        ]
        released_request_ids.extend(
            request_id
            for request_id, input_addr in list(self._affinity.items())
            if self._addr_to_replica_id.get(input_addr) == replica_id
        )
        released_request_ids = list(dict.fromkeys(released_request_ids))
        for request_id in released_request_ids:
            self.release_binding(request_id)
        return released_request_ids

    def mark_replica_unavailable(self, replica_id: int) -> list[str]:
        """Evict a failed replica from admission and release its bindings."""
        if 0 <= replica_id < self.num_replicas:
            self._unavailable_replicas.add(replica_id)
        return self.release_replica_bindings(replica_id)

    def is_replica_available(self, replica_id: int) -> bool:
        return (
            0 <= replica_id < self.num_replicas
            and self.clients[replica_id] is not None
            and replica_id not in self._unavailable_replicas
        )

    def available_replica_ids(self) -> list[int]:
        return [replica_id for replica_id in range(self.num_replicas) if self.is_replica_available(replica_id)]

    def select_replica_id(
        self,
        request_id: str,
        *,
        affinity_request_id: str | None = None,
    ) -> int:
        """Pick a replica id for *request_id* and cache the choice (legacy path)."""
        cached = self.get_bound_replica_id(request_id)
        if cached is not None and self.clients[cached] is not None and self.is_replica_available(cached):
            return cached
        if cached is not None:
            # Cached replica is dead/unavailable: drop the stale binding.
            self.release_binding(request_id)

        chosen: int | None = None
        if affinity_request_id is not None:
            parent = self.get_bound_replica_id(affinity_request_id)
            if parent is not None and self.clients[parent] is not None and self.is_replica_available(parent):
                chosen = parent

        if chosen is None:
            # Prefer replicas that are both live (client up) and available.
            live = [r for r in self.live_replica_ids() if self.is_replica_available(r)]
            if not live:
                raise RuntimeError(f"stage {self.stage_id} has no live replicas")
            if len(live) == 1:
                chosen = live[0]
            else:
                # Round-robin over live, available replicas only.
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
        request_timestamp: float,
        replica_id: int,
        sampling_params: Any | None = None,
    ) -> StageRequestMetrics:
        """Build stage metrics for outputs produced on one replica."""
        now = _time.time()
        stage_gen_time_ms = (now - submit_ts) * 1000.0

        request_id = str(getattr(request_outputs[0], "request_id", "")) if request_outputs else ""
        output_timestamps = self._output_timestamps_by_request.pop(request_id, []) if request_id else []
        non_empty_first_output_ts = (
            self._non_empty_first_output_timestamps_by_request.pop(request_id, None) if request_id else None
        )
        native_text_metrics = {}
        if request_id:
            pop_native_text_metrics = getattr(self.output_processor, "pop_native_text_metrics", None)
            if callable(pop_native_text_metrics):
                native_text_metrics = pop_native_text_metrics(request_id)
        native_generation_tokens = native_text_metrics.get("num_generation_tokens")
        num_tokens_out = (
            max(int(native_generation_tokens), 0)
            if isinstance(native_generation_tokens, int) and not isinstance(native_generation_tokens, bool)
            else count_tokens_from_outputs(request_outputs)
        )
        output_unit_type = self._infer_output_unit_type(request_outputs, token_count=num_tokens_out)
        output_unit_count = self._count_output_units(
            request_outputs,
            unit_type=output_unit_type,
            fallback_token_count=num_tokens_out,
        )
        current_audio_frames, current_audio_sample_rate, _ = self._collect_audio_metrics(request_outputs)
        accumulated_audio_frames = self._audio_frames_by_request.pop(request_id, 0) if request_id else 0
        accumulated_audio_sample_rate = self._audio_sample_rate_by_request.pop(request_id, 0) if request_id else 0
        audio_generated_frames = max(accumulated_audio_frames, current_audio_frames)
        audio_sample_rate = accumulated_audio_sample_rate or current_audio_sample_rate
        audio_duration_s = (
            float(audio_generated_frames) / float(audio_sample_rate)
            if audio_generated_frames > 0 and audio_sample_rate > 0
            else 0.0
        )
        image_pixels = self._count_image_pixels(request_outputs) if output_unit_type == "image" else 0
        num_inference_steps = self._coerce_int_scalar(getattr(sampling_params, "num_inference_steps", None))
        denoise_step_latency_ms = (
            defs.compute_denoise_step_latency(stage_gen_time_ms, num_inference_steps)
            if output_unit_type == "image"
            else 0.0
        )
        has_output_timestamps = bool(output_timestamps)
        first_ts = output_timestamps[0] if has_output_timestamps else now
        serving_time_to_first_output_ms = (
            max((non_empty_first_output_ts - request_timestamp) * 1000.0, 0.0)
            if non_empty_first_output_ts is not None
            else 0.0
        )
        remaining_ms = max((now - first_ts) * 1000.0, 0.0)
        if output_unit_count > 1 and self._is_streaming_output_unit_type(output_unit_type):
            time_per_output_unit_ms = remaining_ms / float(output_unit_count - 1)
        else:
            time_per_output_unit_ms = 0.0
        inter_output_latencies_ms = [
            max((cur_ts - prev_ts) * 1000.0, 0.0) for prev_ts, cur_ts in zip(output_timestamps, output_timestamps[1:])
        ]
        inter_output_latency_ms = (
            sum(inter_output_latencies_ms) / float(len(inter_output_latencies_ms)) if inter_output_latencies_ms else 0.0
        )
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
            audio_generated_frames=audio_generated_frames,
            audio_sample_rate=audio_sample_rate,
            audio_duration_s=audio_duration_s,
            image_pixels=image_pixels,
            denoise_step_latency_ms=denoise_step_latency_ms,
            output_unit_type=output_unit_type,
            output_unit_count=output_unit_count,
            serving_time_to_first_output_ms=serving_time_to_first_output_ms,
            time_per_output_unit_ms=time_per_output_unit_ms,
            inter_output_latency_ms=inter_output_latency_ms,
            inter_output_latencies_ms=inter_output_latencies_ms,
            vllm_ttft_ms=float(native_text_metrics.get("vllm_ttft_ms") or 0.0),
            vllm_tpot_ms=float(native_text_metrics.get("vllm_tpot_ms") or 0.0),
            vllm_itl_ms=float(native_text_metrics.get("vllm_itl_ms") or 0.0),
            vllm_itls_ms=list(native_text_metrics.get("vllm_itls_ms") or []),
        )

    def _infer_output_unit_type(self, request_outputs: list[Any], *, token_count: int) -> str:
        final_output_type = getattr(self.stage_client, "final_output_type", None)

        if self._has_image_output(request_outputs) or final_output_type in {"image", "images"}:
            return "image"
        if self._has_video_output(request_outputs) or final_output_type in {"video", "videos"}:
            return "video"
        if self._has_audio_output(request_outputs) or final_output_type == "audio":
            return "audio"
        if self._has_trajectory_latent_output(request_outputs) or self._has_latent_output(request_outputs):
            return "latent"
        if final_output_type in {"latent", "latents"}:
            return "latent"
        if final_output_type == "text":
            return "text"
        if token_count > 0:
            return "stream"
        return "other"

    @staticmethod
    def _is_streaming_output_unit_type(unit_type: str) -> bool:
        return unit_type in {"text", "stream"}

    def _count_output_units(
        self,
        request_outputs: list[Any],
        *,
        unit_type: str,
        fallback_token_count: int,
    ) -> int:
        if unit_type == "audio":
            total_frames, _, _ = self._collect_audio_metrics(request_outputs)
            if total_frames > 0:
                return total_frames
        if unit_type == "image":
            total_images = sum(self._count_images(ro) for ro in request_outputs)
            if total_images > 0:
                return total_images
        if unit_type == "video":
            total_videos = sum(self._count_videos(ro) for ro in request_outputs)
            if total_videos > 0:
                return total_videos
        if unit_type == "latent":
            total_latents = sum(
                self._count_value_units(getattr(ro, "trajectory_latents", None))
                + self._count_value_units(getattr(ro, "latents", None))
                for ro in request_outputs
            )
            if total_latents > 0:
                return total_latents
        return int(fallback_token_count)

    def _has_audio_output(self, request_outputs: list[Any]) -> bool:
        for ro in request_outputs:
            for mm_output in self._iter_multimodal_outputs(ro):
                if isinstance(mm_output, Mapping) and mm_output.get("audio") is not None:
                    return True
        return False

    def _collect_audio_metrics(self, request_outputs: list[Any]) -> tuple[int, int, float]:
        total_frames = 0
        sample_rate = 0
        for ro in request_outputs:
            for mm_output in self._iter_multimodal_outputs(ro):
                if not isinstance(mm_output, Mapping):
                    continue
                if sample_rate <= 0:
                    sample_rate = self._infer_audio_sample_rate(mm_output)
                audio_output = mm_output.get("audio")
                if audio_output is None:
                    continue
                items = audio_output if isinstance(audio_output, list) else [audio_output]
                for item in items:
                    total_frames += self._count_audio_frames(item)
        duration_s = (float(total_frames) / float(sample_rate)) if total_frames > 0 and sample_rate > 0 else 0.0
        return int(total_frames), int(sample_rate), duration_s

    @staticmethod
    def _count_audio_frames(audio_output: Any) -> int:
        shape = getattr(audio_output, "shape", None)
        if shape is not None and len(shape) > 0:
            # Audio chunks are concatenated on dim=-1 in the output processor,
            # so the frame/sample axis is the last dim (e.g. [channels, frames]).
            # Keep this aligned with serving_chat.py: audio tensors are consumed
            # as (T,), (C, T), or (B, C, T). Flattening would corrupt
            # multi-channel audio.
            return int(shape[-1])
        try:
            return len(audio_output)
        except TypeError:
            return 1

    def _infer_audio_sample_rate(self, mm_output: dict[str, Any]) -> int:
        for key in ("audio_sample_rate", "sample_rate", "sampling_rate", "sr"):
            rate = self._coerce_int_scalar(mm_output.get(key))
            if rate > 0:
                return rate
        for attr in ("audio_sample_rate", "sample_rate", "sampling_rate", "output_sample_rate"):
            rate = self._coerce_int_scalar(getattr(self.stage_client, attr, None))
            if rate > 0:
                return rate
            rate = self._coerce_int_scalar(getattr(self._stage_vllm_config, attr, None))
            if rate > 0:
                return rate
        return 0

    @classmethod
    def _coerce_int_scalar(cls, value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, (list, tuple)):
            for item in value:
                coerced = cls._coerce_int_scalar(item)
                if coerced > 0:
                    return coerced
            return 0
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return int(item())
            except Exception:
                return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _has_image_output(self, request_outputs: list[Any]) -> bool:
        for ro in request_outputs:
            if self._count_images(ro) > 0:
                return True
        return False

    def _has_video_output(self, request_outputs: list[Any]) -> bool:
        for ro in request_outputs:
            if self._count_videos(ro) > 0:
                return True
        return False

    def _has_trajectory_latent_output(self, request_outputs: list[Any]) -> bool:
        return any(self._is_non_empty_value(getattr(ro, "trajectory_latents", None)) for ro in request_outputs)

    def _has_latent_output(self, request_outputs: list[Any]) -> bool:
        return any(self._is_non_empty_value(getattr(ro, "latents", None)) for ro in request_outputs)

    def _count_images(self, request_output: Any) -> int:
        total_images = self._count_value_units(getattr(request_output, "images", None))
        for mm_output in self._iter_multimodal_outputs(request_output):
            if isinstance(mm_output, Mapping):
                total_images += self._count_value_units(mm_output.get("image"))
                total_images += self._count_value_units(mm_output.get("images"))
        return total_images

    def _count_image_pixels(self, request_outputs: list[Any]) -> int:
        total_pixels = 0
        for ro in request_outputs:
            total_pixels += self._count_image_value_pixels(getattr(ro, "images", None))
            for mm_output in self._iter_multimodal_outputs(ro):
                if isinstance(mm_output, Mapping):
                    total_pixels += self._count_image_value_pixels(mm_output.get("image"))
                    total_pixels += self._count_image_value_pixels(mm_output.get("images"))
        return total_pixels

    @classmethod
    def _count_image_value_pixels(cls, value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, (list, tuple)):
            return sum(cls._count_image_value_pixels(item) for item in value)

        size = getattr(value, "size", None)
        if isinstance(size, tuple) and len(size) >= 2:
            try:
                return int(size[0]) * int(size[1])
            except (TypeError, ValueError):
                return 0

        shape = getattr(value, "shape", None)
        if shape is None or len(shape) < 2:
            return 0
        dims = [int(dim) for dim in shape]
        if len(dims) >= 4:
            return dims[0] * dims[-2] * dims[-1]
        if len(dims) == 3 and dims[0] in (1, 3, 4):
            return dims[1] * dims[2]
        if len(dims) == 3 and dims[-1] in (1, 3, 4):
            return dims[0] * dims[1]
        return dims[-2] * dims[-1]

    def _count_videos(self, request_output: Any) -> int:
        total_videos = self._count_video_units(getattr(request_output, "video", None))
        total_videos += self._count_video_units(getattr(request_output, "videos", None))
        for mm_output in self._iter_multimodal_outputs(request_output):
            if isinstance(mm_output, Mapping):
                total_videos += self._count_video_units(mm_output.get("video"))
                total_videos += self._count_video_units(mm_output.get("videos"))
        return total_videos

    def has_non_empty_output(self, request_output: Any) -> bool:
        return self._has_non_empty_output(request_output)

    def _has_non_empty_output(self, request_output: Any) -> bool:
        final_output_type = getattr(request_output, "final_output_type", None)
        if final_output_type is None:
            final_output_type = getattr(self.stage_client, "final_output_type", None)
        if final_output_type == "text":
            return any(bool(getattr(output, "text", "")) for output in getattr(request_output, "outputs", None) or [])

        if getattr(request_output, "images", None):
            return True
        if getattr(request_output, "video", None) or getattr(request_output, "videos", None):
            return True
        if getattr(request_output, "trajectory_latents", None) is not None:
            return True
        custom_output = getattr(request_output, "_custom_output", None)
        if isinstance(custom_output, dict) and custom_output:
            return True
        for mm_output in self._iter_multimodal_outputs(request_output):
            if isinstance(mm_output, Mapping) and any(self._is_non_empty_value(value) for value in mm_output.values()):
                return True
        for output in getattr(request_output, "outputs", None) or []:
            if getattr(output, "text", None):
                return True
            token_ids = getattr(output, "token_ids", None)
            if self._is_non_empty_value(token_ids):
                return True
            cumulative_token_ids = getattr(output, "cumulative_token_ids", None)
            if self._is_non_empty_value(cumulative_token_ids):
                return True
        return False

    @staticmethod
    def _is_non_empty_value(value: Any) -> bool:
        if value is None:
            return False
        shape = getattr(value, "shape", None)
        if shape is not None:
            return all(int(dim) > 0 for dim in shape)
        try:
            return len(value) > 0
        except TypeError:
            return True

    @staticmethod
    def _count_value_units(value: Any) -> int:
        if value is None:
            return 0
        shape = getattr(value, "shape", None)
        if shape is not None:
            return int(shape[0]) if len(shape) > 0 and int(shape[0]) > 0 else 0
        try:
            return len(value)
        except TypeError:
            return 1

    @staticmethod
    def _count_video_units(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, (list, tuple)):
            return len(value)
        shape = getattr(value, "shape", None)
        if shape is not None:
            return 1 if all(int(dim) > 0 for dim in shape) else 0
        try:
            return len(value)
        except TypeError:
            return 1

    def record_output_timestamps(self, request_outputs: list[Any], *, output_ts: float | None = None) -> None:
        """Record all output timestamps and the first non-empty output timestamp."""
        output_ts = _time.time() if output_ts is None else output_ts
        for request_output in request_outputs:
            request_id = getattr(request_output, "request_id", None)
            if request_id is None:
                continue
            rid = str(request_id)
            self._output_timestamps_by_request.setdefault(rid, []).append(output_ts)
            if self._has_non_empty_output(request_output):
                self._non_empty_first_output_timestamps_by_request.setdefault(rid, output_ts)
            audio_frames, audio_sample_rate, _ = self._collect_audio_metrics([request_output])
            if audio_frames > 0:
                self._audio_frames_by_request[rid] = self._audio_frames_by_request.get(rid, 0) + audio_frames
            if self._audio_sample_rate_by_request.get(rid, 0) <= 0 and audio_sample_rate > 0:
                self._audio_sample_rate_by_request[rid] = audio_sample_rate

    def _iter_multimodal_outputs(self, request_output: object) -> list[Mapping[str, Any]]:
        multimodal_outputs: list[Mapping[str, Any]] = []
        outer_mm = getattr(request_output, "multimodal_output", None)
        if isinstance(outer_mm, Mapping) and outer_mm:
            multimodal_outputs.append(outer_mm)
        for output in getattr(request_output, "outputs", None) or []:
            inner_mm = getattr(output, "multimodal_output", None)
            if isinstance(inner_mm, Mapping) and inner_mm:
                multimodal_outputs.append(inner_mm)
        return multimodal_outputs

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
            if isinstance(request, list):
                raise ValueError(
                    "Diffusion list-prompt batch requests are no longer supported. "
                    "Submit multiple independent requests to use scheduler batching."
                )
            replica_id = await self._pick_or_select(
                request_id,
                affinity_request_id=affinity_request_id,
            )
            client = self._diffusion_client(replica_id)
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
            if isinstance(request, list):
                raise ValueError(
                    "Diffusion list-prompt batch requests are no longer supported. "
                    "Submit multiple independent requests to use scheduler batching."
                )
            await self._diffusion_client(replica_id).add_request_async(request_id, request, params)
        else:
            # Refresh the shared output-processor state before yielding to the
            # stage client so streaming segments are merged against the latest
            # prompt/token metadata.
            try:
                self.output_processor.add_request(
                    request=request,
                    prompt=prompt_text,
                    parent_req=None,
                    request_index=0,
                    queue=None,
                )
                await self._llm_client(replica_id).add_request_async(request)
            except Exception:
                rollback = getattr(self.output_processor, "remove_request", None)
                if callable(rollback):
                    try:
                        rollback(request_id)
                    except Exception as rollback_error:
                        logger.warning(
                            "[StagePool] Failed to rollback output processor update for req=%s stage-%s: %s",
                            request_id,
                            self.stage_id,
                            rollback_error,
                        )
                raise
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
        iteration_stats = IterationStats()
        processed = processor.process_outputs(
            raw_outputs.outputs,
            raw_outputs.timestamp,
            iteration_stats,
        )
        # Use the same wall-clock source as OrchestratorRequestState.stage_submit_ts.
        # EngineCoreOutputs.timestamp may use a different clock base, which would
        # make TTFO negative and get clamped to 0.
        self.record_output_timestamps(processed.request_outputs, output_ts=_time.time())

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
        if not self.is_replica_available(replica_id):
            return None
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
        if not self.is_replica_available(replica_id):
            return None
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
