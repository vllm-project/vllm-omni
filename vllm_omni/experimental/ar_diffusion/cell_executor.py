# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cell pipeline slot loop: microbatch, chain P2P, noisy KV exchange."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.experimental.ar_diffusion.cell_schedule import (
    Cell,
    CellPlan,
    Inflight,
    can_admit,
    is_group_boundary,
    rank_work,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import NoisyKVState
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


def activation_use_tensor_dict() -> bool:
    """Legacy Gloo-metadata activation path (disabled; hot path only)."""
    return False


def _pp_global_rank(pp_group: Any, group_rank: int) -> int:
    return int(pp_group.ranks[group_rank])


def _consumer_stash_meta_one(
    req: str,
    cell: Cell,
    *,
    topology: CellTopology,
    src: int,
) -> tuple[str, int, int]:
    """Stash key for one task: consumer ``(req, chunk, step)``.

    Across step replicas the consumer step is ``producer_step + 1`` when leaving
    a layer-group; same-step mid-group boundaries keep the step. ``req`` is
    required when ``R>1`` so concurrent streams do not collide.
    """
    chunk, step, _block = cell
    if topology.is_layer_group_last(src):
        step = int(step) + 1
    return (str(req), int(chunk), int(step))


def _consumer_stash_metas(
    src_tasks: tuple[tuple[str, Cell], ...],
    *,
    topology: CellTopology,
    src: int,
) -> list[tuple[str, int, int]]:
    return [_consumer_stash_meta_one(req, cell, topology=topology, src=src) for req, cell in src_tasks]


def _consumer_stash_meta(
    src_tasks: tuple[tuple[str, Cell], ...],
    *,
    topology: CellTopology,
    src: int,
) -> tuple[str, int, int]:
    """Backward-compatible single-meta helper (first task)."""
    metas = _consumer_stash_metas(src_tasks, topology=topology, src=src)
    if not metas:
        raise ValueError("src_tasks must be non-empty")
    return metas[0]


def _stash_activation_batch(
    adapter: CellAdapter,
    payload: dict[str, Any],
    metas: list[tuple[str, int, int]],
    *,
    handles: list[Any] | None = None,
) -> None:
    """Split a micro-batch activation dict into per-request stash entries."""
    if not metas:
        return
    tensors = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}
    if not tensors:
        return
    n = len(metas)
    for i, meta in enumerate(metas):
        if n == 1:
            sliced = {k: v for k, v in tensors.items()}
        else:
            sliced = {}
            for key, tensor in tensors.items():
                if tensor.shape[0] != n:
                    raise RuntimeError(
                        f"activation {key} batch {tensor.shape[0]} != microbatch {n} (R>1 pad required)"
                    )
                sliced[key] = tensor[i : i + 1]
        adapter.stash_activation(sliced, handles=handles, meta=meta)


def _stash_local_hidden(
    adapter: CellAdapter,
    tasks: tuple[tuple[str, Cell], ...],
    output: Any,
) -> None:
    """Keep mid-group activations on-rank for the next block (no P2P)."""
    if output is None or not tasks:
        return
    metas = [(req, int(cell[0]), int(cell[1])) for req, cell in tasks]
    _stash_activation_batch(adapter, adapter.pack_activation(output), metas)


def _activation_edges(
    *,
    topology: CellTopology,
    inflight: tuple[Any, ...],
    slot: int,
    schedule: Any,
) -> list[tuple[int, int, tuple[tuple[str, Cell], ...]]]:
    """Identical edge list on every rank for this tick."""
    edges: list[tuple[int, int, tuple[tuple[str, Cell], ...]]] = []
    last_rank = topology.world - 1
    if topology.step_replicas == 1:
        peer_tasks = rank_work(inflight, slot, last_rank)
        if _tasks_need_activation_send(peer_tasks, schedule):
            edges.append((last_rank, 0, peer_tasks))
    for src in range(last_rank):
        dst = src + 1
        src_tasks = rank_work(inflight, slot, src)
        if _tasks_need_activation_send(src_tasks, schedule):
            edges.append((src, dst, src_tasks))
    return edges


def _warmup_activation_p2p(pp_group: Any, *, world: int, rank: int, device: torch.device) -> None:
    """One ring P2P so NCCL channels exist before the hot path."""
    if world <= 1 or not torch.cuda.is_available():
        return
    group = pp_group.device_group
    send_buf = torch.zeros(1, device=device, dtype=torch.float32)
    recv_buf = torch.empty(1, device=device, dtype=torch.float32)
    nxt = _pp_global_rank(pp_group, (rank + 1) % world)
    prv = _pp_global_rank(pp_group, (rank - 1 + world) % world)
    ops = [
        dist.P2POp(dist.isend, send_buf, nxt, group),
        dist.P2POp(dist.irecv, recv_buf, prv, group),
    ]
    for work in dist.batch_isend_irecv(ops):
        work.wait()


def _post_activations_ws(
    pp_group: Any,
    edges: list[tuple[int, int, tuple[tuple[str, Cell], ...]]],
    *,
    rank: int,
    output: Any | None,
    adapter: CellAdapter,
    topology: CellTopology,
    device: torch.device,
    comm_stream: torch.cuda.Stream | None = None,
) -> tuple[Any, list[torch.Tensor]] | None:
    """Fixed-shape activation P2P on one ``batch_isend_irecv``.

    With ``comm_stream`` (shared with KV), posts on that stream, waits Works
    there, records a completion Event, and stashes recv buffers with the Event
    as the consume-time ticket — host does not block. Returns ``(event,
    send_keepalives)`` for the producer to poll-retire; ``None`` when sync or
    there were no sends.
    """
    ops: list[dist.P2POp] = []
    recv_jobs: list[tuple[dict[str, torch.Tensor], list[tuple[str, int, int]]]] = []
    send_kept: list[torch.Tensor] = []
    group = pp_group.device_group
    use_stream = comm_stream is not None and device.type == "cuda"

    for src, dst, src_tasks in edges:
        batch = max(1, len(src_tasks))
        include_hidden = not topology.is_layer_group_last(src)
        spec_list = adapter.activation_spec(batch=batch, include_hidden=include_hidden)
        if not spec_list:
            continue
        keys = tuple(k for k, _shape, _dtype in spec_list)
        if rank == src and output is not None:
            packed = adapter.pack_activation(output)
            peer = _pp_global_rank(pp_group, dst)
            for key in keys:
                tensor = packed[key]
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                if tensor.shape[0] != batch:
                    raise RuntimeError(
                        f"activation send {key} batch {tensor.shape[0]} != edge microbatch {batch}"
                    )
                ops.append(dist.P2POp(dist.isend, tensor, peer, group))
                if tensor.is_cuda:
                    tensor.record_stream(comm_stream or torch.cuda.current_stream(device))
                send_kept.append(tensor)
        elif rank == dst:
            peer = _pp_global_rank(pp_group, src)
            payload = {key: torch.empty(shape, dtype=dtype, device=device) for key, shape, dtype in spec_list}
            for key, _shape, _dtype in spec_list:
                tensor = payload[key]
                ops.append(dist.P2POp(dist.irecv, tensor, peer, group))
                if tensor.is_cuda and use_stream:
                    tensor.record_stream(comm_stream)
            recv_jobs.append((payload, _consumer_stash_metas(src_tasks, topology=topology, src=src)))

    if not ops:
        return None

    if use_stream:
        assert comm_stream is not None
        comm_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(comm_stream):
            works = list(dist.batch_isend_irecv(ops))
            _wait(works)
            event = torch.cuda.Event()
            event.record(comm_stream)
        for payload, metas in recv_jobs:
            _stash_activation_batch(adapter, payload, metas, handles=[event])
        return (event, send_kept) if send_kept else None

    _wait(list(dist.batch_isend_irecv(ops)))
    for payload, metas in recv_jobs:
        _stash_activation_batch(adapter, payload, metas)
    return None


def resolve_pp_rank_and_group() -> tuple[int, Any | None]:
    """Read the existing PP group; never creates a new process group."""
    try:
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_pipeline_parallel_rank,
            get_pp_group,
        )

        return int(get_pipeline_parallel_rank()), get_pp_group()
    except Exception:
        return 0, None


@dataclass(frozen=True)
class CellTopology:
    step_replicas: int
    layer_groups: int

    @property
    def world(self) -> int:
        return self.step_replicas * self.layer_groups

    def split_rank(self, rank: int) -> tuple[int, int]:
        if self.step_replicas == 1:
            return 0, rank
        return rank // self.layer_groups, rank % self.layer_groups

    def is_layer_group_first(self, rank: int) -> bool:
        """``g == 0`` for this rank's step replica."""
        _, g = self.split_rank(rank)
        return g == 0

    def is_layer_group_last(self, rank: int) -> bool:
        """``g == G-1`` for this rank's step replica."""
        _, g = self.split_rank(rank)
        return g == self.layer_groups - 1


@dataclass
class CellRunSpec:
    topology: CellTopology
    max_batch_size: int = 1
    rank: int = 0
    pp_group: Any | None = None


@dataclass(frozen=True)
class PendingAdmit:
    req: str
    plan: CellPlan
    chunk_tokens: int


@dataclass
class ARDiffusionCellContext:
    spec: CellRunSpec
    kv: NoisyKVState
    inflight: list[Inflight] = field(default_factory=list)
    pending: list[PendingAdmit] = field(default_factory=list)

    def enqueue(self, req: str, plan: CellPlan, *, chunk_tokens: int) -> None:
        self.pending.append(PendingAdmit(req=req, plan=plan, chunk_tokens=chunk_tokens))

    def admit(self, req: str, plan: CellPlan, *, chunk_tokens: int, slot: int) -> bool:
        pending = tuple(self.inflight)
        if not can_admit(pending, slot, plan, max_batch_size=self.spec.max_batch_size):
            return False
        inflight_tokens = {self.kv._chunk_tokens[item.req] for item in self.inflight}
        if inflight_tokens and chunk_tokens not in inflight_tokens:
            return False
        self.kv.begin_request(req, plan, chunk_tokens=chunk_tokens, t0=slot)
        self.inflight.append(Inflight(req=req, t0=slot, plan=plan))
        self.kv.set_inflight(tuple(self.inflight))
        return True

    def drop(self, req: str) -> None:
        self.inflight = [item for item in self.inflight if item.req != req]
        self.pending = [item for item in self.pending if item.req != req]
        self.kv.end_request(req)
        self.kv.set_inflight(tuple(self.inflight))

    def admit_pending(self, slot: int) -> None:
        """Admit as many pending requests as ``max_batch_size`` lanes allow."""
        still: list[PendingAdmit] = []
        for item in self.pending:
            if still:
                still.append(item)
                continue
            if self.admit(item.req, item.plan, chunk_tokens=item.chunk_tokens, slot=slot):
                continue
            still.append(item)
        self.pending = still


class CellAdapter:
    """Model-side cell step: one block per request on this rank."""

    def forward(
        self,
        tasks: tuple[tuple[str, Cell], ...],
        kv_contexts: list[list[Any]],
        *,
        hidden: Any | None,
    ) -> Any:
        raise NotImplementedError

    def pack_activation(self, output: Any) -> dict:
        raise NotImplementedError

    def unpack_activation(self, payload: dict) -> Any:
        raise NotImplementedError

    def activation_spec(
        self, *, batch: int, include_hidden: bool
    ) -> list[tuple[str, tuple[int, ...], torch.dtype]] | None:
        """Fixed NCCL recv schema ``[(key, shape, dtype), ...]``, or ``None`` for tensor_dict."""
        del batch, include_hidden
        return None

    def stash_activation(
        self,
        payload: dict,
        *,
        handles: list[Any] | None = None,
        meta: tuple[str, int, int] | None = None,
    ) -> None:
        """Buffer peer activation until the matching cell runs.

        ``handles`` may hold a CUDA Event completion ticket (async path) or
        Work objects; ``take_stashed_activation`` waits them before use.
        """
        del payload, handles, meta

    def take_stashed_activation(self, req: str, chunk: int, step: int) -> Any | None:
        del req, chunk, step
        return None


def _wait(handles: list[Any]) -> None:
    for handle in handles:
        if handle is not None:
            handle.wait()


def _assert_i8(tasks: tuple[tuple[str, Cell], ...]) -> None:
    reqs = [req for req, _task in tasks]
    if len(reqs) != len(set(reqs)):
        raise RuntimeError(f"I8 violated: same request appears twice in one microbatch: {tasks}")


def _tasks_need_activation_send(
    tasks: tuple[tuple[str, Cell], ...],
    schedule: Any | None,
) -> bool:
    """Activation P2P only after the last local block of a layer group."""
    if not tasks or schedule is None:
        return bool(tasks)
    return any(is_group_boundary(cell, schedule) for _req, cell in tasks)


def _poll_act_send_retire(retire: deque[tuple[Any, list[torch.Tensor]]]) -> None:
    """Drop keepalive refs for activation sends whose ticket has completed."""
    while retire:
        event, _kept = retire[0]
        if not event.query():
            break
        retire.popleft()


def _drain_act_send_retire(retire: deque[tuple[Any, list[torch.Tensor]]]) -> None:
    for event, _kept in retire:
        event.synchronize()
    retire.clear()


def run_cell_pipeline(
    *,
    ctx: ARDiffusionCellContext,
    adapter: CellAdapter,
    max_slots: int | None = None,
) -> None:
    """Run inflight plans on this rank (one cell per tick).

    Per tick: prepare → forward → publish → post act/KV (async tickets when
    comm stream is on) → poll outbound. Mid-group cells keep ``hidden``
    locally; group-boundary cells P2P.
    """
    spec = ctx.spec
    kv = ctx.kv
    kv.bind_rank(spec.rank, spec.pp_group)
    pp = spec.pp_group
    logger.info(
        "[chunk_pipeline] start: step_replicas=%d layer_groups=%d rank=%d inflight=%d pending=%d",
        spec.topology.step_replicas,
        spec.topology.layer_groups,
        spec.rank,
        len(ctx.inflight),
        len(ctx.pending),
    )
    slot = 0
    limit = max_slots if max_slots is not None else 10**9
    act_send_retire: deque[tuple[Any, list[torch.Tensor]]] = deque()
    try:
        kv_stream = kv.uses_comm_stream
        prev_slot_for_evict: int | None = None
        act_device = getattr(adapter, "device", None)
        if act_device is None:
            act_device = (
                current_omni_platform.get_torch_device()
                if current_omni_platform.get_device_count() > 0
                else torch.device("cpu")
            )
        use_ws_act = (
            pp is not None
            and spec.topology.world > 1
            and not activation_use_tensor_dict()
            and adapter.activation_spec(batch=1, include_hidden=False) is not None
        )
        if use_ws_act:
            _warmup_activation_p2p(pp, world=spec.topology.world, rank=spec.rank, device=act_device)
        while (ctx.inflight or ctx.pending) and slot < limit:
            ctx.admit_pending(slot)
            kv.set_inflight(tuple(ctx.inflight))
            tasks = rank_work(tuple(ctx.inflight), slot, spec.rank)
            _assert_i8(tasks)
            schedule = ctx.inflight[0].plan.schedule if ctx.inflight else None
            contexts: list[Any] = []
            output = None
            if tasks:
                # Consume-time wait: covers prefetch that landed many ticks earlier.
                kv.await_task_sources(tasks)
                contexts = kv.prepare(tasks)
                # Activation comes from per-req stash (local mid-group or P2P).
                output = adapter.forward(tasks, contexts, hidden=None)
                kv.publish(tasks)
                if schedule is not None and not _tasks_need_activation_send(tasks, schedule):
                    _stash_local_hidden(adapter, tasks, output)
            if kv_stream and prev_slot_for_evict is not None:
                # Poll-based retire inside evict; the hot path never host-waits sends.
                kv.evict(prev_slot_for_evict)
                prev_slot_for_evict = None
            _poll_act_send_retire(act_send_retire)
            comm_handles: list[Any] = []
            if pp is not None and spec.topology.world > 1 and schedule is not None:
                edges = _activation_edges(
                    topology=spec.topology,
                    inflight=tuple(ctx.inflight),
                    slot=slot,
                    schedule=schedule,
                )
                if not use_ws_act:
                    for src, dst, src_tasks in edges:
                        if spec.rank == src and output is not None:
                            handles = pp.isend_tensor_dict(adapter.pack_activation(output), dst=dst)
                            comm_handles.extend(handles)
                        elif spec.rank == dst:
                            recv, recv_handles, _ = pp.irecv_tensor_dict(src=src)
                            comm_handles.extend(recv_handles)
                            _stash_activation_batch(
                                adapter,
                                recv,
                                _consumer_stash_metas(src_tasks, topology=spec.topology, src=src),
                            )
                elif edges:
                    # Share KV's comm stream so act+KV keep NCCL peer FIFO order.
                    comm_stream = kv.ensure_comm_stream() if kv_stream else None
                    retire = _post_activations_ws(
                        pp,
                        edges,
                        rank=spec.rank,
                        output=output,
                        adapter=adapter,
                        topology=spec.topology,
                        device=act_device,
                        comm_stream=comm_stream,
                    )
                    if retire is not None:
                        act_send_retire.append(retire)
            kv_handles = kv.exchange(slot)
            if not kv_stream:
                comm_handles.extend(kv_handles)
            # Legacy act / sync KV only. Async act+KV use device tickets.
            _wait(comm_handles)
            still = []
            for item in ctx.inflight:
                local = slot - item.t0
                if local + 1 < item.plan.num_slots:
                    still.append(item)
                else:
                    kv.end_request(item.req)
            ctx.inflight = still
            kv.set_inflight(tuple(ctx.inflight))
            # End-of-slot wait for remotes first consumed at slot+1 (plan wait_ready).
            kv.await_ready(slot)
            if kv_stream:
                prev_slot_for_evict = slot
            else:
                kv.evict(slot)
            slot += 1
        if kv_stream and prev_slot_for_evict is not None:
            kv.evict(prev_slot_for_evict)
        _drain_act_send_retire(act_send_retire)
        kv.drain()
    except Exception:
        _drain_act_send_retire(act_send_retire)
        for req in {item.req for item in ctx.inflight} | {item.req for item in ctx.pending}:
            ctx.drop(req)
        raise


@contextmanager
def bind_ar_diffusion_cell_context(owner: Any, ctx: ARDiffusionCellContext) -> Iterator[ARDiffusionCellContext]:
    prev = getattr(owner, "_ar_diffusion_cell_context", None)
    owner._ar_diffusion_cell_context = ctx
    try:
        yield ctx
    finally:
        owner._ar_diffusion_cell_context = prev
