# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cell pipeline slot loop: microbatch, chain P2P, noisy KV exchange."""

from __future__ import annotations

from collections.abc import Collection, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.batch import (
    ComputeBatch,
    organize_batches,
    slice_activation_for_task,
)
from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import (
    Cell,
    CellPlan,
    Inflight,
    Ordering,
    can_admit,
    is_group_boundary,
    rank_of_block,
    rank_work,
)
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


def activation_use_tensor_dict() -> bool:
    """Legacy Gloo-metadata activation path (disabled; hot path only)."""
    return False


def _pp_global_rank(pp_group: Any, group_rank: int) -> int:
    return int(pp_group.ranks[group_rank])


def _consumer_stash_meta_for_task(
    req: str,
    cell: Cell,
    *,
    topology: CellTopology,
    src: int,
) -> tuple[str, int, int]:
    """Stash key for one member: ``(request_id, consumer_chunk, consumer_step)``.

    Producer schedule meta alone is wrong across step replicas: consumer step is
    ``producer_step + 1`` when leaving a denoise stage. Mid-group boundaries keep
    the step (next block on the same chunk/step).
    """
    chunk, step, _block = cell
    if topology.is_layer_group_last(src):
        step = int(step) + 1
    return (str(req), int(chunk), int(step))


def _stash_activation_rows(
    act_stash: dict[tuple[str, int, int], dict[str, torch.Tensor]],
    payload: Mapping[str, torch.Tensor] | dict,
    src_tasks: tuple[tuple[str, Cell], ...],
    *,
    topology: CellTopology,
    src: int,
) -> None:
    """Split a packed PP activation by send-time member order into per-request rows."""
    n = len(src_tasks)
    for i, (req, cell) in enumerate(src_tasks):
        row = slice_activation_for_task(payload, task_index=i, task_count=n)
        if row is None:
            continue
        meta = _consumer_stash_meta_for_task(req, cell, topology=topology, src=src)
        _stash_activation(act_stash, row, meta=meta)


def _activation_edges(
    *,
    topology: CellTopology,
    inflight: tuple[Any, ...],
    slot: int,
    schedule: Any,
) -> list[tuple[int, int, tuple[tuple[str, Cell], ...]]]:
    """Identical edge list on every rank for this tick."""
    edges: list[tuple[int, int, tuple[tuple[str, Cell], ...]]] = []
    if schedule.ordering is Ordering.SAME_STEP:
        for item in inflight:
            plan = item.plan
            for src in range(topology.world):
                task = plan.task(slot - item.t0, src)
                if task is None or not is_group_boundary(task, plan.schedule):
                    continue
                chunk, step, block = task
                if block + 1 < plan.schedule.num_blocks:
                    next_step, next_block = step, block + 1
                elif step + 1 < plan.schedule.num_denoise_steps or (
                    chunk == 0 and plan.schedule.chunks > 1 and step < plan.schedule.num_denoise_steps
                ):
                    next_step, next_block = step + 1, 0
                else:
                    continue
                dst = rank_of_block(next_step, next_block, plan.schedule)
                if src != dst:
                    edges.append((src, dst, ((item.req, task),)))
        return edges
    schedules = {item.req: item.plan.schedule for item in inflight}
    last_rank = topology.world - 1
    if topology.step_replicas == 1:
        peer_tasks = tuple(
            (req, cell) for req, cell in rank_work(inflight, slot, last_rank) if is_group_boundary(cell, schedules[req])
        )
        if peer_tasks:
            edges.append((last_rank, 0, peer_tasks))
    for src in range(last_rank):
        dst = src + 1
        src_tasks = tuple(
            (req, cell) for req, cell in rank_work(inflight, slot, src) if is_group_boundary(cell, schedules[req])
        )
        if src_tasks:
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


def _stash_activation(
    stash: dict[tuple[str, int, int], dict[str, torch.Tensor]],
    payload: dict,
    *,
    meta: tuple[str, int, int] | None,
) -> None:
    if meta is None:
        raise RuntimeError("stash_activation needs meta")
    tensors = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor) and k not in ("chunk", "step")}
    stash[meta] = tensors


def _post_activations_ws(
    pp_group: Any,
    edges: list[tuple[int, int, tuple[tuple[str, Cell], ...]]],
    *,
    rank: int,
    task_outs: Sequence[dict[str, torch.Tensor] | None],
    tasks: Sequence[tuple[str, Cell]],
    packed_output: Any | None,
    members: Mapping[str, CellAdapter],
    adapter: CellAdapter,
    topology: CellTopology,
    device: torch.device,
    act_stash: dict[tuple[str, int, int], dict[str, torch.Tensor]],
) -> None:
    """Fixed-shape activation P2P: one ``batch_isend_irecv``, wait before return."""
    ops: list[dist.P2POp] = []
    recv_jobs: list[tuple[dict[str, torch.Tensor], tuple[tuple[str, Cell], ...], int]] = []
    group = pp_group.device_group

    for src, dst, src_tasks in edges:
        batch = max(1, len(src_tasks))
        include_hidden = not topology.is_layer_group_last(src)
        send_adapter = members.get(src_tasks[0][0], adapter) if src_tasks else adapter
        spec_list = send_adapter.activation_spec(batch=batch, include_hidden=include_hidden)
        if not spec_list:
            continue
        keys = tuple(k for k, _shape, _dtype in spec_list)
        edge_out = (
            packed_output
            if packed_output is not None and len(src_tasks) == len(tasks)
            else _pack_outs_for_tasks(task_outs, tasks, src_tasks)
        )
        if rank == src:
            if edge_out is None:
                raise RuntimeError(f"Missing activation output for edge {src}->{dst}: {src_tasks}")
            packed = send_adapter.pack_activation(edge_out)
            peer = _pp_global_rank(pp_group, dst)
            for key in keys:
                tensor = packed[key]
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                ops.append(dist.P2POp(dist.isend, tensor, peer, group))
        elif rank == dst:
            peer = _pp_global_rank(pp_group, src)
            payload = {key: torch.empty(shape, dtype=dtype, device=device) for key, shape, dtype in spec_list}
            for key, _shape, _dtype in spec_list:
                ops.append(dist.P2POp(dist.irecv, payload[key], peer, group))
            recv_jobs.append((payload, src_tasks, src))

    if ops:
        _wait(list(dist.batch_isend_irecv(ops)))
    for payload, src_tasks, src_rank in recv_jobs:
        _stash_activation_rows(act_stash, payload, src_tasks, topology=topology, src=src_rank)


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
    retain: frozenset[Cell] = field(default_factory=frozenset)


@runtime_checkable
class SupportsNoisyPPKV(Protocol):
    """KV ops the cell loop needs — filled by experimental ``NoisyKVState``.

    Mainline plan/executor only depend on this structural interface (same idea as
    stepwise ``SupportsStepExecution``), not on ``experimental...kv_cache.noisy``.

    ``end_request(..., retain=None)`` keeps the existing protect set (normal
    plan completion); ``retain=()`` clears retained versions (fail-closed drop).
    ``slot`` arguments are global pipeline tick indices, not inflight lists.
    """

    @property
    def uses_comm_stream(self) -> bool: ...

    def bind_rank(self, rank: int, pp_group: Any | None) -> None: ...

    def begin_request(
        self,
        req: str,
        plan: CellPlan,
        *,
        chunk_tokens: int,
        t0: int = 0,
        retain: Collection[Cell] | None = None,
    ) -> None: ...

    def set_inflight(self, inflight: Sequence[Inflight]) -> None: ...

    def end_request(self, req: str, *, retain: Collection[Cell] | None = None) -> None: ...

    def chunk_tokens(self, req: str) -> int: ...

    def exchange_and_await_prior_handoffs(self, slot: int) -> list[Any]: ...

    def await_task_sources(self, tasks: Sequence[tuple[str, Cell]]) -> int: ...

    def prepare(self, tasks: Sequence[tuple[str, Cell]]) -> list[Any]: ...

    def publish(self, tasks: Sequence[tuple[str, Cell]]) -> None: ...

    def wait_sends(self) -> None: ...

    def evict(self, slot: int) -> None: ...

    def exchange(self, slot: int) -> list[Any]: ...

    def await_ready(self, slot: int) -> int: ...

    def drain(self) -> None: ...


@dataclass
class ARDiffusionCellContext:
    spec: CellRunSpec
    kv: SupportsNoisyPPKV
    inflight: list[Inflight] = field(default_factory=list)
    pending: list[PendingAdmit] = field(default_factory=list)

    def enqueue(
        self,
        req: str,
        plan: CellPlan,
        *,
        chunk_tokens: int,
        retain: Collection[Cell] | None = None,
    ) -> None:
        self.pending.append(
            PendingAdmit(
                req=req,
                plan=plan,
                chunk_tokens=chunk_tokens,
                retain=frozenset(retain or ()),
            )
        )

    def admit(
        self,
        req: str,
        plan: CellPlan,
        *,
        chunk_tokens: int,
        slot: int,
        retain: Collection[Cell] | None = None,
    ) -> bool:
        pending = tuple(self.inflight)
        # Sink bootstrap contains a replica-local wrap. Keep its activation
        # packets request-local until mixed-bootstrap batching is supported.
        if pending and (
            plan.schedule.ordering is Ordering.SAME_STEP
            or any(item.plan.schedule.ordering is Ordering.SAME_STEP for item in pending)
        ):
            return False
        if not can_admit(pending, slot, plan, max_batch_size=self.spec.max_batch_size):
            return False
        inflight_tokens = {self.kv.chunk_tokens(item.req) for item in self.inflight}
        if inflight_tokens and chunk_tokens not in inflight_tokens:
            return False
        self.kv.begin_request(req, plan, chunk_tokens=chunk_tokens, t0=slot, retain=retain)
        self.inflight.append(Inflight(req=req, t0=slot, plan=plan))
        self.kv.set_inflight(tuple(self.inflight))
        return True

    def drop(self, req: str) -> None:
        self.inflight = [item for item in self.inflight if item.req != req]
        self.pending = [item for item in self.pending if item.req != req]
        # Fail-closed: drop session history for this id as well as temporary cells.
        self.kv.end_request(req, retain=())
        self.kv.set_inflight(tuple(self.inflight))

    def admit_pending(self, slot: int) -> None:
        still: list[PendingAdmit] = []
        for item in self.pending:
            if still:
                still.append(item)
                continue
            if self.admit(
                item.req,
                item.plan,
                chunk_tokens=item.chunk_tokens,
                slot=slot,
                retain=item.retain,
            ):
                continue
            still.append(item)
        self.pending = still


class CellAdapter:
    """Model backend for one request's Cell compute.

    The executor organizes ``ComputeBatch`` groups. Adapters only declare
    packing compatibility and run one batch.
    """

    def compatibility_key(self, cell: Cell) -> Hashable:
        """Model-declared key for which Cells may share one compute call."""
        del cell
        raise NotImplementedError

    def forward(
        self,
        batch: ComputeBatch,
        members: Sequence[CellAdapter],
        *,
        member_inputs: Sequence[Any | None],
    ) -> dict[str, torch.Tensor]:
        """Pack inputs → model → split/update per-member state; return packed tensors."""
        del batch, members, member_inputs
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


def _wait(handles: list[Any]) -> None:
    for handle in handles:
        if handle is not None:
            handle.wait()


def _assert_i8(tasks: tuple[tuple[str, Cell], ...]) -> None:
    reqs = [req for req, _task in tasks]
    if len(reqs) != len(set(reqs)):
        raise RuntimeError(f"I8 violated: same request appears twice in one microbatch: {tasks}")


def _tensor_dicts_concatable(typed: Sequence[Mapping[str, torch.Tensor]]) -> bool:
    """True when every dict shares keys and trailing tensor shapes (dim 1+)."""
    if len(typed) <= 1:
        return True
    keys = set(typed[0].keys())
    if any(set(item.keys()) != keys for item in typed[1:]):
        return False
    for key in keys:
        ref = typed[0][key]
        if not isinstance(ref, torch.Tensor):
            return False
        ref_tail = ref.shape[1:]
        for item in typed[1:]:
            tensor = item[key]
            if not isinstance(tensor, torch.Tensor) or tensor.shape[1:] != ref_tail:
                return False
    return True


def _pack_task_outputs(outs: list[dict[str, torch.Tensor] | None]) -> dict[str, torch.Tensor] | None:
    """Cat slot members along dim 0 when geometries match; else keep them unmerged.

    Compatibility grouping already isolates DiT compute. This pack is only for
    PP activation transport. Mixed latent H/W (same token count) must not
    ``torch.cat`` here — callers keep using per-task ``task_outs``.
    """
    typed = [item for item in outs if item is not None]
    if not typed:
        return None
    if len(typed) == 1:
        return typed[0]
    if not _tensor_dicts_concatable(typed):
        return None
    keys = typed[0].keys()
    return {k: torch.cat([o[k] for o in typed], dim=0) for k in keys}


def _pack_outs_for_tasks(
    task_outs: Sequence[dict[str, torch.Tensor] | None],
    tasks: Sequence[tuple[str, Cell]],
    src_tasks: Sequence[tuple[str, Cell]],
) -> dict[str, torch.Tensor] | None:
    index = {task: i for i, task in enumerate(tasks)}
    selected: list[dict[str, torch.Tensor] | None] = []
    for task in src_tasks:
        i = index.get(task)
        selected.append(None if i is None else task_outs[i])
    return _pack_task_outputs(selected)


def _member_pack_geometry(adapter: CellAdapter, *, include_hidden: bool) -> Hashable:
    """Rank-shared activation geometry used to decide whether edge members can pack.

    Must not depend on local ``task_outs``: every rank expands the same
    ``_activation_edges`` list and must agree on message count/size.
    """
    spec = adapter.activation_spec(batch=1, include_hidden=include_hidden)
    if spec:
        return tuple((key, tuple(shape[1:]), str(dtype)) for key, shape, dtype in spec)
    shape = getattr(adapter, "latent_shape", None)
    dtype = getattr(adapter, "dtype", None)
    return ("latent_shape", shape, str(dtype) if dtype is not None else None)


def _expand_activation_edges_for_geometry(
    edges: list[tuple[int, int, tuple[tuple[str, Cell], ...]]],
    *,
    members: Mapping[str, CellAdapter],
    topology: CellTopology,
    fallback: CellAdapter,
) -> list[tuple[int, int, tuple[tuple[str, Cell], ...]]]:
    """Split multi-member edges when member geometries cannot share one packed send.

    Uses only member metadata available on every rank. Local compute outputs are
    intentionally ignored so senders and receivers expand identically even when
    they are working on different Cells this tick.
    """
    expanded: list[tuple[int, int, tuple[tuple[str, Cell], ...]]] = []
    for src, dst, src_tasks in edges:
        if len(src_tasks) <= 1:
            expanded.append((src, dst, src_tasks))
            continue
        include_hidden = not topology.is_layer_group_last(src)
        geos = [
            _member_pack_geometry(members.get(req, fallback), include_hidden=include_hidden) for req, _cell in src_tasks
        ]
        if all(geo == geos[0] for geo in geos[1:]):
            expanded.append((src, dst, src_tasks))
            continue
        for task in src_tasks:
            expanded.append((src, dst, (task,)))
    return expanded


def _run_slot_batches(
    *,
    tasks: tuple[tuple[str, Cell], ...],
    contexts: list[Any],
    members: Mapping[str, CellAdapter],
    max_batch_size: int,
    pending_hidden: dict[str, Any],
    act_stash: dict[tuple[str, int, int], dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor] | None]:
    """Organize compatible Batches, map per-member activations, run model forward."""
    batches = organize_batches(
        tasks,
        contexts,
        compatibility_of=lambda req, cell: members[req].compatibility_key(cell),
        max_batch_size=max_batch_size,
    )
    outs: list[dict[str, torch.Tensor] | None] = [None] * len(tasks)
    for batch in batches:
        member_adapters = [members[m.request_id] for m in batch.members]
        member_inputs: list[Any | None] = []
        for member_ref in batch.members:
            req = member_ref.request_id
            # Local mid-group / clean handoff is request-keyed; PP rows use the
            # consumer (req, chunk, step) written at send time.
            if req in pending_hidden:
                member_inputs.append(pending_hidden.pop(req))
            else:
                member_inputs.append(act_stash.pop((req, int(member_ref.chunk), int(member_ref.step)), None))
        packed = member_adapters[0].forward(
            batch,
            member_adapters,
            member_inputs=member_inputs,
        )
        for local_i, part in enumerate(batch.split_outputs(packed)):
            outs[batch.members[local_i].task_index] = part
    return outs


def run_cell_pipeline(
    *,
    ctx: ARDiffusionCellContext,
    members: Mapping[str, CellAdapter],
    max_slots: int | None = None,
) -> None:
    """Run inflight plans on this rank (one cell per tick).

    Per tick: prepare → organize Batches → compute → publish → post act/KV.
    ``members`` maps request/kv id to the adapter that owns its state.
    """
    member_map = members
    if not member_map:
        raise ValueError("run_cell_pipeline requires a non-empty members mapping")
    primary = next(iter(member_map.values()))
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
    # Request-keyed local handoff (mid-group / same-rank clean). PP rows live in act_stash.
    pending_hidden: dict[str, Any] = {}
    act_stash: dict[tuple[str, int, int], dict[str, torch.Tensor]] = {}
    try:
        kv_stream = kv.uses_comm_stream
        prev_slot_for_evict: int | None = None
        act_device = getattr(primary, "device", None)
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
            and primary.activation_spec(batch=1, include_hidden=False) is not None
        )
        if use_ws_act:
            _warmup_activation_p2p(pp, world=spec.topology.world, rank=spec.rank, device=act_device)
        while (ctx.inflight or ctx.pending) and slot < limit:
            ctx.admit_pending(slot)
            kv.set_inflight(tuple(ctx.inflight))
            # Prior-history handoffs for requests that start on this global slot must
            # complete before their first prepare (exchange normally runs after forward).
            kv.exchange_and_await_prior_handoffs(slot)
            tasks = rank_work(tuple(ctx.inflight), slot, spec.rank)
            _assert_i8(tasks)
            schedule = ctx.inflight[0].plan.schedule if ctx.inflight else None
            contexts: list[Any] = []
            output = None
            task_outs: list[dict[str, torch.Tensor] | None] = []
            if tasks:
                # Consume-time wait: covers prefetch that landed many ticks earlier.
                kv.await_task_sources(tasks)
                contexts = kv.prepare(tasks)
                task_outs = _run_slot_batches(
                    tasks=tasks,
                    contexts=contexts,
                    members=member_map,
                    max_batch_size=max(1, int(spec.max_batch_size)),
                    pending_hidden=pending_hidden,
                    act_stash=act_stash,
                )
                output = _pack_task_outputs(task_outs)
                kv.publish(tasks)
                # A local clean successor consumes the chunk-0 final latent here.
                # The executor owns this handoff; adapters need no latent cache.
                schedules = {item.req: item.plan.schedule for item in ctx.inflight}
                for (req, cell), part in zip(tasks, task_outs, strict=True):
                    member_schedule = schedules[req]
                    keep_clean_input = (
                        member_schedule.ordering is Ordering.SAME_STEP
                        and member_schedule.chunks > 1
                        and rank_of_block(member_schedule.num_denoise_steps, 0, member_schedule) == spec.rank
                        and cell == (0, member_schedule.num_denoise_steps - 1, member_schedule.num_blocks - 1)
                    )
                    if part is not None and (keep_clean_input or not is_group_boundary(cell, member_schedule)):
                        pending_hidden[req] = part
                    else:
                        pending_hidden.pop(req, None)
            if kv_stream and prev_slot_for_evict is not None:
                kv.wait_sends()
                kv.evict(prev_slot_for_evict)
                prev_slot_for_evict = None
            comm_handles: list[Any] = []
            if pp is not None and spec.topology.world > 1 and schedule is not None:
                edges = _activation_edges(
                    topology=spec.topology,
                    inflight=tuple(ctx.inflight),
                    slot=slot,
                    schedule=schedule,
                )
                edges = _expand_activation_edges_for_geometry(
                    edges,
                    members=member_map,
                    topology=spec.topology,
                    fallback=primary,
                )
                if not use_ws_act:
                    for src, dst, src_tasks in edges:
                        edge_out = (
                            output
                            if output is not None and len(src_tasks) == len(tasks)
                            else _pack_outs_for_tasks(task_outs, tasks, src_tasks)
                        )
                        if spec.rank == src:
                            if edge_out is None:
                                raise RuntimeError(f"Missing activation output for edge {src}->{dst}: {src_tasks}")
                            send_adapter = member_map.get(src_tasks[0][0], primary)
                            handles = pp.isend_tensor_dict(send_adapter.pack_activation(edge_out), dst=dst)
                            comm_handles.extend(handles)
                        elif spec.rank == dst:
                            recv, recv_handles, _ = pp.irecv_tensor_dict(src=src)
                            comm_handles.extend(recv_handles)
                            _stash_activation_rows(
                                act_stash,
                                recv,
                                src_tasks,
                                topology=spec.topology,
                                src=src,
                            )
                elif edges:
                    _post_activations_ws(
                        pp,
                        edges,
                        rank=spec.rank,
                        task_outs=task_outs,
                        tasks=tasks,
                        packed_output=output,
                        members=member_map,
                        adapter=primary,
                        topology=spec.topology,
                        device=act_device,
                        act_stash=act_stash,
                    )
            kv_handles = kv.exchange(slot)
            if not kv_stream:
                comm_handles.extend(kv_handles)
            # Outbound KV (+ legacy act). Fixed-shape act already waited in _post_activations_ws.
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
            kv.wait_sends()
            kv.evict(prev_slot_for_evict)
        kv.drain()
    except Exception:
        for req in {item.req for item in ctx.inflight} | {item.req for item in ctx.pending}:
            ctx.drop(req)
        raise
