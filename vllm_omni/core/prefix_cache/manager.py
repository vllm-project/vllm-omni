# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni prefix cache, manager side.

Owns slot occupancy, the request-task table, the hit/span registry,
per-step snapshots, and the merge. The controller owns the staging
pool, copy queues, and writing rows into the CPU block pool. The
state lock covers those tables only — never a wait-for-copy, a
GPU-byte-budget flush, or a memcpy.

Helper docstrings mark ``_state_lock`` (non-reentrant):

    Caller holds   already inside a critical section; do not acquire
    Takes          acquires here (``@_locked`` or ``with``)

Two host stores:

    StagingBufferPool   reusable step-sized pages. save copies this
                        step's immediately-cached keys (hidden +
                        non-deferred mm) device→host into one page.
                        Per-task `chunk.host` is a view into that page,
                        not a second copy.
    PrefixBlockPool     durable (kv_slot, key) prefix cache. The
                        committer only writes into it.

Two write paths (which keys, not how many tokens):

    JOIN_NEXT_STEP      immediately-cached keys. Device→host is already
                        in flight at submit; the committer waits
                        `step_d2h_event` then copies host→pool. The next
                        save waits `done`: reused slots must not leave a
                        pending pool write behind.
    JOIN_ON_FINISH      deferred mm. Stays on the device clone; the
                        committer does that device→host, then writes the
                        pool. Forced onto the high-priority queue on
                        finish/abort or GPU-byte-budget pressure.

Per real scheduler_output, engine-thread order:

    new_step_starts   before _update_states drops finished requests
                      (register hit metadata)
    forward
    save_outputs      clone off live buffers + launch staging copy;
                      returns step id
    materialize or discard_step   exactly one of the two, once

materialize may run on the async output builder while the engine is
already in the next step. Warmup/dummy runs are never fed.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, NamedTuple, NoReturn

import torch

from vllm_omni.core.prefix_cache.adapter import (
    PrefixCacheEventKind,
    PrefixCacheRequestEvent,
    PrefixCacheStep,
    PrefixCacheWriteLayout,
)
from vllm_omni.core.prefix_cache.block_pool import PrefixBlockPool
from vllm_omni.core.prefix_cache.controller import (
    OmniPrefixCacheController,
    StagingBufferHolder,
    StepD2HClaim,
    TaskState,
    WriteTask,
    _BudgetTicket,
    _WriteChunk,
)
from vllm_omni.core.prefix_cache.interface import (
    ModelCachePolicy,
    OmniPrefixCacheStagingTimeoutError,
    OmniPrefixCacheUnmatchError,
    PrefixCacheConfig,
    ReqId,
    StageCacheOutputs,
    StepId,
    TensorName,
    Tid,
    WriteSchedule,
    is_hidden_key,
    without_hidden,
)

logger = logging.getLogger(__name__)


class _Presence(IntEnum):
    """Presence of the current ``(slot, key)`` binding.

    ``UNKNOWN`` means no observation has established this key for the current
    slot generation. ``ABSENT`` is an explicit value for the current tenant,
    rather than an invitation to read an older tenant's CPU-mirror row.
    """

    UNKNOWN = 0
    ABSENT = 1
    PENDING = 2
    IN_TRANSIT = 2  # compatibility alias for the pre-plan occupancy API
    PRESENT = 3
    COMMITTED = 3

# Compatibility name for the existing write/occupancy helpers.  New read
# planning uses the more explicit Presence vocabulary above.
_Occupancy = _Presence


def _is_step_token_tensor(val: Any, n: int, padded: int) -> bool:
    """2D+ tensor whose first dim is this step's token count (``n`` or padded).

    True means callers may take ``val[:n]``. Leftover tensors (``codes.ref``),
    lists, and other shapes are False.
    """
    return isinstance(val, torch.Tensor) and val.ndim >= 2 and int(val.shape[0]) in (n, padded)


def _raise_unreadable_hit(req_id: str, key: str, why: str) -> NoReturn:
    raise OmniPrefixCacheUnmatchError(f"hit span for req {req_id} key={key} is not readable ({why})")


def _snapshot_leftover_mm_cpu(
    mm_outputs: dict[str, Any],
    device_snapshot_keys: set[str],
    num_tokens_unpadded: int,
    num_tokens_padded: int | None = None,
) -> tuple[dict[str, Any], torch.cuda.Event | None]:
    """CPU copy of mm that did not land on the staging page.

    Skip ``device_snapshot_keys`` (those already have a device→host page).
    Copy the rest — deferred mm, lists, ``codes.ref`` — so materialize can
    run after the next forward overwrites graph buffers. Slice ``[:n]``
    only when ``shape[0] == n``; ``>= n`` would clip ``codes.ref``.

    CUDA tensors land in pinned memory through a non-blocking copy on the
    current stream: stream order keeps it ahead of the next forward, and
    the engine thread does not wait for this forward to finish. The
    returned event (None when nothing was on CUDA) must be synchronized
    before the snapshot is read.
    """
    n = num_tokens_unpadded
    padded = n if num_tokens_padded is None else int(num_tokens_padded)
    on_cuda = False

    def _copy(val: Any) -> Any:
        nonlocal on_cuda
        if isinstance(val, torch.Tensor):
            t = val[:n] if _is_step_token_tensor(val, n, padded) and int(val.shape[0]) == n else val
            t = t.detach()
            if t.is_cuda:
                host = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
                host.copy_(t, non_blocking=True)
                on_cuda = True
                return host
            # .cpu() copies other device tensors; CPU/pinned views still share storage.
            copied = t.cpu() if t.device.type != "cpu" else t.clone()
            return copied.contiguous()
        if isinstance(val, Mapping):
            return {k: _copy(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_copy(v) for v in val]
        if isinstance(val, tuple):
            return tuple(_copy(v) for v in val)
        return val

    leftover = {
        key: _copy(val) for key, val in mm_outputs.items() if key not in device_snapshot_keys and not is_hidden_key(key)
    }
    event = None
    if on_cuda:
        event = torch.cuda.Event()
        event.record()
    return leftover, event


def _unpin_leftover(val: Any) -> Any:
    """Clone pinned tensors out so the payload does not hold pinned pages."""
    if isinstance(val, torch.Tensor):
        return val.clone() if val.is_pinned() else val
    if isinstance(val, Mapping):
        return {k: _unpin_leftover(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_unpin_leftover(v) for v in val]
    if isinstance(val, tuple):
        return tuple(_unpin_leftover(v) for v in val)
    return val


@dataclass
class _StepOutputs:
    """This step's outputs split by consumer.

    A tensor whose first dim equals this step's token count (or the
    CUDA-graph padded count) can be sliced ``[:n]`` into per-token rows.
    ``codes.ref`` and lists are not that shape.

    ``immediate``: those per-token rows copied device→host this step
    (hidden + non-deferred mm). ``deferred_chunks``: per-token deferred
    mm, packed per request for JOIN_ON_FINISH. ``leftover``: CPU replica
    for this step's materialize of everything that did not get a staging
    page.
    A deferred per-token key is in both ``deferred_chunks`` (later cache
    write) and ``leftover`` (this-step read) — two consumers, not a
    duplicate store.

    ``immediate_budget`` charges the immediate clones once; every
    JOIN_NEXT_STEP task of the step pins it. Deferred chunks carry their
    own shared ticket.
    """

    immediate: dict[str, torch.Tensor]
    deferred_chunks: list[tuple[str, _WriteChunk]]
    leftover: dict[str, Any]
    # Completion of the leftover device→host copies; None when none ran on CUDA.
    leftover_event: torch.cuda.Event | None = None
    immediate_budget: _BudgetTicket | None = None
    # Pool storage allocated (unlocked) for keys first seen this step;
    # published into the pool / occupancy tables under the state lock.
    new_key_storage: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def deferred_budget(self) -> _BudgetTicket | None:
        return self.deferred_chunks[0][1].budget if self.deferred_chunks else None

    def freeze_targets(self) -> list[torch.Tensor]:
        """Device clones the freeze event must cover."""
        return list(self.immediate.values()) + [t for _, c in self.deferred_chunks for t in c.tensors.values()]

    def budget_bytes(self) -> int:
        return sum(t.nbytes for t in (self.immediate_budget, self.deferred_budget) if t is not None)


def _locked(fn):
    """Serialize public entry points: the async output builder calls
    materialize() while the engine thread is in the next step."""

    def wrapper(self, *args, **kwargs):
        with self._state_lock:
            return fn(self, *args, **kwargs)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


@dataclass(frozen=True, slots=True)
class _SlotBinding:
    """Immutable identity selected for one row of a read plan."""

    slot: int
    version: int
    producer: Tid | None
    presence: _Presence


@dataclass
class _ReadPlan:
    """A hit read bound to exact versions and producers before execution.

    Committed rows are copied into ``resolved`` while the manager lock keeps
    later claims out. Pending rows retain their concrete ``WriteTask``;
    immediate producers additionally lease the staging page. Consequently a
    later claim cannot redirect this plan to a newer tenant.
    """

    read_id: int
    slots: torch.Tensor
    key: TensorName
    req_id: ReqId
    bindings: tuple[_SlotBinding, ...]
    resolved: torch.Tensor
    producers: list[tuple[WriteTask, torch.Tensor]]
    leases: list[tuple[int, StagingBufferHolder]] = field(default_factory=list)
    error: str | None = None
    _closed: bool = False
    _close_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class _SlotRef:
    """Legacy execution source retained for direct controller reads."""
    slots: torch.Tensor
    key: TensorName
    req_id: ReqId
    already_staged: bool
    staged_list: list[tuple[WriteTask, torch.Tensor]]
    join_tids: list[Tid] = field(default_factory=list)
    reserved_version: torch.Tensor | None = None
    preserved: dict[int, torch.Tensor] = field(default_factory=dict)


@dataclass(kw_only=True)
class _StepContext:
    """Save-time snapshot of one step; consumed exactly once.

    Built on the engine thread in save_outputs so materialize (possibly on
    the async builder) never reads the live batch. Host rows live on `d2h`
    (empty views when this save only had leftover mm). materialize or
    discard_step pops it and frees the staging slot. A later save waits
    if all slots are still held.
    """

    # Packed layout in batch order: req -> [start, end) of this step's rows.
    spans: dict[ReqId, tuple[int, int]]
    num_tokens_unpadded: int = 0

    # Hits snapshotted at new_step_starts. Disjoint committed spans prefetch
    # during forward; same-step overlaps bind at save. Materialize writes tail.
    hits: dict[ReqId, tuple[int, list[int]]]  # (hit_upto, blocks)
    hit_plans: dict[ReqId, dict[TensorName, _ReadPlan]] = field(default_factory=dict)
    hit_prefetch: dict[ReqId, dict[TensorName, Future]] = field(default_factory=dict)

    # Key split frozen at save: recompute at materialize races ensure_key.
    cached_keys: set[TensorName] = field(default_factory=set)
    # Leftover mm copied to CPU at save (this-step deferred rows + mm
    # that is not written to the pool).
    mm_cpu_snapshot: dict[TensorName, Any] = field(default_factory=dict)
    # Wait this before reading mm_cpu_snapshot (None when nothing was on CUDA).
    mm_cpu_snapshot_event: torch.cuda.Event | None = None

    # Staging slot for this step id (empty views when only leftover mm).
    d2h: StepD2HClaim | None = None


class _SlotStatus(NamedTuple):
    """Occupancy row for one tensor name (views into the table, not copies)."""

    presence: torch.Tensor  # int8[num_slots]
    producers: torch.Tensor  # Tid per kv slot; 0 = none
    slot_version: torch.Tensor  # int64[num_slots]; bumped each time a new write claims the slot

    @property
    def state(self) -> torch.Tensor:
        return self.presence

    @property
    def tids(self) -> torch.Tensor:
        return self.producers


class _SlotStatusTable:
    """Per (KV slot, tensor name): empty, being written, or already in the pool.

    Hidden and a deferred mm field on the same slot are independent.
    ``map_slots`` marks a write in progress; if another task still owns
    the slot, the manager records those rows as no longer owned by it.
    ``commit`` runs after the pool write: still-owned slots become
    committed.
    """

    def __init__(self, num_slots: int) -> None:
        self.num_slots = num_slots
        self.presence: dict[TensorName, torch.Tensor] = {}  # int8[num_slots]
        self.producers: dict[TensorName, torch.Tensor] = {}  # Tid per kv slot; 0 = none
        # Temporary private-name aliases for focused tests and debugging tools.
        self.state = self.presence
        self.tids = self.producers
        self.slot_versions: dict[TensorName, torch.Tensor] = {}  # int64[num_slots]; bumped on each new write claim
        self.task_bindings: dict[Tid, list[tuple[TensorName, torch.Tensor, torch.Tensor]]] = {}

    def init_table(self, key: TensorName) -> None:
        """Allocate occupancy tensors for ``key`` if they do not exist."""
        if key in self.presence:
            return
        self.presence[key] = torch.zeros(self.num_slots, dtype=torch.int8)
        self.producers[key] = torch.zeros(self.num_slots, dtype=torch.int64)
        self.slot_versions[key] = torch.zeros(self.num_slots, dtype=torch.int64)

    def get_slot_status(self, key: TensorName) -> _SlotStatus:
        """Occupancy tensors for ``key``. ``init_table`` must have run."""
        return _SlotStatus(
            presence=self.presence[key],
            producers=self.producers[key],
            slot_version=self.slot_versions[key],
        )

    def map_slots(
        self, slots: torch.Tensor, tid: Tid, keys: Iterable[TensorName]
    ) -> list[tuple[Tid, TensorName, torch.Tensor]]:
        """Record ``tid`` on these (slot, key). Return in-transit rows
        another write still owned (caller marks them skipped on it)."""
        keys = tuple(keys)
        stolen: list[tuple[Tid, TensorName, torch.Tensor]] = []
        for key in keys:
            status = self.get_slot_status(key)
            cur = status.producers[slots]
            stale = (status.presence[slots] == _Presence.PENDING) & (cur != tid) & (cur != 0)
            if bool(stale.any()):
                for old in {int(o) for o in cur[stale].tolist()}:
                    stolen.append((old, key, slots[stale & (cur == old)]))
            status.presence[slots] = _Presence.PENDING
            status.producers[slots] = tid
            # Bump the version so a reader that captured an older one detects
            # this claim even after it later commits (COMMITTED, not IN_TRANSIT).
            status.slot_version[slots] += 1
            self.task_bindings.setdefault(tid, []).append((key, slots.clone(), status.slot_version[slots].clone()))
        return stolen

    def invalidate(self, slots: torch.Tensor, keys: Iterable[TensorName]) -> list[tuple[Tid, TensorName, torch.Tensor]]:
        """Publish explicit absence for this tenant and return displaced producers."""
        stolen: list[tuple[Tid, TensorName, torch.Tensor]] = []
        for key in keys:
            status = self.get_slot_status(key)
            cur = status.producers[slots]
            pending = (status.presence[slots] == _Presence.PENDING) & (cur != 0)
            for old in {int(value) for value in cur[pending].tolist()}:
                stolen.append((old, key, slots[pending & (cur == old)]))
            status.slot_version[slots] += 1
            status.presence[slots] = _Presence.ABSENT
            status.producers[slots] = 0
        return stolen

    def commit(self, tids: Iterable[Tid]) -> None:
        """Flip still-owned slots to COMMITTED and drop the reverse index."""
        for tid in tids:
            bindings = self.task_bindings.pop(tid, ())
            for key, slots, versions in bindings:
                status = self.get_slot_status(key)
                still_ours = (status.producers[slots] == tid) & (status.slot_version[slots] == versions)
                idx = slots[still_ours]
                status.presence[idx] = _Presence.PRESENT
                status.producers[idx] = 0


class _RequestTaskTable:
    """Per request: still live, which WriteTasks it opened, deferred task.

    Also allocates ``tid`` and increments per-request ``write_n``.
    Device→host copy and pool write stay on the controller.
    """

    def __init__(self) -> None:
        self._next_tid: Tid = 1
        self.write_n: dict[ReqId, int] = {}  # last write_n issued
        self.tasks: dict[ReqId, set[Tid]] = {}
        self.deferred: dict[ReqId, WriteTask] = {}
        self.live_reqs: set[ReqId] = set()

    def alloc_tid(self) -> Tid:
        tid = self._next_tid
        self._next_tid += 1
        return tid

    def increment_write_n(self, req_id: ReqId) -> int:
        n = self.write_n.get(req_id, 0) + 1
        self.write_n[req_id] = n
        return n

    def track(self, req_id: ReqId, tid: Tid) -> None:
        self.tasks.setdefault(req_id, set()).add(tid)

    def finish(self, req_id: ReqId) -> tuple[set[Tid], WriteTask | None]:
        """Drop this request's rows. Returns owned tids + deferred task."""
        self.live_reqs.discard(req_id)
        self.write_n.pop(req_id, None)
        tids = self.tasks.pop(req_id, set())
        dtask = self.deferred.pop(req_id, None)
        return tids, dtask

    def drop_completed(self, tids: Iterable[Tid]) -> None:
        done = set(tids)
        if not done:
            return
        for req_tids in self.tasks.values():
            req_tids -= done


class OmniPrefixCacheManager:
    def __init__(
        self,
        config: PrefixCacheConfig,
        *,
        eager: bool | None = None,
        prefetch_reads: bool = True,
    ):
        self._config = config
        self._pool = PrefixBlockPool(config)
        self._controller = OmniPrefixCacheController(self._pool, config, eager=eager)
        self._policy = ModelCachePolicy()
        # Serializes engine vs async-builder public entries. Non-reentrant:
        # those entries never call each other, and the lock must not cover
        # a wait-for-copy, GPU-byte-budget flush, or device→host copy.
        self._state_lock = threading.Lock()

        self._slot_status = _SlotStatusTable(config.num_blocks * config.block_size)
        # Hidden is known from the default policy; mm rows at init_table.
        if (hk := self._policy.hidden_key) is not None:
            self._slot_status.init_table(hk)
        self._request_tasks = _RequestTaskTable()
        # Join worklists — not occupancy, not the request-task table.
        self._join_next_step_tids: list[Tid] = []
        self._join_finished_tids: set[Tid] = set()  # escalated on finish/abort

        # This step's hits (copied into _StepContext at save).
        self._cur_num_scheduled: dict[ReqId, int] = {}
        self._hit_spans: dict[ReqId, tuple[int, list[int]]] = {}  # (upto, blocks)
        self._hit_plans: dict[ReqId, dict[TensorName, _ReadPlan]] = {}
        self._hit_prefetch: dict[ReqId, dict[TensorName, Future]] = {}
        self._prepared_write_layout: PrefixCacheWriteLayout | None = None
        self._prefetch_reads = prefetch_reads
        self._next_read_id = 1
        self._pending_reads: list[_SlotRef] = []
        # Sticky fatal set on the first drained write failure; every later
        # facade entry re-raises it (see _commit_drained_writes).
        self._fatal_write_failure: str | None = None

        # Prefix gather after the write layout has registered this step's
        # producers (CPU work releases the GIL).
        # One worker: complete in submit order; pop finished work from the head.
        self._prefetch_pool = ThreadPoolExecutor(1, thread_name_prefix="omni-prefix-cache-prefetch")
        self._prefetch_queue: deque[tuple[Future, _ReadPlan]] = deque()

        # One snapshot per step id; consume with materialize or discard_step.
        self._next_step_id: StepId = 1
        self._step_ctxs: dict[StepId, _StepContext] = {}

    # ------------------------------------------------------ public entries

    def register_policy(self, policy: ModelCachePolicy) -> None:
        self._policy = policy
        if (hk := policy.hidden_key) is not None:
            self._slot_status.init_table(hk)

    @torch.inference_mode()
    def new_step_starts(
        self,
        events: Iterable[PrefixCacheRequestEvent] | PrefixCacheStep,
        *,
        num_scheduled_tokens: Mapping[str, int] | None = None,
    ) -> None:
        """Handle one immutable lifecycle event batch.

        Engine thread only; before _update_states removes finished
        requests; exactly once per real step. Registers new-request prefix
        hits (copying their block tables) and forces finished/aborted
        requests' still-open writes onto the high-priority copy queue —
        a block hash that entered the batch must land in the cache,
        abort included. ``escalate`` (eager: the copy + pool write) runs
        after ``_state_lock`` is released.
        """
        to_escalate: list[int] = []
        with self._state_lock:
            if self._prepared_write_layout is not None:
                self._dispose_read_plans(self._hit_plans, self._hit_prefetch)
                self._clear_hit_infos()
            # 1. Publish writes the committer has already written into the pool.
            self._commit_drained_writes()

            # 2. Finished/aborted reqs: collect still-open writes. Abort too:
            #    those block hashes are already in vLLM; dropping the write
            #    would leave future hits ABSENT. The next save waits
            #    join_host_ready. (Not leftover_mm — those are this-step reads.)
            if isinstance(events, PrefixCacheStep):
                step = events
                events = step.events
                num_scheduled_tokens = dict(step.scheduled_tokens)
            else:
                events = tuple(events)
            for event in events:
                if event.kind not in (PrefixCacheEventKind.FINISHED, PrefixCacheEventKind.ABORTED):
                    continue
                req_id = event.req_id
                tids, dtask = self._request_tasks.finish(req_id)
                if dtask is not None:
                    tids.add(dtask.tid)
                pending_tasks = [tid for tid in tids if self._controller.get_task(tid) is not None]
                if pending_tasks:
                    to_escalate.extend(pending_tasks)
                    self._join_finished_tids.update(pending_tasks)

            # 3. Copy this arrival's prefix-hit block ids. The adapter captures
            #    them before _update_states; after that point they sit
            #    on the live request and grow as decode allocates more blocks.
            #    materialize (async builder) must not reread that live table.
            self._clear_hit_infos()
            for event in events:
                if event.kind not in (PrefixCacheEventKind.STARTED,):
                    continue
                req_id = event.req_id
                if req_id in self._request_tasks.live_reqs:
                    # Already live: async_chunk continuation, or a V2-runner
                    # resume after preemption (V1 resumes via
                    # scheduled_cached_reqs). Either way num_computed_tokens is
                    # not mirrored as a hit span; see the design doc.
                    continue
                self._request_tasks.live_reqs.add(req_id)
                num_computed = int(event.hit_end)
                if num_computed > 0:
                    # block_ids is per-kv-group; group 0 only.
                    blocks = event.block_ids
                    if blocks is not None and len(blocks) > 0 and not isinstance(blocks[0], int):
                        blocks = blocks[0]
                    if not blocks:
                        # Fail at the cause: a hit we cannot snapshot now would
                        # crash at materialize time with less context (materialize is
                        # forbidden from reading the live batch).
                        raise OmniPrefixCacheUnmatchError(
                            f"prefix hit for req {req_id} ({num_computed} tokens) carries no block_ids"
                        )
                    bs = self._config.block_size
                    if num_computed % bs != 0:
                        raise OmniPrefixCacheUnmatchError(
                            f"prefix hit not block aligned (req={req_id}, hit_upto={num_computed}, block_size={bs})"
                        )
                    hit_blocks = list(blocks[: num_computed // bs])
                    self._hit_spans[req_id] = (num_computed, hit_blocks)

            # 4. Retire completed work from prior steps.  Planning this step's
            #    reads must wait for save_outputs to register the write layout:
            #    a producer in this same step takes precedence over durable
            #    residency left by an older tenant of the slot.
            while self._prefetch_queue and self._prefetch_queue[0][0].done():
                self._prefetch_queue.popleft()
            self._cur_num_scheduled = dict(
                num_scheduled_tokens or {event.req_id: event.scheduled_tokens for event in events}
            )
        if to_escalate:
            self._controller.escalate(to_escalate)

    def prepare_read_plans(self, write_layout: PrefixCacheWriteLayout) -> None:
        """Register the post-order layout and start safe hit reads pre-forward.

        The layout is available after batch ordering but before model forward.
        Hits disjoint from every slot this step will write can bind committed
        history immediately. Any overlap is deferred until save registers the
        concrete producer, preserving same-step producer precedence.
        """
        with self._state_lock:
            if self._prepared_write_layout is not None:
                raise OmniPrefixCacheUnmatchError("prefix-cache write layout prepared more than once for one step")
            self._commit_drained_writes()
            self._prepared_write_layout = write_layout
            write_slots = {
                int(slot)
                for write in write_layout.writes
                for slot in write.slots
            }
            try:
                if self._hit_spans:
                    self._prefetch_hit_spans(blocked_slots=write_slots)
            except BaseException:
                self._dispose_read_plans(self._hit_plans, self._hit_prefetch)
                self._clear_hit_infos()
                raise

    def abort_prepared_step(self) -> None:
        """Cancel pre-forward reads when model execution fails before save."""
        with self._state_lock:
            self._dispose_read_plans(self._hit_plans, self._hit_prefetch)
            self._clear_hit_infos()

    @torch.inference_mode()
    def save_outputs(
        self,
        hidden_states: torch.Tensor | None,
        mm_outputs: dict[str, Any] | None,
        *,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
        write_layout: PrefixCacheWriteLayout | None = None,
    ) -> int:
        """Write this step's outputs into the cache; returns the step id.

        Engine thread only, after the forward and before materialize.
        Immediately-cached rows: one on-device clone, one whole-step
        device→host into the staging pool, then one JOIN_NEXT_STEP
        WriteTask per request whose `chunk.host` is a view of that page.
        Deferred rows stay on the device clone (JOIN_ON_FINISH); the
        committer copies them later. Leftover mm (this-step deferred rows
        + mm not written to the pool) is copied to CPU here so materialize
        never reads live graph buffers.
        Snapshots everything materialize needs. The returned step id MUST
        be consumed exactly once — by materialize() or discard_step().
        Every step id claims one staging slot (saves with only leftover mm
        included); a later save waits for a free slot and times out if
        none return.

        The state lock never covers a blocking wait: the previous step's
        JOIN_NEXT_STEP wait, the clone build, the GPU-byte-budget reserve
        (which may flush), and the staging-slot claim all run unlocked.
        """
        # 1. Join the previous step's host copies (unlocked).
        self._wait_for_host_ready()

        # 2. Packed batch layout for this step (req -> [start, end)).
        if write_layout is None:
            raise ValueError("save_outputs requires an adapter-produced write_layout")
        with self._state_lock:
            prepared_layout = self._prepared_write_layout
            if prepared_layout is not None and prepared_layout != write_layout:
                raise OmniPrefixCacheUnmatchError("save write layout differs from the pre-forward layout")
        req_order = [write.req_id for write in write_layout.writes]
        num_sched = {write.req_id: write.row_end - write.row_start for write in write_layout.writes}
        query_start = {write.req_id: write.row_start for write in write_layout.writes}

        slots_cpu: torch.Tensor | None = None
        mm_outputs = mm_outputs or {}
        freeze_event = None

        # 3. Slot map, then split into immediate / deferred / leftover.
        if num_tokens_unpadded > 0:
            # Derive the slot mapping on CPU: reading the device one back
            # would need a stream sync that waits on the whole forward.
            slots_cpu = torch.tensor(
                [slot for write in write_layout.writes for slot in write.slots],
                dtype=torch.long,
            )
            if int(slots_cpu.numel()) != num_tokens_unpadded:
                # Fail at the cause: skipping the save would leave rows absent
                # behind hashes vLLM already published — a delayed crash at
                # some future hit instead of a debuggable one here.
                raise OmniPrefixCacheUnmatchError(
                    f"slot mapping covers {int(slots_cpu.numel())} of {num_tokens_unpadded} scheduled tokens; "
                    "CPU-side slot derivation out of sync with the batch"
                )
        step_outputs = self._split_step_outputs(
            hidden_states,
            mm_outputs,
            num_tokens_unpadded,
            num_tokens_padded,
            slots_cpu=slots_cpu,
            req_order=req_order,
            num_sched=num_sched,
            query_start=query_start,
        )

        # 4. Freeze the device clones and reserve the GPU-byte budget (unlocked).
        freezed_tensors = step_outputs.freeze_targets()
        if freezed_tensors:
            if torch.cuda.is_available() and any(t.is_cuda for t in freezed_tensors):
                freeze_event = torch.cuda.Event()
                freeze_event.record()
            # One ticket per clone (immediate step clone, shared deferred
            # clone); per-request slices are views and charge nothing.
            # Reserve may block on a flush: outside the lock. The flush may
            # close a deferred task this step appends to; _stage_deferred
            # then opens a fresh one, so a long request cannot pin the
            # whole budget.
            self._controller.reserve(step_outputs.budget_bytes())

        # 5. Claim a staging slot (unlocked), optional device→host into it,
        #    register the writes + store the step snapshot (locked), then
        #    dispatch the queued writes (unlocked: in eager mode dispatch
        #    is the copy + pool write itself). Saves with only leftover mm
        #    still claim. Full pool waits; timeout lists unused step ids.
        d2h_claim: StepD2HClaim | None = None
        step_holder = StagingBufferHolder.for_step(self._next_step_id)
        transferred = False
        bound_tids: list[int] = []
        try:
            d2h_claim = self._stage_step_host(step_outputs.immediate, num_tokens_unpadded, freeze_event, step_holder)

            step_id, queued = self._publish_saved_step(
                req_order=req_order,
                query_start=query_start,
                num_sched=num_sched,
                num_tokens_unpadded=num_tokens_unpadded,
                step_outputs=step_outputs,
                slots_cpu=slots_cpu,
                mm_keys=set(mm_outputs.keys()),
                freeze_event=freeze_event,
                d2h_claim=d2h_claim,
                bound_tids=bound_tids,
            )
            self._controller.dispatch(queued)
            transferred = True
            return step_id
        finally:
            # Slot claim is outside the lock; a later raise must release
            # the step and any task that already bound this slot.
            if not transferred and d2h_claim is not None:
                # A dispatch failure has no caller that can consume this step.
                with self._state_lock:
                    failed_ctx = self._step_ctxs.pop(step_holder.owner_id, None)
                    if failed_ctx is not None:
                        self._dispose_ctx_read_plans(failed_ctx)
                    else:
                        self._dispose_read_plans(self._hit_plans, self._hit_prefetch)
                        self._clear_hit_infos()
                self._release_staging_on_failed_save(d2h_claim.staging_slot, step_holder, bound_tids)

    @torch.inference_mode()
    def materialize(self, step_id: int, req_ids: list[str]) -> StageCacheOutputs:
        """Per-request merged outputs for the step saved as `step_id`.

        Any thread. `req_ids` must be (a subset of) the save-time snapshot;
        an outside id means the caller is reading the live batch.
        A request without a hit is a plain miss and gets exactly
        this step's rows — normal path, nothing logged. A hit span that
        resolves to absent rows raises OmniPrefixCacheUnmatchError: fatal
        by contract (do not pretend it was a miss).

        Two phases: under the lock, publish finished writes and pin every
        row source (task refs + masks, absent checks included) — not yet
        reading the tensors. Unlocked: wait this step's `step_d2h_event`,
        clone the staging views (then drop the step holder), and merge.
        The engine thread never waits on this thread's device→host copy.
        """
        ctx = None
        step_released = False
        try:
            with self._state_lock:
                ctx = self._take_step_ctx(step_id)
                self._commit_drained_writes()

                # The builder must pass (a subset of) the req list captured at
                # save time — an id outside the snapshot means it is reading the
                # live batch, which the contract forbids. Not a fallback that
                # serves a miss.
                unknown = set(req_ids) - set(ctx.spans)
                if unknown:
                    raise OmniPrefixCacheUnmatchError(
                        f"materialize(step {step_id}) got req ids outside the save snapshot: {sorted(unknown)[:8]}"
                    )

                cached_keys = ctx.cached_keys

                hit_sources: dict[tuple[str, str], _ReadPlan | Future] = {}
                for req_id in req_ids:
                    hit = ctx.hits.get(req_id)
                    if not hit:
                        continue
                    hit_upto, hit_blocks = hit
                    prefetched = ctx.hit_prefetch.get(req_id, {})
                    plans = ctx.hit_plans.get(req_id, {})
                    slots = self._get_hit_slots(hit_upto, hit_blocks)
                    keys = self._policy.get_hit_keys(cached_keys)
                    for key in keys:
                        fut = prefetched.get(key)
                        if fut is not None:
                            hit_sources[(req_id, key)] = fut
                            continue
                        plan = plans.get(key)
                        if plan is None:
                            plan = self._read_plan(slots, key, req_id)
                        hit_sources[(req_id, key)] = plan

            # ---- unlocked: data movement + merge ----
            if ctx.mm_cpu_snapshot_event is not None:
                ctx.mm_cpu_snapshot_event.synchronize()
                ctx.mm_cpu_snapshot = _unpin_leftover(ctx.mm_cpu_snapshot)
            current: dict[str, torch.Tensor] = {}
            if ctx.d2h is not None:
                # Whole-step device→host was launched at save. One event wait
                # (usually already complete), then a contiguous copy-out per
                # key so consumers no longer depend on the reusable slot.
                if ctx.d2h.event is not None:
                    ctx.d2h.event.synchronize()
                current = {k: v.clone() for k, v in ctx.d2h.views.items()}
                self._release_step_staging(ctx, step_id)
                step_released = True

            hidden_out: dict[str, torch.Tensor] | None = None
            hidden_key = self._policy.hidden_key
            if hidden_key is not None and hidden_key in current:
                hidden_out = {}
                for req_id in req_ids:
                    hidden_out[req_id] = self._merge_cached_for_req(
                        ctx, req_id, hidden_key, current[hidden_key], hit_sources
                    )

            mm_out: dict[str, dict[str, Any]] = {}
            for key in cached_keys:
                cur = current.get(key)
                if cur is None:
                    val = ctx.mm_cpu_snapshot.get(key)
                    if not isinstance(val, torch.Tensor):
                        continue
                    # Leftover snapshot; spans stay within [0, n), no re-slice.
                    cur = val
                mm_out[key] = {
                    req_id: self._merge_cached_for_req(ctx, req_id, key, cur, hit_sources) for req_id in req_ids
                }

            self._merge_uncached_mm(ctx, req_ids, cached_keys, mm_out)
            return StageCacheOutputs(hidden_states=hidden_out, mm_outputs=mm_out)
        finally:
            if ctx is not None and not step_released:
                self._release_step_staging(ctx, step_id)
            if ctx is not None:
                self._dispose_ctx_read_plans(ctx)

    @_locked
    def discard_step(self, step_id: int) -> None:
        """Consume the step context when nothing will materialize it.

        Any thread; same exactly-once contract as materialize (unknown or
        duplicate id raises). Only the read-side snapshot is dropped —
        the cache write proceeds unchanged.
        """
        ctx = self._take_step_ctx(step_id)
        self._dispose_ctx_read_plans(ctx)
        self._release_step_staging(ctx, step_id)

    def shutdown(self) -> None:
        # A running prefetch owns its staging read lease until fetch_host has
        # finished.  Cancellation only closes plans that never started; wait
        # for running jobs before stopping the controller they may be joining.
        with self._state_lock:
            ctxs = list(self._step_ctxs.values())
            live_plans = [plan for plans in self._hit_plans.values() for plan in plans.values()]
            for ctx in ctxs:
                self._dispose_ctx_read_plans(ctx)
            self._dispose_read_plans(self._hit_plans, self._hit_prefetch)
        self._prefetch_pool.shutdown(wait=True, cancel_futures=True)
        for ctx in ctxs:
            for plans in ctx.hit_plans.values():
                for plan in plans.values():
                    self._close_read_plan(plan)
        for plan in live_plans:
            self._close_read_plan(plan)
        self._prefetch_queue.clear()
        self._controller.shutdown()

    def _dispose_ctx_read_plans(self, ctx: _StepContext) -> None:
        """Release plans without racing a still-running prefetch.

        A queued future that can be cancelled never entered
        ``_execute_read_plan``, so this path releases its lease. A running
        future owns the lease until its execute ``finally`` block; closing it
        here would let a new save recycle the page underneath ``fetch_host``.
        """
        self._dispose_read_plans(ctx.hit_plans, ctx.hit_prefetch)

    def _dispose_read_plans(
        self,
        hit_plans: Mapping[ReqId, Mapping[TensorName, _ReadPlan]],
        hit_prefetch: Mapping[ReqId, Mapping[TensorName, Future]],
    ) -> None:
        for req_id, plans in hit_plans.items():
            futures = hit_prefetch.get(req_id, {})
            for key, plan in plans.items():
                fut = futures.get(key)
                if fut is None or fut.cancel():
                    self._close_read_plan(plan)
                elif not fut.done():
                    fut.add_done_callback(lambda _fut, read_plan=plan: self._close_read_plan(read_plan))

    # ------------------------------------------------------ new_step

    def _clear_hit_infos(self) -> None:
        """Drop the live hit / prefetch tables. Caller holds ``_state_lock``."""
        self._hit_spans.clear()
        self._hit_plans.clear()
        self._hit_prefetch.clear()
        self._prepared_write_layout = None

    def _prefetch_hit_spans(self, blocked_slots: set[int] | None = None) -> None:
        """Caller holds ``_state_lock``. Plan each hit span not yet planned
        and gather it on the prefetch thread.

        Called first after post-order batch layout, before forward: hit spans
        disjoint from this step's writes start immediately. Called again at
        publish after concrete writers and explicit absences are registered,
        which resolves the overlapping same-step-producer cases.
        """
        keys = self._policy.get_hit_keys(self._pool.keys())
        for req_id, (hit_upto, hit_blocks) in self._hit_spans.items():
            plans = self._hit_plans.setdefault(req_id, {})
            futs = self._hit_prefetch.setdefault(req_id, {})
            if all(key in plans for key in keys):
                continue
            n_new = int(self._cur_num_scheduled.get(req_id, 0))
            slots = self._get_hit_slots(hit_upto, hit_blocks)
            if blocked_slots is not None and any(int(slot) in blocked_slots for slot in slots.tolist()):
                continue
            for key in keys:
                if key in plans:
                    continue
                plan = self._read_plan(slots, key, req_id)
                plans[key] = plan
                if self._prefetch_reads:
                    try:
                        fut = self._prefetch_pool.submit(self._prefetch_hit, plan, n_new)
                    except BaseException:
                        plans.pop(key, None)
                        self._close_read_plan(plan)
                        raise
                    self._prefetch_queue.append((fut, plan))
                    futs[key] = fut
            if not plans:
                del self._hit_plans[req_id]
            if not futs:
                del self._hit_prefetch[req_id]

    @torch.inference_mode()
    def _prefetch_hit(self, plan: _ReadPlan, n_new: int) -> torch.Tensor:
        """Prefetch thread: gather the hit span and pre-build the merged
        buffer with the prefix filled. materialize writes only this step's
        rows at the tail — the gather AND the prefix copy both happen while
        the forward runs, and the cat leaves the critical path."""
        rows = self._execute_read_plan(plan)
        out = torch.empty((rows.shape[0] + n_new, rows.shape[-1]), dtype=rows.dtype)
        out[: rows.shape[0]] = rows
        return out

    # ---------------------------------------------------------- save

    def _stage_step_host(
        self,
        device_snapshot: dict[str, torch.Tensor],
        num_tokens_unpadded: int,
        freeze_event: torch.cuda.Event | None,
        step_holder: StagingBufferHolder,
    ) -> StepD2HClaim:
        """Claim a staging slot (wait + timeout) and copy device→host if this step has rows.

        Unlocked. Timeout is annotated with the unconsumed sids and the
        task count so a leaked consume or a stuck write is visible.
        """
        try:
            return self._controller.stage_step_host(device_snapshot, num_tokens_unpadded, freeze_event, step_holder)
        except OmniPrefixCacheStagingTimeoutError as e:
            with self._state_lock:
                ids = sorted(self._step_ctxs)
            raise OmniPrefixCacheStagingTimeoutError(
                f"{e}; unconsumed step contexts (ids={ids}) or a stuck write "
                f"(in_flight_tasks={self._controller.in_flight_tasks()})"
            ) from e

    def _wait_for_host_ready(self) -> None:
        """Pop last step's join worklists and wait them out, unlocked.

        JOIN_NEXT_STEP waits ``done``, not ``host_ready``: this save may reuse
        those slots, and a pool write still pending behind a reused slot is
        unreadable for a delayed hit read (scatter skips reassigned rows).
        Escalated deferred writes keep ``host_ready`` — their readers hold the
        task in ``staged_list``. Lock covers only the pop.
        """
        with self._state_lock:
            finished_ids = list(self._join_finished_tids)
            next_step_ids = list(self._join_next_step_tids)
            self._join_finished_tids.clear()
            self._join_next_step_tids.clear()
        if finished_ids:
            self._controller.join_host_ready(finished_ids)
        if next_step_ids:
            self._controller.join(next_step_ids)

    @_locked
    def _publish_saved_step(
        self,
        *,
        req_order: list[str],
        query_start: dict[str, int],
        num_sched: dict[str, int],
        num_tokens_unpadded: int,
        step_outputs: _StepOutputs,
        slots_cpu: torch.Tensor | None,
        mm_keys: set[str],
        freeze_event: torch.cuda.Event | None,
        d2h_claim: StepD2HClaim,
        bound_tids: list[int],
    ) -> tuple[StepId, list[WriteTask]]:
        """Takes ``_state_lock``. Register this step's writes and store the
        consume-once snapshot. Copies live hits into the snapshot, then
        clears them. Returns the queued tasks for the caller to dispatch
        unlocked; device→host, budget flush and the eager copy stay outside.
        """
        self._commit_drained_writes()
        for key, storage in step_outputs.new_key_storage.items():
            self._pool.install_key(key, storage)
            self._slot_status.init_table(key)
        queued: list[WriteTask] = []
        if step_outputs.immediate:
            queued = self._submit_step_writes(
                req_order,
                query_start,
                num_sched,
                step_outputs.immediate,
                step_outputs.immediate_budget,
                slots_cpu,
                d2h_claim.views,
                freeze_event,
                d2h_claim.staging_slot,
                d2h_claim.event,
                bound_tids,
            )
        self._stage_deferred(step_outputs.deferred_chunks, freeze_event)
        # Publish explicit sparse absence for every (request, key) that did
        # not produce a token-major value this step.  Without this pass a
        # recycled slot would make a sparse MM read inherit an older tenant's
        # row merely because that key exists somewhere in the mirror.
        written_by_req: dict[str, set[str]] = {req: set() for req in req_order}
        for key in step_outputs.immediate:
            for req in req_order:
                if num_sched[req] > 0:
                    written_by_req[req].add(key)
        for req, chunk in step_outputs.deferred_chunks:
            written_by_req.setdefault(req, set()).update(chunk.tensors)
        all_keys = tuple(self._pool.keys())
        for req in req_order:
            absent_keys = tuple(key for key in all_keys if key not in written_by_req.get(req, ()))
            if not absent_keys or num_sched[req] <= 0:
                continue
            start, end = query_start[req], query_start[req] + num_sched[req]
            absent_slots = slots_cpu[start:end] if slots_cpu is not None else torch.empty(0, dtype=torch.long)
            if absent_slots.numel():
                self._preserve_for_pending_reads(absent_slots, absent_keys)
                for old, key, stolen in self._slot_status.invalidate(absent_slots, absent_keys):
                    old_task = self._controller.get_task(old)
                    if old_task is not None:
                        old_task.add_reassigned(key, stolen)
        if self._hit_spans:
            # Same-step hits: their rows are registered now (IN_TRANSIT on
            # this step's tasks); start the gather before the next step.
            self._prefetch_hit_spans()
        step_id = self._next_step_id
        self._next_step_id += 1
        self._step_ctxs[step_id] = _StepContext(
            spans={r: (query_start[r], query_start[r] + num_sched[r]) for r in req_order},
            num_tokens_unpadded=num_tokens_unpadded,
            hits=dict(self._hit_spans),
            hit_plans=dict(self._hit_plans),
            hit_prefetch=dict(self._hit_prefetch),
            cached_keys=without_hidden(self._pool.keys()) & mm_keys,
            mm_cpu_snapshot=step_outputs.leftover,
            mm_cpu_snapshot_event=step_outputs.leftover_event,
            d2h=d2h_claim,
        )
        self._clear_hit_infos()
        return step_id, queued

    def _split_step_outputs(
        self,
        hidden_states: torch.Tensor | None,
        mm_outputs: dict[str, Any],
        num_tokens_unpadded: int,
        num_tokens_padded: int,
        *,
        slots_cpu: torch.Tensor | None,
        req_order: list[str],
        num_sched: dict[str, int],
        query_start: dict[str, int],
    ) -> _StepOutputs:
        """Split this step's outputs into immediate / deferred / leftover.

        Unlocked. One pass over ``mm_outputs``. ``n==0`` has only leftover
        mm (no device clones). A deferred key whose first dim is this
        step's token count is cloned for the JOIN_ON_FINISH write and
        CPU-copied into leftover for this-step materialize. Talker
        ``codes.audio`` stays unpadded while hidden is padded; both must
        open a pool key.
        Lists and other shapes stay leftover.
        """
        n = num_tokens_unpadded
        immediate: dict[str, torch.Tensor] = {}
        deferred_tensors: dict[str, torch.Tensor] = {}
        new_key_storage: dict[str, torch.Tensor] = {}

        def alloc_key(key: str, val: torch.Tensor) -> None:
            # Pinned allocation, unlocked; installed under the lock at publish.
            storage = self._pool.alloc_key(key, val.dtype, int(val.shape[-1]))
            if storage is not None:
                new_key_storage[key] = storage

        if n > 0:
            if hidden_states is not None and (hk := self._policy.hidden_key) is not None:
                if hidden_states.ndim < 2 or hidden_states.shape[0] < n:
                    rows = 0 if hidden_states.ndim < 2 else int(hidden_states.shape[0])
                    raise OmniPrefixCacheUnmatchError(f"hidden_states has {rows} rows, need {n}")
                alloc_key(hk, hidden_states)
                immediate[hk] = hidden_states[:n].clone()
            for key, val in mm_outputs.items():
                is_step_rows = _is_step_token_tensor(val, n, num_tokens_padded)
                if key in self._policy.deferred_keys:
                    if is_step_rows:
                        alloc_key(key, val)
                        deferred_tensors[key] = val[:n].clone()
                    continue
                if self._policy.skip_immediate_mm(key) or not is_step_rows:
                    continue
                alloc_key(key, val)
                immediate[key] = val[:n].clone()
        leftover, leftover_event = _snapshot_leftover_mm_cpu(mm_outputs, set(immediate), n, num_tokens_padded)
        deferred_chunks: list[tuple[str, _WriteChunk]] = []
        if deferred_tensors:
            assert slots_cpu is not None
            deferred_chunks = self._pack_deferred_chunks(deferred_tensors, slots_cpu, req_order, num_sched, query_start)
        immediate_budget = (
            _BudgetTicket(nbytes=sum(t.numel() * t.element_size() for t in immediate.values())) if immediate else None
        )
        return _StepOutputs(
            immediate=immediate,
            deferred_chunks=deferred_chunks,
            leftover=leftover,
            leftover_event=leftover_event,
            immediate_budget=immediate_budget,
            new_key_storage=new_key_storage,
        )

    def _pack_deferred_chunks(
        self,
        deferred_tensors: dict[str, torch.Tensor],
        slots_cpu: torch.Tensor,
        req_order: list[str],
        num_sched: dict[str, int],
        query_start: dict[str, int],
    ) -> list[tuple[str, _WriteChunk]]:
        """Per-req views of already-cloned deferred tensors. No further clone."""
        ticket = _BudgetTicket(nbytes=sum(t.numel() * t.element_size() for t in deferred_tensors.values()))
        out: list[tuple[str, _WriteChunk]] = []
        for req_id in req_order:
            sched = num_sched[req_id]
            if sched <= 0:
                continue
            start = query_start[req_id]
            end = start + sched
            out.append(
                (
                    req_id,
                    _WriteChunk(
                        slots_cpu=slots_cpu[start:end],
                        tensors={k: v[start:end] for k, v in deferred_tensors.items()},
                        budget=ticket,
                    ),
                )
            )
        return out

    def _submit_step_writes(
        self,
        req_order: list[str],
        query_start: dict[str, int],
        num_sched: dict[str, int],
        device_snapshot: dict[str, torch.Tensor],
        budget: _BudgetTicket | None,
        slots_cpu: torch.Tensor,
        host_views: dict[str, torch.Tensor],
        freeze_event,
        staging_slot: int,
        step_d2h_event,
        bound_tids: list[int],
    ) -> list[WriteTask]:
        """Caller holds ``_state_lock``. Register one WriteTask per request;
        returns them for the caller to dispatch once the lock is released.

        Per-req views of the shared device snapshot: one on-device clone, req-scoped
        finish/abort, reassigned rows, and completion. Appends bound tids to
        `bound_tids` as it goes so a mid-loop raise still unwinds holders.
        """
        tasks: list[WriteTask] = []
        for req_id in req_order:
            start = query_start[req_id]
            end = start + num_sched[req_id]
            if end == start:
                continue
            tensors = {k: v[start:end] for k, v in device_snapshot.items()}
            tid = self._request_tasks.alloc_tid()
            chunk = _WriteChunk(slots_cpu=slots_cpu[start:end], tensors=tensors, budget=budget)
            # Host rows are views into the slot; the committer only waits
            # the shared step event. Device→host is already in flight.
            chunk.host = {k: v[start:end] for k, v in host_views.items()}
            task = WriteTask(
                tid=tid,
                req_id=req_id,
                write_n=self._request_tasks.increment_write_n(req_id),
                schedule=WriteSchedule.JOIN_NEXT_STEP,
                chunks=[chunk],
                freeze_event=freeze_event,
                staging_slot=staging_slot,
                step_d2h_event=step_d2h_event,
            )
            self._map_slots(slots_cpu[start:end], tid, tensors.keys())
            # Bind and pin before register: the slot must never be holder-free
            # while the task is live (released at its pool write).
            self._controller.staging_bind(staging_slot, StagingBufferHolder.for_task(tid))
            bound_tids.append(tid)
            if budget is not None:
                self._controller.pin_budget(budget, tid)
            self._controller.register(task)
            self._request_tasks.track(req_id, tid)
            self._join_next_step_tids.append(tid)
            tasks.append(task)
        return tasks

    def _stage_deferred(self, deferred_chunks: list[tuple[str, _WriteChunk]], freeze_event) -> None:
        """Caller holds ``_state_lock``. Register pre-built deferred `_WriteChunk`s
        (bytes already reserved by save_outputs).

        A new write (``write_n`` + 1) starts when the open task was closed by
        the budget flush, or when a chunk re-acquires a (key, slot) the task
        already lost — appending there would let its own reassigned tombstone
        drop the fresh rows at scatter. Hits read both through ``staged_list``.
        """
        for req_id, chunk in deferred_chunks:
            task = self._request_tasks.deferred.get(req_id)
            if task is not None and (
                task.reassigned_intersects(chunk)
                or self._controller.append_chunk(task, chunk, freeze_event) is not None
            ):
                task = None
            if task is None:
                task = WriteTask(
                    tid=self._request_tasks.alloc_tid(),
                    req_id=req_id,
                    write_n=self._request_tasks.increment_write_n(req_id),
                    schedule=WriteSchedule.JOIN_ON_FINISH,
                    chunks=[chunk],
                    freeze_event=freeze_event,
                )
                self._request_tasks.deferred[req_id] = task
                self._request_tasks.track(req_id, task.tid)
                self._controller.register(task, queued=False)
            if chunk.budget is not None:
                # Safe after register: queued=False keeps the task PENDING
                # until escalate, which only this thread calls.
                self._controller.pin_budget(chunk.budget, task.tid)
            # Block reuse across deferred tenants (preemption path) is
            # handled inside _map_slots: the old tenant's rows are skipped.
            self._map_slots(chunk.slots_cpu, task.tid, chunk.tensors.keys())

    # ----------------------------------------------------- occupancy

    def _map_slots(self, slots: torch.Tensor, tid: int, keys: Iterable[str]) -> None:
        """Caller holds ``_state_lock``. Record `tid` on these (slot, key);
        if another write still owns them, mark those rows skipped on it."""
        keys = tuple(keys)
        # Copy committed rows aside before this claim overwrites them.
        if self._pending_reads:
            self._preserve_for_pending_reads(slots, keys)
        for old, key, stolen in self._slot_status.map_slots(slots, tid, keys):
            old_task = self._controller.get_task(old)
            if old_task is not None:
                old_task.add_reassigned(key, stolen)

    def _preserve_for_pending_reads(self, new_slots: torch.Tensor, keys: tuple[str, ...]) -> None:
        """Caller holds ``_state_lock``. Copy still-valid committed rows into
        each pending ref before ``new_slots`` are claimed.

        Only COMMITTED rows at the reserved version need rescuing: the save
        barrier joins JOIN_NEXT_STEP writes to ``done`` and publish drains
        before claiming, and JOIN_ON_FINISH in-transit rows are read through
        ``staged_list`` task refs directly.
        """
        new_set = {int(s) for s in new_slots.tolist()}
        for ref in self._pending_reads:
            if ref.key not in keys or ref.reserved_version is None or not self._pool.has_key(ref.key):
                continue
            status = self._slot_status.get_slot_status(ref.key)
            take: list[tuple[int, int]] = []  # (row index in ref, slot)
            for i, s in enumerate(ref.slots.tolist()):
                s = int(s)
                if s not in new_set or s in ref.preserved:
                    continue
                committed = int(status.state[s]) == _Occupancy.COMMITTED
                if committed and int(status.slot_version[s]) == int(ref.reserved_version[i]):
                    take.append((i, s))
            if not take:
                continue
            rows = self._pool.rows(ref.key, torch.tensor([s for _, s in take], dtype=torch.long))
            for j, (_, s) in enumerate(take):
                ref.preserved[s] = rows[j].clone()

    @torch.inference_mode()
    def _commit_drained_writes(self) -> None:
        """Fold completed/failed writes into occupancy. Caller holds ``_state_lock``.

        A failed write leaves rows absent behind hashes vLLM already
        published — unservable and unrecoverable, so fatal. The fatality is
        sticky: the drained record is one-shot and the first raise can land
        in a swallowed path (a prefetch Future dropped by discard_step), so
        every later entry must keep raising rather than serve zeros.
        """
        if self._fatal_write_failure is not None:
            raise OmniPrefixCacheUnmatchError(self._fatal_write_failure)
        failed = self._controller.drain_failed()
        if failed:
            self._fatal_write_failure = (
                f"prefix cache write failed for task(s) {failed}; cached rows lost behind published hashes"
            )
            raise OmniPrefixCacheUnmatchError(self._fatal_write_failure)
        drained = self._controller.drain_completed()
        if drained:
            self._slot_status.commit(drained)
            self._request_tasks.drop_completed(drained)

    # --------------------------------------------------- consume-once

    def _take_step_ctx(self, step_id: int) -> _StepContext:
        """Pop the context for this step id (exactly once). Caller holds ``_state_lock``."""
        ctx = self._step_ctxs.pop(step_id, None)
        if ctx is None:
            raise OmniPrefixCacheUnmatchError(
                f"step context {step_id} missing (have {sorted(self._step_ctxs)}); already consumed or never saved"
            )
        return ctx

    def _release_step_staging_slot(self, slot: int, step_id: int) -> None:
        self._controller.staging_release(slot, StagingBufferHolder.for_step(step_id))

    def _release_step_staging(self, ctx: _StepContext, step_id: int) -> None:
        """Drop this step's staging hold. Does not require ``_state_lock``.

        materialize/discard: task holds leave at their pool write.
        """
        if ctx.d2h is not None:
            self._release_step_staging_slot(ctx.d2h.staging_slot, step_id)

    def _release_staging_on_failed_save(
        self, slot: int, step_holder: StagingBufferHolder, bound_tids: list[int]
    ) -> None:
        """save raised after claiming the slot: drop the step hold and any
        task holds. Does not require ``_state_lock``.
        """
        self._release_step_staging_slot(slot, step_holder.owner_id)
        for tid in bound_tids:
            self._controller.staging_release(slot, StagingBufferHolder.for_task(tid))

    # -------------------------------------------------- slot ref / fetch

    def _read_plan(self, slots: torch.Tensor, key: str, req_id: str) -> _ReadPlan:
        """Bind a read to producer/version/presence under ``_state_lock``.

        The returned object is a complete decision, not a promise to repeat
        occupancy lookup later.  Committed rows are copied now; pending rows
        retain their exact producer task.  Sparse MM absence is represented by
        zero rows, while hidden absence remains a contract error.
        """
        status = self._slot_status.get_slot_status(key)
        slots = slots.clone()
        versions = status.slot_version[slots].clone()
        presence = status.state[slots].clone()
        producers = status.tids[slots].clone()
        bindings = [
            _SlotBinding(int(slot), int(ver), int(tid) or None, _Presence(int(present)))
            for slot, ver, tid, present in zip(slots.tolist(), versions.tolist(), producers.tolist(), presence.tolist())
        ]
        read_id = self._next_read_id
        self._next_read_id += 1
        leases: list[tuple[int, StagingBufferHolder]] = []

        def unreadable(why: str) -> _ReadPlan:
            for slot, holder in leases:
                self._controller.staging_release(slot, holder)
            return _ReadPlan(
                read_id,
                slots,
                key,
                req_id,
                tuple(bindings),
                torch.empty(0),
                [],
                error=why,
            )

        unknown = sum(binding.presence is _Presence.UNKNOWN for binding in bindings)
        if unknown:
            return unreadable(f"{unknown} slots have unknown presence")
        if is_hidden_key(key) and any(binding.presence is _Presence.ABSENT for binding in bindings):
            return unreadable("explicitly absent slot")
        if not self._pool.has_key(key):
            return unreadable("key has no producer or committed storage")
        producer_masks: dict[int, torch.Tensor] = {}
        holder = StagingBufferHolder.for_read(read_id)
        try:
            producer_tasks: dict[int, WriteTask] = {}
            for index, binding in enumerate(bindings):
                if binding.presence is not _Presence.PENDING or binding.producer is None:
                    continue
                task = self._controller.get_task(binding.producer)
                if task is None:
                    return unreadable(f"producer {binding.producer} is no longer registered")
                producer_tasks[binding.producer] = task
                mask = producer_masks.setdefault(binding.producer, torch.zeros(slots.numel(), dtype=torch.bool))
                mask[index] = True
            for tid, task in producer_tasks.items():
                if task.schedule is WriteSchedule.JOIN_NEXT_STEP:
                    if not self._controller.staging_bind_read(task, holder):
                        if task.state is TaskState.FAILED:
                            return unreadable(f"producer {task.tid} failed before read binding")
                        # Task-holder release happens after the pool write. Take
                        # the durable snapshot below, after this atomic decision,
                        # rather than retaining a snapshot made before commit.
                        mask = producer_masks[tid]
                        for index in mask.nonzero(as_tuple=False).flatten().tolist():
                            binding = bindings[index]
                            bindings[index] = _SlotBinding(
                                binding.slot,
                                binding.version,
                                binding.producer,
                                _Presence.PRESENT,
                            )
                        mask.zero_()
                        continue
                    leases.append((task.staging_slot, holder))  # type: ignore[arg-type]
            # Read only stable durable rows. Pending producers may be writing
            # the same CPU tensor concurrently; even though fetch_host would
            # overlay those positions later, reading them here would race the
            # committer's index_copy_.
            stable = torch.tensor(
                [binding.presence is _Presence.PRESENT for binding in bindings],
                dtype=torch.bool,
            )
            stable_rows = self._pool.rows(key, slots[stable])
            resolved = torch.empty(
                (slots.numel(), stable_rows.shape[-1]),
                dtype=stable_rows.dtype,
            )
            if bool(stable.any()):
                resolved[stable] = stable_rows
        except BaseException:
            for slot, read_holder in leases:
                self._controller.staging_release(slot, read_holder)
            raise
        # Explicit sparse absence is not stale residency. Keep the row shape
        # from the fixed CPU mirror, but return a deterministic zero payload.
        absent = torch.tensor([binding.presence is _Presence.ABSENT for binding in bindings], dtype=torch.bool)
        if bool(absent.any()):
            resolved[absent] = 0
        producers_list = []
        for tid, mask in producer_masks.items():
            if bool(mask.any()):
                producers_list.append((producer_tasks[tid], mask))
        plan = _ReadPlan(read_id, slots, key, req_id, tuple(bindings), resolved, producers_list, leases)
        return plan

    def _close_read_plan(self, plan: _ReadPlan) -> None:
        with plan._close_lock:
            if plan._closed:
                return
            plan._closed = True
            for slot, holder in plan.leases:
                self._controller.staging_release(slot, holder)

    def _execute_read_plan(self, plan: _ReadPlan) -> torch.Tensor:
        """Resolve only the producer(s) captured by ``_read_plan``."""
        try:
            if plan.error is not None:
                _raise_unreadable_hit(plan.req_id, plan.key, plan.error)
            out = plan.resolved.clone()
            for task, mask in plan.producers:
                rows = self._controller.join([task.tid]) if task.schedule is WriteSchedule.JOIN_NEXT_STEP else None
                del rows
                out[mask] = self._controller.fetch_host(task, plan.slots[mask], plan.key)
            return out
        finally:
            self._close_read_plan(plan)

    def _get_hit_slots(self, hit_upto: int, hit_blocks: list[int]) -> torch.Tensor:
        """Prefix-hit block ids → KV slot ids. Alignment is checked at
        ``new_step_starts``. Does not require ``_state_lock``.
        """
        bs = self._config.block_size
        block_ids = torch.tensor(hit_blocks, dtype=torch.int64)
        return (block_ids.unsqueeze(1) * bs + torch.arange(bs)).reshape(-1)[:hit_upto]

    def _slot_ref(self, slots: torch.Tensor, key: str, req_id: str) -> _SlotRef:
        """Caller holds ``_state_lock``. Pin a ``_SlotRef`` for `slots` (no data movement).

        Rows still being written win over the CPU pool: they may not have
        landed yet, and a pool read would return zero/stale values.
        JOIN_NEXT_STEP tasks go in ``join_tids`` (wait-then-pool at
        fetch). JOIN_ON_FINISH tasks stay as refs for fetch_host.

        Hidden rejects any empty hole (prefetch skips; materialize
        raises). Other keys only need a source — holes fall to the pool.

        The ref is registered in ``_pending_reads`` so a write that later
        reuses one of these slots copies the old row aside first (COW). Reads
        are planned synchronously under the lock at hit time (prefetch runs at
        ``new_step_starts`` / publish), before the block can be reused, so this
        registration always precedes the reuse. ``_fetch_source`` unregisters.
        """
        status = self._slot_status.get_slot_status(key)
        states = status.state[slots]
        tids = status.tids[slots]
        reserved_version = status.slot_version[slots].clone()
        staged_mask = states == _Occupancy.IN_TRANSIT

        staged: list[tuple[WriteTask, torch.Tensor]] = []
        join_tids: list[int] = []
        for tid in {int(t) for t in tids[staged_mask].tolist()}:
            task = self._controller.get_task(tid) if tid != 0 else None
            if task is None:
                _raise_unreadable_hit(req_id, key, f"in-transit entry {tid} cannot serve them")
            if task.schedule is WriteSchedule.JOIN_NEXT_STEP:
                join_tids.append(task.tid)
            else:
                staged.append((task, staged_mask & (tids == tid)))

        already_staged = self._pool.has_key(key)
        has_source = already_staged or bool(staged) or bool(join_tids)
        if is_hidden_key(key):
            n_abs = int((states == _Occupancy.ABSENT).sum())
            if n_abs or not has_source:
                _raise_unreadable_hit(req_id, key, f"{n_abs} absent slots")
        ref = _SlotRef(
            slots=slots,
            key=key,
            req_id=req_id,
            already_staged=already_staged,
            staged_list=staged,
            join_tids=join_tids,
            reserved_version=reserved_version,
        )
        self._pending_reads.append(ref)
        return ref

    def _apply_preserved_rows(self, src: _SlotRef, out: torch.Tensor) -> torch.Tensor:
        """Write ``src.preserved`` rows over the matching slots in ``out``."""
        if not src.preserved:
            return out
        for i, s in enumerate(src.slots.tolist()):
            row = src.preserved.get(int(s))
            if row is not None:
                out[i] = row
        return out

    def _unregister_pending_read(self, src: _SlotRef) -> None:
        """Drop ``src`` from ``_pending_reads``. Identity, not dataclass eq
        (tensors in the ref make ``==`` unusable). Safe if already removed."""
        with self._state_lock:
            self._pending_reads = [ref for ref in self._pending_reads if ref is not src]

    def _fetch_source(self, src: _SlotRef) -> torch.Tensor:
        """Fetch a planned row source (execute phase, no lock).

        One key is one schedule: ``join_tids`` (JOIN_NEXT_STEP) and
        ``staged_list`` (JOIN_ON_FINISH) do not coexist. Immediate: wait
        ``done``, drain, read the pool. Deferred: pool rows already
        written, overlay ``fetch_host`` on the still-in-progress mask.
        Slots a later write reused are served from ``src.preserved``
        (copied under the lock before that write claimed them).
        """
        try:
            return self._fetch_source_inner(src)
        finally:
            self._unregister_pending_read(src)

    def _fetch_source_inner(self, src: _SlotRef) -> torch.Tensor:
        # For JOIN_NEXT_STEP, wait `done`, drain, read the pool
        if src.join_tids:
            self._controller.join(src.join_tids)
            with self._state_lock:
                self._commit_drained_writes()
            joined = self._apply_preserved_rows(src, self._pool.rows(src.key, src.slots))
            self._ensure_not_reassigned(
                src.slots,
                src.key,
                req_id=src.req_id,
                planned_version=src.reserved_version,
                preserved_slots=src.preserved,
            )
            return joined

        # For JOIN_ON_FINISH, pool rows already written, overlay `fetch_host`
        n = int(src.slots.numel())
        out: torch.Tensor | None = None
        if src.already_staged:
            out = self._pool.rows(src.key, src.slots)
        in_transit = None
        for task, mask in src.staged_list:
            try:
                rows = self._controller.fetch_host(task, src.slots[mask], src.key)
            except KeyError:
                _raise_unreadable_hit(
                    src.req_id,
                    src.key,
                    f"entry {task.tid} (req {task.req_id}, write_n {task.write_n}) cannot serve them",
                )
            if out is None:
                out = torch.zeros((n, rows.shape[-1]), dtype=rows.dtype)
            out[mask] = rows
            in_transit = mask if in_transit is None else in_transit | mask
        if out is not None:
            out = self._apply_preserved_rows(src, out)
        self._ensure_not_reassigned(
            src.slots,
            src.key,
            in_transit_mask=in_transit,
            req_id=src.req_id,
            planned_version=src.reserved_version,
            preserved_slots=src.preserved,
        )
        if out is None:
            _raise_unreadable_hit(src.req_id, src.key, "no source")
        return out

    def _ensure_not_reassigned(
        self,
        slots: torch.Tensor,
        key: str,
        *,
        in_transit_mask: torch.Tensor | None = None,
        req_id: str = "?",
        planned_version: torch.Tensor | None = None,
        preserved_slots: dict[int, torch.Tensor] | None = None,
    ) -> None:
        """Takes ``_state_lock``. Post-fetch check: rows read unlocked may
        have been given to a newer write between plan and read (block reuse).

        vLLM frees a request's blocks when it finishes or is aborted, one
        step before the terminal lifecycle event reaches us, and may reuse them at
        once. A planned ``_SlotRef`` is registered in ``_pending_reads``;
        the reusing write copy-on-writes those COMMITTED rows into
        ``preserved`` before it claims the slots. Those slots are safe.
        A version mismatch with no preserved copy means the pool rows are
        a newer tenant's — raise for live and finished alike, never serve
        them.

        ``planned_version`` is the per-slot version captured at plan time.
        ``in_transit_mask`` excludes JOIN_ON_FINISH slots this task itself
        still serves through ``fetch_host``.
        """
        with self._state_lock:
            status = self._slot_status.get_slot_status(key)
            if planned_version is not None:
                violated = status.slot_version[slots] != planned_version
            else:
                violated = status.state[slots] == _Occupancy.IN_TRANSIT
            if in_transit_mask is not None:
                violated &= ~in_transit_mask
            if preserved_slots:
                for i, s in enumerate(slots.tolist()):
                    if int(s) in preserved_slots:
                        violated[i] = False
            if not bool(violated.any()):
                return
            live = req_id in self._request_tasks.live_reqs
        _raise_unreadable_hit(
            req_id,
            key,
            f"{int(violated.sum())} hit slots reassigned before this delayed read "
            f"({'live' if live else 'finished'} req, block reuse)",
        )

    # ---------------------------------------------------------- merge

    def _merge_cached_for_req(
        self,
        ctx: _StepContext,
        req_id: str,
        key: str,
        current_cpu: torch.Tensor,
        hit_sources: dict[tuple[str, str], _ReadPlan | _SlotRef | Future],
    ) -> torch.Tensor:
        """Hit prefix + this step's rows for one (req, key).

        No hit → this step's slice only. Prefetch Future → write the
        slice into the reserved tail. Else cat(fetch, new).
        """
        start, end = ctx.spans[req_id]
        new_rows = current_cpu[start:end]
        src = hit_sources.get((req_id, key))
        if src is None:
            return new_rows
        if isinstance(src, Future):
            # Prefetched during the forward, prefix already in place; only
            # this step's rows land here. result() re-raises fetch/validation
            # errors — unread hits still raise after the thread hop.
            merged = src.result()
            merged[merged.shape[0] - new_rows.shape[0] :] = new_rows
            return merged
        cached = self._execute_read_plan(src) if isinstance(src, _ReadPlan) else self._fetch_source(src)
        return torch.cat([cached, new_rows], dim=0)

    def _merge_uncached_mm(
        self,
        ctx: _StepContext,
        req_ids: list[str],
        cached_keys: set[str],
        mm_out: dict[str, dict[str, Any]],
    ) -> None:
        """Write leftover mm that is not a pool key into mm_out.

        No hit concat: leftover mm was already copied to CPU at save
        (``ctx.mm_cpu_snapshot``). cached_keys already went through
        _merge_cached_for_req. ``req_ids`` is a subset of ``ctx.spans``.
        """
        leftover = {k: v for k, v in ctx.mm_cpu_snapshot.items() if k not in cached_keys and not is_hidden_key(k)}
        if not leftover:
            return
        from vllm_omni.utils.mm_outputs import to_payload_element

        order = list(ctx.spans)
        total_length = sum(e - s for s, e in ctx.spans.values())
        for key, val in leftover.items():
            per_req: dict[str, Any] = {}
            for req_id in req_ids:
                idx = order.index(req_id)
                start, end = ctx.spans[req_id]
                per_req[req_id] = to_payload_element(
                    val,
                    idx,
                    start=start,
                    end=end,
                    pass_lists_through=True,
                    seq_len=total_length,
                )
            mm_out[key] = per_req
