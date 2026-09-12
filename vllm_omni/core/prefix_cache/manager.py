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
                        save waits `host_ready` only.
    JOIN_ON_FINISH      deferred mm. Stays on the device clone; the
                        committer does that device→host, then writes the
                        pool. Forced onto the high-priority queue on
                        finish/abort or GPU-byte-budget pressure.

Per real scheduler_output, engine-thread order:

    new_step_starts   before _update_states drops finished requests
                      (register hits, start prefix prefetch)
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
from typing import TYPE_CHECKING, Any, NamedTuple, NoReturn

import torch

from vllm_omni.core.prefix_cache.block_pool import PrefixBlockPool
from vllm_omni.core.prefix_cache.controller import (
    OmniPrefixCacheController,
    StagingBufferHolder,
    StepD2HClaim,
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

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_omni.core.prefix_cache.group_view import FullAttentionGroupView

logger = logging.getLogger(__name__)


class _Occupancy(IntEnum):
    ABSENT = 0
    IN_TRANSIT = 1
    COMMITTED = 2


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
) -> tuple[dict[str, Any], object | None]:
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
    leftover_event: object | None = None
    immediate_budget: _BudgetTicket | None = None

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


@dataclass
class _SlotRef:
    """Where one (req, key) span's slots live: planned under the lock, fetched outside it.

    Schedule split, not a single read tier:
    - JOIN_NEXT_STEP in-transit: ``join_tids``. Fetch waits ``done``,
      drains, then reads the pool. Staging views are never sliced.
    - JOIN_ON_FINISH in-transit: ``staged_list`` task refs. Fetch uses
      ``fetch_host`` (device freeze / committer host).
    - Already in the CPU pool: ``already_staged`` → pool.

    A JOIN_NEXT_STEP task may disappear between plan and join (another
    entry already published it into the pool); ``join`` no-ops and the
    pool rows persist.
    """

    slots: torch.Tensor  # KV slot ids (PrefixBlockPool rows)
    key: TensorName
    req_id: ReqId
    already_staged: bool  # this key is already in the CPU pool
    staged_list: list[tuple[WriteTask, torch.Tensor]]  # JOIN_ON_FINISH only
    join_tids: list[Tid] = field(default_factory=list)  # JOIN_NEXT_STEP in-transit


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

    # Hits snapshotted at new_step_starts. Prefetch fills [hit | empty tail]
    # during forward; materialize writes the tail.
    hits: dict[ReqId, tuple[int, list[int] | None]]  # (hit_upto, blocks)
    hit_prefetch: dict[ReqId, dict[TensorName, Future]] = field(default_factory=dict)

    # Key split frozen at save: recompute at materialize races ensure_key.
    cached_keys: set[TensorName] = field(default_factory=set)
    # Leftover mm copied to CPU at save (this-step deferred rows + mm
    # that is not written to the pool).
    mm_cpu_snapshot: dict[TensorName, Any] = field(default_factory=dict)
    # Wait this before reading mm_cpu_snapshot (None when nothing was on CUDA).
    mm_cpu_snapshot_event: object | None = None

    # Staging slot for this step id (empty views when only leftover mm).
    d2h: StepD2HClaim | None = None


class _SlotStatus(NamedTuple):
    """Occupancy row for one tensor name (views into the table, not copies)."""

    state: torch.Tensor  # int8[num_slots]
    tids: torch.Tensor  # Tid per kv slot; 0 = none


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
        self.state: dict[TensorName, torch.Tensor] = {}  # int8[num_slots]
        self.tids: dict[TensorName, torch.Tensor] = {}  # Tid per kv slot; 0 = none
        self.task_slots: dict[Tid, torch.Tensor] = {}  # kv slots
        self.task_keys: dict[Tid, tuple[TensorName, ...]] = {}

    def init_table(self, key: TensorName) -> None:
        """Allocate occupancy tensors for ``key`` if they do not exist."""
        if key in self.state:
            return
        self.state[key] = torch.zeros(self.num_slots, dtype=torch.int8)
        self.tids[key] = torch.zeros(self.num_slots, dtype=torch.int64)

    def get_slot_status(self, key: TensorName) -> _SlotStatus:
        """Occupancy tensors for ``key``. ``init_table`` must have run."""
        return _SlotStatus(state=self.state[key], tids=self.tids[key])

    def map_slots(
        self, slots: torch.Tensor, tid: Tid, keys: Iterable[TensorName]
    ) -> list[tuple[Tid, TensorName, torch.Tensor]]:
        """Record ``tid`` on these (slot, key). Return in-transit rows
        another write still owned (caller marks them skipped on it)."""
        keys = tuple(keys)
        stolen: list[tuple[Tid, TensorName, torch.Tensor]] = []
        for key in keys:
            status = self.get_slot_status(key)
            cur = status.tids[slots]
            stale = (status.state[slots] == _Occupancy.IN_TRANSIT) & (cur != tid) & (cur != 0)
            if bool(stale.any()):
                for old in {int(o) for o in cur[stale].tolist()}:
                    stolen.append((old, key, slots[stale & (cur == old)]))
            status.state[slots] = _Occupancy.IN_TRANSIT
            status.tids[slots] = tid
        prev = self.task_slots.get(tid)
        if prev is None:
            self.task_slots[tid] = slots
            self.task_keys[tid] = keys
        else:
            # Deferred tasks grow one `_WriteChunk` per step.
            self.task_slots[tid] = torch.cat([prev, slots])
            self.task_keys[tid] = tuple(dict.fromkeys(self.task_keys[tid] + keys))
        return stolen

    def commit(self, tids: Iterable[Tid]) -> None:
        """Flip still-owned slots to COMMITTED and drop the reverse index."""
        for tid in tids:
            slots = self.task_slots.pop(tid, None)
            keys = self.task_keys.pop(tid, ())
            if slots is None:
                continue
            for key in keys:
                status = self.get_slot_status(key)
                still_ours = status.tids[slots] == tid
                idx = slots[still_ours]
                status.state[idx] = _Occupancy.COMMITTED
                status.tids[idx] = 0


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
        view: FullAttentionGroupView,
        *,
        eager: bool | None = None,
    ):
        self._config = config
        self._view = view
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
        self._hit_prefetch: dict[ReqId, dict[TensorName, Future]] = {}

        # Prefix gather during forward (CPU work releases the GIL).
        # One worker: complete in submit order; pop finished work from the head.
        self._prefetch_pool = ThreadPoolExecutor(1, thread_name_prefix="omni-prefix-cache-prefetch")
        self._prefetch_queue: deque[tuple[Future, _SlotRef]] = deque()

        # One snapshot per step id; consume with materialize or discard_step.
        self._next_step_id: StepId = 1
        self._step_ctxs: dict[StepId, _StepContext] = {}

    # ------------------------------------------------------ public entries

    def register_policy(self, policy: ModelCachePolicy) -> None:
        self._policy = policy
        if (hk := policy.hidden_key) is not None:
            self._slot_status.init_table(hk)

    @_locked
    @torch.inference_mode()
    def new_step_starts(self, scheduler_output: SchedulerOutput) -> None:
        """Handle one scheduler_output.

        Engine thread only; before _update_states removes finished
        requests; exactly once per real step. Registers new-request prefix
        hits (copying their block tables) and forces finished/aborted
        requests' still-open writes onto the high-priority copy queue —
        a block hash that entered the batch must land in the cache,
        abort included.
        """
        # 1. Publish writes the committer has already written into the pool.
        self._commit_drained_writes()

        # 2. Finished/aborted reqs: force their still-open deferred writes
        #    onto the high-priority copy queue now. The next save waits
        #    join_host_ready. (Not leftover_mm — those are this-step reads.)
        finished = getattr(scheduler_output, "finished_req_ids", None) or ()
        for req_id in finished:
            tids, dtask = self._request_tasks.finish(req_id)
            if dtask is not None:
                tids.add(dtask.tid)
            pending_tasks = [tid for tid in tids if self._controller.get_task(tid) is not None]
            if pending_tasks:
                # Abort too: those block hashes are already in vLLM.
                # Dropping the write would leave future hits ABSENT.
                self._controller.escalate(pending_tasks)
                self._join_finished_tids.update(pending_tasks)

        # 3. Copy this arrival's prefix-hit block ids. scheduled_new_reqs
        #    is the only place they appear; after _update_states they sit
        #    on the live request and grow as decode allocates more blocks.
        #    materialize (async builder) must not reread that live table.
        self._clear_hit_infos()
        for new_req in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            req_id = new_req.req_id
            if req_id in self._request_tasks.live_reqs:
                # Streaming continuation (async_chunk): the id is already
                # live, so num_computed_tokens is its own earlier work, not
                # a cache hit. No hit marking (parity with legacy); a
                # delivered_upto span is Phase 2. Preempt+resume never
                # comes through here (scheduled_cached_reqs), see the
                # design doc.
                continue
            self._request_tasks.live_reqs.add(req_id)
            num_computed = int(getattr(new_req, "num_computed_tokens", 0) or 0)
            if num_computed > 0:
                # block_ids is per-kv-group; group 0 only.
                blocks = getattr(new_req, "block_ids", None)
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

        # 4. Gather those spans on the prefetch thread; overlaps this forward.
        while self._prefetch_queue and self._prefetch_queue[0][0].done():
            self._prefetch_queue.popleft()
        self._cur_num_scheduled = dict(scheduler_output.num_scheduled_tokens)
        if self._hit_spans:
            self._prefetch_hit_spans()

    @torch.inference_mode()
    def save_outputs(
        self,
        hidden_states: torch.Tensor | None,
        mm_outputs: dict[str, Any] | None,
        *,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
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
        req_order = self._view.batch_req_ids()
        num_sched = {r: int(self._cur_num_scheduled.get(r, 0)) for r in req_order}
        query_start: dict[str, int] = {}
        current_start_idx = 0
        for req_id in req_order:
            query_start[req_id] = current_start_idx
            current_start_idx += num_sched[req_id]

        slots_cpu: torch.Tensor | None = None
        mm_outputs = mm_outputs or {}
        freeze_event = None

        # 3. Slot map, then split into immediate / deferred / leftover.
        if num_tokens_unpadded > 0:
            # Derive the slot mapping on CPU: reading the device one back
            # would need a stream sync that waits on the whole forward.
            slots_cpu = self._view.step_slots_cpu(req_order, num_sched)
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
            # Reserve may block on a flush: outside the lock. The flush must
            # not close the deferred entries we are about to append to
            # (main-thread-only reads, safe unlocked).
            exclude = {
                self._request_tasks.deferred[r].tid
                for r, _ in step_outputs.deferred_chunks
                if r in self._request_tasks.deferred
            }
            self._controller.reserve(step_outputs.budget_bytes(), exclude=exclude)

        # 5. Claim a staging slot (unlocked), optional device→host into it,
        #    then submit + store the step snapshot (locked). Saves with only
        #    leftover mm still claim. Full pool waits; timeout lists unused step ids.
        d2h_claim: StepD2HClaim | None = None
        step_holder = StagingBufferHolder.for_step(self._next_step_id)
        transferred = False
        bound_tids: list[int] = []
        try:
            d2h_claim = self._stage_step_host(step_outputs.immediate, num_tokens_unpadded, freeze_event, step_holder)

            step_id = self._publish_saved_step(
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
            transferred = True
            return step_id
        finally:
            # Slot claim is outside the lock; a later raise must release
            # the step and any task that already bound this slot.
            if not transferred and d2h_claim is not None:
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
                # live batch, which the contract forbids (debug assert, not a
                # fallback that serves a miss).
                assert set(req_ids) <= set(ctx.spans), (
                    f"materialize(step {step_id}) got req ids outside the save snapshot: "
                    f"{sorted(set(req_ids) - set(ctx.spans))[:8]}"
                )

                cached_keys = ctx.cached_keys

                hit_sources: dict[tuple[str, str], _SlotRef | Future] = {}
                for req_id in req_ids:
                    hit = ctx.hits.get(req_id)
                    if not hit:
                        continue
                    hit_upto, hit_blocks = hit
                    prefetched = ctx.hit_prefetch.get(req_id, {})
                    slots = self._get_hit_slots(hit_upto, hit_blocks)
                    keys = self._policy.get_hit_keys(cached_keys)
                    for key in keys:
                        fut = prefetched.get(key)
                        if fut is not None:
                            hit_sources[(req_id, key)] = fut
                            continue
                        hit_sources[(req_id, key)] = self._slot_ref(slots, key, req_id)

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

    @_locked
    def discard_step(self, step_id: int) -> None:
        """Consume the step context when nothing will materialize it.

        Any thread; same exactly-once contract as materialize (unknown or
        duplicate id raises). Only the read-side snapshot is dropped —
        the cache write proceeds unchanged.
        """
        ctx = self._take_step_ctx(step_id)
        self._release_step_staging(ctx, step_id)

    def shutdown(self) -> None:
        self._prefetch_pool.shutdown(wait=False, cancel_futures=True)
        self._prefetch_queue.clear()
        self._controller.shutdown()

    # ------------------------------------------------------ new_step

    def _clear_hit_infos(self) -> None:
        """Drop the live hit / prefetch tables. Caller holds ``_state_lock``."""
        self._hit_spans.clear()
        self._hit_prefetch.clear()

    def _prefetch_hit_spans(self) -> None:
        """Caller holds ``_state_lock``. Plan each hit span and gather it on
        the prefetch thread, overlapping the forward. A span that fails to
        plan — same-step hits resolve rows this step's save has not
        registered yet — is left to materialize, which raises if unread.
        """
        keys = self._policy.get_hit_keys(self._pool.keys())
        for req_id, (hit_upto, hit_blocks) in self._hit_spans.items():
            n_new = int(self._cur_num_scheduled.get(req_id, 0))
            slots = self._get_hit_slots(hit_upto, hit_blocks)
            futs: dict[str, Future] = {}
            for key in keys:
                try:
                    src = self._slot_ref(slots, key, req_id)
                except OmniPrefixCacheUnmatchError:
                    continue
                fut = self._prefetch_pool.submit(self._prefetch_hit, src, n_new)
                self._prefetch_queue.append((fut, src))
                futs[key] = fut
            if futs:
                self._hit_prefetch[req_id] = futs

    @torch.inference_mode()
    def _prefetch_hit(self, src: _SlotRef, n_new: int) -> torch.Tensor:
        """Prefetch thread: gather the hit span and pre-build the merged
        buffer with the prefix filled. materialize writes only this step's
        rows at the tail — the gather AND the prefix copy both happen while
        the forward runs, and the cat leaves the critical path."""
        rows = self._fetch_source(src)
        out = torch.empty((rows.shape[0] + n_new, rows.shape[-1]), dtype=rows.dtype)
        out[: rows.shape[0]] = rows
        return out

    # ---------------------------------------------------------- save

    def _stage_step_host(
        self,
        device_snapshot: dict[str, torch.Tensor],
        num_tokens_unpadded: int,
        freeze_event: object | None,
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
        """Pop last step's join worklists, then wait ``host_ready`` unlocked.

        Lock covers only the pop. ``join_host_ready`` may block on device→host.
        """
        with self._state_lock:
            join_ids = list(self._join_finished_tids)
            join_ids.extend(self._join_next_step_tids)
            self._join_finished_tids.clear()
            self._join_next_step_tids.clear()
        if join_ids:
            self._controller.join_host_ready(join_ids)

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
        freeze_event: object | None,
        d2h_claim: StepD2HClaim | None,
        bound_tids: list[int],
    ) -> StepId:
        """Takes ``_state_lock``. Submit this step's writes and store the
        consume-once snapshot. Copies live hits into the snapshot, then
        clears them. Device→host and GPU-byte-budget flush stay outside.
        """
        self._commit_drained_writes()
        if step_outputs.immediate:
            self._submit_step_writes(
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
        step_id = self._next_step_id
        self._next_step_id += 1
        self._step_ctxs[step_id] = _StepContext(
            spans={r: (query_start[r], query_start[r] + num_sched[r]) for r in req_order},
            num_tokens_unpadded=num_tokens_unpadded,
            hits=dict(self._hit_spans),
            hit_prefetch=dict(self._hit_prefetch),
            cached_keys=without_hidden(self._pool.keys()) & mm_keys,
            mm_cpu_snapshot=step_outputs.leftover,
            mm_cpu_snapshot_event=step_outputs.leftover_event,
            d2h=d2h_claim,
        )
        self._clear_hit_infos()
        return step_id

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
        if n > 0:
            if hidden_states is not None and (hk := self._policy.hidden_key) is not None:
                if hidden_states.ndim < 2 or hidden_states.shape[0] < n:
                    rows = 0 if hidden_states.ndim < 2 else int(hidden_states.shape[0])
                    raise OmniPrefixCacheUnmatchError(f"hidden_states has {rows} rows, need {n}")
                self._ensure_cache_key(hk, hidden_states.dtype, int(hidden_states.shape[-1]))
                immediate[hk] = hidden_states[:n].clone()
            for key, val in mm_outputs.items():
                is_step_rows = _is_step_token_tensor(val, n, num_tokens_padded)
                if key in self._policy.deferred_keys:
                    if is_step_rows:
                        if not self._pool.has_key(key):
                            self._ensure_cache_key(key, val.dtype, int(val.shape[-1]))
                        deferred_tensors[key] = val[:n].clone()
                    continue
                if self._policy.skip_immediate_mm(key) or not is_step_rows:
                    continue
                self._ensure_cache_key(key, val.dtype, int(val.shape[-1]))
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
    ) -> None:
        """Caller holds ``_state_lock``. One queued WriteTask per request.

        Per-req views of the shared device snapshot: one on-device clone, req-scoped
        finish/abort, reassigned rows, and completion. Appends bound tids to
        `bound_tids` as it goes so a mid-loop raise still unwinds holders.
        """
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
            # Bind and pin before submit: the slot must never be holder-free
            # while the task is live (released at its pool write), and the
            # committer may reach HOST_READY before this loop returns.
            self._controller.staging_bind(staging_slot, StagingBufferHolder.for_task(tid))
            bound_tids.append(tid)
            if budget is not None:
                self._controller.pin_budget(budget, tid)
            self._controller.submit(task)
            self._request_tasks.track(req_id, tid)
            self._join_next_step_tids.append(tid)

    def _stage_deferred(self, deferred_chunks: list[tuple[str, _WriteChunk]], freeze_event) -> None:
        """Caller holds ``_state_lock``. Register pre-built deferred `_WriteChunk`s
        (bytes already reserved by save_outputs)."""
        for req_id, chunk in deferred_chunks:
            task = self._request_tasks.deferred.get(req_id)
            if task is not None:
                closed = self._controller.append_chunk(task, chunk, freeze_event)
                if closed is not None:
                    # Unreachable by construction: finish removes the entry
                    # before the next save, and reserve() excludes this step's
                    # entries from the budget flush. Recover with a new task,
                    # but say so — it means one of those orderings broke.
                    logger.warning(
                        "omni prefix cache: deferred write for req %s (tid %d, write_n %d) was %s "
                        "before this step's rows were appended; opening a new write",
                        req_id,
                        task.tid,
                        task.write_n,
                        closed.name,
                    )
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
                self._controller.submit(task, queued=False)
            if chunk.budget is not None:
                # Safe after submit: queued=False keeps the task PENDING
                # until escalate, which only this thread calls.
                self._controller.pin_budget(chunk.budget, task.tid)
            # Block reuse across deferred tenants (preemption path) is
            # handled inside _map_slots: the old tenant's rows are skipped.
            self._map_slots(chunk.slots_cpu, task.tid, chunk.tensors.keys())

    # ----------------------------------------------------- occupancy

    def _ensure_cache_key(self, key: TensorName, dtype: torch.dtype, feat: int) -> None:
        """Open the pool storage and the occupancy row for ``key``."""
        self._pool.ensure_key(key, dtype, feat)
        self._slot_status.init_table(key)

    def _map_slots(self, slots: torch.Tensor, tid: int, keys: Iterable[str]) -> None:
        """Caller holds ``_state_lock``. Record `tid` on these (slot, key);
        if another write still owns them, mark those rows skipped on it."""
        for old, key, stolen in self._slot_status.map_slots(slots, tid, keys):
            old_task = self._controller.get_task(old)
            if old_task is not None:
                old_task.add_reassigned(key, stolen)

    @torch.inference_mode()
    def _commit_drained_writes(self) -> None:
        """Fold completed/failed writes into occupancy. Caller holds ``_state_lock``."""
        failed = self._controller.drain_failed()
        if failed:
            # A failed write leaves rows absent behind hashes vLLM already
            # published — unservable and unrecoverable, so fatal. Raise here,
            # once, at the earliest public entry instead of leaving every
            # future hit that touches these slots unreadable.
            raise OmniPrefixCacheUnmatchError(
                f"prefix cache write failed for task(s) {failed}; cached rows lost behind published hashes"
            )
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
        """
        status = self._slot_status.get_slot_status(key)
        states = status.state[slots]
        tids = status.tids[slots]
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
        return _SlotRef(
            slots=slots,
            key=key,
            req_id=req_id,
            already_staged=already_staged,
            staged_list=staged,
            join_tids=join_tids,
        )

    def _fetch_source(self, src: _SlotRef) -> torch.Tensor:
        """Fetch a planned row source (execute phase, no lock).

        One key is one schedule: ``join_tids`` (JOIN_NEXT_STEP) and
        ``staged_list`` (JOIN_ON_FINISH) do not coexist. Immediate: wait
        ``done``, drain, read the pool. Deferred: pool rows already
        written, overlay ``fetch_host`` on the still-in-progress mask.
        """
        # For JOIN_NEXT_STEP, wait `done`, drain, read the pool
        if src.join_tids:
            self._controller.join(src.join_tids)
            with self._state_lock:
                self._commit_drained_writes()
            out = self._pool.rows(src.key, src.slots)
            self._ensure_not_reassigned(src.slots, src.key, req_id=src.req_id)
            return out

        # For JOIN_ON_FINISH, pool rows already written, overlay `fetch_host` on the still-in-progress mask
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
        self._ensure_not_reassigned(src.slots, src.key, in_transit_mask=in_transit, req_id=src.req_id)
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
    ) -> None:
        """Takes ``_state_lock``. Post-fetch check: pool rows read unlocked
        may have been given to a newer write mid-read (block reuse). A torn
        pool read must raise. JOIN_ON_FINISH slots already in-transit at
        plan time are excluded; JOIN_NEXT_STEP slots must be COMMITTED
        after the wait-then-publish.
        """
        with self._state_lock:
            status = self._slot_status.get_slot_status(key)
            violated = status.state[slots] == _Occupancy.IN_TRANSIT
            if in_transit_mask is not None:
                violated &= ~in_transit_mask
            if bool(violated.any()):
                _raise_unreadable_hit(req_id, key, f"reassigned during materialize ({int(violated.sum())} slots)")

    # ---------------------------------------------------------- merge

    def _merge_cached_for_req(
        self,
        ctx: _StepContext,
        req_id: str,
        key: str,
        current_cpu: torch.Tensor,
        hit_sources: dict[tuple[str, str], _SlotRef | Future],
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
        cached = self._fetch_source(src)
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
