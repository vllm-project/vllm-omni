# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runs WriteTasks and the step device→host staging pool.

The manager owns request/slot identity and when to submit. This class
owns the staging pool, the GPU-byte budget, the copy queues, and the
single committer that writes into the CPU block pool.

Two device→host paths — the step path does not write `chunk.host` in
the committer:

    JOIN_NEXT_STEP   save already launched a whole-step device→host into
                     a staging slot and set `chunk.host` as views.
                     Committer waits that `step_d2h_event`, then writes
                     the pool.
    JOIN_ON_FINISH   committer copies the device clone → owned host
                     tensors, then writes the pool.

Async: high-priority then low-priority queues, then pool write.
Eager: submit() does wait+write inline (CPU tests / no CUDA).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal, NamedTuple

import torch

from vllm_omni.core.prefix_cache.block_pool import PrefixBlockPool
from vllm_omni.core.prefix_cache.interface import (
    OmniPrefixCacheStagingTimeoutError,
    OmniPrefixCacheUnmatchError,
    PrefixCacheConfig,
    ReqId,
    TensorName,
    Tid,
    WriteSchedule,
)

logger = logging.getLogger(__name__)


class StagingBufferHolder(NamedTuple):
    """One owner of a staging-buffer slot. The slot is free when none remain.

    Not a buffer state — concurrent owners share the same slot:
    - for_step: claimed at save, released when materialize/discard consumes the ctx
    - for_task: bound before WriteTask submit, released when that task completes
    """

    kind: Literal["step", "task"]
    owner_id: int

    @classmethod
    def for_step(cls, step_id: int) -> StagingBufferHolder:
        return cls("step", step_id)

    @classmethod
    def for_task(cls, tid: int) -> StagingBufferHolder:
        return cls("task", tid)


@dataclass
class _BudgetTicket:
    """GPU-byte-budget charge for one device clone shared by several writes.

    Charged once at ``reserve()``; freed when the last pinned tid releases.
    Immediate: the whole-step clone, pinned by that step's JOIN_NEXT_STEP
    tasks. Deferred: the shared deferred clone, pinned by every task a
    chunk of it was appended to. Does not store tensors.
    """

    nbytes: int
    tids: set[Tid] = field(default_factory=set)
    _freed: bool = False

    def release(self, tid: Tid) -> int:
        """Drop `tid`; return the bytes to uncharge (nbytes once, when the
        last holder leaves; 0 otherwise). Idempotent per tid."""
        self.tids.discard(tid)
        if self.tids or self._freed:
            return 0
        self._freed = True
        return self.nbytes


@dataclass
class _WriteChunk:
    """One save's slots + tensors appended onto a WriteTask.

    Controller identity is the parent ``tid``, not a request. Deferred
    writes grow by appending one chunk per save.
    """

    slots_cpu: torch.Tensor  # int64 flat row ids, in token order
    tensors: dict[TensorName, torch.Tensor]  # freeze (device or eager CPU)
    host: dict[TensorName, torch.Tensor] = field(default_factory=dict)  # host view of rows [0:n)
    # GPU-byte charge of the clone these tensors view; shared across the
    # chunks cut from one save. None for chunks with no device clone.
    budget: _BudgetTicket | None = None


class TaskState(Enum):
    """WriteTask lifecycle: a strict chain, plus FAILED from any non-terminal.

        PENDING -> QUEUED -> COPYING -> HOST_READY -> WRITTEN

    PENDING     registered, not on a copy queue (deferred: waiting for
                finish / budget pressure)
    QUEUED      on a copy queue
    COPYING     one thread owns the copy stage (staging: waits the step
                device→host; deferred: copies the device clone)
    HOST_READY  host rows ready (`host_ready` set); device clone dropped
    WRITTEN     in the CPU pool (`done` set)
    FAILED      committer could not finish (`host_ready` + `done` set;
                manager raises on next entry)

    Moves only through `WriteTask.transition`; skipping a step is illegal.
    """

    PENDING = auto()
    QUEUED = auto()
    COPYING = auto()
    HOST_READY = auto()
    WRITTEN = auto()
    FAILED = auto()

    @property
    def is_terminal(self) -> bool:
        return self in (TaskState.WRITTEN, TaskState.FAILED)


# The one legal forward move from each non-terminal state.
_CHAIN: dict[TaskState, TaskState] = {
    TaskState.PENDING: TaskState.QUEUED,
    TaskState.QUEUED: TaskState.COPYING,
    TaskState.COPYING: TaskState.HOST_READY,
    TaskState.HOST_READY: TaskState.WRITTEN,
}
_NOT_YET_COPYING = frozenset({TaskState.PENDING, TaskState.QUEUED})


def _is_legal_move(frm: TaskState, to: TaskState) -> bool:
    return (to is TaskState.FAILED and not frm.is_terminal) or _CHAIN.get(frm) is to


@dataclass
class WriteTask:
    """One write of (slot, key) rows for a single request.

    Identity: `tid` is the handle in the manager's (slot, key) tables.
    `req_id` + `write_n` mark whose write this is and the nth time that
    request opened a write. One write may cover several keys.

    `state` is the single source of truth for where the write is (see
    `TaskState`). `host_ready` / `done` are wait primitives set by the
    transitions into HOST_READY / WRITTEN / FAILED, never directly.

    How HOST_READY is reached:
    - JOIN_NEXT_STEP: `chunk.host` is a staging view set at save; the
      copy stage only waits `step_d2h_event`.
    - JOIN_ON_FINISH: the copy stage copies device→host into `chunk.host`.

    `JOIN_NEXT_STEP` is queued at submit. `JOIN_ON_FINISH` stays
    PENDING until finish or GPU-byte-budget pressure escalates it;
    budget flush takes the unfinished task with the oldest `enqueued_time`.

    Concurrent readers/writers:
    - Staging readers (materialize clone, committer pool write) all wait
      the same `step_d2h_event` before touching the view.
    - A later task taking the same (slot, key) records those rows in
      `reassigned`; the old pool write skips them.
    - `append_chunk` is refused once the copy stage is claimed; the caller
      opens a fresh task rather than mutating a closed one.
    - `lock` covers `state` / `reassigned` / `append_chunk` / host↔freeze /
      `slot_to_row`. `scatter_rows` snapshots `reassigned`.
    """

    tid: Tid
    req_id: ReqId
    write_n: int  # 1-based: nth write opened by this request
    schedule: WriteSchedule
    chunks: list[_WriteChunk]
    # On-device clone finished on the compute stream. Copy/read streams
    # must wait this before touching `chunks[].tensors`, or they can read
    # the next CUDA-graph static-buffer overwrite.
    freeze_event: object | None = None
    state: TaskState = TaskState.PENDING
    # Slots this write no longer owns (a newer write took them over).
    reassigned: dict[TensorName, torch.Tensor] = field(default_factory=dict)
    # Set on entering HOST_READY (or FAILED): this write's host rows are ready.
    host_ready: threading.Event = field(default_factory=threading.Event)
    # Set on entering WRITTEN (or FAILED): the CPU-pool write is over.
    done: threading.Event = field(default_factory=threading.Event)
    # Guards state / reassigned / append / host↔freeze.
    lock: threading.Lock = field(default_factory=threading.Lock)
    # time.monotonic() at submit; GPU-byte flush picks the oldest of these.
    enqueued_time: float = field(default_factory=time.monotonic)
    # Immediate path: this write's view into the shared step staging page.
    staging_slot: int | None = None
    # Immediate path: this step's CUDA device→host event (shared). None if deferred.
    step_d2h_event: object | None = None
    # slot -> (which chunk, row in that tensor). Built on demand
    # when a write has more than one `_WriteChunk`; pool write uses slot, the tensor uses row.
    _slot_to_row: dict[int, tuple[int, int]] | None = None

    def add_reassigned(self, key: str, slots: torch.Tensor) -> None:
        with self.lock:
            prev = self.reassigned.get(key)
            self.reassigned[key] = slots.clone() if prev is None else torch.cat([prev, slots])

    # ------------------------------------------------------------ lifecycle

    def transition(self, to: TaskState) -> None:
        """Move to `to`; raises unless it is the next chain step or FAILED."""
        with self.lock:
            self._transition_locked(to)

    def try_transition(self, to: TaskState) -> bool:
        """Move to `to` if legal from the current state. True if moved."""
        with self.lock:
            if not _is_legal_move(self.state, to):
                return False
            self._transition_locked(to)
            return True

    def _transition_locked(self, to: TaskState) -> None:
        if not _is_legal_move(self.state, to):
            raise OmniPrefixCacheUnmatchError(f"task {self.tid}: illegal transition {self.state.name} -> {to.name}")
        self.state = to
        if to is TaskState.HOST_READY:
            self.host_ready.set()
        elif to is TaskState.WRITTEN:
            self.done.set()
        elif to is TaskState.FAILED:
            self.host_ready.set()
            self.done.set()

    def claim_copy(self) -> bool:
        """QUEUED -> COPYING. Only one thread may run the copy stage; True if this caller won."""
        return self.try_transition(TaskState.COPYING)

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal

    def append_chunk(self, chunk: _WriteChunk, freeze_event: object | None = None) -> TaskState | None:
        """Grow this write with one save's rows. Returns None when appended,
        else the state that closed the task (COPYING or later).

        `freeze_event` is stored in the same snapshot: events on one compute
        stream are ordered, so the newest also covers every earlier clone.
        """
        with self.lock:
            if self.state not in _NOT_YET_COPYING:
                return self.state
            self.chunks.append(chunk)
            self._slot_to_row = None
            if freeze_event is not None:
                self.freeze_event = freeze_event
            return None

    def get_host_tensor(self, si: int, key: str) -> torch.Tensor | None:
        """`chunks[si]` host if written, else device freeze. One snapshot."""
        with self.lock:
            chunk = self.chunks[si]
            src = chunk.host.get(key)
            if src is None:
                src = chunk.tensors.get(key)
            return src

    def set_host_tensor(self, rows: list[tuple[_WriteChunk, str, torch.Tensor]]) -> None:
        """COPYING -> HOST_READY: write these host tensors, drop the device freeze."""
        with self.lock:
            for chunk, key, tensor in rows:
                chunk.host[key] = tensor
            self._clear_tensors()
            self._transition_locked(TaskState.HOST_READY)

    def mark_host_ready(self) -> None:
        """COPYING -> HOST_READY for staging views: wait the step device→host, drop freeze."""
        if self.step_d2h_event is not None:
            self.step_d2h_event.synchronize()
        with self.lock:
            self._clear_tensors()
            self._transition_locked(TaskState.HOST_READY)

    def mark_failed(self) -> bool:
        """-> FAILED from any non-terminal state; unblocks joiners. False if already terminal."""
        with self.lock:
            if self.state.is_terminal:
                return False
            self._clear_tensors()
            self._transition_locked(TaskState.FAILED)
            return True

    def mark_done(self) -> None:
        """HOST_READY -> WRITTEN."""
        self.transition(TaskState.WRITTEN)

    def scatter_rows(self) -> list[tuple[TensorName, torch.Tensor, torch.Tensor]]:
        """`(key, slots, host)` to write, one entry per key. Omits slots in
        `reassigned`. A slot written twice by this task (preempt + resume
        onto the same block) keeps the later chunk's row.
        """
        with self.lock:
            reassigned = {k: s.clone() for k, s in self.reassigned.items()}
        by_key: dict[TensorName, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        for chunk in self.chunks:
            for k, host in chunk.host.items():
                s, h = by_key.setdefault(k, ([], []))
                s.append(chunk.slots_cpu)
                h.append(host)
        out: list[tuple[TensorName, torch.Tensor, torch.Tensor]] = []
        for k, (s, h) in by_key.items():
            slots = s[0] if len(s) == 1 else torch.cat(s)
            host = h[0] if len(h) == 1 else torch.cat(h, dim=0)
            if len(s) > 1 and torch.unique(slots).numel() != slots.numel():
                last_pos = {int(slot): i for i, slot in enumerate(slots.tolist())}
                keep_idx = torch.tensor(sorted(last_pos.values()), dtype=torch.int64)
                slots, host = slots[keep_idx], host[keep_idx]
            taken = reassigned.get(k)
            if taken is not None and taken.numel():
                keep = ~torch.isin(slots, taken)
                if not bool(keep.any()):
                    continue
                slots, host = slots[keep], host[keep]
            out.append((k, slots, host))
        return out

    def clear_tensors(self) -> None:
        """Drop the device freeze. Host is unchanged (staging wait / fail)."""
        with self.lock:
            self._clear_tensors()

    def _clear_tensors(self) -> None:
        for chunk in self.chunks:
            chunk.tensors = {}

    def budget_tickets(self) -> list[_BudgetTicket]:
        """Distinct tickets across this task's chunks (by identity; one per save)."""
        out: dict[int, _BudgetTicket] = {}
        for chunk in self.chunks:
            if chunk.budget is not None:
                out.setdefault(id(chunk.budget), chunk.budget)
        return list(out.values())

    def slot_to_row(self) -> dict[int, tuple[int, int]]:
        with self.lock:
            if self._slot_to_row is None:
                m: dict[int, tuple[int, int]] = {}
                for si, chunk in enumerate(self.chunks):
                    for ri, slot in enumerate(chunk.slots_cpu.tolist()):
                        m[slot] = (si, ri)
                self._slot_to_row = m
            return self._slot_to_row


class StagingBufferPool:
    """Reusable pinned landing zone for ONE whole-step device→host at save.

    Per-task `chunk.host` is a row-range view into a slot, so the committer
    skips a per-task device→host. Slots recycle; this is not the CPU block pool.

    A slot stays busy while anyone still holds it: the step (until
    materialize/discard) and each immediate write that views the page
    (until its pool write). Prefix hits do not hold a slot —
    they wait for the pool write and read the durable pool.

    Saves with only leftover mm still claim a slot (empty views) so
    every step id shares this bound. A full pool waits; timeout then errors.
    """

    def __init__(self, depth: int, capacity: int):
        self.depth = depth
        self.capacity = capacity  # rows per slot; a larger step raises
        self._bufs: dict[TensorName, torch.Tensor] = {}  # [depth*capacity, width]
        self._busy: list[set[StagingBufferHolder]] = [set() for _ in range(depth)]
        self._slot_free_condition = threading.Condition()
        self._closed = False

    def _buf(self, key: str, width: int, dtype: torch.dtype, pin: bool) -> torch.Tensor:
        buf = self._bufs.get(key)
        if buf is None or buf.shape[-1] != width or buf.dtype != dtype:
            buf = torch.empty((self.depth * self.capacity, width), dtype=dtype, pin_memory=pin)
            self._bufs[key] = buf
        return buf

    def claim(self, holder: StagingBufferHolder, timeout: float) -> int:
        """Grab a free slot for `holder`. Waits until one is free, then
        times out. ``timeout<=0`` fails immediately if none are free.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        with self._slot_free_condition:
            while True:
                if self._closed:
                    raise OmniPrefixCacheUnmatchError("staging pool shut down while waiting for a slot")
                for slot in range(self.depth):
                    if not self._busy[slot]:
                        self._busy[slot].add(holder)
                        return slot
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OmniPrefixCacheStagingTimeoutError(f"timed out waiting for a staging slot after {timeout}s")
                self._slot_free_condition.wait(timeout=remaining)

    def bind(self, slot: int, holder: StagingBufferHolder) -> None:
        with self._slot_free_condition:
            self._busy[slot].add(holder)

    def release(self, slot: int, holder: StagingBufferHolder) -> None:
        with self._slot_free_condition:
            self._busy[slot].discard(holder)
            if not self._busy[slot]:
                self._slot_free_condition.notify()

    def close(self) -> None:
        with self._slot_free_condition:
            self._closed = True
            self._slot_free_condition.notify_all()

    def views(self, slot: int, key: str, n: int, width: int, dtype: torch.dtype, pin: bool) -> torch.Tensor:
        base = slot * self.capacity
        return self._buf(key, width, dtype, pin)[base : base + n]


@dataclass
class StepD2HClaim:
    """One whole-step landing in StagingBufferPool.

    Return of ``stage_step_host``. The manager stores this on
    ``_StepContext`` until materialize/discard releases the step holder.
    """

    staging_slot: int  # StagingBufferPool index
    views: dict[TensorName, torch.Tensor]  # host rows [0:n)
    event: object | None = None  # torch.cuda.Event; None on eager/CPU


class OmniPrefixCacheController:
    """Staging pool + committer. Step device→host is launched at save;
    this thread waits that event (JOIN_NEXT_STEP) or copies deferred
    rows (JOIN_ON_FINISH), then writes into the CPU pool.
    """

    def __init__(self, pool: PrefixBlockPool, config: PrefixCacheConfig, eager: bool | None = None):
        self._pool = pool
        self._config = config
        self._eager = (not torch.cuda.is_available()) if eager is None else eager
        self._tasks: dict[Tid, WriteTask] = {}
        self._completed: deque[Tid] = deque()  # pool write done, awaiting manager drain
        self._failed: deque[Tid] = deque()  # write failed; manager raises on next entry
        self._staged_bytes = 0
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._queue_hi: deque[Tid] = deque()  # JOIN_NEXT_STEP + forced JOIN_ON_FINISH
        self._queue_lo: deque[Tid] = deque()  # JOIN_ON_FINISH waiting for finish/budget
        self._blocked: list[Tid] = []  # host copy done, awaiting pool write
        self._shutdown = False
        self._copy_stream: torch.cuda.Stream | None = None
        self._read_stream: torch.cuda.Stream | None = None
        self._worker: threading.Thread | None = None
        self._staging_pool = StagingBufferPool(config.staging_depth, config.staging_capacity_tokens)
        if not self._eager:
            self._copy_stream = torch.cuda.Stream()
            self._read_stream = torch.cuda.Stream()
            self._worker = threading.Thread(target=self._worker_loop, name="omni-prefix-cache-committer", daemon=True)
            self._worker.start()

    # --------------------------------------------------------- step device→host staging

    def _d2h_on_stream(
        self,
        stream: torch.cuda.Stream,
        freeze_event: object | None,
        copy: Callable[[], None],
    ) -> torch.cuda.Event:
        """Issue device→host on `stream` after freeze; return the done event."""
        with torch.cuda.stream(stream):
            if freeze_event is not None:
                stream.wait_event(freeze_event)
            copy()
            ev = torch.cuda.Event()
            ev.record()
        return ev

    def stage_step_host(
        self, tensors: dict[str, torch.Tensor], n: int, freeze_event: object | None, step_holder: StagingBufferHolder
    ) -> StepD2HClaim:
        """Claim a staging slot and, when `tensors` is non-empty, launch
        ONE whole-step device→host into it.

        Leftover-only saves pass empty `tensors` and still take a slot
        (empty views) so every step id shares this bound. A full pool
        waits for materialize/discard; timeout then errors. A step
        larger than the page overflows the next slot — that is a config
        break. The caller binds tasks after submit; `step_holder` is
        released by materialize/discard via staging_release.
        """
        if tensors and n > self._staging_pool.capacity:
            raise OmniPrefixCacheUnmatchError(
                f"step has {n} tokens; staging capacity is {self._staging_pool.capacity} "
                "(size staging_capacity_tokens to max_num_batched_tokens)"
            )
        slot = self._staging_pool.claim(step_holder, self._config.staging_claim_timeout_s)
        try:
            pin = not self._eager
            views: dict[str, torch.Tensor] = {}
            event: object | None = None
            if self._eager or all(t.device.type == "cpu" for t in tensors.values()):
                for key, src in tensors.items():
                    v = self._staging_pool.views(slot, key, n, int(src.shape[-1]), src.dtype, pin)
                    v.copy_(src)
                    views[key] = v
            else:

                def _copy_to_staging() -> None:
                    for key, src in tensors.items():
                        v = self._staging_pool.views(slot, key, n, int(src.shape[-1]), src.dtype, pin)
                        v.copy_(src, non_blocking=True)
                        views[key] = v

                event = self._d2h_on_stream(self._copy_stream, freeze_event, _copy_to_staging)
            return StepD2HClaim(staging_slot=slot, views=views, event=event)
        except Exception:
            self._staging_pool.release(slot, step_holder)
            raise

    def staging_bind(self, slot: int, holder: StagingBufferHolder) -> None:
        self._staging_pool.bind(slot, holder)

    def staging_release(self, slot: int, holder: StagingBufferHolder) -> None:
        self._staging_pool.release(slot, holder)

    def _release_task_slot(self, task: WriteTask) -> None:
        """Drop the task's staging hold at a terminal state (pool write or failure)."""
        if task.staging_slot is not None:
            self._staging_pool.release(task.staging_slot, StagingBufferHolder.for_task(task.tid))

    def in_flight_tasks(self) -> int:
        """Registered tasks not yet drained (diagnostics only)."""
        return len(self._tasks)

    # ------------------------------------------------------------------ submit

    def submit(self, task: WriteTask, queued: bool = True) -> None:
        """Register a task. queued=False (deferred tasks) stays on the
        GPU clone until finish/abort or the GPU-byte budget forces a copy.

        Caller must reserve() the task bytes first (budget flush can
        block; the manager does that outside the state lock) and pin the
        task on its budget ticket(s) before submit.
        """
        task.enqueued_time = time.monotonic()
        with self._lock:
            self._tasks[task.tid] = task
            if queued:
                task.transition(TaskState.QUEUED)
        if self._eager:
            if queued:
                self._run_eager(task)
            return
        if queued:
            with self._wake:
                (self._queue_hi if task.schedule is WriteSchedule.JOIN_NEXT_STEP else self._queue_lo).append(task.tid)
                self._wake.notify_all()

    def append_chunk(self, task: WriteTask, chunk: _WriteChunk, freeze_event: object | None = None) -> TaskState | None:
        """Append to a pending task. None when appended, else the closing state."""
        return task.append_chunk(chunk, freeze_event)

    def pin_budget(self, ticket: _BudgetTicket, tid: int) -> None:
        """Record that ``tid`` holds a view of the clone ``ticket`` charges."""
        with self._lock:
            ticket.tids.add(tid)

    def reserve(self, nbytes: int, exclude: set[int] | None = None) -> None:
        """Reserve GPU-clone bytes; blocking flush happens here, so callers
        must not hold the manager's state lock."""
        self._reserve_bytes(nbytes, exclude=exclude)

    def _release_staged_bytes(self, task: WriteTask) -> None:
        """Drop this task's hold on its budget ticket(s). Idempotent; a
        ticket uncharges once, when its last holder leaves."""
        tickets = task.budget_tickets()
        with self._wake:
            for ticket in tickets:
                self._staged_bytes -= ticket.release(task.tid)

    def _reserve_bytes(self, nbytes: int, exclude: set[int] | None = None) -> None:
        # GPU-byte budget: force-copy oldest pending tasks until under
        # budget. Bounded wait: their device→host has usually long completed.
        exclude = exclude or set()
        while True:
            with self._lock:
                pending = [tid for tid, t in self._tasks.items() if not t.is_terminal and tid not in exclude]
                if self._staged_bytes + nbytes <= self._config.gpu_staging_bytes or not pending:
                    # Under budget or no pending tasks; admit reservation.
                    self._staged_bytes += nbytes
                    return
                oldest = min(pending, key=lambda tid: self._tasks[tid].enqueued_time)
            logger.warning("omni prefix cache: staging cap hit, force-flushing task %d", oldest)
            self.escalate([oldest])
            self.join([oldest])

    # ------------------------------------------------------------- lifecycle

    def escalate(self, tids: list[int]) -> None:
        """Move pending tasks to the front of the high-priority queue.

        PENDING -> QUEUED at the head; QUEUED on the low-priority queue
        moves up; QUEUED already high-priority is a no-op. COPYING and later
        belong to the worker and are untouched: the worker claims under
        `_wake` too, so a task cannot be popped and re-queued behind its back.
        """
        if self._eager:
            for tid in tids:
                task = self._tasks.get(tid)
                if task is not None and not task.is_terminal:
                    self._run_eager(task)
            return
        with self._wake:
            for tid in tids:
                task = self._tasks.get(tid)
                if task is None:
                    continue
                if task.try_transition(TaskState.QUEUED):
                    self._queue_hi.appendleft(tid)
                elif task.state is TaskState.QUEUED and tid in self._queue_lo:
                    self._queue_lo.remove(tid)
                    self._queue_hi.appendleft(tid)
            self._wake.notify_all()

    def join(self, tids: list[int]) -> None:
        """Block until each task has finished the CPU-pool write (or failed)."""
        for tid in tids:
            task = self._tasks.get(tid)
            if task is not None:
                task.done.wait()

    def join_host_ready(self, tids: list[int]) -> None:
        """Block until each task's device→host is complete (`host_ready`).

        Staging: committer has waited `step_d2h_event`. Deferred: committer
        has written `chunk.host`. Does not wait for the CPU-pool write.
        """
        for tid in tids:
            task = self._tasks.get(tid)
            if task is not None:
                task.host_ready.wait()

    def drain_completed(self) -> list[int]:
        """Pop pool-written tasks from `_completed` and drop them from `_tasks`.
        Staging holders were already released at the pool write."""
        out: list[int] = []
        with self._lock:
            while self._completed:
                out.append(self._completed.popleft())
            for tid in out:
                self._tasks.pop(tid, None)
        return out

    def drain_failed(self) -> list[int]:
        """Pop failed task ids from `_failed`. Does not drop `_tasks`."""
        out: list[int] = []
        with self._lock:
            while self._failed:
                out.append(self._failed.popleft())
        return out

    def get_task(self, tid: int) -> WriteTask | None:
        return self._tasks.get(tid)

    def shutdown(self) -> None:
        self._staging_pool.close()
        with self._wake:
            self._shutdown = True
            self._wake.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=5.0)

    # ------------------------------------------------------------ fetch_host

    @torch.inference_mode()
    def fetch_host(self, task: WriteTask, slots: torch.Tensor, key: str) -> torch.Tensor:
        """Rows for `slots` of one not-yet-done JOIN_ON_FINISH task.

        `_slot_ref` puts JOIN_NEXT_STEP tids in `join_tids` (wait then
        pool). This path reads committer-written `chunk.host`, or the
        device clone if that device→host has not landed.
        """
        if task.step_d2h_event is not None:
            task.step_d2h_event.synchronize()
        return self._rows_from(task, slots, key)

    def _rows_from(self, task: WriteTask, slots: torch.Tensor, key: str) -> torch.Tensor:
        """Map `slots` to rows across one or more `_WriteChunk`s; preserve caller order."""
        s2r = task.slot_to_row()
        idx = [s2r[int(s)] for s in slots.tolist()]
        parts: list[torch.Tensor] = []
        order: list[int] = []
        pos = 0
        rows_groups: dict[int, list[tuple[int, int]]] = {}
        for si, ri in idx:
            rows_groups.setdefault(si, []).append((pos, ri))
            pos += 1
        for si, items in rows_groups.items():
            src = task.get_host_tensor(si, key)
            if src is None:
                continue
            rows_idx = torch.tensor([ri for _, ri in items], dtype=torch.long)
            picked = self._slice_rows(task, src, rows_idx, host=(src.device.type == "cpu"))
            parts.append(picked)
            order.extend(p for p, _ in items)
        if not parts:
            raise KeyError(f"key {key} not present in task {task.tid}")
        cat = torch.cat(parts, dim=0)
        out = torch.empty_like(cat)
        out[torch.tensor(order, dtype=torch.long)] = cat
        return out

    def _slice_rows(self, task: WriteTask, src: torch.Tensor, rows_idx: torch.Tensor, host: bool) -> torch.Tensor:
        n = int(rows_idx.numel())
        # Ascending-run check without materializing an arange: endpoints plus
        # a monotonic diff are enough, and the common case is one long run.
        contiguous = (
            n > 0 and int(rows_idx[-1]) - int(rows_idx[0]) == n - 1 and (n < 2 or bool((rows_idx.diff() == 1).all()))
        )

        def _pick() -> torch.Tensor:
            return src[rows_idx[0] : rows_idx[0] + n] if contiguous else src.index_select(0, rows_idx)

        if host or src.device.type == "cpu":
            return _pick()
        if self._read_stream is None:
            return _pick().detach().cpu()
        copied: list[torch.Tensor] = []

        def _copy_to_cpu() -> None:
            copied.append(_pick().to("cpu", non_blocking=True))

        ev = self._d2h_on_stream(self._read_stream, task.freeze_event, _copy_to_cpu)
        ev.synchronize()
        return copied[0]

    # ------------------------------------------------------------ eager mode

    @torch.inference_mode()
    def _run_eager(self, task: WriteTask) -> None:
        """Eager stand-in for queue + worker: walk the whole chain inline."""
        task.try_transition(TaskState.QUEUED)  # no-op if submit already queued it
        if not task.claim_copy():
            # Copy stage already ran (a held stub, or a prior escalate); finish the write.
            if task.state is TaskState.HOST_READY:
                self._scatter(task)
            return
        if task.schedule is WriteSchedule.JOIN_NEXT_STEP:
            # Host is already a staging view (copied at save). Drop freeze.
            # `step_d2h_event` is None on CPU; `mark_host_ready` skips wait.
            task.mark_host_ready()
        else:
            # JOIN_ON_FINISH: no copy stream; freeze → owned host inline.
            task.set_host_tensor(
                [
                    (chunk, k, t.detach().cpu() if t.device.type != "cpu" else t.clone())
                    for chunk in task.chunks
                    for k, t in chunk.tensors.items()
                ]
            )
        self._release_staged_bytes(task)
        self._scatter(task)

    # ---------------------------------------------------------- worker loop

    def _worker_loop(self) -> None:
        # A dying committer would strand every join() forever, so the loop
        # never propagates: it fails the offending task and keeps serving.
        while True:
            tid = None
            try:
                task: WriteTask | None = None
                with self._wake:
                    while not self._shutdown and not self._queue_hi and not self._queue_lo:
                        if self._blocked:
                            break
                        # submit / escalate / shutdown all notify.
                        self._wake.wait()
                    if self._shutdown and not self._queue_hi and not self._queue_lo and not self._blocked:
                        return
                    if self._queue_hi:
                        tid = self._queue_hi.popleft()
                    elif self._queue_lo:
                        tid = self._queue_lo.popleft()
                    if tid is not None:
                        task = self._tasks.get(tid)
                        if task is not None:
                            # Claim in the same critical section as the pop:
                            # escalate holds `_wake` and sees COPYING.
                            task.transition(TaskState.COPYING)
                if task is not None:
                    self._copy_task(task)
                    with self._wake:
                        assert task.tid not in self._blocked, f"task {task.tid} reached HOST_READY twice"
                        self._blocked.append(task.tid)
                self._scatter_host_ready()
            except BaseException:
                logger.exception("omni prefix cache committer failed on task %s; releasing waiters", tid)
                self._fail_task(tid)

    @torch.inference_mode()
    def _copy_task(self, task: WriteTask) -> None:
        """COPYING -> HOST_READY. Staging: wait the save-time device→host event.
        Deferred: this is the device→host into owned `chunk.host` tensors.
        """
        if task.schedule is WriteSchedule.JOIN_NEXT_STEP:
            # `chunk.host` is already a staging view; device→host ran at save.
            # `mark_host_ready` waits `step_d2h_event` if one was recorded.
            # No per-task copy. Pool write is `_scatter_host_ready`.
            task.mark_host_ready()
            self._release_staged_bytes(task)
            return
        # One device cat + one device→host per key, not one per chunk (a
        # long request has one chunk per step).
        chunk_bytes = self._config.copy_chunk_bytes
        by_key: dict[str, list[_WriteChunk]] = {}
        for chunk in task.chunks:
            for k in chunk.tensors:
                by_key.setdefault(k, []).append(chunk)
        pending: list[tuple[str, list[_WriteChunk], list[torch.Tensor]]] = []
        with torch.cuda.stream(self._copy_stream):
            if task.freeze_event is not None:
                self._copy_stream.wait_event(task.freeze_event)
            for k, chunks in by_key.items():
                srcs = [c.tensors[k] for c in chunks]
                src = srcs[0] if len(srcs) == 1 else torch.cat(srcs, dim=0)
                rows = int(src.shape[0])
                row_bytes = max(1, src[:1].numel() * src.element_size())
                step = max(1, rows if rows * row_bytes <= chunk_bytes else chunk_bytes // row_bytes)
                parts = [src[s : s + step].to("cpu", non_blocking=True) for s in range(0, max(rows, 1), step)]
                pending.append((k, chunks, parts))
            ev = torch.cuda.Event()
            ev.record()
        ev.synchronize()
        rows_out: list[tuple[_WriteChunk, str, torch.Tensor]] = []
        for k, chunks, parts in pending:
            host = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            if len(chunks) == 1:
                rows_out.append((chunks[0], k, host))
                continue
            for chunk, view in zip(chunks, torch.split(host, [int(c.tensors[k].shape[0]) for c in chunks], dim=0)):
                rows_out.append((chunk, k, view))
        task.set_host_tensor(rows_out)
        self._release_staged_bytes(task)

    def _fail_task(self, tid: int | None) -> None:
        """-> FAILED: release waiters for a task the committer could not complete.

        No-op once the task is terminal. Budget release is idempotent, so a
        raise after a successful copy stage does not uncharge twice.
        """
        task = self._tasks.get(tid) if tid is not None else None
        if task is None or not task.mark_failed():
            return
        self._release_staged_bytes(task)
        with self._wake:
            if tid in self._blocked:
                self._blocked.remove(tid)
        self._release_task_slot(task)
        with self._lock:
            # Publish the failure: rows behind already-published block hashes
            # never landed, which the manager must raise on (hiding it would
            # crash on every future hit that touches these slots).
            self._failed.append(task.tid)

    @torch.inference_mode()
    def _scatter_host_ready(self) -> None:
        """HOST_READY -> WRITTEN for every `_blocked` task that has reached HOST_READY."""
        with self._wake:
            ready = [tid for tid in self._blocked if (t := self._tasks.get(tid)) and t.state is TaskState.HOST_READY]
            for tid in ready:
                self._blocked.remove(tid)
        for tid in ready:
            task = self._tasks[tid]
            try:
                self._scatter(task)
            except BaseException:
                # Attribute the failure to THIS task: letting it propagate
                # would fail whichever entry the worker loop happened to be
                # copying, double-release its bytes, and strand this one's
                # join() forever.
                logger.exception("omni prefix cache scatter failed on task %s; releasing waiters", tid)
                self._fail_task(tid)

    @torch.inference_mode()
    def _scatter(self, task: WriteTask) -> None:
        for key, slots, host in task.scatter_rows():
            self._pool.write(key, slots, host)
        task.mark_done()
        # This write was the staging page's last reader. Release before
        # publishing so a join() that sees `done` never sees a busy slot.
        self._release_task_slot(task)
        with self._lock:
            self._completed.append(task.tid)
